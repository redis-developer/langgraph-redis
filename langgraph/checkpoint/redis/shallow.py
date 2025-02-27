from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.constants import TASKS
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)

SCHEMAS = [
    {
        "index": {
            "name": "checkpoints",
            "prefix": CHECKPOINT_PREFIX + REDIS_KEY_SEPARATOR,
            "storage_type": "json",
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_ns", "type": "tag"},
            {"name": "source", "type": "tag"},
            {"name": "step", "type": "numeric"},
        ],
    },
    {
        "index": {
            "name": "checkpoints_blobs",
            "prefix": CHECKPOINT_BLOB_PREFIX + REDIS_KEY_SEPARATOR,
            "storage_type": "json",
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_ns", "type": "tag"},
            {"name": "channel", "type": "tag"},
            {"name": "type", "type": "tag"},
        ],
    },
    {
        "index": {
            "name": "checkpoint_writes",
            "prefix": CHECKPOINT_WRITE_PREFIX + REDIS_KEY_SEPARATOR,
            "storage_type": "json",
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_ns", "type": "tag"},
            {"name": "checkpoint_id", "type": "tag"},
            {"name": "task_id", "type": "tag"},
            {"name": "idx", "type": "numeric"},
            {"name": "channel", "type": "tag"},
            {"name": "type", "type": "tag"},
        ],
    },
]


class ShallowRedisSaver(BaseRedisSaver[Redis, SearchIndex]):
    """Redis implementation that only stores the most recent checkpoint."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> Iterator[ShallowRedisSaver]:
        """Create a new ShallowRedisSaver instance."""
        saver: Optional[ShallowRedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
            )
            yield saver
        finally:
            if saver and saver._owns_its_client:
                saver._redis.close()
                saver._redis.connection_pool.disconnect()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store only the latest checkpoint."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # Store checkpoint data
        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "checkpoint": self._dump_checkpoint(copy),
            "metadata": self._dump_metadata(metadata),
        }

        # store at top-level for filters in list()
        if all(key in metadata for key in ["source", "step"]):
            checkpoint_data["source"] = metadata["source"]
            checkpoint_data["step"] = metadata["step"]

        self.checkpoints_index.load(
            [checkpoint_data],
            keys=[
                ShallowRedisSaver._make_shallow_redis_checkpoint_key(
                    thread_id, checkpoint_ns
                )
            ],
        )

        # Store blob values
        blobs = self._dump_blobs(
            thread_id,
            checkpoint_ns,
            copy.get("channel_values", {}),
            new_versions,
        )

        if blobs:
            # Unzip the list of tuples into separate lists for keys and data
            keys, data = zip(*blobs)
            self.checkpoint_blobs_index.load(list(data), keys=list(keys))

        return next_config

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Redis."""
        # Construct the filter expression
        filter_expression = []
        if config:
            filter_expression.append(
                Tag("thread_id") == config["configurable"]["thread_id"]
            )
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                filter_expression.append(Tag("checkpoint_ns") == checkpoint_ns)

        if filter:
            for k, v in filter.items():
                if k == "source":
                    filter_expression.append(Tag("source") == v)
                elif k == "step":
                    filter_expression.append(Num("step") == v)
                else:
                    raise ValueError(f"Unsupported filter key: {k}")

        # if before:
        #     filter_expression.append(Tag("checkpoint_id") < get_checkpoint_id(before))

        # Combine all filter expressions
        combined_filter = filter_expression[0] if filter_expression else "*"
        for expr in filter_expression[1:]:
            combined_filter &= expr

        # Construct the Redis query
        query = FilterQuery(
            filter_expression=combined_filter,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "$.checkpoint",
                "$.metadata",
            ],
            num_results=limit or 10000,
        )

        # Execute the query
        results = self.checkpoints_index.search(query)

        # Process the results
        for doc in results.docs:
            thread_id = cast(str, getattr(doc, "thread_id", ""))
            checkpoint_ns = cast(str, getattr(doc, "checkpoint_ns", ""))
            checkpoint = json.loads(doc["$.checkpoint"])

            # Fetch channel_values
            channel_values = self.get_channel_values(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint["id"],
            )

            # Fetch pending_sends from parent checkpoint
            pending_sends = self._load_pending_sends(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )

            # Fetch and parse metadata
            raw_metadata = getattr(doc, "$.metadata", "{}")
            metadata_dict = (
                json.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else raw_metadata
            )

            # Ensure metadata matches CheckpointMetadata type
            sanitized_metadata = {
                k.replace("\u0000", ""): (
                    v.replace("\u0000", "") if isinstance(v, str) else v
                )
                for k, v in metadata_dict.items()
            }
            metadata = cast(CheckpointMetadata, sanitized_metadata)

            checkpoint_param = self._load_checkpoint(
                doc["$.checkpoint"],
                channel_values,
                pending_sends,
            )

            config_param: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_param["id"],
                }
            }

            pending_writes = self._load_pending_writes(
                thread_id, checkpoint_ns, checkpoint_param["id"]
            )

            yield CheckpointTuple(
                config=config_param,
                checkpoint=checkpoint_param,
                metadata=metadata,
                parent_config=None,
                pending_writes=pending_writes,
            )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_filter_expression = (Tag("thread_id") == thread_id) & (
            Tag("checkpoint_ns") == checkpoint_ns
        )

        # Construct the query
        checkpoints_query = FilterQuery(
            filter_expression=checkpoint_filter_expression,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "parent_checkpoint_id",
                "$.checkpoint",
                "$.metadata",
            ],
            num_results=1,
        )

        # Execute the query
        results = self.checkpoints_index.search(checkpoints_query)
        if not results.docs:
            return None

        doc = results.docs[0]

        checkpoint = json.loads(doc["$.checkpoint"])

        # Fetch channel_values
        channel_values = self.get_channel_values(
            thread_id=doc["thread_id"],
            checkpoint_ns=doc["checkpoint_ns"],
            checkpoint_id=checkpoint["id"],
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = self._load_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
        )

        # Fetch and parse metadata
        raw_metadata = getattr(doc, "$.metadata", "{}")
        metadata_dict = (
            json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        )

        # Ensure metadata matches CheckpointMetadata type
        sanitized_metadata = {
            k.replace("\u0000", ""): (
                v.replace("\u0000", "") if isinstance(v, str) else v
            )
            for k, v in metadata_dict.items()
        }
        metadata = cast(CheckpointMetadata, sanitized_metadata)

        config_param: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        checkpoint_param = self._load_checkpoint(
            doc["$.checkpoint"],
            channel_values,
            pending_sends,
        )

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_param["id"]
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
            parent_config=None,
            pending_writes=pending_writes,
        )

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None
        self._redis = redis_client or RedisConnectionFactory.get_redis_connection(
            redis_url, **connection_args
        )

    def create_indexes(self) -> None:
        self.checkpoints_index = SearchIndex.from_dict(
            self.SCHEMAS[0], redis_client=self._redis
        )
        self.checkpoint_blobs_index = SearchIndex.from_dict(
            self.SCHEMAS[1], redis_client=self._redis
        )
        self.checkpoint_writes_index = SearchIndex.from_dict(
            self.SCHEMAS[2], redis_client=self._redis
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Optional path info for the task.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Transform writes into appropriate format
        writes_objects = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            write_obj = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "blob": blob,
            }
            writes_objects.append(write_obj)

        # For each write, check existence and then perform appropriate operation
        with self._redis.json().pipeline(transaction=False) as pipeline:
            for write_obj in writes_objects:
                key = self._make_redis_checkpoint_writes_key(
                    thread_id, checkpoint_ns, checkpoint_id, task_id, write_obj["idx"]
                )

                # First check if key exists
                key_exists = self._redis.exists(key) == 1

                if all(w[0] in WRITES_IDX_MAP for w in writes):
                    # UPSERT case - only update specific fields
                    if key_exists:
                        # Update only channel, type, and blob fields
                        pipeline.set(key, "$.channel", write_obj["channel"])
                        pipeline.set(key, "$.type", write_obj["type"])
                        pipeline.set(key, "$.blob", write_obj["blob"])
                    else:
                        # For new records, set the complete object
                        pipeline.set(key, "$", write_obj)  # type: ignore[arg-type]
                else:
                    # INSERT case
                    pipeline.set(key, "$", write_obj)  # type: ignore[arg-type]

            pipeline.execute()

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> List[Tuple[str, dict[str, Any]]]:
        if not versions:
            return []

        return [
            (
                ShallowRedisSaver._make_shallow_redis_checkpoint_blob_key(
                    thread_id, checkpoint_ns, k
                ),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": k,
                    "type": (
                        self._get_type_and_blob(values[k])[0]
                        if k in values
                        else "empty"
                    ),
                    "blob": (
                        self._get_type_and_blob(values[k])[1] if k in values else None
                    ),
                },
            )
            for k, ver in versions.items()
        ]

    def get_channel_values(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> dict[str, Any]:
        """Retrieve channel_values dictionary with properly constructed message objects."""
        checkpoint_query = FilterQuery(
            filter_expression=(Tag("thread_id") == thread_id)
            & (Tag("checkpoint_ns") == checkpoint_ns)
            & (Tag("checkpoint_id") == checkpoint_id),
            return_fields=["$.checkpoint.channel_versions"],
            num_results=1,
        )

        checkpoint_result = self.checkpoints_index.search(checkpoint_query)
        if not checkpoint_result.docs:
            return {}

        channel_versions = json.loads(
            getattr(checkpoint_result.docs[0], "$.checkpoint.channel_versions", "{}")
        )
        if not channel_versions:
            return {}

        channel_values = {}
        for channel, version in channel_versions.items():
            blob_query = FilterQuery(
                filter_expression=(Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
                & (Tag("channel") == channel)
                & (Tag("version") == version),
                return_fields=["type", "$.blob"],
                num_results=1,
            )

            blob_results = self.checkpoint_blobs_index.search(blob_query)
            if blob_results.docs:
                blob_doc = blob_results.docs[0]
                blob_type = blob_doc.type
                blob_data = getattr(blob_doc, "$.blob", None)

                if blob_data and blob_type != "empty":
                    channel_values[channel] = self.serde.loads_typed(
                        (blob_type, blob_data)
                    )

        return channel_values

    def _load_pending_sends(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> List[Tuple[str, bytes]]:
        """Load pending sends for a parent checkpoint.

        Args:
            thread_id: The thread ID
            checkpoint_ns: The checkpoint namespace
            parent_checkpoint_id: The ID of the parent checkpoint

        Returns:
            List of (type, blob) tuples representing pending sends
        """
        # Query checkpoint_writes for parent checkpoint's TASKS channel
        parent_writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == thread_id)
            & (Tag("checkpoint_ns") == checkpoint_ns)
            & (Tag("channel") == TASKS),
            return_fields=["type", "blob", "task_path", "task_id", "idx"],
            num_results=100,
        )
        parent_writes_results = self.checkpoint_writes_index.search(parent_writes_query)

        # Sort results by task_path, task_id, idx (matching Postgres implementation)
        sorted_writes = sorted(
            parent_writes_results.docs,
            key=lambda x: (
                getattr(x, "task_path", ""),
                getattr(x, "task_id", ""),
                getattr(x, "idx", 0),
            ),
        )

        # Extract type and blob pairs
        return [(doc.type, doc.blob) for doc in sorted_writes]

    @staticmethod
    def _make_shallow_redis_checkpoint_key(thread_id: str, checkpoint_ns: str) -> str:
        return REDIS_KEY_SEPARATOR.join([CHECKPOINT_PREFIX, thread_id, checkpoint_ns])

    @staticmethod
    def _make_shallow_redis_checkpoint_blob_key(
        thread_id: str, checkpoint_ns: str, channel: str
    ) -> str:
        return REDIS_KEY_SEPARATOR.join(
            [CHECKPOINT_BLOB_PREFIX, thread_id, checkpoint_ns, channel]
        )
