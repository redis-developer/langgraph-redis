from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.constants import TASKS
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from langgraph.checkpoint.redis.version import __lib_name__, __version__


class RedisSaver(BaseRedisSaver[Redis, SearchIndex]):
    """Standard Redis implementation for checkpoint saving."""

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

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
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
            if checkpoint_id := get_checkpoint_id(config):
                filter_expression.append(Tag("checkpoint_id") == checkpoint_id)

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
                "checkpoint_id",
                "parent_checkpoint_id",
                "$.checkpoint",
                "$.metadata",
            ],
            num_results=limit or 10000,
        )

        # Execute the query
        results = self.checkpoints_index.search(query)

        # Process the results
        for doc in results.docs:
            thread_id = str(getattr(doc, "thread_id", ""))
            checkpoint_ns = str(getattr(doc, "checkpoint_ns", ""))
            checkpoint_id = str(getattr(doc, "checkpoint_id", ""))

            # Fetch channel_values
            channel_values = self.get_channel_values(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )

            # Fetch pending_sends from parent checkpoint
            pending_sends = []
            if doc["parent_checkpoint_id"]:
                pending_sends = self._load_pending_sends(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    parent_checkpoint_id=doc["parent_checkpoint_id"],
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

            config_param: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": doc["checkpoint_id"],
                }
            }

            checkpoint_param = self._load_checkpoint(
                doc["$.checkpoint"],
                channel_values,
                pending_sends,
            )

            pending_writes = self._load_pending_writes(
                thread_id, checkpoint_ns, checkpoint_id
            )

            yield CheckpointTuple(
                config=config_param,
                checkpoint=checkpoint_param,
                metadata=metadata,
                parent_config=None,
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

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
            "parent_checkpoint_id": checkpoint_id,
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
                BaseRedisSaver._make_redis_checkpoint_key(
                    thread_id, checkpoint_ns, checkpoint["id"]
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = str(get_checkpoint_id(config))
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            checkpoint_filter_expression = (
                (Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
                & (Tag("checkpoint_id") == checkpoint_id)
            )
        else:
            checkpoint_filter_expression = (Tag("thread_id") == thread_id) & (
                Tag("checkpoint_ns") == checkpoint_ns
            )

        # Construct the query
        checkpoints_query = FilterQuery(
            filter_expression=checkpoint_filter_expression,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "$.checkpoint",
                "$.metadata",
            ],
            num_results=1,
        )
        checkpoints_query.sort_by("checkpoint_id", asc=False)

        # Execute the query
        results = self.checkpoints_index.search(checkpoints_query)
        if not results.docs:
            return None

        doc = results.docs[0]

        # Fetch channel_values
        channel_values = self.get_channel_values(
            thread_id=doc["thread_id"],
            checkpoint_ns=doc["checkpoint_ns"],
            checkpoint_id=doc["checkpoint_id"],
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = []
        if doc["parent_checkpoint_id"]:
            pending_sends = self._load_pending_sends(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                parent_checkpoint_id=doc["parent_checkpoint_id"],
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
                "checkpoint_id": doc["checkpoint_id"],
            }
        }

        checkpoint_param = self._load_checkpoint(
            doc["$.checkpoint"],
            channel_values,
            pending_sends,
        )

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
            parent_config=None,
            pending_writes=pending_writes,
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> Iterator[RedisSaver]:
        """Create a new RedisSaver instance."""
        saver: Optional[RedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
            )

            yield saver
        finally:
            if saver and saver._owns_its_client:  # Ensure saver is not None
                saver._redis.close()
                saver._redis.connection_pool.disconnect()

    def get_channel_values(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = ""
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
        parent_checkpoint_id: str,
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
            & (Tag("checkpoint_id") == parent_checkpoint_id)
            & (Tag("channel") == TASKS),
            return_fields=["type", "blob", "task_path", "task_id", "idx"],
            num_results=100,  # Adjust as needed
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


__all__ = [
    "__version__",
    "__lib_name__",
    "RedisSaver",
    "AsyncRedisSaver",
    "BaseRedisSaver",
    "ShallowRedisSaver",
    "AsyncShallowRedisSaver",
]
