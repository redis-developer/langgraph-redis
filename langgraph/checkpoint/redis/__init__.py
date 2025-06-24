from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

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
from redis.cluster import RedisCluster
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from langgraph.checkpoint.redis.util import (
    EMPTY_ID_SENTINEL,
    from_storage_safe_id,
    from_storage_safe_str,
    to_storage_safe_id,
    to_storage_safe_str,
)
from langgraph.checkpoint.redis.version import __lib_name__, __version__

logger = logging.getLogger(__name__)


class RedisSaver(BaseRedisSaver[Union[Redis, RedisCluster], SearchIndex]):
    """Standard Redis implementation for checkpoint saving."""

    _redis: Union[Redis, RedisCluster]  # Support both standalone and cluster clients
    # Whether to assume the Redis server is a cluster; None triggers auto-detection
    cluster_mode: Optional[bool] = None

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
        )

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
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

    def setup(self) -> None:
        """Initialize the indices in Redis and detect cluster mode."""
        self._detect_cluster_mode()
        super().setup()

    def _detect_cluster_mode(self) -> None:
        """Detect if the Redis client is a cluster client by inspecting its class."""
        if self.cluster_mode is not None:
            logger.info(
                f"Redis cluster_mode explicitly set to {self.cluster_mode}, skipping detection."
            )
            return

        # Determine cluster mode based on client class
        if isinstance(self._redis, RedisCluster):
            logger.info("Redis client is a cluster client")
            self.cluster_mode = True
        else:
            logger.info("Redis client is a standalone client")
            self.cluster_mode = False

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
                Tag("thread_id")
                == to_storage_safe_id(config["configurable"]["thread_id"])
            )

            # Reproducing the logic from the Postgres implementation, we'll
            # search for checkpoints with any namespace, including an empty
            # string, while `checkpoint_id` has to have a value.
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                filter_expression.append(
                    Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns)
                )
            if checkpoint_id := get_checkpoint_id(config):
                filter_expression.append(
                    Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)
                )

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
            thread_id = from_storage_safe_id(doc["thread_id"])
            checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
            checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
            parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])

            # Fetch channel_values
            channel_values = self.get_channel_values(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )

            # Fetch pending_sends from parent checkpoint
            pending_sends = []
            if parent_checkpoint_id:
                pending_sends = self._load_pending_sends(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    parent_checkpoint_id=parent_checkpoint_id,
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
                    "checkpoint_id": checkpoint_id,
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
        thread_ts = configurable.pop("thread_ts", "")
        checkpoint_id = (
            configurable.pop("checkpoint_id", configurable.pop("thread_ts", ""))
            or thread_ts
        )

        # For values we store in Redis, we need to convert empty strings to the
        # sentinel value.
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        copy = checkpoint.copy()
        # When we return the config, we need to preserve empty strings that
        # were passed in, instead of the sentinel value.
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Store checkpoint data.
        checkpoint_data = {
            "thread_id": storage_safe_thread_id,
            "checkpoint_ns": storage_safe_checkpoint_ns,
            "checkpoint_id": storage_safe_checkpoint_id,
            "parent_checkpoint_id": storage_safe_checkpoint_id,
            "checkpoint": self._dump_checkpoint(copy),
            "metadata": self._dump_metadata(metadata),
        }

        # store at top-level for filters in list()
        if all(key in metadata for key in ["source", "step"]):
            checkpoint_data["source"] = metadata["source"]
            checkpoint_data["step"] = metadata["step"]

        # Create the checkpoint key
        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            storage_safe_thread_id,
            storage_safe_checkpoint_ns,
            storage_safe_checkpoint_id,
        )

        self.checkpoints_index.load(
            [checkpoint_data],
            keys=[checkpoint_key],
        )

        # Store blob values.
        blobs = self._dump_blobs(
            storage_safe_thread_id,
            storage_safe_checkpoint_ns,
            copy.get("channel_values", {}),
            new_versions,
        )

        blob_keys = []
        if blobs:
            # Unzip the list of tuples into separate lists for keys and data
            keys, data = zip(*blobs)
            blob_keys = list(keys)
            self.checkpoint_blobs_index.load(list(data), keys=blob_keys)

        # Apply TTL to checkpoint and blob keys if configured
        if self.ttl_config and "default_ttl" in self.ttl_config:
            self._apply_ttl_to_keys(checkpoint_key, blob_keys)

        return next_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        ascending = True

        if checkpoint_id and checkpoint_id != EMPTY_ID_SENTINEL:
            checkpoint_filter_expression = (
                (Tag("thread_id") == to_storage_safe_id(thread_id))
                & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
                & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id))
            )
        else:
            checkpoint_filter_expression = (
                Tag("thread_id") == to_storage_safe_id(thread_id)
            ) & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            ascending = False

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
        checkpoints_query.sort_by("checkpoint_id", asc=ascending)

        # Execute the query
        results = self.checkpoints_index.search(checkpoints_query)
        if not results.docs:
            return None

        doc = results.docs[0]
        doc_thread_id = from_storage_safe_id(doc["thread_id"])
        doc_checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
        doc_checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
        doc_parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])

        # If refresh_on_read is enabled, refresh TTL for checkpoint key and related keys
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            # Get the checkpoint key
            checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
                to_storage_safe_id(doc_thread_id),
                to_storage_safe_str(doc_checkpoint_ns),
                to_storage_safe_id(doc_checkpoint_id),
            )

            # Get all blob keys related to this checkpoint
            from langgraph.checkpoint.redis.base import (
                CHECKPOINT_BLOB_PREFIX,
                CHECKPOINT_WRITE_PREFIX,
            )

            # Get the blob keys using search index instead of keys()
            blob_query = FilterQuery(
                filter_expression=(
                    Tag("thread_id") == to_storage_safe_id(doc_thread_id)
                )
                & (Tag("checkpoint_ns") == to_storage_safe_str(doc_checkpoint_ns)),
                return_fields=["key"],  # Assuming the key field exists in the index
                num_results=1000,
            )
            blob_results = self.checkpoint_blobs_index.search(blob_query)
            blob_keys = [
                f"{CHECKPOINT_BLOB_PREFIX}:{to_storage_safe_id(doc_thread_id)}:{to_storage_safe_str(doc_checkpoint_ns)}:{getattr(doc, 'channel', '')}:{getattr(doc, 'version', '')}"
                for doc in blob_results.docs
            ]

            # Get checkpoint write keys using search index
            write_query = FilterQuery(
                filter_expression=(
                    Tag("thread_id") == to_storage_safe_id(doc_thread_id)
                )
                & (Tag("checkpoint_ns") == to_storage_safe_str(doc_checkpoint_ns))
                & (Tag("checkpoint_id") == to_storage_safe_id(doc_checkpoint_id)),
                return_fields=["task_id", "idx"],
                num_results=1000,
            )
            write_results = self.checkpoint_writes_index.search(write_query)
            write_keys = [
                BaseRedisSaver._make_redis_checkpoint_writes_key(
                    to_storage_safe_id(doc_thread_id),
                    to_storage_safe_str(doc_checkpoint_ns),
                    to_storage_safe_id(doc_checkpoint_id),
                    getattr(doc, "task_id", ""),
                    getattr(doc, "idx", 0),
                )
                for doc in write_results.docs
            ]

            # Apply TTL to checkpoint, blob keys, and write keys
            all_related_keys = blob_keys + write_keys
            self._apply_ttl_to_keys(checkpoint_key, all_related_keys)

        # Fetch channel_values
        channel_values = self.get_channel_values(
            thread_id=doc_thread_id,
            checkpoint_ns=doc_checkpoint_ns,
            checkpoint_id=doc_checkpoint_id,
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = []
        if doc_parent_checkpoint_id:
            pending_sends = self._load_pending_sends(
                thread_id=doc_thread_id,
                checkpoint_ns=doc_checkpoint_ns,
                parent_checkpoint_id=doc_parent_checkpoint_id,
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
                "checkpoint_id": doc_checkpoint_id,
            }
        }

        checkpoint_param = self._load_checkpoint(
            doc["$.checkpoint"],
            channel_values,
            pending_sends,
        )

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, doc_checkpoint_id
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
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> Iterator[RedisSaver]:
        """Create a new RedisSaver instance."""
        saver: Optional[RedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
                ttl=ttl,
            )

            yield saver
        finally:
            if saver and saver._owns_its_client:  # Ensure saver is not None
                saver._redis.close()
                saver._redis.connection_pool.disconnect()

    def get_channel_values(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = ""
    ) -> Dict[str, Any]:
        """Retrieve channel_values dictionary with properly constructed message objects."""
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        checkpoint_query = FilterQuery(
            filter_expression=(Tag("thread_id") == storage_safe_thread_id)
            & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
            & (Tag("checkpoint_id") == storage_safe_checkpoint_id),
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
                filter_expression=(Tag("thread_id") == storage_safe_thread_id)
                & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
                & (Tag("channel") == channel)
                & (Tag("version") == version),
                return_fields=["type", "$.blob"],
                num_results=1,
            )

            blob_results = self.checkpoint_blobs_index.search(blob_query)
            if blob_results.docs:
                blob_doc = blob_results.docs[0]
                blob_type = getattr(blob_doc, "type", None)
                blob_data = getattr(blob_doc, "$.blob", None)

                if blob_data and blob_type and blob_type != "empty":
                    # Ensure blob_data is bytes for deserialization
                    if isinstance(blob_data, str):
                        blob_data = blob_data.encode("utf-8")
                    channel_values[channel] = self.serde.loads_typed(
                        (str(blob_type), blob_data)
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
        storage_safe_thread_id = to_storage_safe_str(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_parent_checkpoint_id = to_storage_safe_str(parent_checkpoint_id)

        # Query checkpoint_writes for parent checkpoint's TASKS channel
        parent_writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == storage_safe_thread_id)
            & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
            & (Tag("checkpoint_id") == storage_safe_parent_checkpoint_id)
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

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        storage_safe_thread_id = to_storage_safe_id(thread_id)

        # Delete all checkpoints for this thread
        checkpoint_query = FilterQuery(
            filter_expression=Tag("thread_id") == storage_safe_thread_id,
            return_fields=["checkpoint_ns", "checkpoint_id"],
            num_results=10000,  # Get all checkpoints for this thread
        )

        checkpoint_results = self.checkpoints_index.search(checkpoint_query)

        # Collect all keys to delete
        keys_to_delete = []

        for doc in checkpoint_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")

            # Delete checkpoint key
            checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
                storage_safe_thread_id, checkpoint_ns, checkpoint_id
            )
            keys_to_delete.append(checkpoint_key)

        # Delete all blobs for this thread
        blob_query = FilterQuery(
            filter_expression=Tag("thread_id") == storage_safe_thread_id,
            return_fields=["checkpoint_ns", "channel", "version"],
            num_results=10000,
        )

        blob_results = self.checkpoint_blobs_index.search(blob_query)

        for doc in blob_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            channel = getattr(doc, "channel", "")
            version = getattr(doc, "version", "")

            blob_key = BaseRedisSaver._make_redis_checkpoint_blob_key(
                storage_safe_thread_id, checkpoint_ns, channel, version
            )
            keys_to_delete.append(blob_key)

        # Delete all writes for this thread
        writes_query = FilterQuery(
            filter_expression=Tag("thread_id") == storage_safe_thread_id,
            return_fields=["checkpoint_ns", "checkpoint_id", "task_id", "idx"],
            num_results=10000,
        )

        writes_results = self.checkpoint_writes_index.search(writes_query)

        for doc in writes_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")
            task_id = getattr(doc, "task_id", "")
            idx = getattr(doc, "idx", 0)

            write_key = BaseRedisSaver._make_redis_checkpoint_writes_key(
                storage_safe_thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            keys_to_delete.append(write_key)

        # Execute all deletions based on cluster mode
        if self.cluster_mode:
            # For cluster mode, delete keys individually
            for key in keys_to_delete:
                self._redis.delete(key)
        else:
            # For non-cluster mode, use pipeline for efficiency
            pipeline = self._redis.pipeline()
            for key in keys_to_delete:
                pipeline.delete(key)
            pipeline.execute()


__all__ = [
    "__version__",
    "__lib_name__",
    "RedisSaver",
    "AsyncRedisSaver",
    "BaseRedisSaver",
    "ShallowRedisSaver",
    "AsyncShallowRedisSaver",
]
