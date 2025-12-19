from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.constants import TASKS
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory
from ulid import ULID

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)
from langgraph.checkpoint.redis.util import (
    to_storage_safe_id,
    to_storage_safe_str,
)

# Constants
MILLISECONDS_PER_SECOND = 1000

# Logger for this module
logger = logging.getLogger(__name__)


class ShallowRedisSaver(BaseRedisSaver[Redis, SearchIndex]):
    """Redis implementation that only stores the most recent checkpoint."""

    # Default cache size limits
    DEFAULT_KEY_CACHE_MAX_SIZE = 1000
    DEFAULT_CHANNEL_CACHE_MAX_SIZE = 100

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[dict[str, Any]] = None,
        key_cache_max_size: Optional[int] = None,
        channel_cache_max_size: Optional[int] = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_blob_prefix=checkpoint_blob_prefix,
            checkpoint_write_prefix=checkpoint_write_prefix,
        )

        # Instance-level cache for frequently used keys (limited size to prevent memory issues)
        # Using OrderedDict for LRU cache eviction
        self._key_cache: OrderedDict[str, str] = OrderedDict()
        self._key_cache_max_size = key_cache_max_size or self.DEFAULT_KEY_CACHE_MAX_SIZE
        self._channel_cache: OrderedDict[str, Any] = OrderedDict()
        self._channel_cache_max_size = (
            channel_cache_max_size or self.DEFAULT_CHANNEL_CACHE_MAX_SIZE
        )

        # Prefixes are now set in BaseRedisSaver.__init__
        self._separator = REDIS_KEY_SEPARATOR

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[dict[str, Any]] = None,
        key_cache_max_size: Optional[int] = None,
        channel_cache_max_size: Optional[int] = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> Iterator[ShallowRedisSaver]:
        """Create a new ShallowRedisSaver instance."""
        saver: Optional[ShallowRedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
                ttl=ttl,
                key_cache_max_size=key_cache_max_size,
                channel_cache_max_size=channel_cache_max_size,
                checkpoint_prefix=checkpoint_prefix,
                checkpoint_blob_prefix=checkpoint_blob_prefix,
                checkpoint_write_prefix=checkpoint_write_prefix,
            )
            yield saver
        finally:
            if saver and saver._owns_its_client:
                saver._redis.close()
                # RedisCluster doesn't have connection_pool attribute
                if getattr(saver._redis, "connection_pool", None):
                    saver._redis.connection_pool.disconnect()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store checkpoint with inline channel values."""
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

        # Extract timestamp from checkpoint_id (ULID)
        checkpoint_ts = None
        if checkpoint["id"]:
            try:
                ulid_obj = ULID.from_str(checkpoint["id"])
                checkpoint_ts = ulid_obj.timestamp  # milliseconds since epoch
            except Exception as e:
                # If not a valid ULID, use checkpoint's timestamp if available, else current time
                logger.warning(
                    f"Invalid ULID checkpoint_id '{checkpoint['id']}': {e}. "
                    f"Using fallback timestamp."
                )
                # Try to use checkpoint's own timestamp field if available
                ts_value = checkpoint.get("ts")
                if ts_value:
                    # Handle both ISO string and numeric timestamps
                    if isinstance(ts_value, str):
                        try:
                            dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                            checkpoint_ts = dt.timestamp() * MILLISECONDS_PER_SECOND
                        except Exception:
                            checkpoint_ts = time.time() * MILLISECONDS_PER_SECOND
                    else:
                        checkpoint_ts = ts_value
                else:
                    checkpoint_ts = time.time() * MILLISECONDS_PER_SECOND

        # Parse metadata from string to dict to avoid double serialization
        metadata_str = self._dump_metadata(metadata)
        metadata_dict = (
            json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        )

        # Store channel values inline in the checkpoint
        copy["channel_values"] = checkpoint.get("channel_values", {})

        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_ns": to_storage_safe_str(checkpoint_ns),
            "checkpoint_id": checkpoint["id"],
            "checkpoint_ts": checkpoint_ts,
            "checkpoint": self._dump_checkpoint(copy),
            "metadata": metadata_dict,
            # Note: has_writes tracking removed to support put_writes before checkpoint exists
        }

        # Store at top-level for filters in list()
        if all(key in metadata for key in ["source", "step"]):
            checkpoint_data["source"] = metadata["source"]
            checkpoint_data["step"] = metadata["step"]

        checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns
        )

        with self._redis.pipeline(transaction=False) as pipeline:
            pipeline.json().set(checkpoint_key, "$", checkpoint_data)

            # Apply TTL if configured
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                pipeline.expire(checkpoint_key, ttl_seconds)

            pipeline.execute()

        # NOTE: We intentionally do NOT clean up old writes here.
        # In the HITL (Human-in-the-Loop) flow, interrupt writes are saved via
        # put_writes BEFORE the new checkpoint is saved. If we clean up writes
        # when the checkpoint changes, we would delete the interrupt writes
        # before they can be consumed when resuming.
        #
        # Writes are cleaned up in the following scenarios:
        # 1. When delete_thread is called
        # 2. When TTL expires (if configured)
        # 3. When put_writes is called again for the same task/idx (overwrites)
        #
        # See Issue #133 for details on this bug fix.

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
                Tag("thread_id")
                == to_storage_safe_id(config["configurable"]["thread_id"])
            )
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                filter_expression.append(
                    Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns)
                )

        if filter:
            for k, v in filter.items():
                if k == "source":
                    filter_expression.append(Tag("source") == v)
                elif k == "step":
                    filter_expression.append(Num("step") == v)
                else:
                    raise ValueError(f"Unsupported filter key: {k}")

        if before:
            before_checkpoint_id = get_checkpoint_id(before)
            if before_checkpoint_id:
                try:
                    before_ulid = ULID.from_str(before_checkpoint_id)
                    before_ts = before_ulid.timestamp
                    # Use numeric range query: checkpoint_ts < before_ts
                    filter_expression.append(Num("checkpoint_ts") < before_ts)
                except Exception:
                    # If not a valid ULID, ignore the before filter
                    pass

        # Combine all filter expressions
        combined_filter = filter_expression[0] if filter_expression else "*"
        for expr in filter_expression[1:]:
            combined_filter &= expr

        # Get checkpoint data
        # Sort by checkpoint_id in descending order to get most recent checkpoints first
        query = FilterQuery(
            filter_expression=combined_filter,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "$.checkpoint",
                "$.metadata",
            ],
            num_results=limit or 10000,
            sort_by=("checkpoint_id", "DESC"),
        )

        # Execute the query
        results = self.checkpoints_index.search(query)

        # Process the results
        for doc in results.docs:
            thread_id = cast(str, getattr(doc, "thread_id", ""))
            checkpoint_ns = cast(str, getattr(doc, "checkpoint_ns", ""))
            checkpoint = json.loads(doc["$.checkpoint"])

            # Extract channel values from the checkpoint (they're stored inline)
            channel_values: Dict[str, Any] = checkpoint.get("channel_values", {})
            # Deserialize them since they're stored in serialized form
            channel_values = self._deserialize_channel_values(channel_values)

            # Parse metadata
            raw_metadata = getattr(doc, "$.metadata", "{}")
            metadata_dict = (
                json.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else raw_metadata
            )

            # Sanitize metadata
            sanitized_metadata = {
                k.replace("\u0000", ""): (
                    v.replace("\u0000", "") if isinstance(v, str) else v
                )
                for k, v in metadata_dict.items()
            }
            metadata = cast(CheckpointMetadata, sanitized_metadata)

            # Load checkpoint with inline channel values
            checkpoint_param = self._load_checkpoint(
                doc["$.checkpoint"],
                channel_values,  # Pass the extracted channel values
                [],  # No pending_sends in shallow mode
            )

            config_param: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_param["id"],
                }
            }

            # Load pending writes (still uses separate keys - already efficient)
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
        """Get checkpoint with inline channel values."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Single key access gets everything inline
        checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns
        )

        checkpoint_data = self._redis.json().get(checkpoint_key)
        if not checkpoint_data or not isinstance(checkpoint_data, dict):
            return None

        # TTL refresh if enabled - always refresh for shallow implementation
        # Since there's only one checkpoint, the overhead is minimal
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            default_ttl_minutes = self.ttl_config.get("default_ttl", 60)
            ttl_seconds = int(default_ttl_minutes * 60)
            self._redis.expire(checkpoint_key, ttl_seconds)

        # Parse the checkpoint data
        checkpoint = checkpoint_data.get("checkpoint", {})
        if isinstance(checkpoint, str):
            checkpoint = json.loads(checkpoint)

        # Extract channel values from the checkpoint (they're stored inline)
        channel_values: Dict[str, Any] = checkpoint.get("channel_values", {})
        # Deserialize them since they're stored in serialized form
        channel_values = self._deserialize_channel_values(channel_values)

        # Parse metadata
        metadata = checkpoint_data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Sanitize metadata
        sanitized_metadata = {
            k.replace("\u0000", ""): (
                v.replace("\u0000", "") if isinstance(v, str) else v
            )
            for k, v in metadata.items()
        }

        # Load checkpoint with inline channel values
        checkpoint_param = self._load_checkpoint(
            json.dumps(checkpoint),
            channel_values,  # Pass the raw channel values - no deserialization needed
            [],  # No pending_sends in shallow mode
        )

        # Load pending writes (still uses separate keys - already efficient)
        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_param["id"]
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                }
            },
            checkpoint=checkpoint_param,
            metadata=cast(CheckpointMetadata, sanitized_metadata),
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

        # Set client info for Redis monitoring
        self.set_client_info()

    def create_indexes(self) -> None:
        self.checkpoints_index = SearchIndex.from_dict(
            self.checkpoints_schema, redis_client=self._redis
        )
        # Shallow implementation doesn't use blobs, but base class requires the attribute
        self.checkpoint_blobs_index = SearchIndex.from_dict(
            self.blobs_schema, redis_client=self._redis
        )
        self.checkpoint_writes_index = SearchIndex.from_dict(
            self.writes_schema, redis_client=self._redis
        )

    def setup(self) -> None:
        """Initialize the indices in Redis (skip blob index for shallow implementation)."""
        # Create only the indexes we actually use
        self.checkpoints_index.create(overwrite=False)
        # Skip creating blob index since shallow doesn't use separate blobs
        self.checkpoint_writes_index.create(overwrite=False)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint with checkpoint-level registry.

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
                "checkpoint_ns": to_storage_safe_str(checkpoint_ns),
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "blob": self._encode_blob(
                    blob
                ),  # Encode bytes to base64 string for Redis
            }
            writes_objects.append(write_obj)

        # THREAD-LEVEL REGISTRY: Only keep writes for the current checkpoint
        # Use to_storage_safe_str for consistent key naming with delete_thread
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        thread_write_registry_key = (
            f"write_registry:{thread_id}:{safe_checkpoint_ns}:shallow"
        )

        # Collect all write keys
        write_keys = []
        for write_obj in writes_objects:
            key = self._make_redis_checkpoint_writes_key_cached(
                thread_id, checkpoint_ns, checkpoint_id, task_id, write_obj["idx"]
            )
            write_keys.append(key)

        # Create a unified pipeline for all operations
        with self._redis.pipeline(transaction=False) as pipeline:

            # Add all JSON write operations - always overwrite for simplicity
            for idx, write_obj in enumerate(writes_objects):
                key = write_keys[idx]
                # Always set the complete object - simpler and faster than checking existence
                pipeline.json().set(key, "$", write_obj)

            # THREAD-LEVEL REGISTRY: Store write keys in thread-level sorted set
            # These will be cleared when checkpoint changes
            zadd_mapping = {key: idx for idx, key in enumerate(write_keys)}
            pipeline.zadd(thread_write_registry_key, zadd_mapping)  # type: ignore[arg-type]

            # Note: We don't update has_writes on the checkpoint anymore
            # because put_writes can be called before the checkpoint exists

            # Apply TTL to registry key if configured
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                pipeline.expire(thread_write_registry_key, ttl_seconds)
                # Also apply TTL to all write keys
                for key in write_keys:
                    pipeline.expire(key, ttl_seconds)

            # Execute everything in one round trip
            pipeline.execute()

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        """Load pending writes efficiently using thread-level write registry."""
        if checkpoint_id is None:
            return []

        # Use thread-level registry that only contains current checkpoint writes
        # All writes belong to the current checkpoint
        # Use to_storage_safe_str for consistent key naming with delete_thread
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        thread_write_registry_key = (
            f"write_registry:{thread_id}:{safe_checkpoint_ns}:shallow"
        )

        # Get all write keys from the thread's registry (already sorted by index)
        write_keys = self._redis.zrange(thread_write_registry_key, 0, -1)

        if not write_keys:
            return []

        # Batch fetch all writes using pipeline
        with self._redis.pipeline(transaction=False) as pipeline:
            for key in write_keys:
                # Decode bytes to string if needed
                key_str = key.decode() if isinstance(key, bytes) else key
                pipeline.json().get(key_str)

            results = pipeline.execute()

        # Build the writes dictionary
        writes_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for write_data in results:
            if write_data:
                task_id = write_data.get("task_id", "")
                idx = write_data.get("idx", 0)
                writes_dict[(task_id, idx)] = write_data

        # Use base class method to deserialize
        return BaseRedisSaver._load_writes(self.serde, writes_dict)

    def get_channel_values(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channel_versions: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Retrieve channel_values dictionary from inline checkpoint data."""
        # For shallow checkpoints, channel values are stored inline in the checkpoint
        checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns
        )

        # Single JSON.GET operation to retrieve checkpoint with inline channel_values
        checkpoint_data = self._redis.json().get(checkpoint_key, "$.checkpoint")

        if not checkpoint_data:
            return {}

        # checkpoint_data[0] is already a deserialized dict
        checkpoint = (
            checkpoint_data[0] if isinstance(checkpoint_data, list) else checkpoint_data
        )
        channel_values = checkpoint.get("channel_values", {})

        # Deserialize channel values since they're stored in serialized form
        # Cast to dict[str, Any] as we know this is the correct type from checkpoint structure
        from typing import cast

        return self._deserialize_channel_values(
            cast(dict[str, Any], channel_values) if channel_values else {}
        )

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
            & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            & (Tag("channel") == TASKS),
            return_fields=["type", "$.blob", "task_path", "task_id", "idx"],
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
        # Handle both direct attribute access and JSON path access
        # Filter out documents where blob is None (similar to RedisSaver in __init__.py)
        return [
            (getattr(doc, "type", ""), blob)
            for doc in sorted_writes
            if (blob := getattr(doc, "$.blob", getattr(doc, "blob", None))) is not None
        ]

    def _make_shallow_redis_checkpoint_key_cached(
        self, thread_id: str, checkpoint_ns: str
    ) -> str:
        """Create a cached key for shallow checkpoints using only thread_id and checkpoint_ns."""
        cache_key = f"shallow_checkpoint:{thread_id}:{checkpoint_ns}"
        if cache_key in self._key_cache:
            # Move to end for LRU (most recently used)
            self._key_cache.move_to_end(cache_key)
        else:
            # Add new entry, evicting oldest if necessary
            if len(self._key_cache) >= self._key_cache_max_size:
                # Remove least recently used (first item)
                self._key_cache.popitem(last=False)
            self._key_cache[cache_key] = self._separator.join(
                [self._checkpoint_prefix, thread_id, checkpoint_ns]
            )
        return self._key_cache[cache_key]

    @staticmethod
    def _make_shallow_redis_checkpoint_key(thread_id: str, checkpoint_ns: str) -> str:
        """Create a key for shallow checkpoints using only thread_id and checkpoint_ns."""
        return REDIS_KEY_SEPARATOR.join([CHECKPOINT_PREFIX, thread_id, checkpoint_ns])

    def _make_redis_checkpoint_writes_key_cached(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: Optional[int],
    ) -> str:
        """Create a cached key for checkpoint writes."""
        cache_key = (
            f"writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"
        )
        if cache_key in self._key_cache:
            # Move to end for LRU (most recently used)
            self._key_cache.move_to_end(cache_key)
        else:
            # Add new entry, evicting oldest if necessary
            if len(self._key_cache) >= self._key_cache_max_size:
                # Remove least recently used (first item)
                self._key_cache.popitem(last=False)
            self._key_cache[cache_key] = self._make_redis_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
        return self._key_cache[cache_key]

    @staticmethod
    def _make_shallow_redis_checkpoint_writes_key_pattern(
        thread_id: str, checkpoint_ns: str
    ) -> str:
        """Create a pattern to match all writes keys for a thread and namespace."""
        return (
            REDIS_KEY_SEPARATOR.join(
                [
                    CHECKPOINT_WRITE_PREFIX,
                    str(to_storage_safe_id(thread_id)),
                    to_storage_safe_str(checkpoint_ns),
                ]
            )
            + ":*"
        )

    @staticmethod
    def _make_shallow_redis_checkpoint_blob_key_pattern(
        thread_id: str, checkpoint_ns: str
    ) -> str:
        """Create a pattern to match all blob keys for a thread and namespace."""
        return (
            REDIS_KEY_SEPARATOR.join(
                [
                    CHECKPOINT_BLOB_PREFIX,
                    str(to_storage_safe_id(thread_id)),
                    to_storage_safe_str(checkpoint_ns),
                ]
            )
            + ":*"
        )

    def _make_shallow_redis_checkpoint_blob_key_cached(
        self, thread_id: str, checkpoint_ns: str, channel: str, version: str
    ) -> str:
        """Create a cached key for checkpoint blobs."""
        cache_key = f"shallow_blob:{thread_id}:{checkpoint_ns}:{channel}:{version}"
        if cache_key in self._key_cache:
            # Move to end for LRU (most recently used)
            self._key_cache.move_to_end(cache_key)
        else:
            # Add new entry, evicting oldest if necessary
            if len(self._key_cache) >= self._key_cache_max_size:
                # Remove least recently used (first item)
                self._key_cache.popitem(last=False)
            self._key_cache[cache_key] = self._make_redis_checkpoint_blob_key(
                thread_id, checkpoint_ns, channel, version
            )
        return self._key_cache[cache_key]

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a specific thread ID.

        Args:
            thread_id: The thread ID whose checkpoints should be deleted.
        """
        # Only one checkpoint per thread/namespace combination
        # Find all namespaces for this thread and delete them
        storage_safe_thread_id = to_storage_safe_id(thread_id)

        # Find all checkpoints for this thread to get checkpoint IDs
        checkpoint_query = FilterQuery(
            filter_expression=Tag("thread_id") == storage_safe_thread_id,
            return_fields=["checkpoint_ns", "checkpoint_id"],
            num_results=10000,
        )

        checkpoint_results = self.checkpoints_index.search(checkpoint_query)

        # Collect namespaces and checkpoint IDs
        checkpoint_data = []
        for doc in checkpoint_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")
            checkpoint_data.append((checkpoint_ns, checkpoint_id))

        # Delete all checkpoints and related data
        if checkpoint_data:
            with self._redis.pipeline(transaction=False) as pipeline:
                for checkpoint_ns, checkpoint_id in checkpoint_data:
                    # Delete the main checkpoint key
                    checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
                        thread_id, checkpoint_ns
                    )
                    pipeline.delete(checkpoint_key)

                    # Delete thread-level write registry and its writes
                    # Each namespace has its own thread-level registry
                    thread_write_registry_key = (
                        f"write_registry:{thread_id}:{checkpoint_ns}:shallow"
                    )

                    # Get all write keys from the thread registry before deleting
                    write_keys = self._redis.zrange(thread_write_registry_key, 0, -1)
                    for write_key in write_keys:
                        write_key_str = (
                            write_key.decode()
                            if isinstance(write_key, bytes)
                            else write_key
                        )
                        pipeline.delete(write_key_str)

                    # Delete the registry itself
                    pipeline.delete(thread_write_registry_key)

                    # Delete the current checkpoint tracker
                    current_checkpoint_key = (
                        f"current_checkpoint:{thread_id}:{checkpoint_ns}:shallow"
                    )
                    pipeline.delete(current_checkpoint_key)

                # Execute all deletions
                pipeline.execute()
