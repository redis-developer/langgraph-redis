"""Async shallow Redis implementation for LangGraph checkpoint saving."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Type, cast

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
from redis.asyncio import Redis as AsyncRedis
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
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


class AsyncShallowRedisSaver(BaseRedisSaver[AsyncRedis, AsyncSearchIndex]):
    """Async Redis implementation that only stores the most recent checkpoint."""

    _redis_url: str
    checkpoints_index: AsyncSearchIndex
    checkpoint_blobs_index: AsyncSearchIndex
    checkpoint_writes_index: AsyncSearchIndex

    _redis: AsyncRedis  # Override the type from the base class

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[dict[str, Any]] = None,
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
        self.loop = asyncio.get_running_loop()

        # Instance-level cache for frequently used keys (limited size to prevent memory issues)
        self._key_cache: Dict[str, str] = {}
        self._key_cache_max_size = 1000  # Configurable limit
        self._channel_cache: Dict[str, Any] = {}

        self._separator = REDIS_KEY_SEPARATOR

    async def __aenter__(self) -> AsyncShallowRedisSaver:
        """Async context manager enter."""
        await self.asetup()

        # Set client info once Redis is set up
        await self.aset_client_info()

        return self

    async def __aexit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        if self._owns_its_client:
            await self._redis.aclose()  # type: ignore[attr-defined]
            # RedisCluster doesn't have connection_pool attribute
            if getattr(self._redis, "connection_pool", None):
                coro = self._redis.connection_pool.disconnect()
                if coro:
                    await coro

            # Prevent RedisVL from attempting to close the client
            # on an event loop in a separate thread.
            self.checkpoints_index._redis_client = None
            self.checkpoint_blobs_index._redis_client = None
            self.checkpoint_writes_index._redis_client = None

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[dict[str, Any]] = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> AsyncIterator[AsyncShallowRedisSaver]:
        """Create a new AsyncShallowRedisSaver instance."""
        async with cls(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_blob_prefix=checkpoint_blob_prefix,
            checkpoint_write_prefix=checkpoint_write_prefix,
        ) as saver:
            yield saver

    async def asetup(self) -> None:
        """Initialize Redis indexes asynchronously (skip blob index for shallow implementation)."""
        # Create only the indexes we actually use
        await self.checkpoints_index.create(overwrite=False)
        # Skip creating blob index since shallow doesn't use separate blobs
        await self.checkpoint_writes_index.create(overwrite=False)

    async def setup(self) -> None:  # type: ignore[override]
        """Set up the checkpoint saver asynchronously.

        This method creates the necessary indices in Redis.
        It MUST be called before using the checkpointer.

        This async method follows the canonical pattern used by other
        async checkpointers in the LangGraph ecosystem. The type ignore is necessary because
        the base class defines a sync setup() method, but async checkpointers require
        an async setup() method to properly handle coroutines.
        """
        await self.asetup()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store checkpoint with INLINE channel values

        Stores all channel values directly in main checkpoint JSON

        Args:
            config: The config to associate with the checkpoint
            checkpoint: The checkpoint data to store
            metadata: Additional metadata to save with the checkpoint
            new_versions: New channel versions as of this write

        Returns:
            Updated configuration after storing the checkpoint

        Raises:
            asyncio.CancelledError: If the operation is cancelled/interrupted
        """
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

        try:
            # Extract timestamp from checkpoint_id (ULID)
            checkpoint_ts = None
            if checkpoint["id"]:
                try:
                    from ulid import ULID

                    ulid_obj = ULID.from_str(checkpoint["id"])
                    checkpoint_ts = ulid_obj.timestamp  # milliseconds since epoch
                except Exception:
                    # If not a valid ULID, use current time
                    import time

                    checkpoint_ts = time.time() * 1000

            # Store channel values inline in the checkpoint
            copy["channel_values"] = checkpoint.get("channel_values", {})

            checkpoint_data = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
                "checkpoint_ts": checkpoint_ts,
                "checkpoint": self._dump_checkpoint(copy),
                "metadata": self._dump_metadata(metadata),
                # Note: has_writes tracking removed to support put_writes before checkpoint exists
            }

            # Store at top-level for filters in list()
            if all(key in metadata for key in ["source", "step"]):
                checkpoint_data["source"] = metadata["source"]
                checkpoint_data["step"] = metadata["step"]

            # SHALLOW MODE: Only one key needed - overwrite everything atomically
            checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
                thread_id, checkpoint_ns
            )

            # Create pipeline for all operations
            pipeline = self._redis.pipeline(transaction=False)

            # Set the new checkpoint data
            pipeline.json().set(checkpoint_key, "$", checkpoint_data)

            # Apply TTL if configured
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                pipeline.expire(checkpoint_key, ttl_seconds)

            # Execute pipeline to set new checkpoint data
            await pipeline.execute()

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

        except asyncio.CancelledError:
            # Handle cancellation/interruption
            # Pipeline will be automatically discarded
            # Either all operations succeed or none do
            raise

        except Exception as e:
            # Re-raise other exceptions
            raise e

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,  # noqa: ARG002
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from Redis asynchronously."""
        query_filter = []

        if config:
            query_filter.append(
                Tag("thread_id")
                == to_storage_safe_id(config["configurable"]["thread_id"])
            )
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                query_filter.append(
                    Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns)
                )

        if filter:
            for key, value in filter.items():
                if key == "source":
                    query_filter.append(Tag("source") == value)
                elif key == "step":
                    query_filter.append(Num("step") == value)

        if before:
            before_checkpoint_id = get_checkpoint_id(before)
            if before_checkpoint_id:
                try:
                    before_ulid = ULID.from_str(before_checkpoint_id)
                    before_ts = before_ulid.timestamp
                    # Use numeric range query: checkpoint_ts < before_ts
                    query_filter.append(Num("checkpoint_ts") < before_ts)
                except Exception:
                    # If not a valid ULID, ignore the before filter
                    pass

        combined_filter = query_filter[0] if query_filter else "*"
        for expr in query_filter[1:]:
            combined_filter &= expr

        # Sort by checkpoint_id in descending order to get most recent checkpoints first
        query = FilterQuery(
            filter_expression=combined_filter,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "source",
                "step",
                "score",
                "ts",
            ],
            num_results=limit or 100,  # Set higher limit to retrieve more results
            sort_by=("checkpoint_id", "DESC"),
        )

        results = await self.checkpoints_index.search(query)
        for doc in results.docs:
            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": getattr(doc, "thread_id", ""),
                        "checkpoint_ns": getattr(doc, "checkpoint_ns", ""),
                        "checkpoint_id": getattr(doc, "checkpoint_id", ""),
                    }
                },
                checkpoint={
                    "v": 1,
                    "ts": getattr(doc, "ts", ""),
                    "id": getattr(doc, "checkpoint_id", ""),
                    "channel_values": {},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
                metadata={
                    "source": getattr(doc, "source", "input"),
                    "step": int(getattr(doc, "step", 0)),
                    "writes": {},
                    "score": float(getattr(doc, "score", 0)),
                },
                pending_writes=[],
            )

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from Redis asynchronously."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Use direct key access for shallow checkpoints
        # Shallow checkpoints only store the latest checkpoint per thread/namespace
        checkpoint_key = self._make_shallow_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns
        )

        # Single fetch gets everything inline - matching sync implementation
        full_checkpoint_data = await self._redis.json().get(checkpoint_key)  # type: ignore[misc]
        if not full_checkpoint_data or not isinstance(full_checkpoint_data, dict):
            return None

        # If refresh_on_read is enabled, refresh TTL for checkpoint key
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            # TTL refresh if enabled - always refresh for shallow implementation
            # Since there's only one checkpoint per thread/namespace, the overhead is minimal
            default_ttl_minutes = self.ttl_config.get("default_ttl", 60)
            ttl_seconds = int(default_ttl_minutes * 60)
            await self._redis.expire(checkpoint_key, ttl_seconds)

        # Parse the checkpoint data
        checkpoint = full_checkpoint_data.get("checkpoint", {})
        if isinstance(checkpoint, str):
            checkpoint = json.loads(checkpoint)

        # Extract channel values from the checkpoint (they're stored inline)
        # NO NEED TO CALL aget_channel_values - we already have the data!
        channel_values: Dict[str, Any] = checkpoint.get("channel_values", {})
        # Deserialize them since they're stored in serialized form
        channel_values = self._deserialize_channel_values(channel_values)

        # Parse metadata
        metadata = full_checkpoint_data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Ensure metadata matches CheckpointMetadata type
        sanitized_metadata = {
            k.replace("\u0000", ""): (
                v.replace("\u0000", "") if isinstance(v, str) else v
            )
            for k, v in metadata.items()
        }

        # For shallow mode, pending_sends is always empty
        pending_sends: list[tuple[str, bytes]] = []

        config_param: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        checkpoint_param = self._load_checkpoint(
            json.dumps(checkpoint),
            channel_values,
            pending_sends,  # No pending_sends in shallow mode
        )

        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_param["id"]
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=cast(CheckpointMetadata, sanitized_metadata),
            parent_config=None,
            pending_writes=pending_writes,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes for the latest checkpoint and clean up old writes with transaction handling.

        This method uses Redis pipeline with transaction=True to ensure atomicity of all
        write operations. In case of interruption, all operations will be aborted.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.

        Raises:
            asyncio.CancelledError: If the operation is cancelled/interrupted
        """
        if not writes:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        try:
            # Create a pipeline without transaction to avoid lock contention
            pipeline = self._redis.pipeline(transaction=False)

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
                    "blob": self._encode_blob(
                        blob
                    ),  # Encode bytes to base64 string for Redis
                }
                writes_objects.append(write_obj)

            # Thread-level sorted set for write keys
            # Use to_storage_safe_str for consistent key naming
            safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
            thread_zset_key = (
                f"write_keys_zset:{thread_id}:{safe_checkpoint_ns}:shallow"
            )

            # Collect all write keys
            write_keys = []
            for write_obj in writes_objects:
                key = self._make_redis_checkpoint_writes_key_cached(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_obj["idx"],
                )
                write_keys.append(key)

            # No cleanup in put_writes - we do it in aput() when checkpoint changes

            # Add new writes to the pipeline - always overwrite for simplicity
            for idx, write_obj in enumerate(writes_objects):
                key = write_keys[idx]
                # Always set the complete object - simpler and faster than checking existence
                pipeline.json().set(key, "$", write_obj)

            # Use thread-level sorted set
            zadd_mapping = {key: idx for idx, key in enumerate(write_keys)}
            pipeline.zadd(thread_zset_key, zadd_mapping)  # type: ignore[arg-type]

            # Apply TTL to registry key if configured
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                pipeline.expire(thread_zset_key, ttl_seconds)

            # Execute everything in one round trip
            await pipeline.execute()

        except asyncio.CancelledError:
            # Handle cancellation/interruption
            # Pipeline will be automatically discarded
            # Either all operations succeed or none do
            raise

        except Exception as e:
            # Re-raise other exceptions
            raise e

    async def aget_channel_values(
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
        checkpoint_data = await self._redis.json().get(checkpoint_key, "$.checkpoint")  # type: ignore[misc]

        if not checkpoint_data:
            return {}

        # checkpoint_data[0] is already a deserialized dict
        checkpoint = (
            checkpoint_data[0] if isinstance(checkpoint_data, list) else checkpoint_data
        )
        channel_values = checkpoint.get("channel_values", {})

        # Deserialize channel values since they're stored in serialized form
        return self._deserialize_channel_values(channel_values)

    async def _aload_pending_sends(
        self,
        thread_id: str,
        checkpoint_ns: str,
    ) -> list[tuple[str, bytes]]:
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
        parent_writes_results = await self.checkpoint_writes_index.search(
            parent_writes_query
        )

        # Sort results by task_path, task_id, idx
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
        # Filter out documents where blob is None (similar to AsyncRedisSaver in aio.py)
        return [
            (getattr(doc, "type", ""), blob)
            for doc in sorted_writes
            if (blob := getattr(doc, "$.blob", getattr(doc, "blob", None))) is not None
        ]

    async def _aload_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        """Load pending writes using thread-level sorted set registry."""
        if checkpoint_id is None:
            return []

        # Use thread-level sorted set
        # Use to_storage_safe_str for consistent key naming
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        thread_zset_key = f"write_keys_zset:{thread_id}:{safe_checkpoint_ns}:shallow"

        try:
            # Check if we have any writes in the thread sorted set
            write_count = await self._redis.zcard(thread_zset_key)

            if write_count == 0:
                # No writes for this thread
                return []

            # Get all write keys from the thread sorted set
            write_keys = await self._redis.zrange(thread_zset_key, 0, -1)

            if write_keys:
                # All keys in the set belong to current checkpoint
                decoded_keys = [
                    key.decode() if isinstance(key, bytes) else key
                    for key in write_keys
                ]

                # Fetch all writes using pipeline
                pipeline = self._redis.pipeline(transaction=False)
                for key in decoded_keys:
                    pipeline.json().get(key)

                results = await pipeline.execute()

                # Build the writes dictionary
                writes_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

                for write_data in results:
                    if write_data:
                        task_id = write_data.get("task_id", "")
                        idx = write_data.get("idx", 0)
                        writes_dict[(task_id, idx)] = write_data

                # Use base class method to deserialize
                return BaseRedisSaver._load_writes(self.serde, writes_dict)

        except Exception:
            pass

        # Return empty list if registry not available
        return []

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None

        if redis_client is None:
            if not redis_url:
                redis_url = os.environ.get("REDIS_URL")
                if not redis_url:
                    raise ValueError("REDIS_URL env var not set")
            self._redis = AsyncRedis.from_url(redis_url, **(connection_args or {}))
        else:
            self._redis = redis_client

    def create_indexes(self) -> None:
        """Create indexes without connecting to Redis."""
        self.checkpoints_index = AsyncSearchIndex.from_dict(
            self.checkpoints_schema, redis_client=self._redis
        )
        # Shallow implementation doesn't use blobs, but base class requires the attribute
        self.checkpoint_blobs_index = AsyncSearchIndex.from_dict(
            self.blobs_schema, redis_client=self._redis
        )
        self.checkpoint_writes_index = AsyncSearchIndex.from_dict(
            self.writes_schema, redis_client=self._redis
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from Redis synchronously."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncShallowRedisSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store only the latest checkpoint synchronously."""
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes synchronously."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

    def get_channel_values(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        channel_versions: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Retrieve channel_values dictionary with properly constructed message objects (sync wrapper)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncShallowRedisSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aget_channel_values(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_channel_values(
                thread_id, checkpoint_ns, checkpoint_id, channel_versions
            ),
            self.loop,
        ).result()

    def _make_shallow_redis_checkpoint_key_cached(
        self, thread_id: str, checkpoint_ns: str
    ) -> str:
        """Create a cached key for shallow checkpoints using only thread_id and checkpoint_ns."""
        cache_key = f"shallow_checkpoint:{thread_id}:{checkpoint_ns}"
        if cache_key not in self._key_cache:
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
        if cache_key not in self._key_cache:
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
        if cache_key not in self._key_cache:
            if len(self._key_cache) >= self._key_cache_max_size:
                # Remove oldest entry when cache is full
                self._key_cache.pop(next(iter(self._key_cache)))
            self._key_cache[cache_key] = self._make_redis_checkpoint_blob_key(
                thread_id, checkpoint_ns, channel, version
            )
        return self._key_cache[cache_key]
