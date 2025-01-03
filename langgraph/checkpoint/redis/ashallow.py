"""Async shallow Redis implementation for LangGraph checkpoint saving."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.redis.base import BaseRedisSaver
from redis.asyncio import Redis


class AsyncShallowRedisSaver(BaseRedisSaver[Redis, AsyncSearchIndex]):
    """Async Redis implementation that only stores the most recent checkpoint."""

    SCHEMAS = [
        {
            "index": {
                "name": "checkpoints",
                "prefix": "checkpoint",
                "storage_type": "json",
            },
            "fields": [
                {"name": "thread_id", "type": "tag"},
                {"name": "checkpoint_ns", "type": "tag"},
                {"name": "checkpoint_id", "type": "tag"},
                {"name": "source", "type": "tag", "path": "$.metadata.source"},
                {"name": "step", "type": "numeric", "path": "$.metadata.step"},
                {"name": "v", "type": "numeric"},
                {"name": "ts", "type": "text"},
                {"name": "score", "type": "numeric", "path": "$.metadata.score"},
            ],
        }
    ]

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            index_prefix=index_prefix,
            connection_args=connection_args,
        )
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[AsyncShallowRedisSaver]:
        """Create a new AsyncShallowRedisSaver instance."""
        saver: Optional[AsyncShallowRedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                index_prefix=index_prefix,
                connection_args=connection_args,
            )
            yield saver
        finally:
            if saver and saver._owns_its_client:
                await saver._redis.aclose()  # type: ignore[attr-defined]
                await saver._redis.connection_pool.disconnect()

    async def asetup(self) -> None:
        """Initialize the checkpoint_index in Redis asynchronously."""
        await self.checkpoint_index.set_client(self._redis)
        await self.checkpoint_index.create(overwrite=False)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store only the latest checkpoint asynchronously."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        # Sanitize metadata
        sanitized_metadata = {
            k.replace("\x00", ""): v.replace("\x00", "") if isinstance(v, str) else v
            for k, v in metadata.items()
        }
        sanitized_metadata["score"] = sanitized_metadata.get("score", 0) or 0

        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "v": checkpoint.get("v", 1),
            "ts": checkpoint.get("ts", ""),
            "metadata": sanitized_metadata,
        }

        # Construct a unique key using thread_id, checkpoint_ns, and checkpoint_id
        key = f"{self.index_prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

        # Clean up old keys
        previous_key_pattern = f"{self.index_prefix}:{thread_id}:{checkpoint_ns}:*"
        async for old_key in self._redis.scan_iter(match=previous_key_pattern):
            if old_key != key:
                await self._redis.delete(old_key)

        # Store the new checkpoint
        await self._redis.json().set(key, "$", checkpoint_data)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from Redis asynchronously."""
        query_filter = []

        if config:
            query_filter.append(Tag("thread_id") == config["configurable"]["thread_id"])
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                query_filter.append(Tag("checkpoint_ns") == checkpoint_ns)

        if filter:
            for key, value in filter.items():
                if key == "source":
                    query_filter.append(Tag("source") == value)
                elif key == "step":
                    query_filter.append(Num("step") == value)

        combined_filter = query_filter[0] if query_filter else "*"
        for expr in query_filter[1:]:
            combined_filter &= expr

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
        )

        results = await self.checkpoint_index.search(query)
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
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Construct the key for Redis JSON retrieval
        key = f"{self.index_prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}".strip(
            ":"
        )
        checkpoint_data = await self._redis.json().get(key)

        if not checkpoint_data:
            return None

        # Sanitize metadata
        metadata_dict = checkpoint_data.get("metadata", {})
        metadata = cast(CheckpointMetadata, metadata_dict)

        # Prepare pending writes
        writes_key = f"{self.index_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}".strip(
            ":"
        )
        writes_data = await self._redis.json().get(writes_key)
        pending_writes = (
            [
                (
                    write["task_id"],
                    write["channel"],
                    write["value"],
                )
                for write in writes_data.get("writes", [])
            ]
            if writes_data
            else []
        )

        return CheckpointTuple(
            config=config,
            checkpoint={
                "v": checkpoint_data.get("v", 1),
                "ts": checkpoint_data.get("ts", ""),
                "id": checkpoint_id,
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
            metadata=metadata,
            pending_writes=pending_writes,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes for the latest checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Construct the key for writes
        key = f"{self.index_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}".strip(
            ":"
        )

        # Prepare the writes data, preserving order and task_id
        writes_objects = []
        for idx, (channel, value) in enumerate(writes):
            writes_objects.append(
                {
                    "task_id": task_id,
                    "channel": channel,
                    "value": value,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                }
            )

        # Store in Redis
        await self._redis.json().set(key, "$", {"writes": writes_objects})

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None
        self._redis = redis_client or RedisConnectionFactory.get_async_redis_connection(
            redis_url, **connection_args
        )

    def create_indexes(self) -> None:
        """Create the necessary Redis indexes."""
        self.checkpoint_index = AsyncSearchIndex.from_dict(self.SCHEMAS[0])

    def setup(self) -> None:
        """Initialize the checkpoint_index in Redis."""
        asyncio.run_coroutine_threadsafe(self.asetup(), self.loop).result()

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
    ) -> None:
        """Store intermediate writes synchronously."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()
