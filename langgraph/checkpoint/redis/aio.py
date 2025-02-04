"""Async implementation of Redis checkpoint saver."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from types import TracebackType
from typing import Any, Optional, Sequence, Tuple, Type, cast

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
    get_checkpoint_id,
)
from langgraph.checkpoint.redis.base import BaseRedisSaver
from redis.asyncio import Redis as AsyncRedis


class AsyncRedisSaver(BaseRedisSaver[AsyncRedis, AsyncSearchIndex]):
    """Async Redis implementation for checkpoint saver."""

    _redis_url: Optional[str] = None
    _redis: Optional[AsyncRedis] = None
    _owns_its_client: bool = False

    checkpoint_index: AsyncSearchIndex
    channel_index: AsyncSearchIndex
    writes_index: AsyncSearchIndex

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            index_prefix=index_prefix,
            connection_args=connection_args,
        )
        self._exit_stack = AsyncExitStack()
        self.loop = asyncio.get_running_loop()

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None
        self._redis_url = redis_url
        self._connection_args = connection_args or {}
        self._redis = redis_client

    async def _redis_client(self) -> AsyncRedis:
        if self._redis is None:
            self._owns_its_client = True
            self._redis = RedisConnectionFactory.get_async_redis_connection(
                self._redis_url, **self._connection_args
            )
            await self._exit_stack.enter_async_context(self._redis)
        return self._redis

    def create_indexes(self) -> None:
        """Create indexes without connecting to Redis."""
        self.checkpoint_index = AsyncSearchIndex.from_dict(self.SCHEMAS[0])
        self.channel_index = AsyncSearchIndex.from_dict(self.SCHEMAS[1])
        self.writes_index = AsyncSearchIndex.from_dict(self.SCHEMAS[2])

    async def __aenter__(self) -> AsyncRedisSaver:
        """Async context manager enter."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        # Close client connections
        if self._owns_its_client and self._redis_client is not None:
            await self._exit_stack.aclose()

    async def asetup(self) -> None:
        """Initialize Redis indexes asynchronously."""
        # Connect Redis client to indices asynchronously
        redis = await self._redis_client()
        await self.checkpoint_index.set_client(redis)
        await self.channel_index.set_client(redis)
        await self.writes_index.set_client(redis)

        # Create indexes in Redis asynchronously
        await self.checkpoint_index.create(overwrite=False)
        await self.channel_index.create(overwrite=False)
        await self.writes_index.create(overwrite=False)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis asynchronously."""
        configurable = config.get("configurable")
        if not configurable:
            raise ValueError('The "configurable" key was not found in RunnableConfig')

        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]

        # Construct the query
        query = FilterQuery(
            filter_expression=(
                (Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
                & (Tag("checkpoint_id") == checkpoint_id)
            ),
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "checkpoint",
                "$.metadata",
            ],
            num_results=1,
        )

        # Execute the query
        results = await self.checkpoint_index.search(query)
        if not results.docs:
            return None

        doc = results.docs[0]

        # Fetch and parse metadata
        raw_metadata = getattr(doc, "$.metadata", "{}")
        metadata_dict = (
            json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        )
        metadata = cast(CheckpointMetadata, metadata_dict)

        # Get writes for this checkpoint
        write_key = (
            f"{self.index_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        )
        writes = await self._aload_writes_from_redis(write_key)

        return CheckpointTuple(
            config=config,
            checkpoint=self._load_checkpoint(getattr(doc, "checkpoint", {}), {}, []),
            metadata=metadata,
            parent_config=None,
            pending_writes=writes,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from Redis asynchronously."""
        # Construct the filter expression
        filter_expression = []
        if config:
            configurable = config.get("configurable")
            if not configurable:
                raise ValueError('The "configurable" key was not found in RunnableConfig')

            filter_expression.append(Tag("thread_id") == configurable["thread_id"])
            if checkpoint_ns := configurable.get("checkpoint_ns"):
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

        if before:
            filter_expression.append(Tag("checkpoint_id") < get_checkpoint_id(before))

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
                "checkpoint",
                "metadata",
                "pending_sends",
                "pending_writes",
            ],
            num_results=limit or 10,
        )

        # Execute the query asynchronously
        results = await self.checkpoint_index.search(query)

        # Process the results
        for doc in results.docs:
            # Access fields directly using attributes
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")
            thread_id = getattr(doc, "thread_id", "")
            metadata = cast(
                CheckpointMetadata, getattr(doc, "metadata", {})
            )  # Enforce type

            # Extract channel values
            channel_query = FilterQuery(
                filter_expression=(Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
            )
            channel_values = {}
            channel_results = await self.channel_index.search(channel_query)
            for val in channel_results.docs:
                channel_values[getattr(val, "channel", "")] = {
                    "type": getattr(val, "type", ""),
                    "blob": getattr(val, "blob", ""),
                }

            # Load checkpoint data
            checkpoint = self._load_checkpoint(
                getattr(doc, "checkpoint", {}),
                channel_values,
                getattr(doc, "pending_sends", []),
            )

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,  # Metadata is now properly typed
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": getattr(doc, "parent_checkpoint_id", ""),
                        }
                    }
                    if hasattr(doc, "parent_checkpoint_id")
                    else None
                ),
                pending_writes=getattr(doc, "pending_writes", []),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis asynchronously."""
        configurable = config.get("configurable")
        if not configurable:
            raise ValueError('The "configurable" key was not found in RunnableConfig')

        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Sanitize metadata
        sanitized_metadata = {
            k.replace("\x00", ""): v.replace("\x00", "") if isinstance(v, str) else v
            for k, v in metadata.items()
        }

        cast_sanitize_metadata = cast(CheckpointMetadata, sanitized_metadata)

        # Store checkpoint data
        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint": self._dump_checkpoint(copy),
            "metadata": self._dump_metadata(cast_sanitize_metadata),
            "pending_sends": [],
            "pending_writes": [],
        }

        # Save to Redis asynchronously
        await self.checkpoint_index.load([checkpoint_data])

        # Store channel values
        blobs = self._dump_blobs(
            thread_id,
            checkpoint_ns,
            copy.get("channel_values", {}),
            new_versions,
        )

        if blobs:
            await self.channel_index.load(blobs)

        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint using Redis JSON.

        Args:
            config: Configuration of the related checkpoint
            writes: List of writes as (channel, value) pairs
            task_id: Identifier for the task creating the writes
        """
        configurable = config.get("configurable")
        if not configurable:
            raise ValueError('The "configurable" key was not found in RunnableConfig')

        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]

        # Transform writes into JSON structure
        writes_objects = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            writes_objects.append(
                {
                    "task_id": task_id,
                    "idx": WRITES_IDX_MAP.get(channel, idx),
                    "channel": channel,
                    "type": type_,
                    "blob": self._encode_blob(blob),
                }
            )

        if writes_objects:
            write_key = f"{self.index_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

            # Use pipeline for atomic operations
            client = await self._redis_client()
            async with client.pipeline() as pipeline:
                try:
                    await pipeline.watch(write_key)

                    # Check if document exists
                    pipeline.json().type(write_key, "$")
                    exists = (await pipeline.execute())[0] is not None

                    if exists:
                        # Append to writes array
                        # TODO: What's up with the type error here?
                        await pipeline.json().arrappend(
                            write_key, "$.writes", *writes_objects  # type: ignore
                        )
                    else:
                        # Create new document
                        await pipeline.json().set(
                            write_key,
                            "$",
                            {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                                "writes": writes_objects,
                            },
                        )
                    await pipeline.execute()
                finally:
                    await pipeline.reset()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Synchronous wrapper for aput_writes.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

    async def _aload_writes_from_redis(
        self, write_key: str
    ) -> list[tuple[str, str, Any]]:
        """Load writes from Redis JSON storage by key asynchronously."""
        if not write_key:
            return []

        # Get the full JSON document
        client = await self._redis_client()
        result = await client.json().get(write_key)
        if not result:
            return []

        writes = []
        for write in result["writes"]:
            writes.append(
                (
                    write["task_id"],
                    write["channel"],
                    self.serde.loads_typed(
                        (write["type"], self._decode_blob(write["blob"]))
                    ),
                )
            )
        return writes

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.

        Raises:
            asyncio.InvalidStateError: If called from the wrong thread/event loop
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncRedisSaver are only allowed from a "
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
        """Store a checkpoint to Redis.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.

        Raises:
            asyncio.InvalidStateError: If called from the wrong thread/event loop
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncRedisSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.aput(...)` or `await "
                    "graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[AsyncRedisSaver]:
        async with cls(
            redis_url=redis_url,
            redis_client=redis_client,
            index_prefix=index_prefix,
            connection_args=connection_args,
        ) as saver:
            yield saver
