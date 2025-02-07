"""Async shallow Redis implementation for LangGraph checkpoint saving."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple, cast

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
    PendingWrite,
)
from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)
from langgraph.constants import TASKS
from redis.asyncio import Redis as AsyncRedis

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
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
        )
        self.loop = asyncio.get_running_loop()

    async def __aenter__(self) -> AsyncShallowRedisSaver:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_its_client:
            await self._redis.aclose()  # type: ignore[attr-defined]
            await self._redis.connection_pool.disconnect()

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
    ) -> AsyncIterator[AsyncShallowRedisSaver]:
        """Create a new AsyncShallowRedisSaver instance."""
        async with cls(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
        ) as saver:
            yield saver

    async def asetup(self) -> None:
        """Initialize Redis indexes asynchronously."""
        # Connect Redis client to indices asynchronously
        await self.checkpoints_index.set_client(self._redis)
        await self.checkpoint_blobs_index.set_client(self._redis)
        await self.checkpoint_writes_index.set_client(self._redis)

        # Create indexes in Redis asynchronously
        await self.checkpoints_index.create(overwrite=False)
        await self.checkpoint_blobs_index.create(overwrite=False)
        await self.checkpoint_writes_index.create(overwrite=False)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store only the latest checkpoint asynchronously."""
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

        await self.checkpoints_index.load(
            [checkpoint_data],
            keys=[
                AsyncShallowRedisSaver._make_shallow_redis_checkpoint_key(
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
            await self.checkpoint_blobs_index.load(list(data), keys=list(keys))

        return next_config

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
        results = await self.checkpoints_index.search(checkpoints_query)
        if not results.docs:
            return None

        doc = results.docs[0]

        checkpoint = json.loads(doc["$.checkpoint"])

        # Fetch channel_values
        channel_values = await self.aget_channel_values(
            thread_id=doc["thread_id"],
            checkpoint_ns=doc["checkpoint_ns"],
            checkpoint_id=checkpoint["id"],
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = await self._aload_pending_sends(
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

        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_param["id"]
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
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
        """Store intermediate writes for the latest checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
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

            upsert_case = all(w[0] in WRITES_IDX_MAP for w in writes)
            for write_obj in writes_objects:
                key = self._make_redis_checkpoint_writes_key(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_obj["idx"],
                )
                if upsert_case:

                    async def tx(pipe, key=key, write_obj=write_obj):
                        exists = await pipe.exists(key)
                        if exists:
                            await pipe.json().set(
                                key, "$.channel", write_obj["channel"]
                            )
                            await pipe.json().set(key, "$.type", write_obj["type"])
                            await pipe.json().set(key, "$.blob", write_obj["blob"])
                        else:
                            await pipe.json().set(key, "$", write_obj)

                    await self._redis.transaction(tx, key)
                else:
                    # Unlike AsyncRedisSaver, the shallow implementation always overwrites
                    # the full object, so we don't check for key existence here.
                    await self._redis.json().set(key, "$", write_obj)

    async def aget_channel_values(
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

        checkpoint_result = await self.checkpoints_index.search(checkpoint_query)
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

            blob_results = await self.checkpoint_blobs_index.search(blob_query)
            if blob_results.docs:
                blob_doc = blob_results.docs[0]
                blob_type = blob_doc.type
                blob_data = getattr(blob_doc, "$.blob", None)

                if blob_data and blob_type != "empty":
                    channel_values[channel] = self.serde.loads_typed(
                        (blob_type, blob_data)
                    )

        return channel_values

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
            & (Tag("checkpoint_ns") == checkpoint_ns)
            & (Tag("channel") == TASKS),
            return_fields=["type", "blob", "task_path", "task_id", "idx"],
            num_results=100,
        )
        parent_writes_results = await self.checkpoint_writes_index.search(
            parent_writes_query
        )

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

    async def _aload_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        if checkpoint_id is None:
            return []  # Early return if no checkpoint_id

        writes_key = BaseRedisSaver._make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self._redis.keys(pattern=writes_key)
        parsed_keys = [
            BaseRedisSaver._parse_redis_checkpoint_writes_key(key.decode())
            for key in matching_keys
        ]
        pending_writes = BaseRedisSaver._load_writes(
            self.serde,
            {
                (
                    parsed_key["task_id"],
                    parsed_key["idx"],
                ): await self._redis.json().get(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None
        self._redis = redis_client or RedisConnectionFactory.get_async_redis_connection(
            redis_url, **connection_args
        )

    def create_indexes(self) -> None:
        """Create indexes without connecting to Redis."""
        self.checkpoints_index = AsyncSearchIndex.from_dict(self.SCHEMAS[0])
        self.checkpoint_blobs_index = AsyncSearchIndex.from_dict(self.SCHEMAS[1])
        self.checkpoint_writes_index = AsyncSearchIndex.from_dict(self.SCHEMAS[2])

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
        task_path: str = "",
    ) -> None:
        """Store intermediate writes synchronously."""
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

    @staticmethod
    def _make_shallow_redis_checkpoint_key(thread_id: str, checkpoint_ns: str) -> str:
        return REDIS_KEY_SEPARATOR.join([CHECKPOINT_PREFIX, thread_id, checkpoint_ns])
