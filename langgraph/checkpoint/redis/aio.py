"""Async implementation of Redis checkpoint saver."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

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
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag

from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.util import (
    EMPTY_ID_SENTINEL,
    from_storage_safe_id,
    from_storage_safe_str,
    safely_decode,
    to_storage_safe_id,
    to_storage_safe_str,
)

logger = logging.getLogger(__name__)


class AsyncRedisSaver(
    BaseRedisSaver[Union[AsyncRedis, AsyncRedisCluster], AsyncSearchIndex]
):
    """Async Redis implementation for checkpoint saver."""

    _redis_url: str
    checkpoints_index: AsyncSearchIndex
    checkpoint_blobs_index: AsyncSearchIndex
    checkpoint_writes_index: AsyncSearchIndex

    _redis: Union[
        AsyncRedis, AsyncRedisCluster
    ]  # Support both standalone and cluster clients
    # Whether to assume the Redis server is a cluster; None triggers auto-detection
    cluster_mode: Optional[bool] = None

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
        )
        self.loop = asyncio.get_running_loop()

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None

        # Use direct AsyncRedis.from_url to avoid the deprecated get_async_redis_connection
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
            self.SCHEMAS[0], redis_client=self._redis
        )
        self.checkpoint_blobs_index = AsyncSearchIndex.from_dict(
            self.SCHEMAS[1], redis_client=self._redis
        )
        self.checkpoint_writes_index = AsyncSearchIndex.from_dict(
            self.SCHEMAS[2], redis_client=self._redis
        )

    async def __aenter__(self) -> AsyncRedisSaver:
        """Async context manager enter."""
        await self.asetup()

        # Set client info once Redis is set up
        await self.aset_client_info()

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        if self._owns_its_client:
            await self._redis.aclose()
            coro = self._redis.connection_pool.disconnect()
            if coro:
                await coro

            # Prevent RedisVL from attempting to close the client
            # on an event loop in a separate thread.
            self.checkpoints_index._redis_client = None
            self.checkpoint_blobs_index._redis_client = None
            self.checkpoint_writes_index._redis_client = None

    async def asetup(self) -> None:
        """Set up the checkpoint saver."""
        self.create_indexes()
        await self.checkpoints_index.create(overwrite=False)
        await self.checkpoint_blobs_index.create(overwrite=False)
        await self.checkpoint_writes_index.create(overwrite=False)

        # Detect cluster mode if not explicitly set
        await self._detect_cluster_mode()

    async def _detect_cluster_mode(self) -> None:
        """Detect if the Redis client is a cluster client by inspecting its class."""
        if self.cluster_mode is not None:
            logger.info(
                f"Redis cluster_mode explicitly set to {self.cluster_mode}, skipping detection."
            )
            return

        # Determine cluster mode based on client class
        if isinstance(self._redis, AsyncRedisCluster):
            logger.info("Redis client is a cluster client")
            self.cluster_mode = True
        else:
            logger.info("Redis client is a standalone client")
            self.cluster_mode = False

    async def _apply_ttl_to_keys(
        self,
        main_key: str,
        related_keys: Optional[list[str]] = None,
        ttl_minutes: Optional[float] = None,
    ) -> Any:
        """Apply Redis native TTL to keys asynchronously.

        Args:
            main_key: The primary Redis key
            related_keys: Additional Redis keys that should expire at the same time
            ttl_minutes: Time-to-live in minutes, overrides default_ttl if provided

        Returns:
            Result of the Redis operation
        """
        if ttl_minutes is None:
            # Check if there's a default TTL in config
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

        if ttl_minutes is not None:
            ttl_seconds = int(ttl_minutes * 60)

            if self.cluster_mode:
                # For cluster mode, execute TTL operations individually
                await self._redis.expire(main_key, ttl_seconds)

                if related_keys:
                    for key in related_keys:
                        await self._redis.expire(key, ttl_seconds)

                return True
            else:
                # For non-cluster mode, use pipeline for efficiency
                pipeline = self._redis.pipeline()

                # Set TTL for main key
                pipeline.expire(main_key, ttl_seconds)

                # Set TTL for related keys
                if related_keys:
                    for key in related_keys:
                        pipeline.expire(key, ttl_seconds)

                return await pipeline.execute()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis asynchronously."""
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
        results = await self.checkpoints_index.search(checkpoints_query)
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

            # Get the blob keys
            blob_key_pattern = f"{CHECKPOINT_BLOB_PREFIX}:{to_storage_safe_id(doc_thread_id)}:{to_storage_safe_str(doc_checkpoint_ns)}:*"
            blob_keys = await self._redis.keys(blob_key_pattern)
            # Use safely_decode to handle both string and bytes responses
            blob_keys = [safely_decode(key) for key in blob_keys]

            # Also get checkpoint write keys that should have the same TTL
            write_key_pattern = f"{CHECKPOINT_WRITE_PREFIX}:{to_storage_safe_id(doc_thread_id)}:{to_storage_safe_str(doc_checkpoint_ns)}:{to_storage_safe_id(doc_checkpoint_id)}:*"
            write_keys = await self._redis.keys(write_key_pattern)
            # Use safely_decode to handle both string and bytes responses
            write_keys = [safely_decode(key) for key in write_keys]

            # Apply TTL to checkpoint, blob keys, and write keys
            all_related_keys = blob_keys + write_keys
            await self._apply_ttl_to_keys(
                checkpoint_key, all_related_keys if all_related_keys else None
            )

        # Fetch channel_values
        channel_values = await self.aget_channel_values(
            thread_id=doc_thread_id,
            checkpoint_ns=doc_checkpoint_ns,
            checkpoint_id=doc_checkpoint_id,
        )

        # Fetch pending_sends from parent checkpoint
        pending_sends = []
        if doc_parent_checkpoint_id:
            pending_sends = await self._aload_pending_sends(
                thread_id=thread_id,
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

        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, doc_checkpoint_id
        )

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
            parent_config=None,
            pending_writes=pending_writes,
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

        # Execute the query asynchronously
        results = await self.checkpoints_index.search(query)

        # Process the results
        for doc in results.docs:
            thread_id = from_storage_safe_id(doc["thread_id"])
            checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
            checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
            parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])

            # Fetch channel_values
            channel_values = await self.aget_channel_values(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            )

            # Fetch pending_sends from parent checkpoint
            pending_sends = []
            if parent_checkpoint_id:
                pending_sends = await self._aload_pending_sends(
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

            pending_writes = await self._aload_pending_writes(
                thread_id, checkpoint_ns, checkpoint_id
            )

            yield CheckpointTuple(
                config=config_param,
                checkpoint=checkpoint_param,
                metadata=metadata,
                parent_config=None,
                pending_writes=pending_writes,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
        stream_mode: str = "values",
    ) -> RunnableConfig:
        """Store a checkpoint to Redis with proper transaction handling.

        This method ensures that all Redis operations are performed atomically
        using Redis transactions. In case of interruption (asyncio.CancelledError),
        the transaction will be aborted, ensuring consistency.

        Args:
            config: The config to associate with the checkpoint
            checkpoint: The checkpoint data to store
            metadata: Additional metadata to save with the checkpoint
            new_versions: New channel versions as of this write
            stream_mode: The streaming mode being used (values, updates, etc.)

        Returns:
            Updated configuration after storing the checkpoint

        Raises:
            asyncio.CancelledError: If the operation is cancelled/interrupted
        """
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
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Store checkpoint data with cluster-aware handling
        try:
            # Store checkpoint data
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

            # Prepare checkpoint key
            checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                storage_safe_checkpoint_id,
            )

            # Store blob values
            blobs = self._dump_blobs(
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                copy.get("channel_values", {}),
                new_versions,
            )

            if self.cluster_mode:
                # For cluster mode, execute operations individually
                await self._redis.json().set(checkpoint_key, "$", checkpoint_data)  # type: ignore[misc]

                if blobs:
                    for key, data in blobs:
                        await self._redis.json().set(key, "$", data)  # type: ignore[misc]

                # Apply TTL if configured
                if self.ttl_config and "default_ttl" in self.ttl_config:
                    await self._apply_ttl_to_keys(
                        checkpoint_key,
                        [key for key, _ in blobs] if blobs else None,
                    )
            else:
                # For non-cluster mode, use pipeline with transaction for atomicity
                pipeline = self._redis.pipeline(transaction=True)

                # Add checkpoint data to pipeline
                pipeline.json().set(checkpoint_key, "$", checkpoint_data)

                if blobs:
                    # Add all blob operations to the pipeline
                    for key, data in blobs:
                        pipeline.json().set(key, "$", data)

                # Execute all operations atomically
                await pipeline.execute()

                # Apply TTL to checkpoint and blob keys if configured
                if self.ttl_config and "default_ttl" in self.ttl_config:
                    await self._apply_ttl_to_keys(
                        checkpoint_key,
                        [key for key, _ in blobs] if blobs else None,
                    )

            return next_config

        except asyncio.CancelledError:
            # Handle cancellation/interruption based on stream mode
            if stream_mode in ("values", "messages"):
                # For these modes, we want to ensure any partial state is committed
                # to allow resuming the stream later
                try:
                    # Store minimal checkpoint data
                    checkpoint_data = {
                        "thread_id": storage_safe_thread_id,
                        "checkpoint_ns": storage_safe_checkpoint_ns,
                        "checkpoint_id": storage_safe_checkpoint_id,
                        "parent_checkpoint_id": storage_safe_checkpoint_id,
                        "checkpoint": self._dump_checkpoint(copy),
                        "metadata": self._dump_metadata(
                            {
                                **metadata,
                                "interrupted": True,
                                "stream_mode": stream_mode,
                            }
                        ),
                    }

                    # Prepare checkpoint key
                    checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
                        storage_safe_thread_id,
                        storage_safe_checkpoint_ns,
                        storage_safe_checkpoint_id,
                    )

                    if self.cluster_mode:
                        # For cluster mode, execute operation directly
                        await self._redis.json().set(  # type: ignore[misc]
                            checkpoint_key, "$", checkpoint_data
                        )
                    else:
                        # For non-cluster mode, use pipeline
                        pipeline = self._redis.pipeline(transaction=True)
                        pipeline.json().set(checkpoint_key, "$", checkpoint_data)
                        await pipeline.execute()
                except Exception:
                    # If this also fails, we just propagate the original cancellation
                    pass

            # Re-raise the cancellation
            raise

        except Exception as e:
            # Re-raise other exceptions
            raise e

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint using Redis JSON with transaction handling.

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

        # Transform writes into appropriate format
        writes_objects = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            write_obj = {
                "thread_id": to_storage_safe_id(thread_id),
                "checkpoint_ns": to_storage_safe_str(checkpoint_ns),
                "checkpoint_id": to_storage_safe_id(checkpoint_id),
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "blob": blob,
            }
            writes_objects.append(write_obj)

        try:
            # Determine if this is an upsert case
            upsert_case = all(w[0] in WRITES_IDX_MAP for w in writes)
            created_keys = []

            if self.cluster_mode:
                # For cluster mode, execute operations individually
                for write_obj in writes_objects:
                    key = self._make_redis_checkpoint_writes_key(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_obj["idx"],  # type: ignore[arg-type]
                    )

                    if upsert_case:
                        # For upsert case, check if key exists and update differently
                        exists = await self._redis.exists(key)
                        if exists:
                            # Update existing key
                            await self._redis.json().set(key, "$.channel", write_obj["channel"])  # type: ignore[misc, arg-type]
                            await self._redis.json().set(key, "$.type", write_obj["type"])  # type: ignore[misc, arg-type]
                            await self._redis.json().set(key, "$.blob", write_obj["blob"])  # type: ignore[misc, arg-type]
                        else:
                            # Create new key
                            await self._redis.json().set(key, "$", write_obj)  # type: ignore[misc]
                            created_keys.append(key)
                    else:
                        # For non-upsert case, only set if key doesn't exist
                        exists = await self._redis.exists(key)
                        if not exists:
                            await self._redis.json().set(key, "$", write_obj)  # type: ignore[misc]
                            created_keys.append(key)

                # Apply TTL to newly created keys
                if (
                    created_keys
                    and self.ttl_config
                    and "default_ttl" in self.ttl_config
                ):
                    await self._apply_ttl_to_keys(
                        created_keys[0],
                        created_keys[1:] if len(created_keys) > 1 else None,
                    )
            else:
                # For non-cluster mode, use transaction pipeline for atomicity
                pipeline = self._redis.pipeline(transaction=True)

                # Add all write operations to the pipeline
                for write_obj in writes_objects:
                    key = self._make_redis_checkpoint_writes_key(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_obj["idx"],  # type: ignore[arg-type]
                    )

                    if upsert_case:
                        # For upsert case, we need to check if the key exists and update differently
                        exists = await self._redis.exists(key)
                        if exists:
                            # Update existing key
                            pipeline.json().set(
                                key,
                                "$.channel",
                                write_obj["channel"],  # type: ignore[arg-type]
                            )
                            pipeline.json().set(
                                key,
                                "$.type",
                                write_obj["type"],  # type: ignore[arg-type]
                            )
                            pipeline.json().set(
                                key,
                                "$.blob",
                                write_obj["blob"],  # type: ignore[arg-type]
                            )
                        else:
                            # Create new key
                            pipeline.json().set(key, "$", write_obj)
                            created_keys.append(key)
                    else:
                        # For non-upsert case, only set if key doesn't exist
                        exists = await self._redis.exists(key)
                        if not exists:
                            pipeline.json().set(key, "$", write_obj)
                            created_keys.append(key)

                # Execute all operations atomically
                await pipeline.execute()

                # Apply TTL to newly created keys
                if (
                    created_keys
                    and self.ttl_config
                    and "default_ttl" in self.ttl_config
                ):
                    await self._apply_ttl_to_keys(
                        created_keys[0],
                        created_keys[1:] if len(created_keys) > 1 else None,
                    )

        except asyncio.CancelledError:
            # Handle cancellation/interruption
            # Pipeline will be automatically discarded
            # Either all operations succeed or none do
            raise

        except Exception as e:
            # Re-raise other exceptions
            raise e

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Synchronous wrapper for aput_writes.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (List[Tuple[str, Any]]): List of writes to store.
            task_id (str): Identifier for the task creating the writes.
            task_path (str): Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()

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
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AsyncRedisSaver]:
        async with cls(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args,
            ttl=ttl,
        ) as saver:
            yield saver

    async def aget_channel_values(
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
                filter_expression=(Tag("thread_id") == storage_safe_thread_id)
                & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
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
        self, thread_id: str, checkpoint_ns: str = "", parent_checkpoint_id: str = ""
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
            filter_expression=(
                (Tag("thread_id") == to_storage_safe_id(thread_id))
                & (Tag("checkpoint_ns") == checkpoint_ns)
                & (Tag("checkpoint_id") == to_storage_safe_id(parent_checkpoint_id))
                & (Tag("channel") == TASKS)
            ),
            return_fields=["type", "$.blob", "task_path", "task_id", "idx"],
            num_results=100,
        )
        res = await self.checkpoint_writes_index.search(parent_writes_query)

        # Sort results for deterministic order
        docs = sorted(
            res.docs,
            key=lambda d: (
                getattr(d, "task_path", ""),
                getattr(d, "task_id", ""),
                getattr(d, "idx", 0),
            ),
        )

        # Convert to expected format
        return [
            (d.type.encode(), blob)
            for d in docs
            if (blob := getattr(d, "$.blob", getattr(d, "blob", None))) is not None
        ]

    async def _aload_pending_writes(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        checkpoint_id: str = "",
    ) -> List[PendingWrite]:
        if checkpoint_id is None:
            return []  # Early return if no checkpoint_id

        # Use search index instead of keys() to avoid CrossSlot errors
        # Note: For checkpoint_ns, we use the raw value for tag searches
        # because RediSearch may not handle sentinel values correctly in tag fields
        writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == to_storage_safe_id(thread_id))
            & (Tag("checkpoint_ns") == checkpoint_ns)
            & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)),
            return_fields=["task_id", "idx", "channel", "type", "$.blob"],
            num_results=1000,  # Adjust as needed
        )

        writes_results = await self.checkpoint_writes_index.search(writes_query)

        # Sort results by idx to maintain order
        sorted_writes = sorted(writes_results.docs, key=lambda x: getattr(x, "idx", 0))

        # Build the writes dictionary
        writes_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for doc in sorted_writes:
            task_id = str(getattr(doc, "task_id", ""))
            idx = str(getattr(doc, "idx", 0))
            blob_data = getattr(doc, "$.blob", "")
            # Ensure blob is bytes for deserialization
            if isinstance(blob_data, str):
                blob_data = blob_data.encode("utf-8")
            writes_dict[(task_id, idx)] = {
                "task_id": task_id,
                "idx": idx,
                "channel": str(getattr(doc, "channel", "")),
                "type": str(getattr(doc, "type", "")),
                "blob": blob_data,
            }

        pending_writes = BaseRedisSaver._load_writes(self.serde, writes_dict)
        return pending_writes

    async def adelete_thread(self, thread_id: str) -> None:
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

        checkpoint_results = await self.checkpoints_index.search(checkpoint_query)

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

        blob_results = await self.checkpoint_blobs_index.search(blob_query)

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

        writes_results = await self.checkpoint_writes_index.search(writes_query)

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
                await self._redis.delete(key)
        else:
            # For non-cluster mode, use pipeline for efficiency
            pipeline = self._redis.pipeline()
            for key in keys_to_delete:
                pipeline.delete(key)
            await pipeline.execute()
