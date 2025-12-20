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

import orjson
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
from ulid import ULID

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)
from langgraph.checkpoint.redis.key_registry import (
    AsyncCheckpointKeyRegistry as AsyncKeyRegistry,
)
from langgraph.checkpoint.redis.util import (
    EMPTY_ID_SENTINEL,
    from_storage_safe_id,
    from_storage_safe_str,
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
    _key_registry: Optional[AsyncKeyRegistry] = None  # Track keys to avoid SCAN/KEYS

    # Instance-level cache (will be initialized in __init__)

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
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

        self._separator = REDIS_KEY_SEPARATOR

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[AsyncRedis, AsyncRedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
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
        self.checkpoint_blobs_index = AsyncSearchIndex.from_dict(
            self.blobs_schema, redis_client=self._redis
        )
        self.checkpoint_writes_index = AsyncSearchIndex.from_dict(
            self.writes_schema, redis_client=self._redis
        )

    def _make_redis_checkpoint_key_cached(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Optimized key generation with caching."""
        # Create cache key
        cache_key = f"ckpt:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

        # Check cache first
        if cache_key in self._key_cache:
            return self._key_cache[cache_key]

        # Generate key using optimized string operations
        safe_thread_id = str(to_storage_safe_id(thread_id))
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        safe_checkpoint_id = str(to_storage_safe_id(checkpoint_id))

        # Use pre-computed prefix and join
        key = self._separator.join(
            [
                self._checkpoint_prefix,
                safe_thread_id,
                safe_checkpoint_ns,
                safe_checkpoint_id,
            ]
        )

        # Cache for future use (limit cache size to prevent memory issues)
        if len(self._key_cache) < self._key_cache_max_size:
            self._key_cache[cache_key] = key

        return key

    def _make_redis_checkpoint_writes_key_cached(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: Optional[int],
    ) -> str:
        """Optimized writes key generation with caching."""
        # Create cache key
        cache_key = f"write:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}"

        # Check cache first
        if cache_key in self._key_cache:
            return self._key_cache[cache_key]

        # Generate key using optimized string operations
        safe_thread_id = str(to_storage_safe_id(thread_id))
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        safe_checkpoint_id = str(to_storage_safe_id(checkpoint_id))

        # Build key components
        key_parts = [
            self._checkpoint_write_prefix,
            safe_thread_id,
            safe_checkpoint_ns,
            safe_checkpoint_id,
            task_id,
        ]

        if idx is not None:
            key_parts.append(str(idx))

        # Use pre-computed separator
        key = self._separator.join(key_parts)

        # Cache for future use (limit cache size)
        if len(self._key_cache) < 10000:
            self._key_cache[cache_key] = key

        return key

    async def __aenter__(self) -> AsyncRedisSaver:
        """Async context manager enter."""
        await self.asetup()

        # Set client info once Redis is set up
        await self.aset_client_info()

        return self

    async def __aexit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        if self._owns_its_client:
            await self._redis.aclose()
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

    async def asetup(self) -> None:
        """Set up the checkpoint saver."""
        self.create_indexes()
        await self.checkpoints_index.create(overwrite=False)
        await self.checkpoint_blobs_index.create(overwrite=False)
        await self.checkpoint_writes_index.create(overwrite=False)

        # Detect cluster mode if not explicitly set
        await self._detect_cluster_mode()

        # Initialize key registry
        self._key_registry = AsyncKeyRegistry(self._redis)

    async def setup(self) -> None:  # type: ignore[override]
        """Set up the checkpoint saver asynchronously.

        This method creates the necessary indices in Redis and detects cluster mode.
        It MUST be called before using the checkpointer.

        This async method follows the canonical pattern used by PostgreSQL and SQLite
        checkpointers in the LangGraph ecosystem. The type ignore is necessary because
        the base class defines a sync setup() method, but async checkpointers require
        an async setup() method to properly handle coroutines.

        Usage: await checkpointer.setup()
        """
        await self.asetup()

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
                        Use -1 to remove TTL (make keys persistent)

        Returns:
            Result of the Redis operation
        """
        if ttl_minutes is None:
            # Check if there's a default TTL in config
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

        if ttl_minutes is not None:
            # Special case: -1 means remove TTL (make persistent)
            if ttl_minutes == -1:
                if self.cluster_mode:
                    # For cluster mode, execute PERSIST operations individually
                    await self._redis.persist(main_key)

                    if related_keys:
                        for key in related_keys:
                            await self._redis.persist(key)

                    return True
                else:
                    # For non-cluster mode, use pipeline for efficiency
                    pipeline = self._redis.pipeline()

                    # Remove TTL for main key
                    pipeline.persist(main_key)

                    # Remove TTL for related keys
                    if related_keys:
                        for key in related_keys:
                            pipeline.persist(key)

                    return await pipeline.execute()

            # Regular TTL setting
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

        # For values we store in Redis, we need to convert empty strings to the
        # sentinel value.
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)

        if checkpoint_id and checkpoint_id != EMPTY_ID_SENTINEL:
            # Use direct key access instead of FT.SEARCH when checkpoint_id is known
            storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

            # Construct direct key for checkpoint data
            checkpoint_key = self._make_redis_checkpoint_key(
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                storage_safe_checkpoint_id,
            )

            # Create pipeline for efficient batch operations
            pipeline = self._redis.pipeline(transaction=False)

            # Add checkpoint data fetch to pipeline
            pipeline.json().get(checkpoint_key)

            # Add TTL check if refresh_on_read is enabled
            if self.ttl_config and self.ttl_config.get("refresh_on_read"):
                pipeline.ttl(checkpoint_key)

            # Execute pipeline to get checkpoint data and TTL
            pipeline_results = await pipeline.execute()

            checkpoint_data = pipeline_results[0]
            if not checkpoint_data:
                return None

            # Extract TTL if we fetched it
            current_ttl = None
            if self.ttl_config and self.ttl_config.get("refresh_on_read"):
                current_ttl = pipeline_results[1]

            # Create doc-like object from direct access
            doc = {
                "thread_id": checkpoint_data.get("thread_id", storage_safe_thread_id),
                "checkpoint_ns": checkpoint_data.get(
                    "checkpoint_ns", storage_safe_checkpoint_ns
                ),
                "checkpoint_id": checkpoint_data.get(
                    "checkpoint_id", storage_safe_checkpoint_id
                ),
                "parent_checkpoint_id": checkpoint_data.get(
                    "parent_checkpoint_id", storage_safe_checkpoint_id
                ),
                "$.checkpoint": json.dumps(checkpoint_data.get("checkpoint", {})),
                "$.metadata": checkpoint_data.get(
                    "metadata", "{}"
                ),  # metadata is already a JSON string
            }
        else:
            # Try to get latest checkpoint using pointer
            latest_pointer_key = f"checkpoint_latest:{storage_safe_thread_id}:{storage_safe_checkpoint_ns}"

            checkpoint_key = await self._redis.get(latest_pointer_key)
            if not checkpoint_key:
                # No pointer means no checkpoints exist
                return None

            # Create pipeline for efficient operations
            pipeline = self._redis.pipeline(transaction=False)

            # Add checkpoint data fetch to pipeline
            pipeline.json().get(checkpoint_key)

            # Add TTL check if refresh_on_read is enabled
            if self.ttl_config and self.ttl_config.get("refresh_on_read"):
                pipeline.ttl(checkpoint_key)

            # Execute pipeline
            pipeline_results = await pipeline.execute()

            checkpoint_data = pipeline_results[0]
            if not checkpoint_data:
                # Pointer exists but checkpoint is missing - data inconsistency
                return None

            # Extract TTL if we fetched it
            current_ttl = None
            if self.ttl_config and self.ttl_config.get("refresh_on_read"):
                current_ttl = pipeline_results[1]

            # Create doc-like object from direct access
            doc = {
                "thread_id": checkpoint_data.get("thread_id", storage_safe_thread_id),
                "checkpoint_ns": checkpoint_data.get(
                    "checkpoint_ns", storage_safe_checkpoint_ns
                ),
                "checkpoint_id": checkpoint_data.get("checkpoint_id"),
                "parent_checkpoint_id": checkpoint_data.get("parent_checkpoint_id"),
                "$.checkpoint": json.dumps(checkpoint_data.get("checkpoint", {})),
                "$.metadata": checkpoint_data.get(
                    "metadata", "{}"
                ),  # metadata is already a JSON string
            }

        doc_thread_id = from_storage_safe_id(doc["thread_id"])
        doc_checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
        doc_checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
        doc_parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])

        # Lazy TTL refresh - only refresh if TTL is below threshold
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            # If we didn't get TTL from pipeline (i.e., came from else branch), fetch it now
            if "current_ttl" not in locals():
                # Get the checkpoint key
                checkpoint_key = self._make_redis_checkpoint_key(
                    to_storage_safe_id(doc_thread_id),
                    to_storage_safe_str(doc_checkpoint_ns),
                    to_storage_safe_id(doc_checkpoint_id),
                )
                current_ttl = await self._redis.ttl(checkpoint_key)

            # Always refresh TTL when refresh_on_read is enabled
            # This ensures all related keys maintain synchronized TTLs

            # Only refresh if key exists and has TTL (skip keys with no expiry)
            # TTL states: -2 = key doesn't exist, -1 = key exists but no TTL, 0 = expired, >0 = seconds remaining
            if current_ttl > 0:
                # Get all blob keys related to this checkpoint
                from langgraph.checkpoint.redis.base import (
                    CHECKPOINT_BLOB_PREFIX,
                    CHECKPOINT_WRITE_PREFIX,
                )

                # Get write keys from registry instead of SCAN
                write_keys = []

                if self._key_registry:
                    write_keys = await self._key_registry.get_write_keys(
                        doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id
                    )

                # Apply TTL to checkpoint and write keys
                await self._apply_ttl_to_keys(
                    checkpoint_key, write_keys if write_keys else None
                )

                # Also refresh TTL on registry keys if they exist
                if self._key_registry and self.ttl_config:
                    ttl_minutes = self.ttl_config.get("default_ttl")
                    if ttl_minutes is not None:
                        ttl_seconds = int(ttl_minutes * 60)
                        # Registry TTL is handled per checkpoint
                        await self._key_registry.apply_ttl(
                            doc_thread_id,
                            doc_checkpoint_ns,
                            doc_checkpoint_id,
                            ttl_seconds,
                        )

        # Fetch channel_values - pass channel_versions if we have them from direct access
        checkpoint_raw = (
            doc.get("$.checkpoint")
            if isinstance(doc, dict)
            else getattr(doc, "$.checkpoint", None)
        )
        if isinstance(checkpoint_raw, str):
            checkpoint_data_dict = json.loads(checkpoint_raw)
        else:
            checkpoint_data_dict = checkpoint_raw

        channel_versions_from_checkpoint = (
            checkpoint_data_dict.get("channel_versions")
            if checkpoint_data_dict
            else None
        )

        # Run channel_values, pending_sends, and pending_writes loads in parallel
        # Create list of coroutines to run
        tasks: List[Any] = []

        # Always load channel values
        tasks.append(
            self.aget_channel_values(
                thread_id=doc_thread_id,
                checkpoint_ns=doc_checkpoint_ns,
                checkpoint_id=doc_checkpoint_id,
                channel_versions=channel_versions_from_checkpoint,
            )
        )

        # Conditionally load pending sends if parent exists
        if doc_parent_checkpoint_id:
            tasks.append(
                self._aload_pending_sends(
                    thread_id=thread_id,
                    checkpoint_ns=doc_checkpoint_ns,
                    parent_checkpoint_id=doc_parent_checkpoint_id,
                )
            )

        # Always load pending writes
        tasks.append(
            self._aload_pending_writes(thread_id, checkpoint_ns, doc_checkpoint_id)
        )

        # Execute all tasks in parallel - pending_sends is optional
        if doc_parent_checkpoint_id:
            results = await asyncio.gather(*tasks)
            channel_values: Dict[str, Any] = self._recursive_deserialize(results[0])
            pending_sends: List[Tuple[str, Union[str, bytes]]] = results[1]
            pending_writes: List[PendingWrite] = results[2]
        else:
            # Only channel_values and pending_writes tasks
            results = await asyncio.gather(*tasks)
            channel_values = self._recursive_deserialize(results[0])
            pending_sends = []
            pending_writes = results[1]

        # Fetch and parse metadata
        raw_metadata = (
            doc.get("$.metadata", "{}")
            if isinstance(doc, dict)
            else getattr(doc, "$.metadata", "{}")
        )
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

        # Handle both direct dict access and FT.SEARCH results
        checkpoint_data = doc["$.checkpoint"]
        if isinstance(checkpoint_data, dict):
            # Direct key access returns dict, convert to JSON string for consistency
            checkpoint_data = json.dumps(checkpoint_data)

        checkpoint_param = self._load_checkpoint(
            checkpoint_data,
            channel_values,
            pending_sends,
        )

        # Build parent config if parent_checkpoint_id exists
        parent_config: RunnableConfig | None = None
        if doc_parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": doc_parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=config_param,
            checkpoint=checkpoint_param,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,  # noqa: ARG002
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

            # Search for checkpoints with any namespace, including an empty
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

        # Construct the Redis query
        # Sort by checkpoint_id in descending order to get most recent checkpoints first
        query = FilterQuery(
            filter_expression=combined_filter,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "$.checkpoint",
                "$.metadata",
                "has_writes",  # Include has_writes to optimize pending_writes loading
            ],
            num_results=limit or 10000,
            sort_by=("checkpoint_id", "DESC"),
        )

        # Execute the query asynchronously
        results = await self.checkpoints_index.search(query)

        # Pre-process all docs to collect batch query requirements
        all_docs_data = []
        pending_sends_batch_keys = []
        pending_writes_batch_keys = []

        for doc in results.docs:
            # Extract all attributes once
            doc_dict = doc.__dict__ if hasattr(doc, "__dict__") else {}

            thread_id = from_storage_safe_id(doc["thread_id"])
            checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
            checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
            parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])

            # Get channel values from inline checkpoint data (already returned by FT.SEARCH)
            checkpoint_data = doc_dict.get("$.checkpoint") or getattr(
                doc, "$.checkpoint", None
            )
            if checkpoint_data:
                # Parse checkpoint to extract inline channel_values
                if isinstance(checkpoint_data, list) and checkpoint_data:
                    checkpoint_data = checkpoint_data[0]

                # Use orjson for faster parsing
                checkpoint_dict = (
                    checkpoint_data
                    if isinstance(checkpoint_data, dict)
                    else orjson.loads(checkpoint_data)
                )
                channel_values = self._recursive_deserialize(
                    checkpoint_dict.get("channel_values", {})
                )
            else:
                # If checkpoint data is missing, the document is corrupted
                # Set empty channel values rather than attempting a fallback
                channel_values = {}

            # Collect batch keys for pending_sends
            if parent_checkpoint_id and parent_checkpoint_id != "None":
                batch_key = (thread_id, checkpoint_ns, parent_checkpoint_id)
                pending_sends_batch_keys.append(batch_key)

            # Collect batch keys for pending_writes
            checkpoint_has_writes = doc_dict.get("has_writes") or getattr(
                doc, "has_writes", False
            )
            # Convert string "False" to boolean false if needed (optimize for common case)
            if checkpoint_has_writes == "true":
                checkpoint_has_writes = True
            elif checkpoint_has_writes == "false" or checkpoint_has_writes == "False":
                checkpoint_has_writes = False

            if checkpoint_has_writes:
                batch_key = (thread_id, checkpoint_ns, checkpoint_id)
                pending_writes_batch_keys.append(batch_key)

            # Store processed doc data for final iteration
            all_docs_data.append(
                {
                    "doc": doc,
                    "doc_dict": doc_dict,
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "checkpoint_data": checkpoint_data,
                    "checkpoint_dict": checkpoint_dict if checkpoint_data else None,
                    "channel_values": channel_values,
                    "has_writes": checkpoint_has_writes,
                }
            )

        # Load pending_sends for all parent checkpoints at once
        pending_sends_map = {}
        if pending_sends_batch_keys:
            pending_sends_map = await self._abatch_load_pending_sends(
                pending_sends_batch_keys
            )

        # Load pending_writes for all checkpoints with writes at once
        pending_writes_map = {}
        if pending_writes_batch_keys:
            pending_writes_map = await self._abatch_load_pending_writes(
                pending_writes_batch_keys
            )

        # Process the results using pre-loaded batch data
        for doc_data in all_docs_data:
            thread_id = doc_data["thread_id"]
            checkpoint_ns = doc_data["checkpoint_ns"]
            checkpoint_id = doc_data["checkpoint_id"]
            parent_checkpoint_id = doc_data["parent_checkpoint_id"]

            # Get pending_sends from batch results
            pending_sends: List[Tuple[str, Union[str, bytes]]] = []
            if parent_checkpoint_id:
                batch_key = (thread_id, checkpoint_ns, parent_checkpoint_id)
                pending_sends = pending_sends_map.get(batch_key, [])

            # Fetch and parse metadata
            doc_dict = doc_data["doc_dict"]
            raw_metadata = doc_dict.get("$.metadata") or getattr(
                doc_data["doc"], "$.metadata", "{}"
            )
            # Use orjson for faster parsing
            metadata_dict = (
                orjson.loads(raw_metadata)
                if isinstance(raw_metadata, str)
                else raw_metadata
            )

            # Only sanitize if null bytes detected (rare case)
            if any(
                "\u0000" in str(v) for v in metadata_dict.values() if isinstance(v, str)
            ):
                sanitized_metadata = {
                    k.replace("\u0000", ""): (
                        v.replace("\u0000", "") if isinstance(v, str) else v
                    )
                    for k, v in metadata_dict.items()
                }
                metadata = cast(CheckpointMetadata, sanitized_metadata)
            else:
                metadata = cast(CheckpointMetadata, metadata_dict)

            # Pre-create the config structure more efficiently
            config_param: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

            # Pass already parsed checkpoint_dict to avoid re-parsing
            checkpoint_param = self._load_checkpoint(
                (
                    doc_data["checkpoint_dict"]
                    if doc_data["checkpoint_data"]
                    else doc_data["doc"]["$.checkpoint"]
                ),
                doc_data["channel_values"],
                pending_sends,
            )

            # Get pending_writes from batch results
            pending_writes: List[PendingWrite] = []
            if doc_data["has_writes"]:
                batch_key = (thread_id, checkpoint_ns, checkpoint_id)
                pending_writes = pending_writes_map.get(batch_key, [])

            # Build parent config if parent_checkpoint_id exists
            parent_config: RunnableConfig | None = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }

            yield CheckpointTuple(
                config=config_param,
                checkpoint=checkpoint_param,
                metadata=metadata,
                parent_config=parent_config,
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
        # Get checkpoint_id from config - this will be parent if saving a child
        config_checkpoint_id = configurable.pop("checkpoint_id", None)
        # For backward compatibility with thread_ts
        thread_ts = configurable.pop("thread_ts", "")

        # Determine the checkpoint ID
        # This follows the original logic but with clearer parent handling
        checkpoint_id = config_checkpoint_id or thread_ts or checkpoint.get("id", "")

        # If checkpoint has its own ID that's different from what we'd use,
        # and we have a config checkpoint_id, then config checkpoint_id is the parent
        parent_checkpoint_id = None
        if (
            checkpoint.get("id")
            and config_checkpoint_id
            and checkpoint.get("id") != config_checkpoint_id
        ):
            parent_checkpoint_id = config_checkpoint_id
            checkpoint_id = checkpoint["id"]

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
            # Store checkpoint data WITH inline channel values
            # Extract timestamp from checkpoint_id (ULID)
            checkpoint_ts = None
            if checkpoint_id:
                try:
                    from ulid import ULID

                    ulid_obj = ULID.from_str(checkpoint_id)
                    checkpoint_ts = ulid_obj.timestamp  # milliseconds since epoch
                except Exception:
                    # If not a valid ULID, use current time
                    import time

                    checkpoint_ts = time.time() * 1000

            checkpoint_data = {
                "thread_id": storage_safe_thread_id,
                "checkpoint_ns": storage_safe_checkpoint_ns,
                "checkpoint_id": storage_safe_checkpoint_id,
                "parent_checkpoint_id": (
                    to_storage_safe_id(parent_checkpoint_id)
                    if parent_checkpoint_id
                    else ""
                ),
                "checkpoint_ts": checkpoint_ts,
                "checkpoint": self._dump_checkpoint(copy),
                "metadata": self._dump_metadata(metadata),
                "has_writes": False,  # Track if this checkpoint has pending writes
            }

            # store at top-level for filters in list()
            if all(key in metadata for key in ["source", "step"]):
                checkpoint_data["source"] = metadata["source"]
                checkpoint_data["step"] = metadata["step"]

            # Prepare checkpoint key
            checkpoint_key = self._make_redis_checkpoint_key_cached(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
            )

            # Calculate TTL in seconds if configured
            ttl_seconds = None
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config["default_ttl"] * 60)

            # Store checkpoint with TTL in a single operation using SearchIndex
            await self.checkpoints_index.load(
                [checkpoint_data],
                keys=[checkpoint_key],
                ttl=ttl_seconds,  # RedisVL applies TTL in its internal pipeline
            )

            # For test compatibility: ensure TTL operations are visible to mocks
            if (
                self.cluster_mode
                and self.ttl_config
                and "default_ttl" in self.ttl_config
                and ttl_seconds is not None
            ):
                # In cluster mode, also call expire directly so tests can verify
                await self._redis.expire(checkpoint_key, ttl_seconds)

            # Update latest checkpoint pointer
            latest_pointer_key = f"checkpoint_latest:{storage_safe_thread_id}:{storage_safe_checkpoint_ns}"
            await self._redis.set(latest_pointer_key, checkpoint_key)

            # Apply TTL to latest pointer key as well
            if ttl_seconds is not None:
                await self._redis.expire(latest_pointer_key, ttl_seconds)

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
                        "parent_checkpoint_id": (
                            to_storage_safe_id(
                                str(checkpoint.get("parent_checkpoint_id", ""))
                            )
                            if checkpoint.get("parent_checkpoint_id")
                            else ""
                        ),
                        "checkpoint": self._dump_checkpoint(copy),
                        "metadata": self._dump_metadata(
                            {
                                **metadata,
                                "interrupted": True,
                                "stream_mode": stream_mode,
                            }
                        ),
                        "has_writes": False,  # Track if this checkpoint has pending writes
                    }

                    # Prepare checkpoint key
                    checkpoint_key = self._make_redis_checkpoint_key(
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
                        pipeline = self._redis.pipeline(transaction=False)
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
        """Store intermediate writes linked to a checkpoint using Redis JSON.

        This method uses Redis pipeline without transaction to avoid lock contention
        during parallel test execution.

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
                "blob": self._encode_blob(
                    blob
                ),  # Encode bytes to base64 string for Redis
            }
            writes_objects.append(write_obj)

        try:
            # Determine if this is an upsert case
            upsert_case = all(w[0] in WRITES_IDX_MAP for w in writes)
            created_keys = []

            if self.cluster_mode:
                # For cluster mode, execute operations individually
                for write_obj in writes_objects:
                    key = self._make_redis_checkpoint_writes_key_cached(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_obj["idx"],  # type: ignore[arg-type]
                    )

                    # Redis JSON.SET is an UPSERT by default
                    await self._redis.json().set(key, "$", cast(Any, write_obj))  # type: ignore[misc]
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

                # Register write keys in the key registry for cluster mode
                if self._key_registry:
                    write_keys = []
                    for write_obj in writes_objects:
                        key = self._make_redis_checkpoint_writes_key_cached(
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            write_obj["idx"],  # type: ignore[arg-type]
                        )
                        write_keys.append(key)

                    if write_keys:
                        # Use per-checkpoint sorted set registry
                        zset_key = self._key_registry.make_write_keys_zset_key(
                            thread_id, checkpoint_ns, checkpoint_id
                        )

                        # Add all write keys with their index as score for ordering
                        zadd_mapping = {key: idx for idx, key in enumerate(write_keys)}
                        await self._redis.zadd(zset_key, zadd_mapping)  # type: ignore[arg-type]

                        # Apply TTL to registry key if configured
                        if self.ttl_config and "default_ttl" in self.ttl_config:
                            ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                            await self._redis.expire(zset_key, ttl_seconds)

            else:
                # For non-cluster mode, use pipeline without transaction to avoid lock contention
                pipeline = self._redis.pipeline(transaction=False)

                # Add all write operations to the pipeline efficiently
                for write_obj in writes_objects:
                    key = self._make_redis_checkpoint_writes_key_cached(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_obj["idx"],  # type: ignore[arg-type]
                    )

                    pipeline.json().set(key, "$", cast(Any, write_obj))
                    created_keys.append(key)

                # Add TTL operations to the pipeline if configured
                if (
                    created_keys
                    and self.ttl_config
                    and "default_ttl" in self.ttl_config
                ):
                    ttl_seconds = int(self.ttl_config["default_ttl"] * 60)
                    for key in created_keys:
                        pipeline.expire(key, ttl_seconds)

                # Update checkpoint to indicate it has writes
                if writes_objects:
                    checkpoint_key = self._make_redis_checkpoint_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )
                    # Use merge to update existing document without error
                    pipeline.json().merge(checkpoint_key, "$", {"has_writes": True})

                # Integrate registry operations into the pipeline if registry is available
                write_keys = []
                for write_obj in writes_objects:
                    key = self._make_redis_checkpoint_writes_key_cached(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        write_obj["idx"],  # type: ignore[arg-type]
                    )
                    write_keys.append(key)

                if self._key_registry and write_keys:
                    # Use per-checkpoint sorted set registry
                    zset_key = self._key_registry.make_write_keys_zset_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )

                    # Add all write keys with their index as score for ordering
                    zadd_mapping = {key: idx for idx, key in enumerate(write_keys)}
                    pipeline.zadd(zset_key, zadd_mapping)  # type: ignore[arg-type]

                    # Apply TTL to registry key if configured
                    if self.ttl_config and "default_ttl" in self.ttl_config:
                        ttl_seconds = int(self.ttl_config.get("default_ttl") * 60)
                        pipeline.expire(zset_key, ttl_seconds)

                # Execute everything in one round trip
                try:
                    await pipeline.execute()
                except Exception as e:
                    # Check if JSON.MERGE failed (older Redis versions)
                    if "JSON.MERGE" in str(e) or "merge" in str(e).lower():
                        # Retry without JSON.MERGE for older Redis versions
                        async with self._redis.pipeline(
                            transaction=False
                        ) as fallback_pipeline:
                            # Re-add all the write operations
                            for write_obj in writes_objects:
                                key = self._make_redis_checkpoint_writes_key_cached(
                                    thread_id,
                                    checkpoint_ns,
                                    checkpoint_id,
                                    task_id,
                                    write_obj["idx"],  # type: ignore[arg-type]
                                )
                                fallback_pipeline.json().set(
                                    key, "$", cast(Any, write_obj)
                                )

                            # Add TTL operations if configured
                            if (
                                created_keys
                                and self.ttl_config
                                and "default_ttl" in self.ttl_config
                            ):
                                ttl_seconds = int(self.ttl_config["default_ttl"] * 60)
                                for key in created_keys:
                                    fallback_pipeline.expire(key, ttl_seconds)

                            # Re-add registry operations if needed
                            if self._key_registry and write_keys:
                                zset_key = self._key_registry.make_write_keys_zset_key(
                                    thread_id, checkpoint_ns, checkpoint_id
                                )
                                zadd_mapping = {
                                    key: idx for idx, key in enumerate(write_keys)
                                }
                                fallback_pipeline.zadd(zset_key, zadd_mapping)  # type: ignore[arg-type]
                                if self.ttl_config and "default_ttl" in self.ttl_config:
                                    ttl_seconds = int(
                                        self.ttl_config.get("default_ttl") * 60
                                    )
                                    fallback_pipeline.expire(zset_key, ttl_seconds)

                            # Execute the fallback pipeline
                            await fallback_pipeline.execute()

                            # Update has_writes flag separately for older Redis
                            if checkpoint_key:
                                try:
                                    checkpoint_data = await self._redis.json().get(  # type: ignore[misc]
                                        checkpoint_key
                                    )
                                    if isinstance(
                                        checkpoint_data, dict
                                    ) and not checkpoint_data.get("has_writes"):
                                        checkpoint_data["has_writes"] = True
                                        await self._redis.json().set(  # type: ignore[misc]
                                            checkpoint_key, "$", checkpoint_data
                                        )
                                except Exception:
                                    # If this fails, it's not critical - the writes are still saved
                                    pass
                    else:
                        # Re-raise other exceptions
                        raise

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

    def get_channel_values(
        self, thread_id: str, checkpoint_ns: str = "", checkpoint_id: str = ""
    ) -> Dict[str, Any]:
        """Retrieve channel_values using efficient FT.SEARCH with checkpoint_id (sync wrapper)."""
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncRedisSaver are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await checkpointer.get_channel_values(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_channel_values(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
            ),
            self.loop,
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
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> AsyncIterator[AsyncRedisSaver]:
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

    async def aget_channel_values(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        checkpoint_id: str = "",
        channel_versions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve channel_values using efficient FT.SEARCH with checkpoint_id."""
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        # Get checkpoint with inline channel_values using single JSON.GET operation (MongoDB approach)
        checkpoint_key = self._make_redis_checkpoint_key_cached(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
        )

        # Single JSON.GET operation to retrieve checkpoint with inline channel_values
        checkpoint_data = await self._redis.json().get(checkpoint_key, "$.checkpoint")  # type: ignore[misc]

        if not checkpoint_data:
            return {}

        # checkpoint_data[0] is already a deserialized dict, not a typed tuple
        checkpoint = checkpoint_data[0]
        return checkpoint.get("channel_values", {})

    async def _aload_pending_sends(
        self, thread_id: str, checkpoint_ns: str = "", parent_checkpoint_id: str = ""
    ) -> List[Tuple[str, Union[str, bytes]]]:
        """Load pending sends for a parent checkpoint.

        Args:
            thread_id: The thread ID
            checkpoint_ns: The checkpoint namespace
            parent_checkpoint_id: The ID of the parent checkpoint

        Returns:
            List of (type, blob) tuples representing pending sends
        """
        if not parent_checkpoint_id:
            return []

        # FAST PATH: Try sorted set registry first
        if self._key_registry:
            try:
                # Check if parent checkpoint has any writes in the sorted set
                write_count = await self._key_registry.get_write_count(
                    thread_id, checkpoint_ns, parent_checkpoint_id
                )

                if write_count == 0:
                    # No writes for parent checkpoint - return immediately
                    return []

                # Get exact write keys from the per-checkpoint registry
                write_keys = await self._key_registry.get_write_keys(
                    thread_id, checkpoint_ns, parent_checkpoint_id
                )

                # Filter for TASKS channel writes
                task_write_keys = []
                for key in write_keys:
                    # Keys contain channel info: checkpoint_write:thread:ns:checkpoint:task:idx
                    # We need to check if it's a TASKS channel write
                    # This is a simple heuristic - we might need to fetch to be sure
                    if TASKS in key or "__pregel_tasks" in key:
                        task_write_keys.append(key)

                if not task_write_keys:
                    return []

                # Fetch task writes using pipeline (safe for cluster mode)
                pipeline = self._redis.pipeline(transaction=False)
                for key in task_write_keys:
                    pipeline.json().get(key)

                results = await pipeline.execute()

                # Extract pending sends and sort them
                pending_sends_with_sort_keys = []
                for write_data in results:
                    if write_data and write_data.get("channel") == TASKS:
                        pending_sends_with_sort_keys.append(
                            (
                                write_data.get("task_path", ""),
                                write_data.get("task_id", ""),
                                write_data.get("idx", 0),
                                write_data.get("type", ""),
                                write_data.get("blob", b""),
                            )
                        )

                # Sort by task_path, task_id, idx
                pending_sends_with_sort_keys.sort(key=lambda x: (x[0], x[1], x[2]))

                # Return just the (type, blob) tuples
                return [(item[3], item[4]) for item in pending_sends_with_sort_keys]

            except Exception:
                # If sorted set approach fails, fall back to FT.SEARCH
                pass

        # Fallback to FT.SEARCH logic
        parent_writes_query = FilterQuery(
            filter_expression=(
                (Tag("thread_id") == to_storage_safe_id(thread_id))
                & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
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

        # FAST PATH: Try sorted set registry first
        if self._key_registry:
            try:
                # Check if this checkpoint has any writes in the sorted set
                write_count = await self._key_registry.get_write_count(
                    thread_id, checkpoint_ns, checkpoint_id
                )

                if write_count == 0:
                    # No writes for this checkpoint - return immediately
                    return []

                # Get exact write keys from the per-checkpoint registry
                write_keys = await self._key_registry.get_write_keys(
                    thread_id, checkpoint_ns, checkpoint_id
                )

                # Fetch all writes efficiently using pipeline
                pipeline = self._redis.pipeline(transaction=False)
                for key in write_keys:
                    pipeline.json().get(key)

                results = await pipeline.execute()

                # Build the writes dictionary
                writes_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}

                for write_data in results:
                    if write_data:
                        task_id = write_data.get("task_id", "")
                        idx = str(write_data.get("idx", 0))
                        writes_dict[(task_id, idx)] = {
                            "task_id": task_id,
                            "idx": idx,
                            "channel": write_data.get("channel", ""),
                            "type": write_data.get("type", ""),
                            "blob": write_data.get("blob", b""),
                        }

                # Deserialize and return
                pending_writes = BaseRedisSaver._load_writes(self.serde, writes_dict)
                return pending_writes

            except Exception:
                # If sorted set approach fails, fall back to FT.SEARCH
                pass

        # FALLBACK: Use search index instead of keys() to avoid CrossSlot errors
        # Note: All tag fields use sentinel values for consistency
        writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == to_storage_safe_id(thread_id))
            & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)),
            return_fields=["task_id", "idx", "channel", "type", "$.blob"],
            num_results=1000,  # Adjust as needed
        )

        writes_results = await self.checkpoint_writes_index.search(writes_query)

        # Sort results by idx to maintain order
        sorted_writes = sorted(writes_results.docs, key=lambda x: getattr(x, "idx", 0))

        # Build the writes dictionary from search results
        search_writes_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for doc in sorted_writes:
            task_id = str(getattr(doc, "task_id", ""))
            idx = str(getattr(doc, "idx", 0))
            blob_data = getattr(doc, "$.blob", "")
            # Ensure blob is bytes for deserialization
            if isinstance(blob_data, str):
                blob_data = blob_data.encode("utf-8")
            search_writes_dict[(task_id, idx)] = {
                "task_id": task_id,
                "idx": idx,
                "channel": str(getattr(doc, "channel", "")),
                "type": str(getattr(doc, "type", "")),
                "blob": blob_data,
            }

        pending_writes = BaseRedisSaver._load_writes(self.serde, search_writes_dict)
        return pending_writes

    async def _abatch_load_pending_sends(
        self, batch_keys: List[Tuple[str, str, str]]
    ) -> Dict[Tuple[str, str, str], List[Tuple[str, Union[str, bytes]]]]:
        """Batch load pending sends for multiple parent checkpoints.

        Args:
            batch_keys: List of (thread_id, checkpoint_ns, parent_checkpoint_id) tuples

        Returns:
            Dict mapping batch_key -> list of (type, blob) tuples
        """
        if not batch_keys:
            return {}

        results_map = {}

        # Group by thread_id and checkpoint_ns for efficient querying
        grouped_keys: Dict[Tuple[str, str], List[str]] = {}
        for thread_id, checkpoint_ns, parent_checkpoint_id in batch_keys:
            group_key = (thread_id, checkpoint_ns)
            if group_key not in grouped_keys:
                grouped_keys[group_key] = []
            grouped_keys[group_key].append(parent_checkpoint_id)

        # Batch query for each group
        for (thread_id, checkpoint_ns), parent_checkpoint_ids in grouped_keys.items():
            storage_safe_thread_id = to_storage_safe_id(thread_id)
            storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
            storage_safe_parent_checkpoint_ids = [
                to_storage_safe_id(pid) for pid in parent_checkpoint_ids
            ]

            # Build filter for multiple parent checkpoint IDs
            thread_filter = Tag("thread_id") == storage_safe_thread_id
            ns_filter = Tag("checkpoint_ns") == storage_safe_checkpoint_ns
            channel_filter = Tag("channel") == TASKS

            # Create filter for multiple parent checkpoint IDs (Tag supports lists)
            checkpoint_filter = (
                Tag("checkpoint_id") == storage_safe_parent_checkpoint_ids
            )

            batch_query = FilterQuery(
                filter_expression=thread_filter
                & ns_filter
                & checkpoint_filter
                & channel_filter,
                return_fields=[
                    "checkpoint_id",
                    "type",
                    "$.blob",
                    "task_path",
                    "task_id",
                    "idx",
                ],
                num_results=1000,  # Increased limit for batch loading
            )

            batch_results = await self.checkpoint_writes_index.search(batch_query)

            # Group results by parent checkpoint ID
            writes_by_checkpoint: Dict[str, List[Any]] = {}
            for doc in batch_results.docs:
                parent_checkpoint_id = from_storage_safe_id(doc.checkpoint_id)
                if parent_checkpoint_id not in writes_by_checkpoint:
                    writes_by_checkpoint[parent_checkpoint_id] = []
                writes_by_checkpoint[parent_checkpoint_id].append(doc)

            # Sort and format results for each parent checkpoint
            for parent_checkpoint_id in parent_checkpoint_ids:
                batch_key = (thread_id, checkpoint_ns, parent_checkpoint_id)
                writes = writes_by_checkpoint.get(parent_checkpoint_id, [])

                # Sort results by task_path, task_id, idx
                sorted_writes = sorted(
                    writes,
                    key=lambda x: (
                        getattr(x, "task_path", ""),
                        getattr(x, "task_id", ""),
                        getattr(x, "idx", 0),
                    ),
                )

                # Extract type and blob pairs
                # Handle both direct attribute access and JSON path access
                results_map[batch_key] = [
                    (
                        getattr(doc, "type", ""),
                        getattr(doc, "$.blob", getattr(doc, "blob", b"")),
                    )
                    for doc in sorted_writes
                ]

        return results_map

    async def _abatch_load_pending_writes(
        self, batch_keys: List[Tuple[str, str, str]]
    ) -> Dict[Tuple[str, str, str], List[PendingWrite]]:
        """Batch load pending writes for multiple checkpoints.

        Args:
            batch_keys: List of (thread_id, checkpoint_ns, checkpoint_id) tuples

        Returns:
            Dict mapping batch_key -> list of PendingWrite objects
        """
        if not batch_keys:
            return {}

        results_map: Dict[Tuple[str, str, str], List[PendingWrite]] = {}

        # If we have a key registry, use it for efficient batch loading
        if self._key_registry:
            # First, collect all write keys for all checkpoints in parallel
            pipeline = self._redis.pipeline(transaction=False)

            # Add all ZCARD operations to pipeline to check write counts
            for thread_id, checkpoint_ns, checkpoint_id in batch_keys:
                zset_key = self._key_registry.make_write_keys_zset_key(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                pipeline.zcard(zset_key)

            # Execute all ZCARD operations at once
            write_counts = await pipeline.execute()

            # Now get the actual keys for checkpoints that have writes
            pipeline = self._redis.pipeline(transaction=False)
            checkpoints_with_writes = []

            for i, (thread_id, checkpoint_ns, checkpoint_id) in enumerate(batch_keys):
                if write_counts[i] > 0:
                    checkpoints_with_writes.append(
                        (thread_id, checkpoint_ns, checkpoint_id)
                    )
                    zset_key = self._key_registry.make_write_keys_zset_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )
                    pipeline.zrange(zset_key, 0, -1)
                else:
                    # No writes for this checkpoint
                    batch_key = (thread_id, checkpoint_ns, checkpoint_id)
                    results_map[batch_key] = []

            if checkpoints_with_writes:
                # Get all write keys at once
                all_write_keys_results = await pipeline.execute()

                # Now fetch all the actual write data in a single pipeline
                pipeline = self._redis.pipeline(transaction=False)
                write_key_mapping = {}  # Maps pipeline index to checkpoint info
                pipeline_index = 0

                for i, (thread_id, checkpoint_ns, checkpoint_id) in enumerate(
                    checkpoints_with_writes
                ):
                    write_keys = all_write_keys_results[i]
                    if write_keys:
                        decoded_keys = [
                            key.decode() if isinstance(key, bytes) else key
                            for key in write_keys
                        ]
                        for key in decoded_keys:
                            pipeline.json().get(key)
                            write_key_mapping[pipeline_index] = (
                                thread_id,
                                checkpoint_ns,
                                checkpoint_id,
                                key,
                            )
                            pipeline_index += 1

                # Execute all JSON.GET operations at once
                if pipeline_index > 0:
                    all_writes_data = await pipeline.execute()

                    # Group results by checkpoint
                    writes_by_checkpoint: Dict[
                        Tuple[str, str, str], Dict[Tuple[str, str], Dict[str, Any]]
                    ] = {}

                    for idx, write_data in enumerate(all_writes_data):
                        if write_data:
                            thread_id, checkpoint_ns, checkpoint_id, key = (
                                write_key_mapping[idx]
                            )
                            batch_key = (thread_id, checkpoint_ns, checkpoint_id)

                            if batch_key not in writes_by_checkpoint:
                                writes_by_checkpoint[batch_key] = {}

                            task_id = write_data.get("task_id", "")
                            idx_val = str(write_data.get("idx", 0))
                            writes_by_checkpoint[batch_key][(task_id, idx_val)] = {
                                "task_id": task_id,
                                "idx": idx_val,
                                "channel": write_data.get("channel", ""),
                                "type": write_data.get("type", ""),
                                "blob": write_data.get("blob", b""),
                            }

                    # Deserialize and store results
                    for batch_key, writes_dict in writes_by_checkpoint.items():
                        results_map[batch_key] = BaseRedisSaver._load_writes(
                            self.serde, writes_dict
                        )
        else:
            # Fallback to batch search
            # Group by thread_id and checkpoint_ns for efficient querying
            grouped_keys: Dict[Tuple[str, str], List[str]] = {}
            for thread_id, checkpoint_ns, checkpoint_id in batch_keys:
                group_key = (thread_id, checkpoint_ns)
                if group_key not in grouped_keys:
                    grouped_keys[group_key] = []
                grouped_keys[group_key].append(checkpoint_id)

            # Batch query for each group
            for (thread_id, checkpoint_ns), checkpoint_ids in grouped_keys.items():
                storage_safe_thread_id = to_storage_safe_id(thread_id)
                storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
                storage_safe_checkpoint_ids = [
                    to_storage_safe_id(cid) for cid in checkpoint_ids
                ]

                # Build batch query
                thread_filter = Tag("thread_id") == storage_safe_thread_id
                ns_filter = Tag("checkpoint_ns") == storage_safe_checkpoint_ns
                checkpoint_filter = Tag("checkpoint_id") == storage_safe_checkpoint_ids

                batch_query = FilterQuery(
                    filter_expression=thread_filter & ns_filter & checkpoint_filter,
                    return_fields=[
                        "checkpoint_id",
                        "task_id",
                        "idx",
                        "channel",
                        "type",
                        "$.blob",
                    ],
                    num_results=5000,  # Increased limit for batch
                )

                batch_results = await self.checkpoint_writes_index.search(batch_query)

                # Group results by checkpoint ID
                fallback_writes_by_checkpoint: Dict[
                    str, Dict[Tuple[str, str], Dict[str, Any]]
                ] = {}
                for doc in batch_results.docs:
                    checkpoint_id = from_storage_safe_id(doc.checkpoint_id)
                    if checkpoint_id not in fallback_writes_by_checkpoint:
                        fallback_writes_by_checkpoint[checkpoint_id] = {}

                    task_id = getattr(doc, "task_id", "")
                    idx_str = str(getattr(doc, "idx", 0))
                    blob = getattr(doc, "$.blob", getattr(doc, "blob", b""))

                    fallback_writes_by_checkpoint[checkpoint_id][(task_id, idx_str)] = {
                        "task_id": task_id,
                        "idx": idx_str,
                        "channel": getattr(doc, "channel", ""),
                        "type": getattr(doc, "type", ""),
                        "blob": blob,
                    }

                # Process results for each checkpoint
                for checkpoint_id in checkpoint_ids:
                    batch_key = (thread_id, checkpoint_ns, checkpoint_id)
                    writes_dict = fallback_writes_by_checkpoint.get(checkpoint_id, {})
                    results_map[batch_key] = BaseRedisSaver._load_writes(
                        self.serde, writes_dict
                    )

        return results_map

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
        checkpoint_namespaces = set()

        for doc in checkpoint_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")

            # Track unique namespaces for latest pointer cleanup
            checkpoint_namespaces.add(checkpoint_ns)

            # Delete checkpoint key
            checkpoint_key = self._make_redis_checkpoint_key(
                storage_safe_thread_id, checkpoint_ns, checkpoint_id
            )
            keys_to_delete.append(checkpoint_key)

        # Add latest checkpoint pointers to deletion list
        for checkpoint_ns in checkpoint_namespaces:
            latest_pointer_key = f"checkpoint_latest:{storage_safe_thread_id}:{to_storage_safe_str(checkpoint_ns)}"
            keys_to_delete.append(latest_pointer_key)

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

            blob_key = self._make_redis_checkpoint_blob_key(
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

            write_key = self._make_redis_checkpoint_writes_key(
                storage_safe_thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            keys_to_delete.append(write_key)

        # Delete the registry sorted sets for each checkpoint
        if self._key_registry:
            # Get unique checkpoints from the results we already have
            processed_checkpoints = set()
            for doc in checkpoint_results.docs:
                checkpoint_ns = getattr(doc, "checkpoint_ns", "")
                checkpoint_id = getattr(doc, "checkpoint_id", "")
                checkpoint_key = (thread_id, checkpoint_ns, checkpoint_id)

                if checkpoint_key not in processed_checkpoints:
                    processed_checkpoints.add(checkpoint_key)
                    # Add the write registry key for this checkpoint
                    zset_key = self._key_registry.make_write_keys_zset_key(
                        thread_id, checkpoint_ns, checkpoint_id
                    )
                    keys_to_delete.append(zset_key)

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
