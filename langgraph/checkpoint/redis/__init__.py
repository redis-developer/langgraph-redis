from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

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
from redis import Redis
from redis.cluster import RedisCluster
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory
from ulid import ULID

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)
from langgraph.checkpoint.redis.key_registry import SyncCheckpointKeyRegistry
from langgraph.checkpoint.redis.message_exporter import (
    LangChainRecipe,
    MessageExporter,
    MessageRecipe,
)
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
        # Prefixes are now set in BaseRedisSaver.__init__
        self._separator = REDIS_KEY_SEPARATOR

        # Instance-level cache for frequently used keys (limited size to prevent memory issues)
        self._key_cache: Dict[str, str] = {}
        self._key_cache_max_size = 1000  # Configurable limit

        # Key registry will be initialized in setup()
        self._key_registry: Optional[SyncCheckpointKeyRegistry] = None

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        from redis.exceptions import ResponseError

        from langgraph.checkpoint.redis.version import __full_lib_name__

        self._owns_its_client = redis_client is None
        self._redis = redis_client or RedisConnectionFactory.get_redis_connection(
            redis_url, **connection_args
        )

        # Set client info for Redis monitoring
        try:
            self._redis.client_setinfo("LIB-NAME", __full_lib_name__)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                self._redis.echo(__full_lib_name__)
            except Exception:
                # Silently fail if even echo doesn't work
                pass

    def create_indexes(self) -> None:
        self.checkpoints_index = SearchIndex.from_dict(
            self.checkpoints_schema, redis_client=self._redis
        )
        self.checkpoint_blobs_index = SearchIndex.from_dict(
            self.blobs_schema, redis_client=self._redis
        )
        self.checkpoint_writes_index = SearchIndex.from_dict(
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

        key = self._separator.join(key_parts)

        # Cache for future use (limit cache size)
        if len(self._key_cache) < self._key_cache_max_size:
            self._key_cache[cache_key] = key

        return key

    def setup(self) -> None:
        """Initialize the indices in Redis and detect cluster mode."""
        self._detect_cluster_mode()
        super().setup()

        # Initialize key registry for this instance
        if self._redis and not self._key_registry:
            self._key_registry = SyncCheckpointKeyRegistry(self._redis)

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
        before: Optional[RunnableConfig] = None,  # noqa: ARG002
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

        # Execute the query
        results = self.checkpoints_index.search(query)

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
                channel_values = checkpoint_dict.get("channel_values", {})
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
            pending_sends_map = self._batch_load_pending_sends(pending_sends_batch_keys)

        # Load pending_writes for all checkpoints with writes at once
        pending_writes_map = {}
        if pending_writes_batch_keys:
            pending_writes_map = self._batch_load_pending_writes(
                pending_writes_batch_keys
            )

        # Process the results using pre-loaded batch data
        for doc_data in all_docs_data:
            thread_id = doc_data["thread_id"]
            checkpoint_ns = doc_data["checkpoint_ns"]
            checkpoint_id = doc_data["checkpoint_id"]
            parent_checkpoint_id = doc_data["parent_checkpoint_id"]

            # Get pending_sends from batch results
            pending_sends: List[Tuple[str, bytes]] = []
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

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis with separate blob storage."""
        configurable = config["configurable"].copy()

        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        # Get checkpoint_id from config - this will be parent if saving a child
        config_checkpoint_id = configurable.pop("checkpoint_id", None)
        # For backward compatibility with thread_ts
        thread_ts = configurable.pop("thread_ts", "")

        # Determine the checkpoint ID
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

        # Convert empty strings to the sentinel value.
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
                to_storage_safe_id(parent_checkpoint_id) if parent_checkpoint_id else ""
            ),
            "checkpoint_ts": checkpoint_ts,
            "checkpoint": self._dump_checkpoint(copy),  # Includes channel_values inline
            "metadata": self._dump_metadata(metadata),
            "has_writes": False,  # Track if this checkpoint has pending writes
        }

        # Store at top-level for filters in list()
        if all(key in metadata for key in ["source", "step"]):
            checkpoint_data["source"] = metadata["source"]
            checkpoint_data["step"] = metadata["step"]

        # Create the checkpoint key
        checkpoint_key = self._make_redis_checkpoint_key_cached(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
        )

        # Calculate TTL in seconds if configured
        ttl_seconds = None
        if self.ttl_config and "default_ttl" in self.ttl_config:
            ttl_seconds = int(self.ttl_config["default_ttl"] * 60)

        # Store checkpoint with TTL in a single pipeline operation
        self.checkpoints_index.load(
            [checkpoint_data],
            keys=[checkpoint_key],
            ttl=ttl_seconds,  # RedisVL applies TTL in its internal pipeline
        )

        # Update latest checkpoint pointer
        latest_pointer_key = (
            f"checkpoint_latest:{storage_safe_thread_id}:{storage_safe_checkpoint_ns}"
        )
        self._redis.set(latest_pointer_key, checkpoint_key)

        # Apply TTL to latest pointer key as well
        if ttl_seconds is not None:
            self._redis.expire(latest_pointer_key, ttl_seconds)

        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint with integrated key registry."""
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

        with self._redis.pipeline(transaction=False) as pipeline:
            # Keep track of keys we're creating
            created_keys = []
            write_keys = []

            for write_obj in writes_objects:
                idx_value = write_obj["idx"]
                assert isinstance(idx_value, int)
                key = self._make_redis_checkpoint_writes_key_cached(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx_value,
                )
                write_keys.append(key)
                pipeline.json().set(key, "$", cast(Any, write_obj))
                created_keys.append(key)

            # Add TTL operations to the pipeline if configured
            if created_keys and self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config["default_ttl"] * 60)
                for key in created_keys:
                    pipeline.expire(key, ttl_seconds)

            # Update checkpoint to indicate it has writes
            if writes_objects:
                checkpoint_key = self._make_redis_checkpoint_key_cached(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                # Use merge to update existing document
                pipeline.json().merge(checkpoint_key, "$", {"has_writes": True})

            try:
                pipeline.execute()
            except Exception as e:
                # Check if JSON.MERGE failed (older Redis versions)
                if "JSON.MERGE" in str(e) or "merge" in str(e).lower():
                    # Retry without JSON.MERGE for older Redis versions
                    with self._redis.pipeline(transaction=False) as fallback_pipeline:
                        # Re-add all the write operations
                        for write_obj in writes_objects:
                            idx_value = write_obj["idx"]
                            assert isinstance(idx_value, int)
                            key = self._make_redis_checkpoint_writes_key_cached(
                                thread_id,
                                checkpoint_ns,
                                checkpoint_id,
                                task_id,
                                idx_value,
                            )
                            fallback_pipeline.json().set(key, "$", cast(Any, write_obj))

                        # Add TTL operations if configured
                        if (
                            created_keys
                            and self.ttl_config
                            and "default_ttl" in self.ttl_config
                        ):
                            ttl_seconds = int(self.ttl_config["default_ttl"] * 60)
                            for key in created_keys:
                                fallback_pipeline.expire(key, ttl_seconds)

                        # Execute the fallback pipeline
                        fallback_pipeline.execute()

                        # Update has_writes flag separately for older Redis
                        if checkpoint_key:
                            try:
                                checkpoint_data = self._redis.json().get(checkpoint_key)
                                if isinstance(
                                    checkpoint_data, dict
                                ) and not checkpoint_data.get("has_writes"):
                                    checkpoint_data["has_writes"] = True
                                    self._redis.json().set(
                                        checkpoint_key, "$", checkpoint_data
                                    )
                            except Exception:
                                # If this fails, it's not critical - the writes are still saved
                                pass
                else:
                    # Re-raise other exceptions
                    raise

        # Update key registry with the write keys
        if self._key_registry and write_keys:
            self._key_registry.register_write_keys_batch(
                thread_id, checkpoint_ns, checkpoint_id, write_keys
            )

            # Apply TTL to registry key if configured
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_seconds = int(self.ttl_config["default_ttl"] * 60)
                self._key_registry.apply_ttl(
                    thread_id, checkpoint_ns, checkpoint_id, ttl_seconds
                )

    def _get_checkpoint_document_by_id(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> Optional[dict]:
        """Get checkpoint document by specific ID using direct key access."""
        checkpoint_key = self._make_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns, checkpoint_id
        )

        checkpoint_data = self._redis.json().get(checkpoint_key)
        if not checkpoint_data or not isinstance(checkpoint_data, dict):
            return None

        # Extract the actual checkpoint data
        checkpoint_inner = checkpoint_data.get("checkpoint", {})

        return {
            "thread_id": checkpoint_data.get(
                "thread_id", to_storage_safe_id(thread_id)
            ),
            "checkpoint_ns": checkpoint_data.get(
                "checkpoint_ns", to_storage_safe_str(checkpoint_ns)
            ),
            "checkpoint_id": checkpoint_data.get(
                "checkpoint_id", to_storage_safe_id(checkpoint_id)
            ),
            "parent_checkpoint_id": checkpoint_data.get(
                "parent_checkpoint_id", to_storage_safe_id(checkpoint_id)
            ),
            "$.checkpoint": (
                json.dumps(checkpoint_inner)
                if isinstance(checkpoint_inner, dict)
                else checkpoint_inner
            ),
            "$.metadata": checkpoint_data.get("metadata", "{}"),
            "_channel_versions": (
                checkpoint_inner.get("channel_versions")
                if isinstance(checkpoint_inner, dict)
                else None
            ),
            "has_writes": checkpoint_data.get("has_writes", False),
        }

    def _get_latest_checkpoint_document(
        self, thread_id: str, checkpoint_ns: str
    ) -> Optional[dict]:
        """Get latest checkpoint document using pointer."""
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)

        # Get latest checkpoint using pointer
        latest_pointer_key = (
            f"checkpoint_latest:{storage_safe_thread_id}:{storage_safe_checkpoint_ns}"
        )
        checkpoint_key_bytes = self._redis.get(latest_pointer_key)

        if not checkpoint_key_bytes:
            # No pointer means no checkpoints exist
            return None

        # Decode bytes to string
        checkpoint_key = (
            checkpoint_key_bytes.decode()
            if isinstance(checkpoint_key_bytes, bytes)
            else checkpoint_key_bytes
        )
        checkpoint_data = self._redis.json().get(str(checkpoint_key))
        if not checkpoint_data or not isinstance(checkpoint_data, dict):
            # Pointer exists but checkpoint is missing - data inconsistency
            return None

        checkpoint_inner = checkpoint_data.get("checkpoint", {})
        return {
            "thread_id": checkpoint_data.get("thread_id", storage_safe_thread_id),
            "checkpoint_ns": checkpoint_data.get(
                "checkpoint_ns", storage_safe_checkpoint_ns
            ),
            "checkpoint_id": checkpoint_data.get("checkpoint_id"),
            "parent_checkpoint_id": checkpoint_data.get("parent_checkpoint_id"),
            "$.checkpoint": (
                json.dumps(checkpoint_inner)
                if isinstance(checkpoint_inner, dict)
                else checkpoint_inner
            ),
            "$.metadata": checkpoint_data.get("metadata", "{}"),
            "_channel_versions": (
                checkpoint_inner.get("channel_versions")
                if isinstance(checkpoint_inner, dict)
                else None
            ),
            "has_writes": checkpoint_data.get("has_writes", False),
            # Store the full checkpoint data to avoid re-fetching
            "_checkpoint_data": checkpoint_data,
        }

    def _refresh_checkpoint_ttl(
        self, doc_thread_id: str, doc_checkpoint_ns: str, doc_checkpoint_id: str
    ) -> None:
        """Refresh TTL for checkpoint and all related keys."""
        if not self.ttl_config or not self.ttl_config.get("refresh_on_read"):
            return

        checkpoint_key = self._make_redis_checkpoint_key_cached(
            doc_thread_id,
            doc_checkpoint_ns,
            doc_checkpoint_id,
        )

        # Get write keys
        write_keys = []

        if self._key_registry:
            write_keys = self._key_registry.get_write_keys(
                doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id
            )
        else:
            # Use search indices as fallback
            write_keys = self._get_write_keys_from_search(
                doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id
            )

        # Apply TTL to all keys
        self._apply_ttl_to_keys(checkpoint_key, write_keys)

        # Refresh registry key TTL
        if self._key_registry and self.ttl_config:
            ttl_minutes = self.ttl_config.get("default_ttl")
            if ttl_minutes is not None:
                ttl_seconds = int(ttl_minutes * 60)
                # Registry TTL is handled per checkpoint
                self._key_registry.apply_ttl(
                    doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id, ttl_seconds
                )

    def _get_write_keys_from_search(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[str]:
        """Get write keys using search index."""
        write_query = FilterQuery(
            filter_expression=(Tag("thread_id") == to_storage_safe_id(thread_id))
            & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)),
            return_fields=["task_id", "idx"],
            num_results=1000,
        )
        write_results = self.checkpoint_writes_index.search(write_query)

        return [
            self._make_redis_checkpoint_writes_key(
                to_storage_safe_id(thread_id),
                to_storage_safe_str(checkpoint_ns),
                to_storage_safe_id(checkpoint_id),
                getattr(doc, "task_id", ""),
                getattr(doc, "idx", 0),
            )
            for doc in write_results.docs
        ]

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

        # For values we store in Redis, we need to convert empty strings to the
        # sentinel value.
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)

        if checkpoint_id and checkpoint_id != EMPTY_ID_SENTINEL:
            # Direct key access when checkpoint_id is known - no fallback needed
            storage_safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

            # Construct direct key for checkpoint data
            checkpoint_key = self._make_redis_checkpoint_key_cached(
                thread_id, checkpoint_ns, checkpoint_id
            )

            # Direct key access only
            checkpoint_data = self._redis.json().get(checkpoint_key)

            if not checkpoint_data or not isinstance(checkpoint_data, dict):
                # Checkpoint doesn't exist
                return None

            # Process checkpoint data from direct access
            # Create doc-like object from direct access
            # Extract the actual checkpoint data
            checkpoint_inner = checkpoint_data.get("checkpoint", {})

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
                "$.checkpoint": (
                    json.dumps(checkpoint_inner)
                    if isinstance(checkpoint_inner, dict)
                    else checkpoint_inner
                ),
                "$.metadata": checkpoint_data.get(
                    "metadata", "{}"
                ),  # metadata is already a JSON string
                # Store channel_versions for easy access
                "_channel_versions": (
                    checkpoint_inner.get("channel_versions")
                    if isinstance(checkpoint_inner, dict)
                    else None
                ),
                # Store has_writes flag
                "has_writes": checkpoint_data.get(
                    "has_writes", False
                ),  # Default to False to avoid expensive searches
                # Store the full checkpoint data to avoid re-fetching
                "_checkpoint_data": checkpoint_data,
            }
        else:
            # Get latest checkpoint using the helper method
            doc = self._get_latest_checkpoint_document(thread_id, checkpoint_ns)
            if not doc:
                return None
        # Handle both dict (from direct access) and Document objects (from FT.SEARCH)
        if isinstance(doc, dict):
            doc_thread_id = from_storage_safe_id(doc["thread_id"])
            doc_checkpoint_ns = from_storage_safe_str(doc["checkpoint_ns"])
            doc_checkpoint_id = from_storage_safe_id(doc["checkpoint_id"])
            doc_parent_checkpoint_id = from_storage_safe_id(doc["parent_checkpoint_id"])
        else:
            doc_thread_id = from_storage_safe_id(doc.thread_id)
            doc_checkpoint_ns = from_storage_safe_str(doc.checkpoint_ns)
            doc_checkpoint_id = from_storage_safe_id(doc.checkpoint_id)
            doc_parent_checkpoint_id = from_storage_safe_id(doc.parent_checkpoint_id)

        # Lazy TTL refresh - only refresh if TTL is below threshold
        if self.ttl_config and self.ttl_config.get("refresh_on_read"):
            # Get the checkpoint key
            checkpoint_key = self._make_redis_checkpoint_key_cached(
                doc_thread_id,
                doc_checkpoint_ns,
                doc_checkpoint_id,
            )

            # Always refresh TTL when refresh_on_read is enabled
            # This ensures all related keys maintain synchronized TTLs
            current_ttl = self._redis.ttl(checkpoint_key)

            # Only refresh if key exists and has TTL (skip keys with no expiry)
            # TTL states: -2 = key doesn't exist, -1 = key exists but no TTL, 0 = expired, >0 = seconds remaining
            if current_ttl > 0:
                # Note: We don't refresh TTL for keys with no expiry (TTL = -1)
                # Get all blob keys related to this checkpoint
                from langgraph.checkpoint.redis.base import (
                    CHECKPOINT_BLOB_PREFIX,
                    CHECKPOINT_WRITE_PREFIX,
                )

                # Get write keys - use key registry if available, otherwise fall back to search
                write_keys = []

                if self._key_registry:
                    # Use key registry for faster lookup
                    write_keys = self._key_registry.get_write_keys(
                        doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id
                    )
                else:
                    # Fallback to search index
                    write_keys = self._get_write_keys_from_search(
                        doc_thread_id, doc_checkpoint_ns, doc_checkpoint_id
                    )

                # Apply TTL to checkpoint and write keys
                self._apply_ttl_to_keys(checkpoint_key, write_keys)

        # Fetch channel_values - pass channel_versions if we have them from direct access
        # First check if we stored channel_versions during direct access
        channel_versions_from_checkpoint = doc.get("_channel_versions")

        if channel_versions_from_checkpoint is None:
            # Fall back to extracting from checkpoint data
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

        # Get channel values from the checkpoint we already fetched
        # Extract the checkpoint data based on doc type
        if isinstance(doc, dict):
            # From direct access - we have the full data
            checkpoint_inner = doc.get("_checkpoint_data", {}).get("checkpoint", {})
            if isinstance(checkpoint_inner, str):
                checkpoint_inner = json.loads(checkpoint_inner)
        else:
            # From search - parse the checkpoint
            checkpoint_str = getattr(doc, "$.checkpoint", "{}")
            checkpoint_inner = (
                json.loads(checkpoint_str)
                if isinstance(checkpoint_str, str)
                else checkpoint_str
            )

        # Channel values are already inline in the checkpoint
        channel_values = checkpoint_inner.get("channel_values", {})
        # Deserialize them since they're stored in serialized form
        channel_values = self._deserialize_channel_values(channel_values)

        # Fetch pending_sends from parent checkpoint
        pending_sends = []
        if doc_parent_checkpoint_id:
            pending_sends = self._load_pending_sends_with_registry_check(
                thread_id=doc_thread_id,
                checkpoint_ns=doc_checkpoint_ns,
                parent_checkpoint_id=doc_parent_checkpoint_id,
            )

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

        # Handle both direct dict access and FT.SEARCH results efficiently
        checkpoint_data = (
            doc.get("$.checkpoint")
            if isinstance(doc, dict)
            else getattr(doc, "$.checkpoint")
        )

        checkpoint_param = self._load_checkpoint(
            checkpoint_data or {},
            channel_values,
            pending_sends,
        )

        # Skip pending_writes if we can determine there are none
        checkpoint_has_writes = (
            doc.get("has_writes")
            if isinstance(doc, dict)
            else getattr(doc, "has_writes", False)
        )
        pending_writes = self._load_pending_writes_with_registry_check(
            doc_thread_id,
            doc_checkpoint_ns,
            doc_checkpoint_id,
            checkpoint_has_writes=bool(checkpoint_has_writes),
            registry_has_writes=False,  # We don't have registry info here
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

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Union[Redis, RedisCluster]] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> Iterator[RedisSaver]:
        """Create a new RedisSaver instance."""
        saver: Optional[RedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                connection_args=connection_args,
                ttl=ttl,
                checkpoint_prefix=checkpoint_prefix,
                checkpoint_blob_prefix=checkpoint_blob_prefix,
                checkpoint_write_prefix=checkpoint_write_prefix,
            )

            yield saver
        finally:
            if saver and saver._owns_its_client:  # Ensure saver is not None
                saver._redis.close()
                # RedisCluster doesn't have connection_pool attribute
                if getattr(saver._redis, "connection_pool", None):
                    saver._redis.connection_pool.disconnect()

    def get_channel_values(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        checkpoint_id: str = "",
        channel_versions: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve channel_values using efficient FT.SEARCH with checkpoint_id."""
        # Get checkpoint with inline channel_values using single JSON.GET operation
        checkpoint_key = self._make_redis_checkpoint_key_cached(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
        )

        # Single JSON.GET operation to retrieve checkpoint with inline channel_values
        checkpoint_data = self._redis.json().get(checkpoint_key, "$.checkpoint")

        if not checkpoint_data:
            return {}

        # checkpoint_data[0] is already a deserialized dict, not a typed tuple
        checkpoint = checkpoint_data[0]
        return checkpoint.get("channel_values", {})

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        """Load pending writes using sorted set registry."""
        return self._load_pending_writes_with_registry_check(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            checkpoint_has_writes=True,  # Assume writes exist if we're calling this
            registry_has_writes=False,
        )

    def _load_pending_sends(
        self,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str,
    ) -> List[Tuple[str, Union[str, bytes]]]:
        """Load pending sends for a parent checkpoint.

        Args:
            thread_id: The thread ID
            checkpoint_ns: The checkpoint namespace
            parent_checkpoint_id: The ID of the parent checkpoint

        Returns:
            List of (type, blob) tuples representing pending sends
        """
        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_parent_checkpoint_id = to_storage_safe_id(parent_checkpoint_id)

        parent_writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == storage_safe_thread_id)
            & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
            & (Tag("checkpoint_id") == storage_safe_parent_checkpoint_id)
            & (Tag("channel") == TASKS),
            return_fields=["type", "$.blob", "task_path", "task_id", "idx"],
            num_results=100,  # Adjust as needed
        )
        parent_writes_results = self.checkpoint_writes_index.search(parent_writes_query)

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
        return [
            (
                getattr(doc, "type", ""),
                getattr(doc, "$.blob", getattr(doc, "blob", b"")),
            )
            for doc in sorted_writes
        ]

    def _batch_load_pending_sends(
        self, batch_keys: List[Tuple[str, str, str]]
    ) -> Dict[Tuple[str, str, str], List[Tuple[str, bytes]]]:
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

            batch_results = self.checkpoint_writes_index.search(batch_query)

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

    def _batch_load_pending_writes(
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

        results_map = {}

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

            # Build filter for multiple checkpoint IDs
            thread_filter = Tag("thread_id") == storage_safe_thread_id
            ns_filter = Tag("checkpoint_ns") == storage_safe_checkpoint_ns

            # Create filter for multiple checkpoint IDs (Tag supports lists)
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
                num_results=10000,  # Large limit for batch loading
            )

            batch_results = self.checkpoint_writes_index.search(batch_query)

            # Group results by checkpoint ID
            writes_by_checkpoint: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
            for doc in batch_results.docs:
                checkpoint_id = from_storage_safe_id(doc.checkpoint_id)
                if checkpoint_id not in writes_by_checkpoint:
                    writes_by_checkpoint[checkpoint_id] = {}

                task_id = str(doc.task_id)
                idx = str(doc.idx)
                writes_by_checkpoint[checkpoint_id][(task_id, idx)] = {
                    "task_id": task_id,
                    "idx": idx,
                    "channel": getattr(doc, "channel", ""),
                    "type": getattr(doc, "type", ""),
                    "blob": getattr(doc, "$.blob", b""),
                }

            # Format results for each checkpoint
            for checkpoint_id in checkpoint_ids:
                batch_key = (thread_id, checkpoint_ns, checkpoint_id)
                writes_dict = writes_by_checkpoint.get(checkpoint_id, {})

                # Use base class method to deserialize
                results_map[batch_key] = BaseRedisSaver._load_writes(
                    self.serde, writes_dict
                )

        return results_map

    def _load_pending_writes_with_registry_check(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        checkpoint_has_writes: bool,
        registry_has_writes: bool,
    ) -> List[PendingWrite]:
        """Load pending writes with registry optimization and fallback."""
        if not checkpoint_has_writes:
            return []

        # FAST PATH: Try sorted set registry first
        if self._key_registry:
            try:
                # Check write count from registry
                write_count = self._key_registry.get_write_count(
                    thread_id, checkpoint_ns, checkpoint_id
                )

                if write_count == 0:
                    return []

                # Get write keys from registry
                write_keys = self._key_registry.get_write_keys(
                    thread_id, checkpoint_ns, checkpoint_id
                )

                if write_keys:
                    # Batch fetch all writes using pipeline
                    with self._redis.pipeline(transaction=False) as pipeline:
                        for key in write_keys:
                            pipeline.json().get(key)

                        results = pipeline.execute()

                    # Build writes dictionary
                    writes_dict = {}
                    for write_data in results:
                        if write_data:
                            task_id = write_data.get("task_id", "")
                            idx = write_data.get("idx", 0)
                            writes_dict[(task_id, idx)] = write_data

                    # Use base class method to deserialize
                    return BaseRedisSaver._load_writes(self.serde, writes_dict)

            except Exception:
                # Fall through to FT.SEARCH fallback
                pass

        # FALLBACK: Use FT.SEARCH if registry not available or failed
        # Call the base class implementation to avoid recursion
        return super()._load_pending_writes(thread_id, checkpoint_ns, checkpoint_id)

    def _load_pending_sends_with_registry_check(
        self,
        thread_id: str,
        checkpoint_ns: str,
        parent_checkpoint_id: str,
    ) -> List[Tuple[str, Union[str, bytes]]]:
        """Load pending sends for a parent checkpoint with pre-computed registry check."""
        if not parent_checkpoint_id:
            return []

        # FAST PATH: Try sorted set registry first
        if self._key_registry:
            try:
                # Check if parent checkpoint has any writes in the sorted set
                write_count = self._key_registry.get_write_count(
                    thread_id, checkpoint_ns, parent_checkpoint_id
                )

                if write_count == 0:
                    # No writes for parent checkpoint - return immediately
                    return []

                # Get exact write keys from the per-checkpoint registry
                write_keys = self._key_registry.get_write_keys(
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
                with self._redis.pipeline(transaction=False) as pipeline:
                    for key in task_write_keys:
                        pipeline.json().get(key)

                    results = pipeline.execute()

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

        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_parent_checkpoint_id = to_storage_safe_id(parent_checkpoint_id)

        parent_writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == storage_safe_thread_id)
            & (Tag("checkpoint_ns") == storage_safe_checkpoint_ns)
            & (Tag("checkpoint_id") == storage_safe_parent_checkpoint_id)
            & (Tag("channel") == TASKS),
            return_fields=["type", "$.blob", "task_path", "task_id", "idx"],
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
        # Handle both direct attribute access and JSON path access
        return [
            (
                getattr(doc, "type", ""),
                getattr(doc, "$.blob", getattr(doc, "blob", b"")),
            )
            for doc in sorted_writes
        ]

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
        checkpoint_namespaces = set()

        for doc in checkpoint_results.docs:
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")

            # Track unique namespaces for latest pointer cleanup
            checkpoint_namespaces.add(checkpoint_ns)

            # Delete checkpoint key
            checkpoint_key = self._make_redis_checkpoint_key_cached(
                thread_id, checkpoint_ns, checkpoint_id
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

        blob_results = self.checkpoint_blobs_index.search(blob_query)

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

        writes_results = self.checkpoint_writes_index.search(writes_query)

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
    "MessageExporter",
    "LangChainRecipe",
    "MessageRecipe",
]
