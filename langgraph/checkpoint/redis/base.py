import base64
import binascii
import logging
import random
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, Union, cast

import orjson
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    PendingWrite,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import ChannelProtocol
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag

from langgraph.checkpoint.redis.util import (
    safely_decode,
    to_storage_safe_id,
    to_storage_safe_str,
)

from .jsonplus_redis import JsonPlusRedisSerializer
from .types import IndexType, RedisClientType

logger = logging.getLogger(__name__)

REDIS_KEY_SEPARATOR = ":"
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_BLOB_PREFIX = "checkpoint_blob"
CHECKPOINT_WRITE_PREFIX = "checkpoint_write"


class BaseRedisSaver(BaseCheckpointSaver[str], Generic[RedisClientType, IndexType]):
    """Base Redis implementation for checkpoint saving.

    Uses Redis JSON for storing checkpoints and related data, with RediSearch for querying.
    """

    _redis: RedisClientType
    _owns_its_client: bool = False
    _key_registry: Optional[Any] = None

    checkpoints_index: IndexType
    checkpoint_blobs_index: IndexType
    checkpoint_writes_index: IndexType

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[RedisClientType] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        ttl: Optional[Dict[str, Any]] = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
        checkpoint_blob_prefix: str = CHECKPOINT_BLOB_PREFIX,
        checkpoint_write_prefix: str = CHECKPOINT_WRITE_PREFIX,
    ) -> None:
        """Initialize Redis-backed checkpoint saver.

        Args:
            redis_url: Redis connection URL
            redis_client: Redis client instance to use (alternative to redis_url)
            connection_args: Additional arguments for Redis connection
            ttl: Optional TTL configuration dict with optional keys:
                - default_ttl: TTL in minutes for all checkpoint keys
                - refresh_on_read: Whether to refresh TTL on reads
            checkpoint_prefix: Prefix for checkpoint keys (default: "checkpoint")
            checkpoint_blob_prefix: Prefix for checkpoint blob keys (default: "checkpoint_blob")
            checkpoint_write_prefix: Prefix for checkpoint write keys (default: "checkpoint_write")
        """
        super().__init__(serde=JsonPlusRedisSerializer())
        if redis_url is None and redis_client is None:
            raise ValueError("Either redis_url or redis_client must be provided")

        # Store TTL configuration
        self.ttl_config = ttl

        # Store custom prefixes
        self._checkpoint_prefix = checkpoint_prefix
        self._checkpoint_blob_prefix = checkpoint_blob_prefix
        self._checkpoint_write_prefix = checkpoint_write_prefix

        self.configure_client(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args or {},
        )

        # Initialize indexes
        self.checkpoints_index: IndexType
        self.checkpoint_blobs_index: IndexType
        self.checkpoint_writes_index: IndexType
        self.create_indexes()

    @property
    def checkpoints_schema(self) -> Dict[str, Any]:
        """Schema for the checkpoints index."""
        return {
            "index": {
                "name": self._checkpoint_prefix,
                "prefix": self._checkpoint_prefix + REDIS_KEY_SEPARATOR,
                "storage_type": "json",
            },
            "fields": [
                {"name": "thread_id", "type": "tag"},
                {"name": "checkpoint_ns", "type": "tag"},
                {"name": "checkpoint_id", "type": "tag"},
                {"name": "parent_checkpoint_id", "type": "tag"},
                {"name": "checkpoint_ts", "type": "numeric"},
                {"name": "source", "type": "tag"},
                {"name": "step", "type": "numeric"},
                {"name": "has_writes", "type": "tag"},
            ],
        }

    @property
    def blobs_schema(self) -> Dict[str, Any]:
        """Schema for the checkpoint blobs index."""
        return {
            "index": {
                "name": self._checkpoint_blob_prefix,
                "prefix": self._checkpoint_blob_prefix + REDIS_KEY_SEPARATOR,
                "storage_type": "json",
            },
            "fields": [
                {"name": "thread_id", "type": "tag"},
                {"name": "checkpoint_ns", "type": "tag"},
                {"name": "checkpoint_id", "type": "tag"},
                {"name": "channel", "type": "tag"},
                {"name": "version", "type": "tag"},
                {"name": "type", "type": "tag"},
            ],
        }

    @property
    def writes_schema(self) -> Dict[str, Any]:
        """Schema for the checkpoint writes index."""
        return {
            "index": {
                "name": self._checkpoint_write_prefix,
                "prefix": self._checkpoint_write_prefix + REDIS_KEY_SEPARATOR,
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
        }

    @abstractmethod
    def create_indexes(self) -> None:
        """Create appropriate SearchIndex instances."""
        pass

    @abstractmethod
    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[RedisClientType] = None,
        connection_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        pass

    def set_client_info(self) -> None:
        """Set client info for Redis monitoring."""
        from redis.exceptions import ResponseError

        from langgraph.checkpoint.redis.version import __full_lib_name__

        try:
            # Try to use client_setinfo command if available
            self._redis.client_setinfo("LIB-NAME", __full_lib_name__)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                self._redis.echo(__full_lib_name__)
            except Exception:
                # Silently fail if even echo doesn't work
                pass

    async def aset_client_info(self) -> None:
        """Set client info for Redis monitoring asynchronously."""
        from redis.exceptions import ResponseError

        from langgraph.checkpoint.redis.version import __full_lib_name__

        try:
            # Try to use client_setinfo command if available
            await self._redis.client_setinfo("LIB-NAME", __full_lib_name__)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                # Call with await to ensure it's an async call
                echo_result = self._redis.echo(__full_lib_name__)
                if hasattr(echo_result, "__await__"):
                    await echo_result
            except Exception:
                # Silently fail if even echo doesn't work
                pass

    def setup(self) -> None:
        """Initialize the indices in Redis."""
        # Create indexes in Redis
        self.checkpoints_index.create(overwrite=False)
        self.checkpoint_blobs_index.create(overwrite=False)
        self.checkpoint_writes_index.create(overwrite=False)

    def _load_checkpoint(
        self,
        checkpoint: Union[Dict[str, Any], str],
        channel_values: Dict[str, Any],
        pending_sends: List[Any],
    ) -> Checkpoint:
        if not checkpoint:
            return {}

        # OPTIMIZED: Handle both dict and string inputs efficiently
        loaded = (
            checkpoint
            if isinstance(checkpoint, dict)
            else cast(dict, orjson.loads(checkpoint))
        )

        return {
            **loaded,
            "pending_sends": [
                self.serde.loads_typed((safely_decode(c), b))
                for c, b in pending_sends or []
            ],
            "channel_values": channel_values,
        }

    def _apply_ttl_to_keys(
        self,
        main_key: str,
        related_keys: Optional[list[str]] = None,
        ttl_minutes: Optional[float] = None,
    ) -> Any:
        """Apply Redis native TTL to keys.

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
                # Check if cluster mode is detected (for sync checkpoint savers)
                cluster_mode = getattr(self, "cluster_mode", False)

                if cluster_mode:
                    # For cluster mode, execute PERSIST operations individually
                    self._redis.persist(main_key)

                    if related_keys:
                        for key in related_keys:
                            self._redis.persist(key)

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

                    return pipeline.execute()

            # Regular TTL setting
            ttl_seconds = int(ttl_minutes * 60)

            # Check if cluster mode is detected (for sync checkpoint savers)
            cluster_mode = getattr(self, "cluster_mode", False)

            if cluster_mode:
                # For cluster mode, execute TTL operations individually
                self._redis.expire(main_key, ttl_seconds)

                if related_keys:
                    for key in related_keys:
                        self._redis.expire(key, ttl_seconds)

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

                return pipeline.execute()

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        """Convert checkpoint to Redis format."""
        type_, data = self.serde.dumps_typed(checkpoint)

        # Decode the serialized data - handle both JSON and msgpack
        if type_ == "json":
            checkpoint_data = cast(dict, orjson.loads(data))
        else:
            # For msgpack or other types, deserialize with loads_typed
            checkpoint_data = cast(dict, self.serde.loads_typed((type_, data)))

            # When using msgpack, bytes are preserved - but Redis JSON.SET can't handle them
            # Encode bytes in channel_values with type marker for JSON storage
            if "channel_values" in checkpoint_data:
                for key, value in checkpoint_data["channel_values"].items():
                    if isinstance(value, bytes):
                        checkpoint_data["channel_values"][key] = {
                            "__bytes__": self._encode_blob(value)
                        }

        # Ensure channel_versions are always strings to fix issue #40
        if "channel_versions" in checkpoint_data:
            checkpoint_data["channel_versions"] = {
                k: str(v) for k, v in checkpoint_data["channel_versions"].items()
            }

        return {"type": type_, **checkpoint_data, "pending_sends": []}

    def _load_blobs(self, blob_values: dict[str, Any]) -> dict[str, Any]:
        """Load binary data from Redis."""
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((v["type"], v["blob"]))
            for k, v in blob_values.items()
            if v["type"] != "empty"
        }

    def _deserialize_channel_values(
        self, channel_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Deserialize channel values that were stored inline.

        When channel values are stored inline in the checkpoint, they're in their
        serialized form. This method deserializes them back to their original types.

        This specifically handles LangChain message objects that may be stored in their
        serialized format: {'lc': 1, 'type': 'constructor', 'id': [...], 'kwargs': {...}}
        and ensures they are properly reconstructed as message objects.
        """
        if not channel_values:
            return {}

        try:
            # Apply recursive deserialization to handle nested structures and LangChain objects
            return self._recursive_deserialize(channel_values)
        except Exception as e:
            logger.warning(
                f"Error deserializing channel values, attempting recovery: {e}"
            )
            # Attempt to recover by processing each channel individually
            recovered = {}
            for key, value in channel_values.items():
                try:
                    recovered[key] = self._recursive_deserialize(value)
                except Exception as inner_e:
                    logger.error(
                        f"Failed to deserialize channel '{key}': {inner_e}. "
                        f"Value will be returned as-is."
                    )
                    recovered[key] = value
            return recovered

    def _recursive_deserialize(self, obj: Any) -> Any:
        """Recursively deserialize LangChain objects and nested structures.

        This method specifically handles the deserialization of LangChain message objects
        that may be stored in their serialized format to prevent MESSAGE_COERCION_FAILURE.

        Args:
            obj: The object to deserialize, which may be a dict, list, or primitive.

        Returns:
            The deserialized object, with LangChain objects properly reconstructed.
        """
        if isinstance(obj, dict):
            # Check if this is a bytes marker from msgpack storage
            if "__bytes__" in obj and len(obj) == 1:
                # Decode base64-encoded bytes
                return self._decode_blob(obj["__bytes__"])

            # Check if this is a Send object marker (issue #94)
            if (
                obj.get("__send__") is True
                and "node" in obj
                and "arg" in obj
                and len(obj) == 3
            ):
                try:
                    from langgraph.types import Send

                    return Send(
                        node=obj["node"],
                        arg=self._recursive_deserialize(obj["arg"]),
                    )
                except (ImportError, TypeError, ValueError) as e:
                    logger.debug(
                        "Failed to deserialize Send object: %s", e, exc_info=True
                    )

            # Check if this is a LangChain serialized object
            if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                try:
                    # Use the serde's reviver to reconstruct the object

                    if hasattr(self.serde, "_revive_if_needed"):
                        return self.serde._revive_if_needed(obj)
                    elif hasattr(self.serde, "_reviver"):
                        return self.serde._reviver(obj)
                    else:
                        # Log warning if serde doesn't have reviver
                        logger.warning(
                            "Serializer does not have a reviver method. "
                            "LangChain object may not be properly deserialized. "
                            f"Object ID: {obj.get('id')}"
                        )
                        return obj
                except Exception as e:
                    # Provide detailed error message for debugging
                    obj_id = obj.get("id", "unknown")
                    obj_type = (
                        obj.get("id", ["unknown"])[-1]
                        if isinstance(obj.get("id"), list)
                        else "unknown"
                    )
                    logger.error(
                        f"Failed to deserialize LangChain object of type '{obj_type}'. "
                        f"This may cause MESSAGE_COERCION_FAILURE. Error: {e}. "
                        f"Object structure: lc={obj.get('lc')}, type={obj.get('type')}, "
                        f"id={obj_id}"
                    )
                    # Return the object as-is to prevent complete failure
                    return obj
            # Recursively process nested dicts
            return {k: self._recursive_deserialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process lists
            return [self._recursive_deserialize(item) for item in obj]
        else:
            # Return primitives as-is
            return obj

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert write operations for Redis storage."""
        return [
            {
                "thread_id": to_storage_safe_id(thread_id),
                "checkpoint_ns": to_storage_safe_str(checkpoint_ns),
                "checkpoint_id": to_storage_safe_id(checkpoint_id),
                "task_id": task_id,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": t,
                "blob": self._encode_blob(b),  # Encode bytes to base64 string for Redis
            }
            for idx, (channel, value) in enumerate(writes)
            for t, b in [self.serde.dumps_typed(value)]
        ]

    def _load_metadata(self, metadata: dict[str, Any]) -> CheckpointMetadata:
        """Load metadata from Redis-compatible dictionary.

        Args:
            metadata: Dictionary representation from Redis.

        Returns:
            Original metadata dictionary.
        """
        # Roundtrip through serializer to ensure proper type handling
        type_str, data_bytes = self.serde.dumps_typed(metadata)
        return self.serde.loads_typed((type_str, data_bytes))

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        """Convert metadata to a Redis-compatible dictionary.

        Args:
            metadata: Metadata to convert.

        Returns:
            Dictionary representation of metadata for Redis storage.
        """
        type_str, serialized_bytes = self.serde.dumps_typed(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
        return serialized_bytes.decode().replace("\\u0000", "")

    def get_next_version(  # type: ignore[override]
        self, current: Optional[str], channel: ChannelProtocol[Any, Any, Any]
    ) -> str:
        """Generate next version number."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _encode_blob(self, blob: Any) -> str:
        """Encode blob data for Redis storage."""
        if isinstance(blob, bytes):
            return base64.b64encode(blob).decode()
        return blob

    def _decode_blob(self, blob: str) -> bytes:
        """Decode blob data from Redis storage."""
        try:
            return base64.b64decode(blob)
        except (binascii.Error, TypeError):
            # Handle both malformed base64 data and incorrect input types
            return blob.encode() if isinstance(blob, str) else blob

    def _load_writes_from_redis(self, write_key: str) -> List[Tuple[str, str, Any]]:
        """Load writes from Redis JSON storage by key."""
        if not write_key:
            return []

        # Get the full JSON document
        # Cast needed: redis-py types json().get() as List[JsonType] but returns dict
        result = cast(Optional[Dict[str, Any]], self._redis.json().get(write_key))
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

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

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

        # For each write, check existence and then perform appropriate operation
        with self._redis.json().pipeline(transaction=False) as pipeline:
            # Keep track of keys we're creating
            created_keys = []

            for write_obj in writes_objects:
                idx_value = write_obj["idx"]
                assert isinstance(idx_value, int)
                key = self._make_redis_checkpoint_writes_key(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    idx_value,
                )

                # First check if key exists
                key_exists = self._redis.exists(key) == 1

                if all(w[0] in WRITES_IDX_MAP for w in writes):
                    # UPSERT case - only update specific fields
                    if key_exists:
                        # Update only channel, type, and blob fields
                        pipeline.json().set(key, "$.channel", write_obj["channel"])
                        pipeline.json().set(key, "$.type", write_obj["type"])
                        pipeline.json().set(key, "$.blob", write_obj["blob"])
                    else:
                        # For new records, set the complete object
                        pipeline.json().set(key, "$", write_obj)
                        created_keys.append(key)
                else:
                    # INSERT case - only insert if doesn't exist
                    if not key_exists:
                        pipeline.json().set(key, "$", write_obj)
                        created_keys.append(key)

            pipeline.execute()

            # Apply TTL to newly created keys
            if created_keys and self.ttl_config and "default_ttl" in self.ttl_config:
                self._apply_ttl_to_keys(
                    created_keys[0], created_keys[1:] if len(created_keys) > 1 else None
                )

            # Update checkpoint to indicate it has writes
            if writes_objects:
                checkpoint_key = self._make_redis_checkpoint_key(
                    to_storage_safe_id(thread_id),
                    to_storage_safe_str(checkpoint_ns),
                    to_storage_safe_id(checkpoint_id),
                )
                # Check if the checkpoint exists before updating
                if self._redis.exists(checkpoint_key):
                    # JSON.SET can add new fields at non-root paths for existing documents
                    # Use JSONPath $ to update at root level
                    self._redis.json().set(checkpoint_key, "$.has_writes", True)

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        if checkpoint_id is None:
            return []  # Early return if no checkpoint_id

        # Most checkpoints don't have writes, return empty list quickly
        # Quick check: see if write registry exists and has any keys
        write_registry_key = self._key_registry.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        registry_exists = self._redis.exists(write_registry_key)

        if not registry_exists:
            # No writes registry means no writes
            return []

        # Use search index instead of keys() to avoid CrossSlot errors
        # Note: All tag fields use sentinel values for consistency
        writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == to_storage_safe_id(thread_id))
            & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)),
            return_fields=["task_id", "idx", "channel", "type", "$.blob"],
            num_results=1000,  # Adjust as needed
        )

        writes_results = self.checkpoint_writes_index.search(writes_query)

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

    def _load_pending_writes_with_registry_check(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        checkpoint_has_writes: bool,
        registry_has_writes: bool,
    ) -> List[PendingWrite]:
        """Load pending writes with pre-computed registry check to avoid duplicate Redis calls."""
        if checkpoint_id is None:
            return []  # Early return if no checkpoint_id

        # Pre-computed registry check instead of making another Redis call
        if not registry_has_writes:
            # No writes in registry means no writes to load
            return []

        # Also check checkpoint-level has_writes flag for additional optimization
        if not checkpoint_has_writes:
            return []

        # Fallback to original FT.SEARCH logic since registry indicates writes exist
        # Use search index instead of keys() to avoid CrossSlot errors
        # Note: All tag fields use sentinel values for consistency
        writes_query = FilterQuery(
            filter_expression=(Tag("thread_id") == to_storage_safe_id(thread_id))
            & (Tag("checkpoint_ns") == to_storage_safe_str(checkpoint_ns))
            & (Tag("checkpoint_id") == to_storage_safe_id(checkpoint_id)),
            return_fields=["task_id", "idx", "channel", "type", "$.blob"],
            num_results=1000,  # Adjust as needed
        )

        writes_results = self.checkpoint_writes_index.search(writes_query)

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

    @staticmethod
    def _load_writes(
        serde: SerializerProtocol, task_id_to_data: dict[tuple[str, str], dict]
    ) -> list[PendingWrite]:
        """Deserialize pending writes."""
        writes = [
            (
                task_id,
                data["channel"],
                serde.loads_typed(
                    (data["type"], BaseRedisSaver._decode_blob_static(data["blob"]))
                ),
            )
            for (task_id, _), data in task_id_to_data.items()
        ]
        return writes

    @staticmethod
    def _decode_blob_static(blob: bytes | str) -> bytes:
        """Decode blob data from Redis storage (static method)."""
        try:
            # If it's already bytes, try to decode as base64
            if isinstance(blob, bytes):
                return base64.b64decode(blob)
            # If it's a string, encode to bytes first then decode
            return base64.b64decode(blob.encode("utf-8"))
        except (binascii.Error, TypeError, ValueError):
            # Handle both malformed base64 data and incorrect input types
            return blob.encode("utf-8") if isinstance(blob, str) else blob

    @staticmethod
    def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
        # Ensure redis_key is a string
        redis_key = safely_decode(redis_key)

        parts = redis_key.split(REDIS_KEY_SEPARATOR)
        # Ensure we have at least 6 parts
        if len(parts) < 6:
            raise ValueError(
                f"Expected at least 6 parts in Redis key, got {len(parts)}"
            )

        # Extract the first 6 parts regardless of total length
        namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = parts[:6]

        if namespace != CHECKPOINT_WRITE_PREFIX:
            raise ValueError("Expected checkpoint key to start with 'checkpoint'")

        return {
            "thread_id": to_storage_safe_str(thread_id),
            "checkpoint_ns": to_storage_safe_str(checkpoint_ns),
            "checkpoint_id": to_storage_safe_str(checkpoint_id),
            "task_id": task_id,
            "idx": idx,
        }

    def _make_redis_checkpoint_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        return REDIS_KEY_SEPARATOR.join(
            [
                self._checkpoint_prefix,
                str(to_storage_safe_id(thread_id)),
                to_storage_safe_str(checkpoint_ns),
                str(to_storage_safe_id(checkpoint_id)),
            ]
        )

    def _make_redis_checkpoint_blob_key(
        self, thread_id: str, checkpoint_ns: str, channel: str, version: str
    ) -> str:
        return REDIS_KEY_SEPARATOR.join(
            [
                self._checkpoint_blob_prefix,
                str(to_storage_safe_id(thread_id)),
                to_storage_safe_str(checkpoint_ns),
                channel,
                version,
            ]
        )

    def _make_redis_checkpoint_writes_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: Optional[int],
    ) -> str:
        storage_safe_thread_id = str(to_storage_safe_id(thread_id))
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        storage_safe_checkpoint_id = str(to_storage_safe_id(checkpoint_id))

        if idx is None:
            return REDIS_KEY_SEPARATOR.join(
                [
                    self._checkpoint_write_prefix,
                    storage_safe_thread_id,
                    storage_safe_checkpoint_ns,
                    storage_safe_checkpoint_id,
                    task_id,
                ]
            )

        return REDIS_KEY_SEPARATOR.join(
            [
                self._checkpoint_write_prefix,
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                storage_safe_checkpoint_id,
                task_id,
                str(idx),
            ]
        )
