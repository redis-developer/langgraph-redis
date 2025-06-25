import base64
import binascii
import json
import random
from abc import abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
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

REDIS_KEY_SEPARATOR = ":"
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_BLOB_PREFIX = "checkpoint_blob"
CHECKPOINT_WRITE_PREFIX = "checkpoint_write"


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
            {"name": "checkpoint_id", "type": "tag"},
            {"name": "parent_checkpoint_id", "type": "tag"},
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
            {"name": "version", "type": "tag"},
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


class BaseRedisSaver(BaseCheckpointSaver[str], Generic[RedisClientType, IndexType]):
    """Base Redis implementation for checkpoint saving.

    Uses Redis JSON for storing checkpoints and related data, with RediSearch for querying.
    """

    _redis: RedisClientType
    _owns_its_client: bool = False
    SCHEMAS = SCHEMAS

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
    ) -> None:
        """Initialize Redis-backed checkpoint saver.

        Args:
            redis_url: Redis connection URL
            redis_client: Redis client instance to use (alternative to redis_url)
            connection_args: Additional arguments for Redis connection
            ttl: Optional TTL configuration dict with optional keys:
                - default_ttl: TTL in minutes for all checkpoint keys
                - refresh_on_read: Whether to refresh TTL on reads
        """
        super().__init__(serde=JsonPlusRedisSerializer())
        if redis_url is None and redis_client is None:
            raise ValueError("Either redis_url or redis_client must be provided")

        # Store TTL configuration
        self.ttl_config = ttl

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

        from langgraph.checkpoint.redis.version import __lib_name__, __redisvl_version__

        # Create the client info string with only the redisvl version
        client_info = f"redis-py(redisvl_v{__redisvl_version__})"

        try:
            # Try to use client_setinfo command if available
            await self._redis.client_setinfo("LIB-NAME", client_info)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                # Call with await to ensure it's an async call
                echo_result = self._redis.echo(client_info)
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
        checkpoint: Dict[str, Any],
        channel_values: Dict[str, Any],
        pending_sends: List[Any],
    ) -> Checkpoint:
        if not checkpoint:
            return {}

        loaded = json.loads(checkpoint)  # type: ignore[arg-type]

        # Note: TTL refresh is now handled in get_tuple() to ensure it works
        # with all Redis operations, not just internal deserialization

        return {
            **loaded,
            "pending_sends": [
                self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends or []
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

        Returns:
            Result of the Redis operation
        """
        if ttl_minutes is None:
            # Check if there's a default TTL in config
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

        if ttl_minutes is not None:
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

        # Decode bytes to avoid double serialization
        checkpoint_data = json.loads(data)

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

    def _get_type_and_blob(self, value: Any) -> Tuple[str, Optional[bytes]]:
        """Helper to get type and blob from a value."""
        t, b = self.serde.dumps_typed(value)
        return t, b

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: Dict[str, Any],
        versions: ChannelVersions,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert blob data for Redis storage."""
        if not versions:
            return []

        storage_safe_thread_id = to_storage_safe_id(thread_id)
        storage_safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)

        # Ensure all versions are converted to strings to avoid TypeError with Tag filters
        str_versions = {k: str(v) for k, v in versions.items()}

        return [
            (
                BaseRedisSaver._make_redis_checkpoint_blob_key(
                    storage_safe_thread_id,
                    storage_safe_checkpoint_ns,
                    k,
                    str_versions[k],  # Use the string version
                ),
                {
                    "thread_id": storage_safe_thread_id,
                    "checkpoint_ns": storage_safe_checkpoint_ns,
                    "channel": k,
                    "version": str_versions[k],  # Use the string version
                    "type": (
                        self._get_type_and_blob(values[k])[0]
                        if k in values
                        else "empty"
                    ),
                    "blob": (
                        self._get_type_and_blob(values[k])[1] if k in values else None
                    ),
                },
            )
            for k in str_versions.keys()
        ]

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
                "blob": b,
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
        return self.serde.loads(self.serde.dumps(metadata))

    def _dump_metadata(self, metadata: CheckpointMetadata) -> str:
        """Convert metadata to a Redis-compatible dictionary.

        Args:
            metadata: Metadata to convert.

        Returns:
            Dictionary representation of metadata for Redis storage.
        """
        serialized_metadata = self.serde.dumps(metadata)
        # NOTE: we're using JSON serializer (not msgpack), so we need to remove null characters before writing
        return serialized_metadata.decode().replace("\\u0000", "")

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
        result = self._redis.json().get(write_key)
        if not result:
            return []

        writes = []
        for write in result["writes"]:  # type: ignore[call-overload]
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
                "checkpoint_ns": checkpoint_ns,  # Don't use sentinel for tag fields in RediSearch
                "checkpoint_id": to_storage_safe_id(checkpoint_id),
                "task_id": task_id,
                "task_path": task_path,
                "idx": WRITES_IDX_MAP.get(channel, idx),
                "channel": channel,
                "type": type_,
                "blob": blob,
            }
            writes_objects.append(write_obj)

        # For each write, check existence and then perform appropriate operation
        with self._redis.json().pipeline(transaction=False) as pipeline:
            # Keep track of keys we're creating
            created_keys = []

            for write_obj in writes_objects:
                key = self._make_redis_checkpoint_writes_key(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    write_obj["idx"],
                )

                # First check if key exists
                key_exists = self._redis.exists(key) == 1

                if all(w[0] in WRITES_IDX_MAP for w in writes):
                    # UPSERT case - only update specific fields
                    if key_exists:
                        # Update only channel, type, and blob fields
                        pipeline.set(key, "$.channel", write_obj["channel"])
                        pipeline.set(key, "$.type", write_obj["type"])
                        pipeline.set(key, "$.blob", write_obj["blob"])
                    else:
                        # For new records, set the complete object
                        pipeline.set(key, "$", write_obj)
                        created_keys.append(key)
                else:
                    # INSERT case - only insert if doesn't exist
                    if not key_exists:
                        pipeline.set(key, "$", write_obj)
                        created_keys.append(key)

            pipeline.execute()

            # Apply TTL to newly created keys
            if created_keys and self.ttl_config and "default_ttl" in self.ttl_config:
                self._apply_ttl_to_keys(
                    created_keys[0], created_keys[1:] if len(created_keys) > 1 else None
                )

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
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
                serde.loads_typed((data["type"], data["blob"])),
            )
            for (task_id, _), data in task_id_to_data.items()
        ]
        return writes

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

    @staticmethod
    def _make_redis_checkpoint_key(
        thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        return REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_PREFIX,
                str(to_storage_safe_id(thread_id)),
                to_storage_safe_str(checkpoint_ns),
                str(to_storage_safe_id(checkpoint_id)),
            ]
        )

    @staticmethod
    def _make_redis_checkpoint_blob_key(
        thread_id: str, checkpoint_ns: str, channel: str, version: str
    ) -> str:
        return REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_BLOB_PREFIX,
                str(to_storage_safe_id(thread_id)),
                to_storage_safe_str(checkpoint_ns),
                channel,
                version,
            ]
        )

    @staticmethod
    def _make_redis_checkpoint_writes_key(
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
                    CHECKPOINT_WRITE_PREFIX,
                    storage_safe_thread_id,
                    storage_safe_checkpoint_ns,
                    storage_safe_checkpoint_id,
                    task_id,
                ]
            )

        return REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                storage_safe_thread_id,
                storage_safe_checkpoint_ns,
                storage_safe_checkpoint_id,
                task_id,
                str(idx),
            ]
        )
