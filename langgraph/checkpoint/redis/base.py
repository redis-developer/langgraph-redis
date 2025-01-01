import base64
import binascii
import random
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Optional, Tuple, cast

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import ChannelProtocol

from .types import IndexType, RedisClientType

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
            {"name": "checkpoint.v", "type": "numeric"},
            {
                "name": "checkpoint.ts",  # TODO: convert to a number and index as numeric?
                "type": "tag",
            },
            {
                "name": "source",
                "type": "tag",
                "path": "$.metadata.source",
            },
            {
                "name": "step",
                "type": "numeric",
                "path": "$.metadata.step",
            },
            {"name": "parent_checkpoint_id", "type": "tag"},
        ],
    },
    {
        "index": {
            "name": "channel_values",
            "prefix": "channel_value",
            "storage_type": "json",
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_ns", "type": "tag"},
            {"name": "channel", "type": "tag"},
            {"name": "version", "type": "tag"},
            {"name": "type", "type": "tag"},
            {"name": "blob", "type": "text"},
        ],
    },
    {
        "index": {
            "name": "checkpoint_writes",
            "prefix": "checkpoint:writes",
            "storage_type": "json",
        },
        "fields": [
            {"name": "thread_id", "type": "tag"},
            {"name": "checkpoint_ns", "type": "tag"},
            {"name": "checkpoint_id", "type": "tag"},
            {"name": "task_id", "type": "tag", "path": "$.writes[*].task_id"},
            {"name": "channel", "type": "tag", "path": "$.writes[*].channel"},
            {"name": "idx", "type": "numeric", "path": "$.writes[*].idx"},
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
    jsonplus_serde = JsonPlusSerializer()

    checkpoint_index: IndexType
    channel_index: IndexType
    writes_index: IndexType

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[RedisClientType] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if redis_url is None and redis_client is None:
            raise ValueError("Either redis_url or redis_client must be provided")

        self.configure_client(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args or {},
        )
        self.index_prefix = index_prefix

        # Initialize indexes
        self.checkpoint_index: IndexType
        self.channel_index: IndexType
        self.writes_index: IndexType
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
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        pass

    def setup(self) -> None:
        """Initialize the indices in Redis."""
        # Create indexes in Redis
        self.checkpoint_index.create(overwrite=False)
        self.channel_index.create(overwrite=False)
        self.writes_index.create(overwrite=False)

    def _load_checkpoint(
        self,
        checkpoint: dict[str, Any],
        channel_values: dict[str, Any],
        pending_sends: list[Any],
    ) -> Checkpoint:
        """Load checkpoint from Redis data."""
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((send["type"], send["blob"]))
                for send in (pending_sends or [])
            ],
            "channel_values": self._load_blobs(channel_values),
        }

    def _dump_checkpoint(self, checkpoint: Checkpoint) -> dict[str, Any]:
        """Convert checkpoint to Redis format."""
        return {**checkpoint, "pending_sends": []}

    def _load_blobs(self, blob_values: dict[str, Any]) -> dict[str, Any]:
        """Load binary data from Redis."""
        if not blob_values:
            return {}
        return {
            k: self.serde.loads_typed((v["type"], v["blob"]))
            for k, v in blob_values.items()
            if v["type"] != "empty"
        }

    def _get_type_and_blob(self, value: Any) -> tuple[str, Optional[bytes]]:
        """Helper to get type and blob from a value."""
        t, b = self.serde.dumps_typed(value)
        return t, b

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[dict[str, Any]]:
        """Convert blob data for Redis storage."""
        if not versions:
            return []

        return [
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "channel": k,
                "version": cast(str, ver),
                "type": self._get_type_and_blob(values[k])[0]
                if k in values
                else "empty",
                "blob": self._get_type_and_blob(values[k])[1] if k in values else None,
            }
            for k, ver in versions.items()
        ]

    def _load_writes(self, writes: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
        """Load write operations from Redis."""
        if not writes:
            return []
        return [
            (
                write["task_id"],
                write["channel"],
                self.serde.loads_typed((write["type"], write["blob"])),
            )
            for write in writes
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
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
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
        return metadata

    def _dump_metadata(self, metadata: CheckpointMetadata) -> dict[str, Any]:
        """Convert metadata to a Redis-compatible dictionary.

        Args:
            metadata: Metadata to convert.

        Returns:
            Dictionary representation of metadata for Redis storage.
        """
        return metadata

    def get_next_version(self, current: Optional[str], channel: ChannelProtocol) -> str:
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

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        # Sanitize metadata
        sanitized_metadata = {
            k.replace("\x00", ""): v.replace("\x00", "") if isinstance(v, str) else v
            for k, v in metadata.items()
        }

        # Store sanitized metadata
        checkpoint_data = {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": config["configurable"].get("parent_checkpoint_id"),
            "checkpoint": self._dump_checkpoint(checkpoint.copy()),
            "metadata": sanitized_metadata,
            "pending_sends": [],
            "pending_writes": [],
        }
        self.checkpoint_index.load([checkpoint_data])
        return config

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

    def _load_writes_from_redis(self, write_key: str) -> list[tuple[str, str, Any]]:
        """Load writes from Redis JSON storage by key."""
        if not write_key:
            return []

        # Get the full JSON document
        result = self._redis.json().get(write_key)
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
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint using Redis JSON.

        Args:
            config: Configuration of the related checkpoint
            writes: List of writes as (channel, value) pairs
            task_id: Identifier for the task creating the writes
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

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
            pipeline = self.checkpoint_index.client.pipeline()
            try:
                # Check if document exists
                pipeline.json().type(write_key, "$")
                exists = pipeline.execute()[0] is not None

                if exists:
                    # Append to writes array
                    pipeline.json().arrappend(write_key, "$.writes", *writes_objects)
                else:
                    # Create new document
                    pipeline.json().set(
                        write_key,
                        "$",
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                            "writes": writes_objects,
                        },
                    )
                pipeline.execute()
            finally:
                pipeline.reset()
