"""Key registry using sorted sets per checkpoint.

This module provides a registry for tracking writes per checkpoint using Redis
sorted sets, eliminating the need for some FT.SEARCH operations.
"""

from typing import List, Optional, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

from langgraph.checkpoint.redis.util import to_storage_safe_id, to_storage_safe_str

WRITE_KEYS_ZSET_PREFIX = "write_keys_zset"
REDIS_KEY_SEPARATOR = ":"


class CheckpointKeyRegistry:
    """Base class for checkpoint-based key registry using sorted sets."""

    @staticmethod
    def make_write_keys_zset_key(
        thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        """Create the key for the write keys sorted set for a specific checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace (will be converted to storage-safe format)
            checkpoint_id: Checkpoint identifier (will be converted to storage-safe format)

        Returns:
            The Redis key for the write keys sorted set
        """
        # Convert empty strings to sentinel values for RediSearch compatibility
        safe_thread_id = to_storage_safe_id(thread_id)
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        safe_checkpoint_id = to_storage_safe_id(checkpoint_id)

        return REDIS_KEY_SEPARATOR.join(
            [
                WRITE_KEYS_ZSET_PREFIX,
                safe_thread_id,
                safe_checkpoint_ns,
                safe_checkpoint_id,
            ]
        )


class SyncCheckpointKeyRegistry(CheckpointKeyRegistry):
    """Synchronous checkpoint key registry using sorted sets."""

    def __init__(self, redis_client: Union[Redis, RedisCluster]):
        self._redis = redis_client

    def register_write_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        write_key: str,
        score: Optional[float] = None,
    ) -> None:
        """Register a write key in the checkpoint's sorted set.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            write_key: The write key to register
            score: Optional score (defaults to current timestamp)
        """
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        if score is None:
            # Use current timestamp as score for ordering
            import time

            score = time.time()
        self._redis.zadd(zset_key, {write_key: score})

    def register_write_keys_batch(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        write_keys: List[str],
    ) -> None:
        """Register multiple write keys at once.

        Args:
            thread_id: Thread identifier
            checkpoint_ns: Checkpoint namespace
            checkpoint_id: Checkpoint identifier
            write_keys: List of write keys to register
        """
        if not write_keys:
            return

        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        # Use index as score to maintain order
        mapping = {key: idx for idx, key in enumerate(write_keys)}
        self._redis.zadd(zset_key, mapping)  # type: ignore[arg-type]

    def get_write_keys(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[str]:
        """Get all write keys for a specific checkpoint.

        Returns:
            List of write keys in score order
        """
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        # Get all members sorted by score
        keys = self._redis.zrange(zset_key, 0, -1)
        return [key.decode() if isinstance(key, bytes) else key for key in keys]

    def get_write_count(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> int:
        """Get count of write keys for a checkpoint.

        Returns:
            Number of writes registered for this checkpoint
        """
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        # Check if key exists first to avoid unnecessary ZCARD calls
        if not self._redis.exists(zset_key):
            return 0
        return self._redis.zcard(zset_key)

    def has_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> bool:
        """Check if checkpoint has any writes.

        Returns:
            True if checkpoint has writes, False otherwise
        """
        return self.get_write_count(thread_id, checkpoint_ns, checkpoint_id) > 0

    def remove_write_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, write_key: str
    ) -> None:
        """Remove a specific write key from the checkpoint's registry."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        self._redis.zrem(zset_key, write_key)

    def clear_checkpoint_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> None:
        """Clear all write keys for a specific checkpoint."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        self._redis.delete(zset_key)

    def apply_ttl(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, ttl_seconds: int
    ) -> None:
        """Apply TTL to the checkpoint's write registry."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        self._redis.expire(zset_key, ttl_seconds)


class AsyncCheckpointKeyRegistry(CheckpointKeyRegistry):
    """Asynchronous checkpoint key registry using sorted sets."""

    def __init__(self, redis_client: Union[AsyncRedis, AsyncRedisCluster]):
        self._redis = redis_client

    async def register_write_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        write_key: str,
        score: Optional[float] = None,
    ) -> None:
        """Register a write key in the checkpoint's sorted set."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        if score is None:
            import time

            score = time.time()
        await self._redis.zadd(zset_key, {write_key: score})

    async def register_write_keys_batch(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        write_keys: List[str],
    ) -> None:
        """Register multiple write keys at once."""
        if not write_keys:
            return

        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        mapping = {key: idx for idx, key in enumerate(write_keys)}
        await self._redis.zadd(zset_key, mapping)  # type: ignore[arg-type]

    async def get_write_keys(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[str]:
        """Get all write keys for a specific checkpoint."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        keys = await self._redis.zrange(zset_key, 0, -1)
        return [key.decode() if isinstance(key, bytes) else key for key in keys]

    async def get_write_count(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> int:
        """Get count of write keys for a checkpoint."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return await self._redis.zcard(zset_key)

    async def has_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> bool:
        """Check if checkpoint has any writes."""
        count = await self.get_write_count(thread_id, checkpoint_ns, checkpoint_id)
        return count > 0

    async def remove_write_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, write_key: str
    ) -> None:
        """Remove a specific write key from the checkpoint's registry."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        await self._redis.zrem(zset_key, write_key)

    async def clear_checkpoint_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> None:
        """Clear all write keys for a specific checkpoint."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        await self._redis.delete(zset_key)

    async def apply_ttl(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, ttl_seconds: int
    ) -> None:
        """Apply TTL to the checkpoint's write registry."""
        zset_key = self.make_write_keys_zset_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        await self._redis.expire(zset_key, ttl_seconds)
