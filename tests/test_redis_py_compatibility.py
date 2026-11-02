"""Tests for redis-py compatibility across supported Python versions.

Tests cover protocol configuration and basic synchronous and asynchronous
checkpointer and store operations.

Originally added to verify redis-py 7.x support after a get_protocol_version()
TypeError was reported on Python 3.13.

See: https://github.com/redis/redis-py/issues/3501
See: https://github.com/redis/redis-py/pull/3510
"""

from __future__ import annotations

import sys
from typing import AsyncIterator, Iterator

import pytest
import redis
import redis.asyncio as aredis
from redis.commands.helpers import get_protocol_version

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis import RedisStore
from langgraph.store.redis.aio import AsyncRedisStore


# redis-py 8 leaves the protocol configuration unset by default; earlier
# supported releases configure RESP2. These tests inspect configuration only.
_PROTOCOL_CASES = [
    pytest.param(None, None if redis.VERSION >= (8, 0, 0) else 2, id="default"),
    pytest.param(2, 2, id="resp2"),
    pytest.param(3, 3, id="resp3"),
]


class TestRedisPyCompatibility:
    """Test redis-py version information and protocol configuration."""

    def test_redis_version_info(self) -> None:
        """Log redis-py version for debugging."""
        redis_version = redis.__version__
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"\nredis-py version: {redis_version}")
        print(f"Python version: {python_version}")

        # Verify redis version is at least 5.2.1
        major, minor, patch = map(int, redis_version.split(".")[:3])
        assert major >= 5, f"redis-py {redis_version} is too old, need >= 5.2.1"

    @pytest.mark.parametrize("protocol,expected_protocol", _PROTOCOL_CASES)
    def test_get_protocol_version_sync_client(
        self, protocol: int | None, expected_protocol: int | None
    ) -> None:
        """Read the exact sync protocol configuration without connecting.

        This was the function causing TypeError on Python 3.13 before the fix.
        """
        client = redis.Redis() if protocol is None else redis.Redis(protocol=protocol)
        try:
            # This should not raise TypeError on Python 3.13
            version = get_protocol_version(client)
            assert version == expected_protocol
        finally:
            client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("protocol,expected_protocol", _PROTOCOL_CASES)
    async def test_get_protocol_version_async_client(
        self, protocol: int | None, expected_protocol: int | None
    ) -> None:
        """Read the exact async protocol configuration without connecting.

        This was the function causing TypeError on Python 3.13 before the fix.
        """
        client = aredis.Redis() if protocol is None else aredis.Redis(protocol=protocol)
        try:
            # This should not raise TypeError on Python 3.13
            version = get_protocol_version(client)
            assert version == expected_protocol
        finally:
            await client.aclose()


class TestRedisPyWithCheckpointer:
    """Test checkpointer functionality with the installed redis-py version."""

    @pytest.fixture
    def saver(self, redis_url: str) -> Iterator[RedisSaver]:
        """Create a RedisSaver for testing."""
        with RedisSaver.from_conn_string(redis_url) as saver:
            saver.setup()
            yield saver

    @pytest.fixture
    async def async_saver(self, redis_url: str) -> AsyncIterator[AsyncRedisSaver]:
        """Create an AsyncRedisSaver for testing."""
        async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
            await saver.setup()
            yield saver

    def test_sync_checkpointer_basic_operations(
        self, saver: RedisSaver, redis_url: str
    ) -> None:
        """Test basic sync checkpointer operations work with current redis-py."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        # Create a test checkpoint
        config = {
            "configurable": {
                "thread_id": "test-redis7-sync",
                "checkpoint_ns": "",
            }
        }
        checkpoint = Checkpoint(
            v=1,
            id="test-checkpoint-1",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        metadata = CheckpointMetadata()

        # Put and get checkpoint
        result = saver.put(config, checkpoint, metadata, {})
        assert result is not None

        # Verify we can retrieve it
        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint["id"]

    @pytest.mark.asyncio
    async def test_async_checkpointer_basic_operations(
        self, async_saver: AsyncRedisSaver
    ) -> None:
        """Test basic async checkpointer operations work with current redis-py."""
        from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

        config = {
            "configurable": {
                "thread_id": "test-redis7-async",
                "checkpoint_ns": "",
            }
        }
        checkpoint = Checkpoint(
            v=1,
            id="test-checkpoint-2",
            ts="2024-01-01T00:00:00Z",
            channel_values={},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        metadata = CheckpointMetadata()

        result = await async_saver.aput(config, checkpoint, metadata, {})
        assert result is not None

        retrieved = await async_saver.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint["id"]


class TestRedisPyWithStore:
    """Test store functionality with the installed redis-py version."""

    @pytest.fixture
    def store(self, redis_url: str) -> Iterator[RedisStore]:
        """Create a RedisStore for testing."""
        with RedisStore.from_conn_string(redis_url) as store:
            store.setup()
            yield store

    @pytest.fixture
    async def async_store(self, redis_url: str) -> AsyncIterator[AsyncRedisStore]:
        """Create an AsyncRedisStore for testing."""
        async with AsyncRedisStore.from_conn_string(redis_url) as store:
            await store.setup()
            yield store

    def test_sync_store_basic_operations(self, store: RedisStore) -> None:
        """Test basic sync store operations work with current redis-py."""
        namespace = ("test", "redis7", "sync")
        key = "test-key"
        value = {"data": "test-value", "count": 42}

        store.put(namespace, key, value)

        item = store.get(namespace, key)
        assert item is not None
        assert item.value == value

    @pytest.mark.asyncio
    async def test_async_store_basic_operations(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test basic async store operations work with current redis-py."""
        namespace = ("test", "redis7", "async")
        key = "test-key"
        value = {"data": "test-value", "count": 42}

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert item.value == value
