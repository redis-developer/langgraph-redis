"""Tests for issue #124: Support redis-py 7.x (Python 3.13 compatibility).

This test verifies that the library works correctly with redis-py 7.x,
which includes fixes for Python 3.13's stricter isinstance() behavior.

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


class TestRedis7Compatibility:
    """Test suite for redis-py 7.x compatibility with Python 3.13+."""

    def test_redis_version_info(self) -> None:
        """Log redis-py version for debugging."""
        redis_version = redis.__version__
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"\nredis-py version: {redis_version}")
        print(f"Python version: {python_version}")

        # Verify redis version is at least 5.2.1
        major, minor, patch = map(int, redis_version.split(".")[:3])
        assert major >= 5, f"redis-py {redis_version} is too old, need >= 5.2.1"

    def test_get_protocol_version_sync_client(self) -> None:
        """Test that get_protocol_version works with sync Redis client.

        This was the function causing TypeError on Python 3.13 before the fix.
        """
        client = redis.Redis()
        try:
            # This should not raise TypeError on Python 3.13
            version = get_protocol_version(client)
            assert version in (2, 3), f"Unexpected protocol version: {version}"
        finally:
            client.close()

    def test_get_protocol_version_async_client(self) -> None:
        """Test that get_protocol_version works with async Redis client.

        This was the function causing TypeError on Python 3.13 before the fix.
        """
        client = aredis.Redis()
        try:
            # This should not raise TypeError on Python 3.13
            version = get_protocol_version(client)
            assert version in (2, 3), f"Unexpected protocol version: {version}"
        finally:
            # Async client cleanup
            import asyncio

            try:
                asyncio.get_event_loop().run_until_complete(client.aclose())
            except RuntimeError:
                # Event loop may not be available in sync test
                pass


class TestRedis7WithCheckpointer:
    """Test checkpointer functionality with redis-py 7.x compatibility."""

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


class TestRedis7WithStore:
    """Test store functionality with redis-py 7.x compatibility."""

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
