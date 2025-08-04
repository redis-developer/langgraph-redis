"""Integration tests for checkpoint key registry functionality.

This module tests the key registry which tracks write keys per checkpoint
using Redis sorted sets, providing efficient write tracking without FT.SEARCH.
"""

import asyncio
import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis.key_registry import (
    AsyncCheckpointKeyRegistry,
    SyncCheckpointKeyRegistry,
)


@pytest.fixture
def redis_container() -> Generator[RedisContainer, None, None]:
    """Redis container with all required modules."""
    with RedisContainer("redis:8") as container:
        yield container


@pytest.fixture
def redis_url(redis_container: RedisContainer) -> str:
    """Get Redis URL from container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


@contextmanager
def sync_registry(redis_url: str) -> Generator[SyncCheckpointKeyRegistry, None, None]:
    """Create a sync key registry with Redis client."""
    client = Redis.from_url(redis_url, decode_responses=True)
    registry = SyncCheckpointKeyRegistry(client)
    try:
        yield registry
    finally:
        client.close()


@asynccontextmanager
async def async_registry(
    redis_url: str,
) -> AsyncGenerator[AsyncCheckpointKeyRegistry, None]:
    """Create an async key registry with Redis client."""
    client = AsyncRedis.from_url(redis_url, decode_responses=True)
    registry = AsyncCheckpointKeyRegistry(client)
    try:
        yield registry
    finally:
        await client.aclose()


def test_register_and_get_write_keys(redis_url: str) -> None:
    """Test registering and retrieving write keys."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Test single key registration with automatic timestamp
        registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "write_key_1"
        )

        # Test single key registration with custom score
        registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "write_key_2", score=100.0
        )

        # Retrieve keys
        keys = registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id)
        assert len(keys) == 2
        # Keys should be ordered by score (write_key_2 first due to lower score)
        assert keys[0] == "write_key_2"
        assert keys[1] == "write_key_1"


def test_register_write_keys_batch(redis_url: str) -> None:
    """Test batch registration of write keys."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Test batch registration
        write_keys = ["key1", "key2", "key3", "key4"]
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, write_keys
        )

        # Retrieve and verify order
        retrieved_keys = registry.get_write_keys(
            thread_id, checkpoint_ns, checkpoint_id
        )
        assert retrieved_keys == write_keys

        # Test empty batch (should handle gracefully)
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id + "_empty", []
        )
        empty_keys = registry.get_write_keys(
            thread_id, checkpoint_ns, checkpoint_id + "_empty"
        )
        assert empty_keys == []


def test_write_count_and_has_writes(redis_url: str) -> None:
    """Test counting writes and checking existence."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Initially no writes
        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id) == 0
        assert not registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)

        # Add some writes
        write_keys = ["write1", "write2", "write3"]
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, write_keys
        )

        # Check count and existence
        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id) == 3
        assert registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)


def test_remove_write_key(redis_url: str) -> None:
    """Test removing specific write keys."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Add multiple keys
        write_keys = ["keep1", "remove_me", "keep2", "also_remove"]
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, write_keys
        )

        # Remove specific keys
        registry.remove_write_key(thread_id, checkpoint_ns, checkpoint_id, "remove_me")
        registry.remove_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "also_remove"
        )

        # Verify remaining keys
        remaining = registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id)
        assert remaining == ["keep1", "keep2"]
        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id) == 2


def test_clear_checkpoint_writes(redis_url: str) -> None:
    """Test clearing all writes for a checkpoint."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"

        # Create multiple checkpoints with writes
        checkpoint_id1 = str(uuid4())
        checkpoint_id2 = str(uuid4())

        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id1, ["w1", "w2", "w3"]
        )
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id2, ["w4", "w5"]
        )

        # Clear writes for checkpoint1 only
        registry.clear_checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id1)

        # Verify checkpoint1 is cleared but checkpoint2 remains
        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id1) == 0
        assert not registry.has_writes(thread_id, checkpoint_ns, checkpoint_id1)

        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id2) == 2
        assert registry.has_writes(thread_id, checkpoint_ns, checkpoint_id2)


def test_apply_ttl(redis_url: str) -> None:
    """Test applying TTL to write registry."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Add writes
        registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, ["ttl1", "ttl2"]
        )

        # Apply short TTL
        registry.apply_ttl(thread_id, checkpoint_ns, checkpoint_id, ttl_seconds=2)

        # Verify keys exist
        assert registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)

        # Wait for TTL to expire
        time.sleep(3)

        # Keys should be gone
        assert not registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)
        assert registry.get_write_count(thread_id, checkpoint_ns, checkpoint_id) == 0


def test_key_generation(redis_url: str) -> None:
    """Test the key generation for sorted sets."""
    # Test static method
    key = SyncCheckpointKeyRegistry.make_write_keys_zset_key(
        "thread123", "namespace", "checkpoint456"
    )
    assert key == "write_keys_zset:thread123:namespace:checkpoint456"

    # Test with special characters
    key = SyncCheckpointKeyRegistry.make_write_keys_zset_key(
        "thread:with:colons", "ns/with/slash", "check-point"
    )
    assert key == "write_keys_zset:thread:with:colons:ns/with/slash:check-point"


def test_score_ordering(redis_url: str) -> None:
    """Test that write keys are properly ordered by score."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Register keys with different scores
        registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "third", score=3.0
        )
        registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "first", score=1.0
        )
        registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "second", score=2.0
        )

        # Keys should be returned in score order
        keys = registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id)
        assert keys == ["first", "second", "third"]


# Async tests


@pytest.mark.asyncio
async def test_async_register_and_get_write_keys(redis_url: str) -> None:
    """Test async registering and retrieving write keys."""
    async with async_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Test single key registration with automatic timestamp
        await registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "async_write_1"
        )

        # Small delay to ensure different timestamp
        await asyncio.sleep(0.01)

        # Test single key registration with custom score
        await registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "async_write_2", score=50.0
        )

        # Retrieve keys
        keys = await registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id)
        assert len(keys) == 2
        # async_write_2 should come first due to lower score
        assert keys[0] == "async_write_2"
        assert keys[1] == "async_write_1"


@pytest.mark.asyncio
async def test_async_batch_operations(redis_url: str) -> None:
    """Test async batch registration operations."""
    async with async_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "async-ns"
        checkpoint_id = str(uuid4())

        # Test batch registration
        write_keys = ["async_key1", "async_key2", "async_key3"]
        await registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, write_keys
        )

        # Verify
        retrieved = await registry.get_write_keys(
            thread_id, checkpoint_ns, checkpoint_id
        )
        assert retrieved == write_keys

        # Test empty batch
        await registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id + "_empty", []
        )
        empty_result = await registry.get_write_keys(
            thread_id, checkpoint_ns, checkpoint_id + "_empty"
        )
        assert empty_result == []


@pytest.mark.asyncio
async def test_async_has_writes(redis_url: str) -> None:
    """Test async has_writes functionality."""
    async with async_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "async-ns"
        checkpoint_id = str(uuid4())

        # No writes initially
        assert not await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)

        # Add a write
        await registry.register_write_key(
            thread_id, checkpoint_ns, checkpoint_id, "write1"
        )

        # Now has writes
        assert await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)


@pytest.mark.asyncio
async def test_async_remove_and_clear(redis_url: str) -> None:
    """Test async remove and clear operations."""
    async with async_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "async-ns"
        checkpoint_id1 = str(uuid4())
        checkpoint_id2 = str(uuid4())

        # Add writes to two checkpoints
        await registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id1, ["a1", "a2", "a3"]
        )
        await registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id2, ["b1", "b2"]
        )

        # Remove specific key from checkpoint1
        await registry.remove_write_key(thread_id, checkpoint_ns, checkpoint_id1, "a2")

        keys1 = await registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id1)
        assert keys1 == ["a1", "a3"]

        # Clear all writes from checkpoint2
        await registry.clear_checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id2)

        assert not await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id2)
        assert await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id1)


@pytest.mark.asyncio
async def test_async_apply_ttl(redis_url: str) -> None:
    """Test async TTL application."""
    async with async_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "async-ns"
        checkpoint_id = str(uuid4())

        # Add writes
        await registry.register_write_keys_batch(
            thread_id, checkpoint_ns, checkpoint_id, ["ttl_test1", "ttl_test2"]
        )

        # Apply 1 second TTL
        await registry.apply_ttl(thread_id, checkpoint_ns, checkpoint_id, ttl_seconds=1)

        # Verify exists
        assert await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)

        # Wait for expiry
        await asyncio.sleep(2)

        # Should be gone
        assert not await registry.has_writes(thread_id, checkpoint_ns, checkpoint_id)


def test_namespace_isolation(redis_url: str) -> None:
    """Test that different namespaces are properly isolated."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_id = str(uuid4())

        # Add writes to different namespaces
        registry.register_write_keys_batch(
            thread_id, "ns1", checkpoint_id, ["ns1_write1", "ns1_write2"]
        )
        registry.register_write_keys_batch(
            thread_id, "ns2", checkpoint_id, ["ns2_write1", "ns2_write2", "ns2_write3"]
        )

        # Verify isolation
        ns1_keys = registry.get_write_keys(thread_id, "ns1", checkpoint_id)
        ns2_keys = registry.get_write_keys(thread_id, "ns2", checkpoint_id)

        assert len(ns1_keys) == 2
        assert len(ns2_keys) == 3
        assert set(ns1_keys) == {"ns1_write1", "ns1_write2"}
        assert set(ns2_keys) == {"ns2_write1", "ns2_write2", "ns2_write3"}


def test_concurrent_writes(redis_url: str) -> None:
    """Test that concurrent writes are handled properly."""
    with sync_registry(redis_url) as registry:
        thread_id = str(uuid4())
        checkpoint_ns = "test-ns"
        checkpoint_id = str(uuid4())

        # Simulate concurrent writes by adding keys with very close timestamps
        for i in range(10):
            registry.register_write_key(
                thread_id, checkpoint_ns, checkpoint_id, f"concurrent_{i}"
            )

        # All writes should be preserved
        keys = registry.get_write_keys(thread_id, checkpoint_ns, checkpoint_id)
        assert len(keys) == 10
        assert all(f"concurrent_{i}" in keys for i in range(10))
