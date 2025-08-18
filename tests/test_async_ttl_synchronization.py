"""Test async TTL synchronization behavior for AsyncRedisSaver."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint

from langgraph.checkpoint.redis.aio import AsyncRedisSaver


@asynccontextmanager
async def _saver(
    redis_url: str, ttl_config: dict
) -> AsyncGenerator[AsyncRedisSaver, None]:
    """Create an AsyncRedisSaver with the given TTL configuration."""
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl_config) as saver:
        await saver.setup()
        yield saver


@pytest.mark.asyncio
async def test_async_ttl_refresh_on_read(redis_url: str) -> None:
    """Test that TTL is always refreshed when refresh_on_read is enabled (async)."""

    # Configure with TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": True,
    }

    async with _saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_ns = ""
        checkpoint_id = str(uuid4())

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts="2024-01-01T00:00:00+00:00",
            channel_values={"test": "value"},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        # Save the checkpoint
        saved_config = await saver.aput(config, checkpoint, {"test": "metadata"}, {})

        # Get the checkpoint key
        from langgraph.checkpoint.redis.base import BaseRedisSaver
        from langgraph.checkpoint.redis.util import (
            to_storage_safe_id,
            to_storage_safe_str,
        )

        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            to_storage_safe_id(thread_id),
            to_storage_safe_str(checkpoint_ns),
            to_storage_safe_id(saved_config["configurable"]["checkpoint_id"]),
        )

        # Check initial TTL (should be around 120 seconds)
        initial_ttl = await saver._redis.ttl(checkpoint_key)
        assert (
            115 <= initial_ttl <= 120
        ), f"Initial TTL should be ~120s, got {initial_ttl}"

        # Wait a bit (simulate time passing)
        await asyncio.sleep(2)

        # Read the checkpoint - this should refresh TTL to full value
        result = await saver.aget_tuple(saved_config)
        assert result is not None

        # Check TTL after read - should be refreshed to full value
        refreshed_ttl = await saver._redis.ttl(checkpoint_key)
        assert (
            115 <= refreshed_ttl <= 120
        ), f"TTL should be refreshed to ~120s, got {refreshed_ttl}"


@pytest.mark.asyncio
async def test_async_ttl_no_refresh_when_disabled(redis_url: str) -> None:
    """Test that TTL is not refreshed when refresh_on_read is disabled (async)."""

    # Configure without TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": False,  # Don't refresh TTL on read
    }

    async with _saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_ns = ""
        checkpoint_id = str(uuid4())

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts="2024-01-01T00:00:00+00:00",
            channel_values={"test": "value"},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        # Save the checkpoint
        saved_config = await saver.aput(config, checkpoint, {"test": "metadata"}, {})

        # Get the checkpoint key
        from langgraph.checkpoint.redis.base import BaseRedisSaver
        from langgraph.checkpoint.redis.util import (
            to_storage_safe_id,
            to_storage_safe_str,
        )

        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            to_storage_safe_id(thread_id),
            to_storage_safe_str(checkpoint_ns),
            to_storage_safe_id(saved_config["configurable"]["checkpoint_id"]),
        )

        # Check initial TTL
        initial_ttl = await saver._redis.ttl(checkpoint_key)
        assert (
            115 <= initial_ttl <= 120
        ), f"Initial TTL should be ~120s, got {initial_ttl}"

        # Wait a bit
        await asyncio.sleep(2)

        # Read the checkpoint - should NOT refresh TTL when refresh_on_read=False
        result = await saver.aget_tuple(saved_config)
        assert result is not None

        # Check TTL after read - should NOT be refreshed
        current_ttl = await saver._redis.ttl(checkpoint_key)
        assert (
            current_ttl < initial_ttl - 1
        ), f"TTL should have decreased, got {current_ttl}"


@pytest.mark.asyncio
async def test_async_ttl_synchronization_with_external_keys(redis_url: str) -> None:
    """Test TTL synchronization between checkpoint keys and external user keys (async)."""

    # Configure with TTL refresh on read for synchronization
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": True,
    }

    async with _saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_ns = ""
        checkpoint_id = str(uuid4())

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts="2024-01-01T00:00:00+00:00",
            channel_values={"test": "value"},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        # Save the checkpoint
        saved_config = await saver.aput(config, checkpoint, {"test": "metadata"}, {})

        # Create external keys that should expire together
        external_key1 = f"user:metadata:{thread_id}"
        external_key2 = f"user:context:{thread_id}"

        # Set external keys with same TTL
        await saver._redis.set(external_key1, "metadata_value", ex=120)
        await saver._redis.set(external_key2, "context_value", ex=120)

        # Wait a bit
        await asyncio.sleep(2)

        # Read checkpoint - should refresh its TTL
        result = await saver.aget_tuple(saved_config)
        assert result is not None

        # Manually refresh external keys' TTL (simulating user's synchronization logic)
        await saver._redis.expire(external_key1, 120)
        await saver._redis.expire(external_key2, 120)

        # Check that all TTLs are synchronized
        from langgraph.checkpoint.redis.base import BaseRedisSaver
        from langgraph.checkpoint.redis.util import (
            to_storage_safe_id,
            to_storage_safe_str,
        )

        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            to_storage_safe_id(thread_id),
            to_storage_safe_str(checkpoint_ns),
            to_storage_safe_id(saved_config["configurable"]["checkpoint_id"]),
        )

        checkpoint_ttl = await saver._redis.ttl(checkpoint_key)
        external_ttl1 = await saver._redis.ttl(external_key1)
        external_ttl2 = await saver._redis.ttl(external_key2)

        # All TTLs should be close to each other (within 2 seconds)
        assert (
            abs(checkpoint_ttl - external_ttl1) <= 2
        ), f"TTLs not synchronized: {checkpoint_ttl} vs {external_ttl1}"
        assert (
            abs(checkpoint_ttl - external_ttl2) <= 2
        ), f"TTLs not synchronized: {checkpoint_ttl} vs {external_ttl2}"
        assert (
            115 <= checkpoint_ttl <= 120
        ), f"Checkpoint TTL should be ~120s, got {checkpoint_ttl}"


@pytest.mark.asyncio
async def test_async_ttl_no_refresh_for_persistent_keys(redis_url: str) -> None:
    """Test that keys without TTL (persistent) are not affected by refresh logic (async)."""

    # Configure with TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes
        "refresh_on_read": True,
    }

    async with _saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_ns = ""
        checkpoint_id = str(uuid4())

        # Create a checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=checkpoint_id,
            ts="2024-01-01T00:00:00+00:00",
            channel_values={"test": "value"},
            channel_versions={},
            versions_seen={},
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        # Save the checkpoint
        saved_config = await saver.aput(config, checkpoint, {"test": "metadata"}, {})

        # Remove TTL to make it persistent
        from langgraph.checkpoint.redis.base import BaseRedisSaver
        from langgraph.checkpoint.redis.util import (
            to_storage_safe_id,
            to_storage_safe_str,
        )

        checkpoint_key = BaseRedisSaver._make_redis_checkpoint_key(
            to_storage_safe_id(thread_id),
            to_storage_safe_str(checkpoint_ns),
            to_storage_safe_id(saved_config["configurable"]["checkpoint_id"]),
        )
        await saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)

        # Verify key is persistent (TTL = -1)
        ttl_before = await saver._redis.ttl(checkpoint_key)
        assert ttl_before == -1, f"Key should be persistent (TTL=-1), got {ttl_before}"

        # Read the checkpoint
        result = await saver.aget_tuple(saved_config)
        assert result is not None

        # Verify key is still persistent (not affected by refresh)
        ttl_after = await saver._redis.ttl(checkpoint_key)
        assert (
            ttl_after == -1
        ), f"Key should remain persistent (TTL=-1), got {ttl_after}"
