"""Test TTL synchronization behavior for RedisSaver."""

import time
from contextlib import contextmanager
from typing import Generator
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import Checkpoint

from langgraph.checkpoint.redis import RedisSaver


@contextmanager
def _saver(redis_url: str, ttl_config: dict) -> Generator[RedisSaver, None, None]:
    """Create a RedisSaver with the given TTL configuration."""
    with RedisSaver.from_conn_string(redis_url, ttl=ttl_config) as saver:
        saver.setup()
        yield saver


def test_ttl_refresh_on_read(redis_url: str) -> None:
    """Test that TTL is always refreshed when refresh_on_read is enabled."""

    # Configure with TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": True,
    }

    with _saver(redis_url, ttl_config) as saver:
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
        saved_config = saver.put(config, checkpoint, {"test": "metadata"}, {})

        # Get the checkpoint key
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns, saved_config["configurable"]["checkpoint_id"]
        )

        # Check initial TTL (should be around 120 seconds)
        initial_ttl = saver._redis.ttl(checkpoint_key)
        assert (
            115 <= initial_ttl <= 120
        ), f"Initial TTL should be ~120s, got {initial_ttl}"

        # Wait a bit (simulate time passing)
        time.sleep(2)

        # Read the checkpoint - this should refresh TTL to full value
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Check TTL after read - should be refreshed to full value
        refreshed_ttl = saver._redis.ttl(checkpoint_key)
        assert (
            115 <= refreshed_ttl <= 120
        ), f"TTL should be refreshed to ~120s, got {refreshed_ttl}"


def test_ttl_no_refresh_when_disabled(redis_url: str) -> None:
    """Test that TTL is not refreshed when refresh_on_read is disabled."""

    # Configure without TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": False,  # Don't refresh TTL on read
    }

    with _saver(redis_url, ttl_config) as saver:
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
        saved_config = saver.put(config, checkpoint, {"test": "metadata"}, {})

        # Get the checkpoint key
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns, saved_config["configurable"]["checkpoint_id"]
        )

        # Check initial TTL
        initial_ttl = saver._redis.ttl(checkpoint_key)
        assert (
            115 <= initial_ttl <= 120
        ), f"Initial TTL should be ~120s, got {initial_ttl}"

        # Wait a bit
        time.sleep(2)

        # Read the checkpoint - should NOT refresh TTL when refresh_on_read=False
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Check TTL after read - should NOT be refreshed
        current_ttl = saver._redis.ttl(checkpoint_key)
        assert (
            current_ttl < initial_ttl - 1
        ), f"TTL should have decreased, got {current_ttl}"


def test_ttl_synchronization_with_external_keys(redis_url: str) -> None:
    """Test TTL synchronization between checkpoint keys and external user keys."""

    # Configure with TTL refresh on read for synchronization
    ttl_config = {
        "default_ttl": 2,  # 2 minutes = 120 seconds
        "refresh_on_read": True,
    }

    with _saver(redis_url, ttl_config) as saver:
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
        saved_config = saver.put(config, checkpoint, {"test": "metadata"}, {})

        # Create external keys that should expire together
        external_key1 = f"user:metadata:{thread_id}"
        external_key2 = f"user:context:{thread_id}"

        # Set external keys with same TTL
        saver._redis.set(external_key1, "metadata_value", ex=120)
        saver._redis.set(external_key2, "context_value", ex=120)

        # Wait a bit
        time.sleep(2)

        # Read checkpoint - should refresh its TTL
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Manually refresh external keys' TTL (simulating user's synchronization logic)
        saver._redis.expire(external_key1, 120)
        saver._redis.expire(external_key2, 120)

        # Check that all TTLs are synchronized
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns, saved_config["configurable"]["checkpoint_id"]
        )

        checkpoint_ttl = saver._redis.ttl(checkpoint_key)
        external_ttl1 = saver._redis.ttl(external_key1)
        external_ttl2 = saver._redis.ttl(external_key2)

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


def test_ttl_no_refresh_for_persistent_keys(redis_url: str) -> None:
    """Test that keys without TTL (persistent) are not affected by refresh logic."""

    # Configure with TTL refresh on read
    ttl_config = {
        "default_ttl": 2,  # 2 minutes
        "refresh_on_read": True,
    }

    with _saver(redis_url, ttl_config) as saver:
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
        saved_config = saver.put(config, checkpoint, {"test": "metadata"}, {})

        # Remove TTL to make it persistent
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            thread_id, checkpoint_ns, saved_config["configurable"]["checkpoint_id"]
        )
        saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)

        # Verify key is persistent (TTL = -1)
        ttl_before = saver._redis.ttl(checkpoint_key)
        assert ttl_before == -1, f"Key should be persistent (TTL=-1), got {ttl_before}"

        # Read the checkpoint
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Verify key is still persistent (not affected by refresh)
        ttl_after = saver._redis.ttl(checkpoint_key)
        assert (
            ttl_after == -1
        ), f"Key should remain persistent (TTL=-1), got {ttl_after}"
