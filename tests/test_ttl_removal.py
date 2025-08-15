"""Tests for TTL removal feature (issue #66)."""

import time
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver


def test_ttl_removal_with_negative_one(redis_url: str) -> None:
    """Test that ttl_minutes=-1 removes TTL from keys."""
    saver = RedisSaver(redis_url, ttl={"default_ttl": 1})  # 1 minute default TTL
    saver.setup()

    thread_id = str(uuid4())
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": ["test"]}, step=1
    )
    checkpoint["channel_values"]["messages"] = ["test"]

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Save checkpoint (will have TTL)
    saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

    checkpoint_key = f"checkpoint:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}"

    # Verify TTL is set
    ttl = saver._redis.ttl(checkpoint_key)
    assert 50 <= ttl <= 60, f"TTL should be around 60 seconds, got {ttl}"

    # Remove TTL using -1
    saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)

    # Verify TTL is removed
    ttl_after = saver._redis.ttl(checkpoint_key)
    assert ttl_after == -1, "Key should be persistent after setting ttl_minutes=-1"


def test_ttl_removal_with_related_keys(redis_url: str) -> None:
    """Test that TTL removal works for main key and related keys."""
    saver = RedisSaver(redis_url, ttl={"default_ttl": 1})
    saver.setup()

    thread_id = str(uuid4())

    # Create a checkpoint with writes (to have related keys)
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": ["test"]}, step=1
    )
    checkpoint["channel_values"]["messages"] = ["test"]

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "test-checkpoint",
        }
    }

    # Save checkpoint and writes
    saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})
    saver.put_writes(
        saved_config, [("channel1", "value1"), ("channel2", "value2")], "task-1"
    )

    # Get the keys
    checkpoint_key = f"checkpoint:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}"
    write_key1 = f"checkpoint_write:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}:task-1:0"
    write_key2 = f"checkpoint_write:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}:task-1:1"

    # All keys should have TTL
    assert 50 <= saver._redis.ttl(checkpoint_key) <= 60
    assert 50 <= saver._redis.ttl(write_key1) <= 60
    assert 50 <= saver._redis.ttl(write_key2) <= 60

    # Remove TTL from all keys
    saver._apply_ttl_to_keys(checkpoint_key, [write_key1, write_key2], ttl_minutes=-1)

    # All keys should be persistent
    assert saver._redis.ttl(checkpoint_key) == -1
    assert saver._redis.ttl(write_key1) == -1
    assert saver._redis.ttl(write_key2) == -1


def test_no_ttl_means_persistent(redis_url: str) -> None:
    """Test that no TTL configuration means keys are persistent."""
    # Create saver with no TTL config
    saver = RedisSaver(redis_url)  # No TTL config
    saver.setup()

    thread_id = str(uuid4())
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": ["test"]}, step=1
    )
    checkpoint["channel_values"]["messages"] = ["test"]

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Save checkpoint
    saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

    # Check TTL
    checkpoint_key = f"checkpoint:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}"
    ttl = saver._redis.ttl(checkpoint_key)

    # Should be -1 (persistent) when no TTL configured
    assert ttl == -1, "Key should be persistent when no TTL configured"


def test_ttl_removal_preserves_data(redis_url: str) -> None:
    """Test that removing TTL doesn't affect the data."""
    saver = RedisSaver(redis_url, ttl={"default_ttl": 1})
    saver.setup()

    thread_id = str(uuid4())
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": ["original data"]}, step=1
    )
    checkpoint["channel_values"]["messages"] = ["original data"]

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Save checkpoint
    saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

    # Load data before TTL removal
    loaded_before = saver.get_tuple(saved_config)
    assert loaded_before.checkpoint["channel_values"]["messages"] == ["original data"]

    # Remove TTL
    checkpoint_key = f"checkpoint:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}"
    saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)

    # Load data after TTL removal
    loaded_after = saver.get_tuple(saved_config)
    assert loaded_after.checkpoint["channel_values"]["messages"] == ["original data"]

    # Verify TTL is removed
    assert saver._redis.ttl(checkpoint_key) == -1


@pytest.mark.asyncio
async def test_async_ttl_removal(redis_url: str) -> None:
    """Test TTL removal with async saver."""
    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": 1}
    ) as saver:
        thread_id = str(uuid4())
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels={"messages": ["async test"]}, step=1
        )
        checkpoint["channel_values"]["messages"] = ["async test"]

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        # Save checkpoint
        saved_config = await saver.aput(
            config, checkpoint, {"source": "test", "step": 1}, {}
        )

        checkpoint_key = f"checkpoint:{thread_id}:__empty__:{saved_config['configurable']['checkpoint_id']}"

        # Verify TTL is set
        ttl = await saver._redis.ttl(checkpoint_key)
        assert 50 <= ttl <= 60, f"TTL should be around 60 seconds, got {ttl}"

        # Remove TTL using -1
        await saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)

        # Verify TTL is removed
        ttl_after = await saver._redis.ttl(checkpoint_key)
        assert ttl_after == -1, "Key should be persistent after setting ttl_minutes=-1"


def test_pin_thread_use_case(redis_url: str) -> None:
    """Test the 'pin thread' use case from issue #66.

    This simulates pinning a specific thread by removing its TTL,
    making it persistent while other threads expire.
    """
    saver = RedisSaver(
        redis_url, ttl={"default_ttl": 0.1}
    )  # 6 seconds TTL for quick test
    saver.setup()

    # Create two threads
    thread_to_pin = str(uuid4())
    thread_to_expire = str(uuid4())

    # Store checkpoint IDs to avoid using wildcards (more efficient and precise)
    checkpoint_ids = {}

    for thread_id in [thread_to_pin, thread_to_expire]:
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": [f"Thread {thread_id}"]},
            step=1,
        )
        checkpoint["channel_values"]["messages"] = [f"Thread {thread_id}"]

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})
        checkpoint_ids[thread_id] = saved_config["configurable"]["checkpoint_id"]

    # Pin the first thread by removing its TTL using exact key
    pinned_checkpoint_key = (
        f"checkpoint:{thread_to_pin}:__empty__:{checkpoint_ids[thread_to_pin]}"
    )
    saver._apply_ttl_to_keys(pinned_checkpoint_key, ttl_minutes=-1)

    # Verify pinned thread has no TTL
    assert saver._redis.exists(pinned_checkpoint_key) == 1
    assert saver._redis.ttl(pinned_checkpoint_key) == -1

    # Verify other thread still has TTL
    expiring_checkpoint_key = (
        f"checkpoint:{thread_to_expire}:__empty__:{checkpoint_ids[thread_to_expire]}"
    )
    assert saver._redis.exists(expiring_checkpoint_key) == 1
    ttl = saver._redis.ttl(expiring_checkpoint_key)
    assert 0 < ttl <= 6

    # Wait for expiring thread to expire
    time.sleep(7)

    # Pinned thread should still exist
    assert saver._redis.exists(pinned_checkpoint_key) == 1

    # Expiring thread should be gone
    assert saver._redis.exists(expiring_checkpoint_key) == 0
