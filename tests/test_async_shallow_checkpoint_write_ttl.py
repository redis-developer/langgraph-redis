"""Test TTL functionality for checkpoint_write keys in AsyncShallowRedisSaver.

This test verifies that checkpoint_write keys have TTL applied when default_ttl is configured.
"""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from redis.asyncio import Redis as AsyncRedis

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver


@asynccontextmanager
async def async_shallow_saver(redis_url: str, ttl_config: dict = None):
    """Create an async shallow checkpoint saver with proper setup."""
    async with AsyncShallowRedisSaver.from_conn_string(
        redis_url, ttl=ttl_config
    ) as saver:
        yield saver


@pytest.mark.asyncio
async def test_checkpoint_write_keys_have_ttl(redis_url: str) -> None:
    """Test that checkpoint_write keys have TTL applied when default_ttl is configured.

    This test verifies the bug where checkpoint_write keys do not get TTL applied
    in AsyncShallowRedisSaver.aput_writes(), while they do in AsyncRedisSaver.aput_writes().
    """
    # Configure TTL: 0.1 minutes = 6 seconds
    ttl_config = {
        "default_ttl": 0.1,
        "refresh_on_read": False,
    }

    async with async_shallow_saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_id = "test_checkpoint_1"

        # Create a checkpoint
        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        checkpoint["id"] = checkpoint_id

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Save the checkpoint
        result_config = await saver.aput(config, checkpoint, metadata, {})

        # Add writes to the checkpoint
        writes = [("channel1", "value1"), ("channel2", "value2")]
        await saver.aput_writes(result_config, writes, "task1")

        # Get the write keys that were created
        write_key_1 = saver._make_redis_checkpoint_writes_key_cached(
            thread_id, "", checkpoint_id, "task1", 0
        )
        write_key_2 = saver._make_redis_checkpoint_writes_key_cached(
            thread_id, "", checkpoint_id, "task1", 1
        )

        # Verify the write keys exist
        write_1_exists = await saver._redis.exists(write_key_1)
        write_2_exists = await saver._redis.exists(write_key_2)
        assert write_1_exists == 1, "Write key 1 should exist"
        assert write_2_exists == 1, "Write key 2 should exist"

        # Check TTL on checkpoint_write keys
        ttl_write_1 = await saver._redis.ttl(write_key_1)
        ttl_write_2 = await saver._redis.ttl(write_key_2)

        # TTL should be set (positive value, around 6 seconds)
        # -1 means key exists but has no expiry
        # -2 means key does not exist
        assert ttl_write_1 > 0, (
            f"checkpoint_write key should have TTL, got {ttl_write_1}. "
            f"-1 means no expiry set, -2 means key doesn't exist"
        )
        assert ttl_write_2 > 0, (
            f"checkpoint_write key should have TTL, got {ttl_write_2}. "
            f"-1 means no expiry set, -2 means key doesn't exist"
        )

        # TTL should be approximately 6 seconds (with some tolerance)
        assert (
            0 < ttl_write_1 <= 6
        ), f"TTL should be around 6 seconds, got {ttl_write_1}"
        assert (
            0 < ttl_write_2 <= 6
        ), f"TTL should be around 6 seconds, got {ttl_write_2}"


@pytest.mark.asyncio
async def test_checkpoint_write_keys_expire_after_ttl(redis_url: str) -> None:
    """Test that checkpoint_write keys actually expire after the TTL period."""
    # Configure very short TTL: 0.05 minutes = 3 seconds
    ttl_config = {
        "default_ttl": 0.05,
        "refresh_on_read": False,
    }

    async with async_shallow_saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_id = "test_checkpoint_2"

        # Create a checkpoint
        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        checkpoint["id"] = checkpoint_id

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Save the checkpoint
        result_config = await saver.aput(config, checkpoint, metadata, {})

        # Add writes to the checkpoint
        writes = [("channel1", "value1")]
        await saver.aput_writes(result_config, writes, "task1")

        # Get the write key
        write_key = saver._make_redis_checkpoint_writes_key_cached(
            thread_id, "", checkpoint_id, "task1", 0
        )

        # Verify the write key exists initially
        initial_exists = await saver._redis.exists(write_key)
        assert initial_exists == 1, "Write key should exist initially"

        # Wait for TTL to expire (4 seconds > 3 seconds TTL)
        await asyncio.sleep(4)

        # Verify the write key has expired
        final_exists = await saver._redis.exists(write_key)
        assert final_exists == 0, "Write key should have expired after TTL"


@pytest.mark.asyncio
async def test_checkpoint_write_ttl_matches_checkpoint_ttl(redis_url: str) -> None:
    """Test that checkpoint_write keys have the same TTL as the checkpoint itself."""
    ttl_config = {
        "default_ttl": 0.1,  # 6 seconds
        "refresh_on_read": False,
    }

    async with async_shallow_saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())
        checkpoint_id = "test_checkpoint_3"

        # Create a checkpoint
        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        checkpoint["id"] = checkpoint_id

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Save the checkpoint
        result_config = await saver.aput(config, checkpoint, metadata, {})

        # Add writes
        writes = [("channel1", "value1")]
        await saver.aput_writes(result_config, writes, "task1")

        # Get the checkpoint key and write key
        checkpoint_key = saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        write_key = saver._make_redis_checkpoint_writes_key_cached(
            thread_id, "", checkpoint_id, "task1", 0
        )

        # Get TTL for both keys
        checkpoint_ttl = await saver._redis.ttl(checkpoint_key)
        write_ttl = await saver._redis.ttl(write_key)

        # Both should have TTL set
        assert checkpoint_ttl > 0, f"Checkpoint should have TTL, got {checkpoint_ttl}"
        assert write_ttl > 0, f"Write key should have TTL, got {write_ttl}"

        # TTLs should be similar (within 1 second tolerance due to execution time)
        assert abs(checkpoint_ttl - write_ttl) <= 1, (
            f"Checkpoint TTL ({checkpoint_ttl}) and write TTL ({write_ttl}) "
            f"should be similar"
        )
