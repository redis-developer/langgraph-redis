"""
Test for the fix to issue #40 - Fixing numeric version handling with inline storage.
"""

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from redis import Redis

from langgraph.checkpoint.redis import RedisSaver


@pytest.fixture(autouse=True)
async def clear_test_redis(redis_url: str) -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(redis_url)
    try:
        client.flushall()
    finally:
        client.close()


def test_numeric_version_fix(redis_url: str) -> None:
    """
    Test that numeric versions are handled correctly with inline storage.

    With inline storage, channel values are stored directly in the checkpoint
    document, so numeric versions should be automatically converted to strings
    during serialization.
    """
    with RedisSaver.from_conn_string(redis_url) as saver:
        # Set up a basic config
        config = {
            "configurable": {
                "thread_id": "thread-numeric-version-fix",
                "checkpoint_ns": "",
            }
        }

        # Create a basic checkpoint with channel values
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"test_channel": "test_value"}
        checkpoint["channel_versions"] = {
            "test_channel": 1
        }  # Numeric version in checkpoint

        # Store the checkpoint with numeric version
        saved_config = saver.put(
            config,
            checkpoint,
            {},
            {"test_channel": 1},  # Numeric version in new_versions
        )

        # Get the checkpoint ID from the saved config
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]

        # Retrieve the checkpoint
        loaded_checkpoint = saver.get_tuple(saved_config)

        # Verify the checkpoint was stored and retrieved correctly
        assert loaded_checkpoint is not None
        assert (
            loaded_checkpoint.checkpoint["channel_values"]["test_channel"]
            == "test_value"
        )
        assert (
            loaded_checkpoint.checkpoint["channel_versions"]["test_channel"] == "1"
        )  # Should be string

        # Verify inline storage - get the raw checkpoint data
        # Use the actual key format that the saver uses
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            checkpoint_id,
        )
        raw_data = saver._redis.json().get(checkpoint_key)

        assert raw_data is not None
        # Channel values should be stored inline in the checkpoint
        assert "checkpoint" in raw_data
        checkpoint_data = raw_data["checkpoint"]
        if isinstance(checkpoint_data, str):
            import json

            checkpoint_data = json.loads(checkpoint_data)

        # Verify channel_values are inline
        assert "channel_values" in checkpoint_data
        assert checkpoint_data["channel_values"]["test_channel"] == "test_value"

        # Verify no separate blob keys exist
        all_keys = saver._redis.keys("checkpoint_blob:*")
        assert len(all_keys) == 0, "No blob keys should exist with inline storage"
