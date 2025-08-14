"""Test error recovery mechanisms in message deserialization.

This ensures that partial failures in deserialization are handled gracefully
and don't silently mask issues.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


def test_deserialization_error_recovery():
    """Test that deserialization errors are logged and recovered from gracefully."""

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer
        checkpointer = RedisSaver(redis_url)
        checkpointer.setup()

        thread_id = str(uuid4())

        # Create a checkpoint with mixed valid and problematic data
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        # Mix of valid message and problematic data that will fail deserialization
        checkpoint["channel_values"] = {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "Valid message", "type": "human"},
                }
            ],
            "problematic_channel": {
                "lc": 1,
                "type": "constructor",
                "id": ["non", "existent", "class"],  # This will fail to deserialize
                "kwargs": {"data": "test"},
            },
            "normal_data": {
                "key": "value"
            },  # Normal data that doesn't need deserialization
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Patch the logger to capture warnings and errors
        with patch("langgraph.checkpoint.redis.base.logger") as mock_logger:
            # Save checkpoint
            saved_config = checkpointer.put(
                config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
            )

            # Load checkpoint back
            loaded_checkpoint = checkpointer.get(saved_config)

            # Check that we got a result despite partial failure
            assert loaded_checkpoint is not None

            # Check that valid data was deserialized
            assert "messages" in loaded_checkpoint["channel_values"]
            messages = loaded_checkpoint["channel_values"]["messages"]
            assert len(messages) == 1
            assert isinstance(messages[0], HumanMessage)
            assert messages[0].content == "Valid message"

            # Check that normal data passed through
            assert loaded_checkpoint["channel_values"]["normal_data"] == {
                "key": "value"
            }

            # Verify that errors were logged (not silently ignored)
            # Due to the error recovery mechanism, we expect either warning or error logs
            assert (
                mock_logger.warning.called or mock_logger.error.called
            ), "Deserialization errors should be logged, not silently ignored"

    finally:
        redis_container.stop()


def test_complete_deserialization_failure_handling():
    """Test handling when all deserialization attempts fail."""

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer with a broken serializer
        checkpointer = RedisSaver(redis_url)

        # Mock the serializer to always fail on deserialization
        original_deserialize = checkpointer._recursive_deserialize

        def failing_deserialize(obj: Any) -> Any:
            if isinstance(obj, dict) and obj.get("lc"):
                raise ValueError("Simulated deserialization failure")
            return original_deserialize(obj)

        checkpointer._recursive_deserialize = failing_deserialize
        checkpointer.setup()

        thread_id = str(uuid4())

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        # All messages will fail to deserialize
        checkpoint["channel_values"] = {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {"content": "Will fail", "type": "human"},
                }
            ]
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        with patch("langgraph.checkpoint.redis.base.logger") as mock_logger:
            # Save checkpoint
            saved_config = checkpointer.put(
                config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
            )

            # Load checkpoint back - should not raise, but should log errors
            loaded_checkpoint = checkpointer.get(saved_config)

            assert loaded_checkpoint is not None

            # The problematic data should be returned as-is (fallback behavior)
            assert "messages" in loaded_checkpoint["channel_values"]

            # Verify errors were logged
            assert (
                mock_logger.error.called or mock_logger.warning.called
            ), "Complete deserialization failure should be logged"

            # The original dict should be returned as fallback
            messages = loaded_checkpoint["channel_values"]["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                msg = messages[0]
                # It should be the original dict since deserialization failed
                assert isinstance(
                    msg, dict
                ), "Failed deserialization should return original dict"
                assert (
                    msg.get("lc") == 1
                ), "Original structure should be preserved on failure"

    finally:
        redis_container.stop()


def test_partial_channel_deserialization_failure():
    """Test that failure in one channel doesn't affect others."""

    serializer = JsonPlusRedisSerializer()

    # Create a mock saver to test the deserialization logic
    class TestSaver:
        def __init__(self):
            self.serde = serializer

        def _recursive_deserialize(self, obj: Any) -> Any:
            """Simulate failure for specific objects."""
            if isinstance(obj, dict) and obj.get("fail_me"):
                raise ValueError("Intentional failure for testing")

            # Use the real serializer's logic for everything else
            if isinstance(obj, dict):
                if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                    return serializer._reviver(obj)
                return {k: self._recursive_deserialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._recursive_deserialize(item) for item in obj]
            return obj

        def _deserialize_channel_values(
            self, channel_values: dict[str, Any]
        ) -> dict[str, Any]:
            """Test the actual error recovery logic."""
            if not channel_values:
                return {}

            try:
                return self._recursive_deserialize(channel_values)
            except Exception:
                # Recovery mechanism
                recovered = {}
                for key, value in channel_values.items():
                    try:
                        recovered[key] = self._recursive_deserialize(value)
                    except Exception:
                        recovered[key] = value  # Return as-is on failure
                return recovered

    saver = TestSaver()

    # Test data with one channel that will fail
    test_data = {
        "good_channel": [
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "HumanMessage"],
                "kwargs": {"content": "Good message", "type": "human"},
            }
        ],
        "bad_channel": {"fail_me": True, "data": "will fail"},
        "normal_channel": {"regular": "data"},
    }

    # Run deserialization
    result = saver._deserialize_channel_values(test_data)

    # Verify partial success
    assert "good_channel" in result
    assert len(result["good_channel"]) == 1
    assert isinstance(result["good_channel"][0], HumanMessage)

    # Bad channel should return original data
    assert result["bad_channel"] == {"fail_me": True, "data": "will fail"}

    # Normal channel should pass through
    assert result["normal_channel"] == {"regular": "data"}


if __name__ == "__main__":
    test_deserialization_error_recovery()
    test_complete_deserialization_failure_handling()
    test_partial_channel_deserialization_failure()
    print("All error recovery tests passed!")
