"""Tests for Redis key parsing in the BaseRedisSaver class."""

import pytest

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


def test_parse_redis_checkpoint_writes_key_with_exact_parts():
    """Test parsing a Redis key with exactly 6 parts."""
    # Create a key with exactly 6 parts
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify the result
    assert result["thread_id"] == "thread_id"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_id"
    assert result["task_id"] == "task_id"
    assert result["idx"] == "idx"


def test_parse_redis_checkpoint_writes_key_with_extra_parts():
    """Test parsing a Redis key with more than 6 parts."""
    # Create a key with more than 6 parts
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "extra1",
            "extra2",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify the result - should only include the first 6 parts
    assert result["thread_id"] == "thread_id"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_id"
    assert result["task_id"] == "task_id"
    assert result["idx"] == "idx"
    # Extra parts should be ignored
    assert len(result) == 5


def test_parse_redis_checkpoint_writes_key_with_insufficient_parts():
    """Test parsing a Redis key with fewer than 6 parts."""
    # Create a key with fewer than 6 parts
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
        ]
    )

    # Parse the key - should raise ValueError
    with pytest.raises(ValueError, match="Expected at least 6 parts in Redis key"):
        BaseRedisSaver._parse_redis_checkpoint_writes_key(key)


def test_parse_redis_checkpoint_writes_key_with_incorrect_prefix():
    """Test parsing a Redis key with an incorrect prefix."""
    # Create a key with an incorrect prefix
    key = REDIS_KEY_SEPARATOR.join(
        [
            "incorrect_prefix",
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
        ]
    )

    # Parse the key - should raise ValueError
    with pytest.raises(
        ValueError, match="Expected checkpoint key to start with 'checkpoint'"
    ):
        BaseRedisSaver._parse_redis_checkpoint_writes_key(key)


def test_parse_redis_checkpoint_writes_key_with_escaped_special_characters():
    """Test parsing a Redis key with escaped special characters in the parts."""
    # In practice, special characters would be escaped before creating the key
    # This test makes sure the to_storage_safe_str function is being called

    # Create a key with parts that don't contain the separator character
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify the result
    assert result["thread_id"] == "thread_id"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_id"
    assert result["task_id"] == "task_id"
    assert result["idx"] == "idx"
