"""Comprehensive test suite for Redis key parsing fix.

This suite combines all tests into a single file to verify
our fix for the Redis key parsing issue works in all scenarios.
"""

import pytest

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


def test_standard_key_parsing():
    """Test that standard Redis keys with exactly 6 components work correctly."""
    # Create a standard key with exactly 6 components
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_123",
            "checkpoint_ns",
            "checkpoint_456",
            "task_789",
            "0",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify the result structure
    assert len(result) == 5
    assert set(result.keys()) == {
        "thread_id",
        "checkpoint_ns",
        "checkpoint_id",
        "task_id",
        "idx",
    }
    assert result["thread_id"] == "thread_123"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_456"
    assert result["task_id"] == "task_789"
    assert result["idx"] == "0"


def test_key_with_extra_components():
    """Test that keys with extra components are parsed correctly."""
    # Create a key with extra components (8 parts)
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_123",
            "checkpoint_ns",
            "checkpoint_456",
            "task_789",
            "0",
            "extra1",
            "extra2",
        ]
    )

    # Parse the key with the fixed method
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify that only the first 6 components are used
    assert len(result) == 5
    assert result["thread_id"] == "thread_123"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_456"
    assert result["task_id"] == "task_789"
    assert result["idx"] == "0"

    # Verify that extra components are ignored
    assert "extra1" not in result
    assert "extra2" not in result


def test_subgraph_key_pattern():
    """Test that keys with subgraph components are parsed correctly."""
    # Create a key with a pattern seen in subgraph operations
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "parent_thread",
            "checkpoint_ns",
            "checkpoint_id",
            "subgraph_task",
            "1",
            "subgraph",
            "nested",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify parsing works correctly
    assert result["thread_id"] == "parent_thread"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_id"
    assert result["task_id"] == "subgraph_task"
    assert result["idx"] == "1"


def test_semantic_search_key_pattern():
    """Test that keys with semantic search components are parsed correctly."""
    # Create a key with a pattern seen in semantic search operations
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "search_thread",
            "vector_ns",
            "search_checkpoint",
            "search_task",
            "2",
            "vector_embedding",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify parsing works correctly
    assert result["thread_id"] == "search_thread"
    assert result["checkpoint_ns"] == "vector_ns"
    assert result["checkpoint_id"] == "search_checkpoint"
    assert result["task_id"] == "search_task"
    assert result["idx"] == "2"


def test_insufficient_components():
    """Test that keys with fewer than 6 components raise an error."""
    # Create a key with only 5 components
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
        ]
    )

    # Attempt to parse the key - should raise ValueError
    with pytest.raises(ValueError, match="Expected at least 6 parts in Redis key"):
        BaseRedisSaver._parse_redis_checkpoint_writes_key(key)


def test_incorrect_prefix():
    """Test that keys with an incorrect prefix raise an error."""
    # Create a key with an incorrect prefix
    key = REDIS_KEY_SEPARATOR.join(
        ["wrong_prefix", "thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "0"]
    )

    # Attempt to parse the key - should raise ValueError
    with pytest.raises(
        ValueError, match="Expected checkpoint key to start with 'checkpoint'"
    ):
        BaseRedisSaver._parse_redis_checkpoint_writes_key(key)
