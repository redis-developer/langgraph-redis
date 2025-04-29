"""Focused test for Redis key parsing fix.

This verifies only the key parsing changes that address the specific issue that was 
causing the "too many values to unpack (expected 6)" error in the notebooks.
"""

import pytest
from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


def test_key_parsing_handles_extra_components():
    """Test that the fixed key parsing method can handle keys with more than 6 components."""
    # Create various Redis key patterns that would be seen in different scenarios
    
    # Standard key with 6 components (the original expected format)
    standard_key = REDIS_KEY_SEPARATOR.join([
        CHECKPOINT_WRITE_PREFIX,
        "thread_123",
        "checkpoint_ns",
        "checkpoint_456",
        "task_789",
        "0"
    ])
    
    # Key from subgraph state access with 8 components
    subgraph_key = REDIS_KEY_SEPARATOR.join([
        CHECKPOINT_WRITE_PREFIX,
        "thread_123",
        "checkpoint_ns",
        "checkpoint_456",
        "task_789",
        "0",
        "subgraph",
        "nested"
    ])
    
    # Key from semantic search with 7 components
    search_key = REDIS_KEY_SEPARATOR.join([
        CHECKPOINT_WRITE_PREFIX,
        "thread_123",
        "checkpoint_ns",
        "checkpoint_456",
        "task_789",
        "0",
        "vector_embedding"
    ])
    
    # Parse each key with the fixed method
    standard_result = BaseRedisSaver._parse_redis_checkpoint_writes_key(standard_key)
    subgraph_result = BaseRedisSaver._parse_redis_checkpoint_writes_key(subgraph_key)
    search_result = BaseRedisSaver._parse_redis_checkpoint_writes_key(search_key)
    
    # All results should contain exactly the same 5 keys
    assert set(standard_result.keys()) == {"thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"}
    assert set(subgraph_result.keys()) == {"thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"}
    assert set(search_result.keys()) == {"thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"}
    
    # The values should match the first 6 components of each key
    for result, key in [(standard_result, standard_key), 
                         (subgraph_result, subgraph_key),
                         (search_result, search_key)]:
        parts = key.split(REDIS_KEY_SEPARATOR)
        assert result["thread_id"] == parts[1]
        assert result["checkpoint_ns"] == parts[2]
        assert result["checkpoint_id"] == parts[3]
        assert result["task_id"] == parts[4]
        assert result["idx"] == parts[5]
    
    # Verify that additional components are ignored
    assert "subgraph" not in subgraph_result
    assert "nested" not in subgraph_result
    assert "vector_embedding" not in search_result
    
    # Key with fewer than 6 components should raise an error
    invalid_key = REDIS_KEY_SEPARATOR.join([
        CHECKPOINT_WRITE_PREFIX,
        "thread_123",
        "checkpoint_ns",
        "checkpoint_456",
        "task_789"
    ])
    
    with pytest.raises(ValueError, match="Expected at least 6 parts in Redis key"):
        BaseRedisSaver._parse_redis_checkpoint_writes_key(invalid_key)