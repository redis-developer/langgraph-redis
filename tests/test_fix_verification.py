"""Final verification tests for Redis key parsing fixes.

This test specifically tests the key parsing fix that was causing issues in:
1. semantic-search.ipynb
2. subgraphs-manage-state.ipynb
3. subgraph-persistence.ipynb
"""

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


# Test for the specific key parsing issue with extra components
def test_key_parsing_fix_handles_extra_components():
    """Test that our fix for key parsing correctly handles keys with extra components."""
    # Create various keys with different numbers of components
    keys = [
        # Standard key with exactly 6 components (the original expected format)
        REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "task_id",
                "idx",
            ]
        ),
        # Key with 7 components (would have failed before the fix)
        REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "task_id",
                "idx",
                "extra1",
            ]
        ),
        # Key with 8 components (would have failed before the fix)
        REDIS_KEY_SEPARATOR.join(
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
        ),
        # Key with 9 components (would have failed before the fix)
        REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "task_id",
                "idx",
                "extra1",
                "extra2",
                "extra3",
            ]
        ),
        # Key with a common subgraph pattern (from subgraphs-manage-state.ipynb)
        REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                "user_thread",
                "default",
                "checkpoint_123",
                "task_456",
                "1",
                "subgraph",
                "nested",
            ]
        ),
        # Key with a pattern seen in semantic-search.ipynb
        REDIS_KEY_SEPARATOR.join(
            [
                CHECKPOINT_WRITE_PREFIX,
                "search_thread",
                "memory",
                "vector_checkpoint",
                "search_task",
                "2",
                "query",
                "embedding",
            ]
        ),
    ]

    # Test each key with the _parse_redis_checkpoint_writes_key method
    for key in keys:
        # This would have raised a ValueError before the fix
        result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

        # Verify that only the expected fields are present and have the right values
        assert len(result) == 5
        assert set(result.keys()) == {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
        }

        # Check the extracted values match what we expect (only the first 6 components)
        parts = key.split(REDIS_KEY_SEPARATOR)
        assert result["thread_id"] == parts[1]
        assert result["checkpoint_ns"] == parts[2]
        assert result["checkpoint_id"] == parts[3]
        assert result["task_id"] == parts[4]
        assert result["idx"] == parts[5]
