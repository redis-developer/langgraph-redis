"""Test for the semantic search notebook functionality.

This test makes sure that the key parsing fix works with the semantic search
notebook by simulating its exact workflow.
"""

import unittest.mock as mock

import pytest

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


class TestSemanticSearchNotebook:
    """Tests simulating the semantic search notebook."""

    def test_semantic_search_complex_key_parsing(self):
        """Test that the key parsing fix works with complex keys from semantic search."""
        # Create complex keys that would be generated in semantic search
        test_keys = [
            # Simple key with exact number of parts
            f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}thread_123{REDIS_KEY_SEPARATOR}memory_ns{REDIS_KEY_SEPARATOR}checkpoint_id{REDIS_KEY_SEPARATOR}task_id{REDIS_KEY_SEPARATOR}0",
            # Complex key with extra components - this would have failed before our fix
            f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}semantic_search_thread{REDIS_KEY_SEPARATOR}memories{REDIS_KEY_SEPARATOR}user_memories{REDIS_KEY_SEPARATOR}task_123{REDIS_KEY_SEPARATOR}0{REDIS_KEY_SEPARATOR}search_results{REDIS_KEY_SEPARATOR}vector",
            # Very complex key with multiple extra components
            f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}thread_complex{REDIS_KEY_SEPARATOR}memories{REDIS_KEY_SEPARATOR}user/food:prefs{REDIS_KEY_SEPARATOR}task_abc{REDIS_KEY_SEPARATOR}0{REDIS_KEY_SEPARATOR}extra{REDIS_KEY_SEPARATOR}components{REDIS_KEY_SEPARATOR}with{REDIS_KEY_SEPARATOR}many{REDIS_KEY_SEPARATOR}parts",
            # Key with special characters that would be used in tuple keys
            f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}user_123:memories{REDIS_KEY_SEPARATOR}data{REDIS_KEY_SEPARATOR}pizza/pasta:preferences{REDIS_KEY_SEPARATOR}task_456{REDIS_KEY_SEPARATOR}0{REDIS_KEY_SEPARATOR}vector{REDIS_KEY_SEPARATOR}search",
        ]

        # Test parsing each key
        for key in test_keys:
            # This would have failed before our fix for keys with more than 6 components
            result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

            # Verify we get back a proper result dict with all required keys
            assert "thread_id" in result
            assert "checkpoint_ns" in result
            assert "checkpoint_id" in result
            assert "task_id" in result
            assert "idx" in result

            # Verify the first key is parsed correctly (exact number of parts)
            if key == test_keys[0]:
                assert result["thread_id"] == "thread_123"
                assert result["checkpoint_ns"] == "memory_ns"
                assert result["checkpoint_id"] == "checkpoint_id"
                assert result["task_id"] == "task_id"
                assert result["idx"] == "0"

            # Verify the semantic search key parsing (extra components)
            if key == test_keys[1]:
                assert result["thread_id"] == "semantic_search_thread"
                assert result["checkpoint_ns"] == "memories"
                assert result["checkpoint_id"] == "user_memories"
                assert result["task_id"] == "task_123"
                assert result["idx"] == "0"

    def test_semantic_search_insufficient_key_parts(self):
        """Test that we properly raise errors for keys with insufficient parts."""
        # Key with insufficient parts
        insufficient_key = f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}thread_123{REDIS_KEY_SEPARATOR}memory_ns{REDIS_KEY_SEPARATOR}checkpoint_id{REDIS_KEY_SEPARATOR}task_id"

        # This should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            BaseRedisSaver._parse_redis_checkpoint_writes_key(insufficient_key)

        # Verify the error message mentions the right number of parts
        assert "Expected at least 6 parts" in str(excinfo.value)

    def test_semantic_search_incorrect_prefix(self):
        """Test that we properly raise errors for keys with incorrect prefix."""
        # Key with incorrect prefix
        incorrect_prefix_key = f"wrong_prefix{REDIS_KEY_SEPARATOR}thread_123{REDIS_KEY_SEPARATOR}memory_ns{REDIS_KEY_SEPARATOR}checkpoint_id{REDIS_KEY_SEPARATOR}task_id{REDIS_KEY_SEPARATOR}0"

        # This should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            BaseRedisSaver._parse_redis_checkpoint_writes_key(incorrect_prefix_key)

        # Verify the error message mentions the prefix issue
        assert "Expected checkpoint key to start with" in str(excinfo.value)
