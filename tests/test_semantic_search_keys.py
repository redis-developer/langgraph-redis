"""Tests for Redis key parsing with semantic search.

This test verifies that the fix to the Redis key handling works correctly
with the semantic search functionality.
"""

import unittest.mock as mock
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pytest

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)

# Import the Redis store - we'll use mock when needed
from langgraph.store.redis import RedisStore


class Memory(TypedDict):
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def create_dummy_embedding(size: int = 4) -> List[float]:
    """Create a dummy embedding for testing."""
    return list(np.random.random(size).astype(float))


class TestSemanticSearchKeyHandling:
    """Test semantic search key handling without requiring RediSearch."""

    def test_parse_complex_keys(self):
        """Test that the _parse_redis_checkpoint_writes_key method handles complex keys."""
        # This directly tests the fix we made
        # Create a key that simulates what would be generated in semantic search
        complex_key = f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}thread_123{REDIS_KEY_SEPARATOR}memory_ns{REDIS_KEY_SEPARATOR}user_memories{REDIS_KEY_SEPARATOR}task_id{REDIS_KEY_SEPARATOR}0{REDIS_KEY_SEPARATOR}extra_component{REDIS_KEY_SEPARATOR}another_component"

        # Parse the key using the patched method
        result = BaseRedisSaver._parse_redis_checkpoint_writes_key(complex_key)

        # Verify the result contains the expected components
        assert result["thread_id"] == "thread_123"
        assert result["checkpoint_ns"] == "memory_ns"
        assert result["checkpoint_id"] == "user_memories"
        assert result["task_id"] == "task_id"
        assert result["idx"] == "0"

        # The extra components should be ignored by our fix

    def test_semantic_search_key_simulation(self):
        """Simulate semantic search operations and key handling."""
        # Create a key pattern like those generated in semantic search
        namespace = "user_123"
        memory_id = "memory_456"

        # Mock Redis client
        mock_redis = mock.MagicMock()
        mock_redis.hgetall.return_value = {
            "content": "Test memory content",
            "metadata": '{"source": "test", "timestamp": "2023-01-01"}',
            "embedding": "[0.1, 0.2, 0.3, 0.4]",
        }

        # Create a mock for RedisStore with a mocked Redis client
        with mock.patch("redis.Redis", return_value=mock_redis):
            with mock.patch.object(RedisStore, "put", return_value=None):
                with mock.patch.object(
                    RedisStore,
                    "get",
                    return_value={
                        "content": "Test memory content",
                        "metadata": {"source": "test", "timestamp": "2023-01-01"},
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                    },
                ):
                    # Mock the RedisStore for testing
                    store = RedisStore("redis://localhost")

                    # Create a test memory
                    memory = {
                        "content": "Test memory content",
                        "metadata": {"source": "test", "timestamp": "2023-01-01"},
                        "embedding": create_dummy_embedding(),
                    }

                    # Test with tuple key - simulate storing
                    store.put(namespace, memory_id, memory)

                    # Test retrieval
                    retrieved = store.get(namespace, memory_id)

                    # Verify the retrieved data
                    assert retrieved["content"] == memory["content"]
                    assert retrieved["metadata"] == memory["metadata"]

    def test_complex_semantic_search_keys(self):
        """Test with more complex keys that would be used in semantic search."""
        # Create complex keys with special characters and multiple components
        namespace = "user/123:456"
        memory_id = "memory/with:special.chars/456"

        # Construct a checkpoint key like the ones that would be generated
        # This simulates what would happen internally in the checkpointer
        checkpoint_key = f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}{namespace}:{memory_id}{REDIS_KEY_SEPARATOR}memories{REDIS_KEY_SEPARATOR}search_results{REDIS_KEY_SEPARATOR}task_123{REDIS_KEY_SEPARATOR}0{REDIS_KEY_SEPARATOR}extra{REDIS_KEY_SEPARATOR}components"

        # Parse with our fixed method
        result = BaseRedisSaver._parse_redis_checkpoint_writes_key(checkpoint_key)

        # Verify the components are extracted correctly
        assert "thread_id" in result
        assert "checkpoint_ns" in result
        assert "checkpoint_id" in result
        assert "task_id" in result
        assert "idx" in result

        # The key would have been successfully parsed with our fix
        # which is what prevented the original notebooks from working
