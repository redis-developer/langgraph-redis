"""Tests for Redis Cluster mode functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from redis import Redis
from redis.exceptions import ResponseError
from ulid import ULID

from langgraph.store.redis import RedisStore
from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
    get_key_with_hash_tag,
)


class MockRedisCluster(Redis):
    """Mock Redis client that simulates cluster mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []
        self.json_get_calls = []

    def cluster(self, command, *args, **kwargs):
        """Mock cluster command that returns cluster info."""
        if command == "info":
            return {"cluster_state": "ok"}
        raise ResponseError(f"Unknown cluster command: {command}")

    def pipeline(self, transaction=True):
        """Mock pipeline method that records the transaction parameter."""
        self.pipeline_calls.append({"transaction": transaction})
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = []
        mock_pipeline.expire.return_value = True
        mock_pipeline.delete.return_value = 1

        # Mock the pipeline's json method
        mock_json = MagicMock()
        mock_json.get.return_value = {"key": "test", "value": {"test": "data"}}
        mock_pipeline.json.return_value = mock_json

        return mock_pipeline

    def expire(self, key, ttl):
        """Mock expire method that records the key and TTL."""
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    def delete(self, key):
        """Mock delete method that records the key."""
        self.delete_calls.append({"key": key})
        return 1

    def ttl(self, key):
        """Mock ttl method."""
        return 3600  # Return a positive TTL

    def json(self):
        """Mock json method."""
        mock = MagicMock()

        def get(key):
            self.json_get_calls.append({"key": key})
            return {"key": key.split(":")[-1], "value": {"test": "data"}}

        mock.get = get
        return mock


@pytest.fixture
def mock_redis_cluster(redis_url):
    """Fixture to create a mock Redis cluster client."""
    # Use from_url to create the Redis client
    client = Redis.from_url(redis_url)
    # Create a mock that inherits from the real client
    mock = MockRedisCluster()
    # Copy connection attributes from the real client
    mock.connection_pool = client.connection_pool
    return mock


@pytest.fixture
def cluster_store(mock_redis_cluster):
    """Fixture to create a RedisStore with a mock Redis cluster client."""
    # Create a store with the mock Redis client
    store = RedisStore(mock_redis_cluster)

    # Verify that cluster mode was detected
    assert store.cluster_mode is True

    # Mock the store_index and vector_index
    mock_index = MagicMock()
    mock_docs = MagicMock()
    mock_docs.docs = []
    mock_index.search.return_value = mock_docs
    mock_index.load.return_value = None

    # Replace the real indices with mocks
    store.store_index = mock_index
    store.vector_index = mock_index

    return store


def test_cluster_mode_detection(mock_redis_cluster):
    """Test that cluster mode is correctly detected."""
    store = RedisStore(mock_redis_cluster)
    assert store.cluster_mode is True

    # Test with a non-cluster Redis client
    with patch.object(
        mock_redis_cluster,
        "cluster",
        side_effect=ResponseError("cluster command not allowed"),
    ):
        store = RedisStore(mock_redis_cluster)
        assert store.cluster_mode is False


def test_hash_tag_in_keys(cluster_store, mock_redis_cluster):
    """Test that keys have hash tags when in cluster mode."""
    # Test the get_key_with_hash_tag function directly
    key = get_key_with_hash_tag(STORE_PREFIX, REDIS_KEY_SEPARATOR, "test_id", True)
    assert key == f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{{test_id}}"

    # Test without hash tag
    key = get_key_with_hash_tag(STORE_PREFIX, REDIS_KEY_SEPARATOR, "test_id", False)
    assert key == f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}test_id"

    # Put a value in the store to generate keys
    cluster_store.put(("test",), "key1", {"data": "value1"})

    # Check that keys in delete calls have hash tags
    # Note: This is an indirect test as we're checking the mock's recorded calls
    for call in mock_redis_cluster.delete_calls:
        key = call["key"]
        if key.startswith(STORE_PREFIX):
            assert "{" in key and "}" in key, f"Key {key} does not have hash tags"


def test_pipeline_transaction_false(cluster_store, mock_redis_cluster):
    """Test that pipeline is created with transaction=False in cluster mode."""
    # Apply TTL to trigger pipeline creation
    cluster_store._apply_ttl_to_keys("test_key", ["related_key"], 1.0)

    # Check that pipeline was created with transaction=False
    assert len(mock_redis_cluster.pipeline_calls) > 0
    for call in mock_redis_cluster.pipeline_calls:
        assert (
            call["transaction"] is False
        ), "Pipeline should be created with transaction=False in cluster mode"

    # Put a value to trigger more pipeline usage
    cluster_store.put(("test",), "key1", {"data": "value1"})

    # Check again
    assert len(mock_redis_cluster.pipeline_calls) > 0
    for call in mock_redis_cluster.pipeline_calls:
        assert (
            call["transaction"] is False
        ), "Pipeline should be created with transaction=False in cluster mode"


def test_ttl_refresh_in_search(cluster_store, mock_redis_cluster):
    """Test that TTL refresh in search uses transaction=False for pipeline in cluster mode."""
    # Clear the pipeline calls to start fresh
    mock_redis_cluster.pipeline_calls = []

    # Create a main key and related keys to refresh
    doc_id = str(ULID())
    main_key = f"{STORE_PREFIX}:{{{doc_id}}}"  # Main key with hash tag
    related_keys = [f"{STORE_VECTOR_PREFIX}:{{{doc_id}}}"]  # Vector key with hash tag

    # Mock the ttl method to return a positive value to trigger TTL refresh
    original_ttl = mock_redis_cluster.ttl
    mock_redis_cluster.ttl = MagicMock(return_value=3600)

    try:
        # Set up TTL config
        cluster_store.ttl_config = {"default_ttl": 5.0}

        # Use the store's method to apply TTL to keys
        # This is what happens during search with refresh_ttl=True
        cluster_store._apply_ttl_to_keys(main_key, related_keys, 5.0)

        # Check that pipeline was created with transaction=False
        assert len(mock_redis_cluster.pipeline_calls) > 0
        for call in mock_redis_cluster.pipeline_calls:
            assert (
                call["transaction"] is False
            ), "Pipeline should be created with transaction=False in cluster mode"
    finally:
        # Restore the original ttl method
        mock_redis_cluster.ttl = original_ttl
