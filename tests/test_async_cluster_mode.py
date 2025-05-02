"""Tests for Redis Cluster mode functionality with AsyncRedisStore."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError
from ulid import ULID

from langgraph.store.redis import AsyncRedisStore
from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
    get_key_with_hash_tag,
)


class MockAsyncRedisCluster(AsyncRedis):
    """Mock Async Redis client that simulates cluster mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []
        self.json_get_calls = []

    async def cluster(self, command, *args, **kwargs):  # type: ignore
        """Mock asynchronous cluster command that returns cluster info.

        This is called by AsyncRedisStore for async operations.
        """
        if command == "info":
            return {"cluster_state": "ok"}
        raise ResponseError(f"Unknown cluster command: {command}")

    def pipeline(self, transaction=True):
        """Mock pipeline method that records the transaction parameter."""
        self.pipeline_calls.append({"transaction": transaction})
        mock_pipeline = AsyncMock()

        # Mock the pipeline's execute method
        async def execute():
            return []

        mock_pipeline.execute = execute

        # Mock the pipeline's expire method
        async def expire(key, ttl):
            self.expire_calls.append({"key": key, "ttl": ttl})
            return True

        mock_pipeline.expire = expire

        # Mock the pipeline's delete method
        async def delete(key):
            self.delete_calls.append({"key": key})
            return 1

        mock_pipeline.delete = delete

        # Mock the pipeline's json method
        def json():
            mock_json = AsyncMock()

            async def get(key):
                self.json_get_calls.append({"key": key})
                return {"key": key.split(":")[-1], "value": {"test": "data"}}

            mock_json.get = get
            return mock_json

        mock_pipeline.json = json

        return mock_pipeline

    async def expire(self, key, ttl):
        """Mock expire method that records the key and TTL."""
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    async def delete(self, key):
        """Mock delete method that records the key."""
        self.delete_calls.append({"key": key})
        return 1

    async def ttl(self, key):
        """Mock ttl method."""
        return 3600  # Return a positive TTL

    def json(self):
        """Mock json method."""
        mock = AsyncMock()

        async def get(key):
            self.json_get_calls.append({"key": key})
            return {"key": key.split(":")[-1], "value": {"test": "data"}}

        mock.get = get
        return mock


@pytest.fixture
async def mock_async_redis_cluster(redis_url):
    """Fixture to create a mock Async Redis cluster client."""
    # Use from_url to create the Redis client
    client = AsyncRedis.from_url(redis_url)
    # Create a mock that inherits from the real client
    mock = MockAsyncRedisCluster()
    # Copy connection attributes from the real client
    mock.connection_pool = client.connection_pool
    return mock


@pytest.fixture
async def async_cluster_store(mock_async_redis_cluster):
    """Fixture to create an AsyncRedisStore with a mock Redis cluster client."""
    # Create a store with the mock Redis client
    # Pass the mock client explicitly as redis_client to avoid URL parsing
    async with AsyncRedisStore(redis_client=mock_async_redis_cluster) as store:
        # The cluster_mode will be automatically detected during setup
        # Call setup to ensure cluster mode is detected
        await store.detect_cluster_mode()

        # Verify that cluster mode was detected
        assert store.cluster_mode is True

        # Mock the store_index and vector_index
        mock_index = AsyncMock()
        mock_docs = MagicMock()
        mock_docs.docs = []
        mock_index.search.return_value = mock_docs
        mock_index.load.return_value = None
        mock_index.query.return_value = []

        # Replace the real indices with mocks
        store.store_index = mock_index
        store.vector_index = mock_index
        yield store


@pytest.mark.asyncio
async def test_async_cluster_mode_detection(mock_async_redis_cluster):
    """Test that cluster mode is automatically detected."""
    # Pass the mock client explicitly as redis_client to avoid URL parsing
    async with AsyncRedisStore(redis_client=mock_async_redis_cluster) as store:
        # Cluster mode should be initialized to False
        assert store.cluster_mode is False

        # Call detect_cluster_mode to detect cluster mode
        await store.detect_cluster_mode()

        # Cluster mode should be detected as True
        assert store.cluster_mode is True

        # Test with a non-cluster Redis client by patching the cluster method
        with patch.object(
            mock_async_redis_cluster,
            "cluster",
            side_effect=ResponseError("cluster command not allowed"),
        ):
            # Reset cluster_mode to False
            store.cluster_mode = False

            # Call detect_cluster_mode again
            await store.detect_cluster_mode()

            # Cluster mode should remain False
            assert store.cluster_mode is False


@pytest.mark.asyncio
async def test_async_hash_tag_in_keys(async_cluster_store, mock_async_redis_cluster):
    """Test that keys have hash tags when in cluster mode."""
    # Test the get_key_with_hash_tag function directly
    key = get_key_with_hash_tag(STORE_PREFIX, REDIS_KEY_SEPARATOR, "test_id", True)
    assert key == f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{{test_id}}"

    # Test without hash tag
    key = get_key_with_hash_tag(STORE_PREFIX, REDIS_KEY_SEPARATOR, "test_id", False)
    assert key == f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}test_id"

    # Put a value in the store to generate keys
    await async_cluster_store.aput(("test",), "key1", {"data": "value1"})

    # Check that keys in delete calls have hash tags
    # Note: This is an indirect test as we're checking the mock's recorded calls
    for call in mock_async_redis_cluster.delete_calls:
        key = call["key"]
        if key.startswith(STORE_PREFIX):
            assert "{" in key and "}" in key, f"Key {key} does not have hash tags"


@pytest.mark.asyncio
async def test_async_pipeline_transaction_false(
    async_cluster_store, mock_async_redis_cluster
):
    """Test that pipeline is created with transaction=False in cluster mode."""
    # Apply TTL to trigger pipeline creation
    await async_cluster_store._apply_ttl_to_keys("test_key", ["related_key"], 1.0)

    # Check that pipeline was created with transaction=False
    assert len(mock_async_redis_cluster.pipeline_calls) > 0
    for call in mock_async_redis_cluster.pipeline_calls:
        assert (
            call["transaction"] is False
        ), "Pipeline should be created with transaction=False in cluster mode"

    # Put a value to trigger more pipeline usage
    await async_cluster_store.aput(("test",), "key1", {"data": "value1"})

    # Check again
    assert len(mock_async_redis_cluster.pipeline_calls) > 0
    for call in mock_async_redis_cluster.pipeline_calls:
        assert (
            call["transaction"] is False
        ), "Pipeline should be created with transaction=False in cluster mode"


@pytest.mark.asyncio
async def test_async_ttl_refresh_in_search(
    async_cluster_store, mock_async_redis_cluster
):
    """Test that TTL refresh in search uses transaction=False for pipeline in cluster mode."""
    # Clear the pipeline calls to start fresh
    mock_async_redis_cluster.pipeline_calls = []

    # Create a main key and related keys to refresh
    doc_id = str(ULID())
    main_key = f"{STORE_PREFIX}:{{{doc_id}}}"  # Main key with hash tag
    related_keys = [f"{STORE_VECTOR_PREFIX}:{{{doc_id}}}"]  # Vector key with hash tag

    # Mock the ttl method to return a positive value to trigger TTL refresh
    original_ttl = mock_async_redis_cluster.ttl
    mock_async_redis_cluster.ttl = AsyncMock(return_value=3600)

    try:
        # Set up TTL config
        async_cluster_store.ttl_config = {"default_ttl": 5.0}

        # Use the store's method to apply TTL to keys
        # This is what happens during search with refresh_ttl=True
        await async_cluster_store._apply_ttl_to_keys(main_key, related_keys, 5.0)

        # Check that pipeline was created with transaction=False
        assert len(mock_async_redis_cluster.pipeline_calls) > 0
        for call in mock_async_redis_cluster.pipeline_calls:
            assert (
                call["transaction"] is False
            ), "Pipeline should be created with transaction=False in cluster mode"
    finally:
        # Restore the original ttl method
        mock_async_redis_cluster.ttl = original_ttl
