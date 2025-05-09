"""Tests for Redis Cluster mode functionality with AsyncRedisStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import (
    RedisCluster as AsyncRedisCluster,  # Import actual for isinstance checks if needed by store
)

from langgraph.store.redis import AsyncRedisStore


# Override session-scoped redis_container fixture to prevent Docker operations and provide dummy host/port
class DummyCompose:
    def get_service_host_and_port(self, service, port):
        # Return localhost and specified port for dummy usage
        return ("localhost", port)


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    """Override redis_container to use DummyCompose instead of real DockerCompose."""
    yield DummyCompose()


# Basic Mock for non-cluster async client
class AsyncMockRedis(AsyncRedis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []
        # Add other attributes/methods to track if needed

    def pipeline(self, transaction=True):
        # print(f"AsyncMockRedis.pipeline called with transaction={transaction}")
        self.pipeline_calls.append({"transaction": transaction})
        mock_pipeline = AsyncMock()  # Use AsyncMock for awaitable methods
        mock_pipeline.execute = AsyncMock(return_value=[])
        mock_pipeline.expire = AsyncMock(return_value=True)
        mock_pipeline.delete = AsyncMock(return_value=1)

        # Mock json().get() behavior within pipeline
        mock_json_pipeline = AsyncMock()
        mock_json_pipeline.get = AsyncMock(
            return_value={"key": "mock_key", "value": {"data": "mock_data"}}
        )
        mock_pipeline.json = MagicMock(
            return_value=mock_json_pipeline
        )  # json() returns a mock that has async get
        return mock_pipeline

    async def expire(self, key, ttl):
        # print(f"AsyncMockRedis.expire called with key={key}, ttl={ttl}")
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    async def delete(self, key):
        self.delete_calls.append({"key": key})
        return 1

    async def ttl(self, key):
        return 3600  # Default TTL

    def json(self):
        mock_json = AsyncMock()
        mock_json.get = AsyncMock(
            return_value={"key": "mock_key", "value": {"data": "mock_data"}}
        )
        return mock_json

    # Mock cluster method to simulate a non-cluster client
    async def cluster(self, command, *args, **kwargs):
        from redis.exceptions import ResponseError

        if command.lower() == "info":
            raise ResponseError("ERR This instance has cluster support disabled")
        raise ResponseError(f"Unknown cluster command: {command}")


# Mock for cluster async client
class AsyncMockRedisCluster(
    AsyncRedisCluster
):  # Inherit from real to pass isinstance checks in store
    def __init__(self, *args, **kwargs):
        # super().__init__ might be tricky to call if it requires actual cluster setup
        # For mocking purposes, we often bypass the real __init__ or simplify it.
        # If AsyncRedisCluster.__init__ is simple enough or can be called with None/mock args:
        # try:
        #     super().__init__(startup_nodes=None) # Example, adjust as needed
        # except: # pylint: disable=bare-except
        #     pass # Fallback if super().__init__ is problematic
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []

    # Mock pipeline to record calls and simulate async behavior
    def pipeline(self, transaction=True):
        # print(f"AsyncMockRedisCluster.pipeline called with transaction={transaction}")
        self.pipeline_calls.append({"transaction": transaction})
        mock_pipeline = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        mock_pipeline.expire = AsyncMock(return_value=True)
        mock_pipeline.delete = AsyncMock(return_value=1)

        mock_json_pipeline = AsyncMock()
        mock_json_pipeline.get = AsyncMock(
            return_value={"key": "mock_key", "value": {"data": "mock_data"}}
        )
        mock_pipeline.json = MagicMock(return_value=mock_json_pipeline)
        return mock_pipeline

    async def expire(self, key, ttl):
        # print(f"AsyncMockRedisCluster.expire called with key={key}, ttl={ttl}")
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    async def delete(self, key):
        self.delete_calls.append({"key": key})
        return 1

    async def ttl(self, key):
        return 3600  # Default TTL

    def json(self):
        mock_json = AsyncMock()
        mock_json.get = AsyncMock(
            return_value={"key": "mock_key", "value": {"data": "mock_data"}}
        )
        return mock_json

    # Mock cluster method to simulate a cluster client
    async def cluster(self, command, *args, **kwargs):
        if command.lower() == "info":
            return {"cluster_state": "ok"}
        from redis.exceptions import ResponseError

        raise ResponseError(f"Unknown cluster command: {command}")


@pytest.fixture
async def mock_async_redis_cluster_client(redis_url):
    # This fixture provides a mock that IS an instance of AsyncRedisCluster
    # but with mocked methods for testing.
    # For simplicity, we're not trying to fully initialize a real AsyncRedisCluster connection.
    mock_client = AsyncMockRedisCluster(
        host="mockhost"
    )  # host arg may be needed by parent
    # If AsyncRedisStore relies on specific attributes from the client, mock them here:
    # mock_client.connection_pool = AsyncMock()
    return mock_client


@pytest.fixture
async def mock_async_redis_client(redis_url):
    # This provides a mock non-cluster client
    return AsyncMockRedis.from_url(redis_url)  # Standard way to get an async client


@pytest.mark.asyncio
async def test_async_cluster_mode_behavior_differs(
    mock_async_redis_cluster_client, mock_async_redis_client
):
    """Test that AsyncRedisStore behavior differs for cluster vs. non-cluster clients."""

    # --- Test with AsyncMockRedisCluster (simulates cluster) ---
    async_cluster_store = AsyncRedisStore(redis_client=mock_async_redis_cluster_client)
    # Mock indices for async_cluster_store
    mock_index_cluster = AsyncMock()
    mock_index_cluster.search = AsyncMock(return_value=MagicMock(docs=[]))
    mock_index_cluster.load = AsyncMock(return_value=None)
    mock_index_cluster.query = AsyncMock(return_value=[])  # For vector search mocks
    mock_index_cluster.create = AsyncMock(return_value=None)  # For setup
    async_cluster_store.store_index = mock_index_cluster
    async_cluster_store.vector_index = mock_index_cluster
    await async_cluster_store.setup()  # Call setup to initialize indices

    mock_async_redis_cluster_client.expire_calls = []
    mock_async_redis_cluster_client.pipeline_calls = []
    await async_cluster_store.aput(("test_ns",), "key_cluster", {"data": "c"}, ttl=1.0)

    assert len(mock_async_redis_cluster_client.expire_calls) > 0, (
        "Expire should be called directly for async cluster client"
    )
    assert not any(
        call.get("transaction") is True
        for call in mock_async_redis_cluster_client.pipeline_calls
    ), "No transactional pipeline for TTL with async cluster client"

    # --- Test with AsyncMockRedis (simulates non-cluster) ---
    async_non_cluster_store = AsyncRedisStore(redis_client=mock_async_redis_client)
    # Mock indices for async_non_cluster_store
    mock_index_non_cluster = AsyncMock()
    mock_index_non_cluster.search = AsyncMock(return_value=MagicMock(docs=[]))
    mock_index_non_cluster.load = AsyncMock(return_value=None)
    mock_index_non_cluster.query = AsyncMock(return_value=[])
    mock_index_non_cluster.create = AsyncMock(return_value=None)
    async_non_cluster_store.store_index = mock_index_non_cluster
    async_non_cluster_store.vector_index = mock_index_non_cluster
    await async_non_cluster_store.setup()

    mock_async_redis_client.expire_calls = []
    mock_async_redis_client.pipeline_calls = []
    await async_non_cluster_store.aput(
        ("test_ns",), "key_non_cluster", {"data": "nc"}, ttl=1.0
    )

    assert any(
        call.get("transaction") is True
        for call in mock_async_redis_client.pipeline_calls
    ), "Transactional pipeline expected for async non-cluster TTL"
