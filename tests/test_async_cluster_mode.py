"""Tests for Redis Cluster mode functionality with AsyncRedisStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import (
    RedisCluster as AsyncRedisCluster,  # Import actual for isinstance checks if needed by store
)

from langgraph.store.redis import AsyncRedisStore
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


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
        mock_pipeline.expire = MagicMock(return_value=True)
        mock_pipeline.delete = MagicMock(return_value=1)
        mock_pipeline.execute = AsyncMock(return_value=[])

        # Mock json().get() behavior within pipeline
        mock_json_pipeline = AsyncMock()
        mock_json_pipeline.get = MagicMock()
        mock_pipeline.json = MagicMock(return_value=mock_json_pipeline)
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

        # Add required cluster attributes to prevent AttributeError
        self.cluster_error_retry_attempts = 3
        self.connection_pool = AsyncMock()

    # Mock the client_setinfo method that's called during setup
    async def client_setinfo(self, *args, **kwargs):
        return True

    # Mock execute_command to avoid cluster-specific execution
    async def execute_command(self, *args, **kwargs):
        command = args[0] if args else ""
        if command == "CLIENT SETINFO":
            return True
        # Add other command responses as needed
        return None

    # Mock module_list method for Redis modules check
    async def module_list(self):
        # Return mock modules that satisfy the validation requirements
        return [{"name": "search", "ver": 20600}, {"name": "json", "ver": 20600}]

    # Mock pipeline to record calls and simulate async behavior
    def pipeline(self, transaction=True):
        # print(f"AsyncMockRedisCluster.pipeline called with transaction={transaction}")
        self.pipeline_calls.append({"transaction": transaction})
        mock_pipeline = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        mock_pipeline.expire = MagicMock(return_value=True)
        mock_pipeline.delete = MagicMock(return_value=1)

        mock_json_pipeline = MagicMock()
        mock_json_pipeline.get = MagicMock()
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

    async_cluster_store = AsyncRedisStore(redis_client=mock_async_redis_cluster_client)
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

    assert (
        len(mock_async_redis_cluster_client.expire_calls) > 0
    ), "Expire should be called directly for async cluster client"
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


@pytest.fixture(params=[False, True])
async def async_checkpoint_saver(request):
    """Parameterized fixture for AsyncRedisSaver with regular or cluster client."""
    is_cluster = request.param
    client = AsyncMockRedisCluster() if is_cluster else AsyncMockRedis()

    saver = AsyncRedisSaver(redis_client=client)

    # Mock the search indices
    saver.checkpoints_index = AsyncMock()
    saver.checkpoints_index.create = AsyncMock()
    saver.checkpoints_index.search = AsyncMock(return_value=MagicMock(docs=[]))
    saver.checkpoints_index.load = AsyncMock()

    saver.checkpoint_blobs_index = AsyncMock()
    saver.checkpoint_blobs_index.create = AsyncMock()
    saver.checkpoint_blobs_index.search = AsyncMock(return_value=MagicMock(docs=[]))
    saver.checkpoint_blobs_index.load = AsyncMock()

    saver.checkpoint_writes_index = AsyncMock()
    saver.checkpoint_writes_index.create = AsyncMock()
    saver.checkpoint_writes_index.search = AsyncMock(return_value=MagicMock(docs=[]))
    saver.checkpoint_writes_index.load = AsyncMock()

    # Skip asetup() to avoid complex RedisVL index creation, just test cluster detection
    await saver._detect_cluster_mode()
    return saver


@pytest.mark.asyncio
async def test_async_checkpoint_saver_cluster_detection(async_checkpoint_saver):
    """Test that async checkpoint saver cluster_mode is set correctly."""
    is_client_cluster = isinstance(async_checkpoint_saver._redis, AsyncRedisCluster)
    assert async_checkpoint_saver.cluster_mode == is_client_cluster


@pytest.mark.asyncio
async def test_async_checkpoint_saver_aput_ttl_behavior(async_checkpoint_saver):
    """Test TTL behavior in aput for async checkpoint saver in cluster vs. non-cluster mode."""
    from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

    client = async_checkpoint_saver._redis
    client.expire_calls.clear()
    client.pipeline_calls.clear()

    # Set up TTL config
    async_checkpoint_saver.ttl_config = {"default_ttl": 5.0}

    # Mock the JSON operations to avoid actual data operations
    mock_json = AsyncMock()
    mock_json.set = AsyncMock(return_value=True)
    client.json = MagicMock(return_value=mock_json)

    # Create mock checkpoint and metadata
    config = {
        "configurable": {
            "thread_id": "test_thread",
            "checkpoint_ns": "",
            "checkpoint_id": "test_checkpoint",
        }
    }
    checkpoint: Checkpoint = {"channel_values": {}, "version": "1.0"}
    metadata: CheckpointMetadata = {"source": "test", "step": 1}
    new_versions = {}

    # Call aput which should trigger TTL operations
    await async_checkpoint_saver.aput(config, checkpoint, metadata, new_versions)

    if async_checkpoint_saver.cluster_mode:
        # In cluster mode, TTL operations should be called directly
        assert len(client.expire_calls) >= 1  # At least one TTL call for the checkpoint
        # Check that expire was called with correct TTL (5 minutes = 300 seconds)
        ttl_calls = [call for call in client.expire_calls if call.get("ttl") == 300]
        assert len(ttl_calls) >= 1
    else:
        # In non-cluster mode, pipeline should be used for TTL operations
        assert len(client.pipeline_calls) > 0
        # Should have pipeline calls for the main operations and potentially TTL operations


@pytest.mark.asyncio
async def test_async_checkpoint_saver_delete_thread_behavior(async_checkpoint_saver):
    """Test delete_thread behavior for async checkpoint saver in cluster vs. non-cluster mode."""
    client = async_checkpoint_saver._redis
    client.delete_calls.clear()
    client.pipeline_calls.clear()

    # Mock search results to simulate existing data
    mock_checkpoint_doc = MagicMock()
    mock_checkpoint_doc.checkpoint_ns = "test_ns"
    mock_checkpoint_doc.checkpoint_id = "test_checkpoint"

    mock_blob_doc = MagicMock()
    mock_blob_doc.checkpoint_ns = "test_ns"
    mock_blob_doc.channel = "test_channel"
    mock_blob_doc.version = "1"

    mock_write_doc = MagicMock()
    mock_write_doc.checkpoint_ns = "test_ns"
    mock_write_doc.checkpoint_id = "test_checkpoint"
    mock_write_doc.task_id = "test_task"
    mock_write_doc.idx = 0

    async_checkpoint_saver.checkpoints_index.search.return_value = MagicMock(
        docs=[mock_checkpoint_doc]
    )
    async_checkpoint_saver.checkpoint_blobs_index.search.return_value = MagicMock(
        docs=[]
    )
    async_checkpoint_saver.checkpoint_writes_index.search.return_value = MagicMock(
        docs=[]
    )

    await async_checkpoint_saver.adelete_thread("test_thread")

    if async_checkpoint_saver.cluster_mode:
        # In cluster mode, delete operations should be called directly
        assert len(client.delete_calls) > 0  # At least one checkpoint key deletion
        # Pipeline should not be used for deletions in cluster mode
        # (it might be called for other reasons but not for delete operations)
    else:
        # In non-cluster mode, pipeline should be used for deletions
        assert len(client.pipeline_calls) > 0  # At least one pipeline used
        # Direct delete calls should not be made in non-cluster mode
        assert len(client.delete_calls) == 0
