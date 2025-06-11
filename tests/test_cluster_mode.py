"""Tests for RedisStore Redis Cluster mode functionality."""

import json
from datetime import datetime, timezone
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp
from redis import Redis
from redis.cluster import RedisCluster as SyncRedisCluster
from ulid import ULID

from langgraph.store.redis import RedisStore
from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
)
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata


# Override session-scoped redis_container fixture to prevent Docker operations and provide dummy host/port
class DummyCompose:
    def get_service_host_and_port(self, service, port):
        # Return localhost and default port for dummy usage
        return ("localhost", port)


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    """Override redis_container to use DummyCompose instead of real DockerCompose."""
    yield DummyCompose()


# Synchronous Mock Redis Clients
class BaseMockRedis:
    def __init__(self, *args, **kwargs):
        # Do not call super().__init__ to avoid real connection
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []
        self.ttl_calls = []
        self.cluster_info_calls = 0

        # Pipeline mock
        self._pipeline = MagicMock()
        self._pipeline.expire = MagicMock(return_value=self._pipeline)
        self._pipeline.delete = MagicMock(return_value=self._pipeline)
        self._pipeline.json = MagicMock(
            return_value=MagicMock(get=MagicMock(return_value=None))
        )
        self._pipeline.execute = MagicMock(return_value=[])

    def pipeline(self, transaction=True):
        self.pipeline_calls.append({"transaction": transaction})
        return self._pipeline

    def expire(self, key, ttl):
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    def connection_pool(self):
        return MagicMock()

    def delete(self, *keys):
        for key in keys:
            self.delete_calls.append({"key": key})
        return len(keys)

    def ttl(self, key):
        self.ttl_calls.append({"key": key})
        return 3600

    def json(self):
        json_mock = MagicMock()
        json_mock.get = MagicMock(
            return_value={
                "key": "test",
                "value": {"data": "test"},
                "created_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
                "updated_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
            }
        )
        return json_mock

    def cluster(self, subcmd: str, *args, **kwargs):
        self.cluster_info_calls += 1
        from redis.exceptions import ResponseError

        if subcmd.lower() == "info":
            raise ResponseError("ERR This instance has cluster support disabled")
        return {}


class MockRedis(BaseMockRedis, Redis):
    pass


class MockRedisCluster(BaseMockRedis, SyncRedisCluster):
    def __init__(self, *args, **kwargs):
        # Do not call super().__init__ from SyncRedisCluster
        BaseMockRedis.__init__(self)

    def cluster(self, subcmd: str, *args, **kwargs):
        self.cluster_info_calls += 1
        if subcmd.lower() == "info":
            return {"cluster_state": "ok"}
        return {}


@pytest.fixture(params=[False, True])
def store(request):
    """Parameterized fixture for RedisStore with regular or cluster client."""
    is_cluster = request.param
    client = MockRedisCluster() if is_cluster else MockRedis()

    # Basic IndexConfig, embeddings won't be used in these tests
    index_config = {
        "embed": MagicMock(),
        "dims": 128,
        "distance_type": "cosine",
        "fields": ["content"],
    }

    store = RedisStore(conn=client, index=index_config)  # type: ignore

    # Mock the search indices
    store.store_index = MagicMock()
    store.store_index.create = MagicMock()
    store.store_index.search = MagicMock(return_value=MagicMock(docs=[]))
    store.store_index.load = MagicMock()

    store.vector_index = MagicMock()
    store.vector_index.create = MagicMock()
    store.vector_index.query = MagicMock(return_value=[])
    store.vector_index.load = MagicMock()

    store.setup()
    return store


def test_cluster_detection(store):
    """Test that store.cluster_mode is set correctly."""
    is_client_cluster = isinstance(store._redis, SyncRedisCluster)
    assert store.cluster_mode == is_client_cluster


def test_apply_ttl_to_keys_behavior(store):
    """Test _apply_ttl_to_keys behavior for cluster vs. non-cluster."""
    client = store._redis
    client.expire_calls.clear()
    client.pipeline_calls.clear()

    main_key = "main:key"
    related_keys = ["related:key1", "related:key2"]
    ttl_minutes = 10.0

    store._apply_ttl_to_keys(main_key, related_keys, ttl_minutes)

    if store.cluster_mode:
        assert len(client.expire_calls) == 3
        assert {"key": main_key, "ttl": int(ttl_minutes * 60)} in client.expire_calls
        assert {
            "key": related_keys[0],
            "ttl": int(ttl_minutes * 60),
        } in client.expire_calls
        assert {
            "key": related_keys[1],
            "ttl": int(ttl_minutes * 60),
        } in client.expire_calls
        client._pipeline.expire.assert_not_called()
    else:
        assert len(client.pipeline_calls) > 0
        assert client.pipeline_calls[0]["transaction"] is True
        client._pipeline.expire.assert_any_call(main_key, int(ttl_minutes * 60))
        client._pipeline.expire.assert_any_call(related_keys[0], int(ttl_minutes * 60))
        client._pipeline.expire.assert_any_call(related_keys[1], int(ttl_minutes * 60))
        assert len(client.expire_calls) == 0


def test_batch_get_ops_ttl_refresh(store):
    """Test TTL refresh in _batch_get_ops."""
    client = store._redis
    client.expire_calls.clear()
    client.pipeline_calls.clear()
    client.ttl_calls.clear()

    op_idx = 0
    doc_id = str(ULID())
    store_doc_id = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id}"

    # Mock store_index.search to return a document
    mock_doc_data = {
        "key": "test_key",
        "prefix": "test_ns",
        "value": {"data": "content"},
        "id": store_doc_id,
        "created_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
        "updated_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
    }
    mock_redis_doc = MagicMock()
    mock_redis_doc.json = json.dumps(mock_doc_data)
    mock_redis_doc.id = store_doc_id
    store.store_index.search = MagicMock(return_value=MagicMock(docs=[mock_redis_doc]))

    # Mock client.ttl to control TTL refresh logic
    client.ttl_calls.clear()
    client.ttl = lambda key: 3600

    store.ttl_config = {"default_ttl": 5.0}
    get_ops = [
        (op_idx, GetOp(namespace=("test_ns",), key="test_key", refresh_ttl=True))
    ]
    results = [None]

    store._batch_get_ops(get_ops, results)

    if store.cluster_mode:
        assert {
            "key": store_doc_id,
            "ttl": int(store.ttl_config["default_ttl"] * 60),
        } in client.expire_calls
    else:
        assert len(client.pipeline_calls) > 0
        client._pipeline.expire.assert_any_call(
            store_doc_id, int(store.ttl_config["default_ttl"] * 60)
        )
        assert len(client.expire_calls) == 0


def test_batch_put_ops_pre_delete_behavior(store):
    """Test pre-delete behavior in _batch_put_ops."""
    client = store._redis
    client.delete_calls.clear()
    client.pipeline_calls.clear()

    doc_id_to_delete = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{str(ULID())}"
    vector_doc_id_to_delete = f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{str(ULID())}"

    # Mock store_index.search to return a document that needs to be deleted
    mock_store_doc = MagicMock(id=doc_id_to_delete)
    store.store_index.search = MagicMock(return_value=MagicMock(docs=[mock_store_doc]))

    # Mock vector_index.search if index_config is present
    if store.index_config:
        mock_vector_doc = MagicMock(id=vector_doc_id_to_delete)
        store.vector_index.search = MagicMock(
            return_value=MagicMock(docs=[mock_vector_doc])
        )
    else:
        store.vector_index.search = MagicMock(return_value=MagicMock(docs=[]))

    put_ops = [
        (0, PutOp(namespace=("test_ns",), key="test_key", value={"data": "new_val"}))
    ]
    store._batch_put_ops(put_ops)

    if store.cluster_mode:
        assert {"key": doc_id_to_delete} in client.delete_calls
        if store.index_config:
            assert {"key": vector_doc_id_to_delete} in client.delete_calls
        client._pipeline.delete.assert_not_called()
    else:
        assert len(client.pipeline_calls) > 0
        client._pipeline.delete.assert_any_call(doc_id_to_delete)
        if store.index_config:
            client._pipeline.delete.assert_any_call(vector_doc_id_to_delete)
        assert len(client.delete_calls) == 0


def test_batch_search_ops_vector_fetch_behavior(store):
    """Test fetching store docs after vector search in _batch_search_ops."""
    client = store._redis
    client.pipeline_calls.clear()

    if not store.index_config:
        pytest.skip("Skipping vector search test as index_config is not set up for it.")

    store.embeddings = MagicMock()
    store.embeddings.embed_documents = MagicMock(return_value=[[0.1, 0.2]])

    mock_vector_doc_id = str(ULID())
    mock_vector_result_doc = MagicMock()
    mock_vector_result_doc.id = (
        f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{mock_vector_doc_id}"
    )
    mock_vector_result_doc.vector_distance = 0.5
    mock_vector_result_doc.prefix = "test_ns"
    mock_vector_result_doc.key = "test_key"
    store.vector_index.query = MagicMock(return_value=[mock_vector_result_doc])

    expected_store_key = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{mock_vector_doc_id}"
    mock_store_data_search = {
        "prefix": "test_ns",
        "key": "test_key",
        "value": {"content": "data"},
        "created_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
        "updated_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
    }

    mock_json = MagicMock(get=MagicMock(return_value=mock_store_data_search))

    if store.cluster_mode:
        client.json = lambda: mock_json
    else:
        client._pipeline.json.return_value.get.return_value = mock_store_data_search
        client._pipeline.execute.return_value = [mock_store_data_search]

    search_ops = [
        (
            0,
            SearchOp(
                namespace_prefix=("test_ns",), query="some query", limit=1, filter={}
            ),
        )
    ]
    results = [None]

    store._batch_search_ops(search_ops, results)

    if store.cluster_mode:
        assert mock_json.get.call_count == 1
        mock_json.get.assert_called_with(expected_store_key)
        client._pipeline.json.return_value.get.assert_not_called()
    else:
        assert len(client.pipeline_calls) > 0
        client._pipeline.json.return_value.get.assert_called_once_with(
            expected_store_key
        )
        assert not client.json().get.called


def test_batch_list_namespaces_ops_behavior(store):
    """Test listing namespaces in _batch_list_namespaces_ops."""
    mock_doc1 = MagicMock(prefix="test.documents.public")
    mock_doc2 = MagicMock(prefix="test.documents.private")
    mock_doc3 = MagicMock(prefix="test.images.public")
    mock_doc4 = MagicMock(prefix="prod.documents.public")

    mock_search_result = MagicMock(docs=[mock_doc1, mock_doc2, mock_doc3, mock_doc4])
    store.store_index.search = MagicMock(return_value=mock_search_result)

    list_ops = [
        (
            0,
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ),
        (1, ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0)),
    ]
    results: list[Any] = [None, None]
    store._batch_list_namespaces_ops(list_ops, results)

    # Verify results for full depth
    assert len(results[0]) == 4
    assert ("test", "documents", "public") in results[0]
    assert ("test", "documents", "private") in results[0]
    assert ("test", "images", "public") in results[0]
    assert ("prod", "documents", "public") in results[0]

    # Verify results for depth 2
    assert len(results[1]) == 3
    assert all(len(ns) <= 2 for ns in results[1])
    assert ("test", "documents") in results[1]
    assert ("test", "images") in results[1]
    assert ("prod", "documents") in results[1]


@pytest.fixture(params=[False, True])
def checkpoint_saver(request):
    """Parameterized fixture for RedisSaver with regular or cluster client."""
    is_cluster = request.param
    client = MockRedisCluster() if is_cluster else MockRedis()

    saver = RedisSaver(redis_client=client)

    # Mock the search indices
    saver.checkpoints_index = MagicMock()
    saver.checkpoints_index.create = MagicMock()
    saver.checkpoints_index.search = MagicMock(return_value=MagicMock(docs=[]))
    saver.checkpoints_index.load = MagicMock()

    saver.checkpoint_blobs_index = MagicMock()
    saver.checkpoint_blobs_index.create = MagicMock()
    saver.checkpoint_blobs_index.search = MagicMock(return_value=MagicMock(docs=[]))
    saver.checkpoint_blobs_index.load = MagicMock()

    saver.checkpoint_writes_index = MagicMock()
    saver.checkpoint_writes_index.create = MagicMock()
    saver.checkpoint_writes_index.search = MagicMock(return_value=MagicMock(docs=[]))
    saver.checkpoint_writes_index.load = MagicMock()

    saver.setup()
    return saver


def test_checkpoint_saver_cluster_detection(checkpoint_saver):
    """Test that checkpoint saver cluster_mode is set correctly."""
    is_client_cluster = isinstance(checkpoint_saver._redis, SyncRedisCluster)
    assert checkpoint_saver.cluster_mode == is_client_cluster


def test_checkpoint_saver_ttl_behavior(checkpoint_saver):
    """Test TTL behavior for checkpoint saver in cluster vs. non-cluster mode."""
    client = checkpoint_saver._redis
    client.expire_calls.clear()
    client.pipeline_calls.clear()

    # Set up TTL config
    checkpoint_saver.ttl_config = {"default_ttl": 5.0}

    main_key = "checkpoint:test:key"
    blob_keys = ["blob:key1", "blob:key2"]

    checkpoint_saver._apply_ttl_to_keys(main_key, blob_keys)

    if checkpoint_saver.cluster_mode:
        # In cluster mode, TTL operations should be called directly
        assert len(client.expire_calls) == 3
        assert {
            "key": main_key,
            "ttl": 300,
        } in client.expire_calls  # 5 minutes = 300 seconds
        assert {"key": "blob:key1", "ttl": 300} in client.expire_calls
        assert {"key": "blob:key2", "ttl": 300} in client.expire_calls
        # Pipeline should not be used
        assert len(client.pipeline_calls) == 0
    else:
        # In non-cluster mode, pipeline should be used
        assert len(client.pipeline_calls) > 0
        assert client.pipeline_calls[0]["transaction"] is True
        client._pipeline.expire.assert_any_call(main_key, 300)
        client._pipeline.expire.assert_any_call("blob:key1", 300)
        client._pipeline.expire.assert_any_call("blob:key2", 300)
        # Direct expire calls should not be made
        assert len(client.expire_calls) == 0


def test_checkpoint_saver_delete_thread_behavior(checkpoint_saver):
    """Test delete_thread behavior for checkpoint saver in cluster vs. non-cluster mode."""
    client = checkpoint_saver._redis
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

    checkpoint_saver.checkpoints_index.search.return_value = MagicMock(
        docs=[mock_checkpoint_doc]
    )
    checkpoint_saver.checkpoint_blobs_index.search.return_value = MagicMock(
        docs=[mock_blob_doc]
    )
    checkpoint_saver.checkpoint_writes_index.search.return_value = MagicMock(
        docs=[mock_write_doc]
    )

    checkpoint_saver.delete_thread("test_thread")

    if checkpoint_saver.cluster_mode:
        # In cluster mode, delete operations should be called directly
        assert len(client.delete_calls) > 0
        # Pipeline should not be used for deletions
        assert not any(call.get("transaction") for call in client.pipeline_calls)
    else:
        # In non-cluster mode, pipeline should be used for deletions
        assert len(client.pipeline_calls) > 0
        client._pipeline.delete.assert_called()
        # Direct delete calls should not be made
        assert len(client.delete_calls) == 0
