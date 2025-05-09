"""Tests for AsyncRedisStore Redis Cluster mode functionality."""

from __future__ import annotations

import asyncio
import json

# Need datetime and timezone for timestamps
from datetime import datetime, timezone
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.store.base import GetOp, IndexConfig, ListNamespacesOp, PutOp, SearchOp
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from ulid import ULID

from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
    RedisDocument,
)


# Mock Async Redis Clients
class MockAsyncRedis(AsyncMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_calls = []
        self.expire_calls = []
        self.delete_calls = []
        self.json_get_calls = []
        self.ttl_calls = []
        self.cluster_info_calls = 0

        # Mock methods called during AsyncRedisStore initialization or setup
        self.client_setinfo = AsyncMock(return_value=True)
        self.ping = AsyncMock(return_value=True)
        self.echo = AsyncMock(return_value="echo")

        # For pipeline
        self.pipeline_instance = (
            AsyncMock()
        )  # This is the pipeline object, its execute method is awaitable
        self.pipeline_instance.execute = AsyncMock(
            return_value=[]
        )  # The actual async execution

        # Chainable methods on the pipeline should be MagicMocks that return the pipeline_instance for chaining
        self.pipeline_instance.expire = MagicMock(return_value=self.pipeline_instance)
        self.pipeline_instance.delete = MagicMock(return_value=self.pipeline_instance)

        json_commands_mock = MagicMock()
        json_commands_mock.get = MagicMock(
            return_value=json_commands_mock
        )  # a.json().get() returns json_commands_mock
        self.pipeline_instance.json = MagicMock(return_value=json_commands_mock)

        # For json().get() direct calls
        self.json_mock = MagicMock()
        current_time = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        self.json_mock.get = AsyncMock(
            return_value={
                "key": "test",
                "value": {"data": "test"},
                "created_at": current_time,
                "updated_at": current_time,
            }
        )
        self.json = MagicMock(return_value=self.json_mock)

    def pipeline(self, transaction=True):
        self.pipeline_calls.append({"transaction": transaction})
        # Reset chainable MagicMock calls for fresh assertions per pipeline use
        self.pipeline_instance.expire.reset_mock()
        self.pipeline_instance.delete.reset_mock()
        self.pipeline_instance.json.return_value.get.reset_mock()
        return self.pipeline_instance

    async def expire(self, key, ttl):
        self.expire_calls.append({"key": key, "ttl": ttl})
        return True

    async def delete(self, *keys):
        for key in keys:
            self.delete_calls.append({"key": key})
        return len(keys)

    async def ttl(self, key):
        self.ttl_calls.append({"key": key})
        return 3600  # Default TTL

    # Add cluster mock to raise an error to signal non-cluster
    async def cluster(self, subcmd: str, *args, **kwargs) -> dict:
        self.cluster_info_calls += 1
        if subcmd.lower() == "info":
            # Simulate non-cluster by raising a ResponseError
            from redis.exceptions import ResponseError

            raise ResponseError("ERR This instance has cluster support disabled")
        return {}


class MockAsyncRedisCluster(MockAsyncRedis):  # Inherits tracking from MockAsyncRedis
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override or add cluster-specific mocks if any method behaves differently beyond the instance check

    # Override cluster_info to simulate cluster
    async def cluster(self, subcmd: str, *args, **kwargs) -> dict:
        self.cluster_info_calls += 1
        if subcmd.lower() == "info":
            # Simulate cluster by returning some mock cluster data
            return {"cluster_state": "ok"}
        return {}


@pytest.fixture
def mock_async_redis_client():
    return MockAsyncRedis()


@pytest.fixture
def mock_async_redis_cluster_client():
    # This will be identified as a cluster client by AsyncRedisStore
    client = AsyncMock(spec=AsyncRedisCluster)  # Use spec for isinstance checks

    # Add all the mocked methods from MockAsyncRedis to this spec'ed mock
    client.pipeline_calls = []
    client.expire_calls = []
    client.delete_calls = []
    client.json_get_calls = []
    client.ttl_calls = []
    client.cluster_info_calls = 0  # Track calls to cluster info

    client.client_setinfo = AsyncMock(return_value=True)
    client.ping = AsyncMock(return_value=True)
    client.echo = AsyncMock(return_value="echo")

    pipeline_instance = AsyncMock()
    pipeline_instance.execute = AsyncMock(return_value=[])
    pipeline_instance.expire = MagicMock(return_value=pipeline_instance)
    pipeline_instance.delete = MagicMock(return_value=pipeline_instance)
    json_commands_mock_cluster = MagicMock()
    json_commands_mock_cluster.get = MagicMock(return_value=json_commands_mock_cluster)
    pipeline_instance.json = MagicMock(return_value=json_commands_mock_cluster)
    client.pipeline = MagicMock(return_value=pipeline_instance)
    client.pipeline_instance = pipeline_instance

    async def expire_side_effect(key, ttl):
        client.expire_calls.append({"key": key, "ttl": ttl})
        return True

    client.expire = AsyncMock(side_effect=expire_side_effect)

    async def delete_side_effect(key):
        client.delete_calls.append({"key": key})
        return True

    client.delete = AsyncMock(side_effect=delete_side_effect)

    async def ttl_side_effect(key):
        client.ttl_calls.append({"key": key})
        return 3600

    client.ttl = AsyncMock(side_effect=ttl_side_effect)

    json_mock = MagicMock()

    async def json_get_side_effect(key):
        client.json_get_calls.append({"key": key})
        await asyncio.sleep(0)
        current_time_cluster = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
        return {
            "key": key.split(":")[-1],
            "value": {"test": "data"},
            "created_at": current_time_cluster,
            "updated_at": current_time_cluster,
        }

    json_mock.get = AsyncMock(side_effect=json_get_side_effect)
    client.json = MagicMock(return_value=json_mock)

    # Mock for cluster detection
    client.cluster = AsyncMock(
        return_value={"cluster_state": "ok"}
    )  # Simulate cluster response

    # Mock aclose and connection_pool.disconnect for context manager exit
    client.aclose = AsyncMock()
    client.connection_pool = AsyncMock()
    client.connection_pool.disconnect = AsyncMock()

    return client


@pytest.fixture
async def async_store(
    request, mock_async_redis_client, mock_async_redis_cluster_client
):
    """Parameterized fixture for AsyncRedisStore with regular or cluster client."""
    is_cluster = request.param
    client = mock_async_redis_cluster_client if is_cluster else mock_async_redis_client

    # Basic IndexConfig, embeddings won't be used in these tests
    index_config: IndexConfig = {
        "embed": MagicMock(),  # Mock embedder
        "dims": 128,
        "distance_type": "cosine",
        "fields": ["content"],
    }

    store = AsyncRedisStore(redis_client=client, index=index_config)

    # Mock the search indices
    store.store_index = AsyncMock()
    store.store_index.create = AsyncMock()
    store.store_index.search = AsyncMock(return_value=MagicMock(docs=[]))
    store.store_index.load = AsyncMock()

    store.vector_index = AsyncMock()
    store.vector_index.create = AsyncMock()
    store.vector_index.query = AsyncMock(
        return_value=[]
    )  # Assuming query returns a list of docs
    store.vector_index.load = AsyncMock()

    await store.setup()  # This is where cluster_mode is set
    yield store
    # Teardown: close store if necessary (AsyncRedisStore has __aexit__)
    await store.__aexit__()


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_cluster_detection(async_store: AsyncRedisStore, request):
    """Test that store.cluster_mode is set correctly."""
    # The `async_store` fixture is parameterized by `is_cluster` (request.param)
    # `async_store.setup()` (called in the fixture) sets `cluster_mode`
    # So, we check if `cluster_mode` matches how the client was mocked for this test run.
    is_client_actually_cluster_mock = isinstance(async_store._redis, AsyncRedisCluster)
    assert (
        async_store.cluster_mode == is_client_actually_cluster_mock
    ), f"Expected cluster_mode {is_client_actually_cluster_mock} but got {async_store.cluster_mode}"


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_apply_ttl_to_keys_behavior(async_store: AsyncRedisStore):
    """Test _apply_ttl_to_keys behavior for cluster vs. non-cluster."""
    client = cast(MockAsyncRedis, async_store._redis)  # Cast for test attribute access
    client.expire_calls.clear()
    if hasattr(client, "pipeline_calls"):  # pipeline_calls is on MockAsyncRedis
        client.pipeline_calls.clear()

    main_key = "main:key"
    related_keys = ["related:key1", "related:key2"]
    ttl_minutes = 10.0

    await async_store._apply_ttl_to_keys(main_key, related_keys, ttl_minutes)

    if async_store.cluster_mode:
        assert (
            len(client.expire_calls) == 3
        ), "Expire should be called for main and related keys"
        assert {"key": main_key, "ttl": int(ttl_minutes * 60)} in client.expire_calls
        assert {
            "key": related_keys[0],
            "ttl": int(ttl_minutes * 60),
        } in client.expire_calls
        assert {
            "key": related_keys[1],
            "ttl": int(ttl_minutes * 60),
        } in client.expire_calls
        if hasattr(
            client, "pipeline_instance"
        ):  # Ensure pipeline was NOT used for this specific operation if possible
            cast(MagicMock, client.pipeline_instance.expire).assert_not_called()
    else:  # Non-cluster
        assert (
            len(client.pipeline_calls) > 0
        ), "Pipeline should be used for non-cluster TTL"
        assert client.pipeline_calls[0]["transaction"] is True
        cast(MagicMock, client.pipeline_instance.expire).assert_any_call(
            main_key, int(ttl_minutes * 60)
        )
        cast(MagicMock, client.pipeline_instance.expire).assert_any_call(
            related_keys[0], int(ttl_minutes * 60)
        )
        cast(MagicMock, client.pipeline_instance.expire).assert_any_call(
            related_keys[1], int(ttl_minutes * 60)
        )
        assert (
            len(client.expire_calls) == 0
        ), "Direct expire should not be called for non-cluster"


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_batch_get_ops_ttl_refresh(async_store: AsyncRedisStore):
    """Test TTL refresh in _batch_get_ops."""
    client = cast(MockAsyncRedis, async_store._redis)  # Cast for test attribute access
    client.expire_calls.clear()
    if hasattr(client, "pipeline_calls"):
        client.pipeline_calls.clear()
    if hasattr(client, "ttl_calls"):
        client.ttl_calls.clear()

    op_idx = 0
    doc_id = str(ULID())
    store_doc_id = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id}"
    vector_doc_id = f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id}"

    # Mock store_index.search to return a document
    mock_doc_data = {
        "key": "test_key",
        "prefix": "test_ns",
        "value": {"data": "content"},
        "id": store_doc_id,  # This is what RedisVL doc.id would be
        "created_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
        "updated_at": int(datetime.now(timezone.utc).timestamp() * 1_000_000),
    }
    mock_redis_doc = MagicMock()
    mock_redis_doc.json = json.dumps(mock_doc_data)
    mock_redis_doc.id = store_doc_id  # Ensure the mock doc has the .id attribute
    async_store.store_index.search = AsyncMock(
        return_value=MagicMock(docs=[mock_redis_doc])
    )

    # Mock client.ttl to control TTL refresh logic
    client.ttl = AsyncMock(return_value=3600)  # Simulate key exists with TTL

    async_store.ttl_config = {"default_ttl": 5.0}
    get_ops = [
        (op_idx, GetOp(namespace=("test_ns",), key="test_key", refresh_ttl=True))
    ]
    results: list[Any] = [None] * 1  # Corrected typing for results list

    await async_store._batch_get_ops(get_ops, results)

    if async_store.cluster_mode:
        assert len(client.expire_calls) >= 1  # At least one for the main store doc
        # We expect expire to be called for store_doc_id and potentially vector_doc_id if it existed
        assert (
            async_store.ttl_config is not None
        )  # Ensure ttl_config is not None for type checker
        assert {
            "key": store_doc_id,
            "ttl": int(cast(dict, async_store.ttl_config).get("default_ttl", 0) * 60),
        } in client.expire_calls
        # Note: vector_doc_id refresh depends on its existence, which is not mocked here, so we check at least store_doc_id
    else:  # Non-cluster
        assert len(client.pipeline_calls) > 0
        assert async_store.ttl_config is not None
        cast(MagicMock, client.pipeline_instance.expire).assert_any_call(
            store_doc_id,
            int(cast(dict, async_store.ttl_config).get("default_ttl", 0) * 60),
        )
        assert len(client.expire_calls) == 0


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_batch_put_ops_pre_delete_behavior(async_store: AsyncRedisStore):
    """Test pre-delete behavior in _batch_put_ops."""
    client = cast(MockAsyncRedis, async_store._redis)  # Cast for test attribute access
    client.delete_calls.clear()
    if hasattr(client, "pipeline_calls"):
        client.pipeline_calls.clear()

    doc_id_to_delete = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{str(ULID())}"
    vector_doc_id_to_delete = f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{str(ULID())}"

    # Mock store_index.search to return a document that needs to be deleted
    mock_store_doc = MagicMock(id=doc_id_to_delete)
    async_store.store_index.search = AsyncMock(
        return_value=MagicMock(docs=[mock_store_doc])
    )

    # Mock vector_index.search if index_config is present
    if async_store.index_config:
        mock_vector_doc = MagicMock(id=vector_doc_id_to_delete)
        async_store.vector_index.search = AsyncMock(
            return_value=MagicMock(docs=[mock_vector_doc])
        )
    else:
        async_store.vector_index.search = AsyncMock(return_value=MagicMock(docs=[]))

    put_ops = [
        (0, PutOp(namespace=("test_ns",), key="test_key", value={"data": "new_val"}))
    ]
    await async_store._batch_put_ops(put_ops)

    if async_store.cluster_mode:
        assert len(client.delete_calls) >= 1  # At least store doc
        assert {"key": doc_id_to_delete} in client.delete_calls
        if async_store.index_config:
            assert {"key": vector_doc_id_to_delete} in client.delete_calls
        if hasattr(client, "pipeline_instance"):
            cast(MagicMock, client.pipeline_instance.delete).assert_not_called()
    else:  # Non-cluster
        assert len(client.pipeline_calls) > 0
        cast(MagicMock, client.pipeline_instance.delete).assert_any_call(
            doc_id_to_delete
        )
        if async_store.index_config:
            cast(MagicMock, client.pipeline_instance.delete).assert_any_call(
                vector_doc_id_to_delete
            )
        assert len(client.delete_calls) == 0


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_batch_search_ops_vector_fetch_behavior(async_store: AsyncRedisStore):
    """Test fetching store docs after vector search in _batch_search_ops."""
    client = cast(MockAsyncRedis, async_store._redis)  # Cast for test attribute access
    client.json_get_calls.clear()
    if hasattr(client, "pipeline_calls"):
        client.pipeline_calls.clear()
    if hasattr(client, "pipeline_instance") and hasattr(
        cast(MagicMock, client.pipeline_instance.json).return_value, "get"
    ):
        cast(
            MagicMock, cast(MagicMock, client.pipeline_instance.json).return_value.get
        ).reset_mock()

    # Mock embeddings and vector_index.query
    if not async_store.index_config:  # This test requires index_config
        pytest.skip("Skipping vector search test as index_config is not set up for it.")

    async_store.embeddings = AsyncMock()
    async_store.embeddings.aembed_documents = AsyncMock(
        return_value=[[0.1, 0.2]]
    )  # Single vector

    mock_vector_doc_id = str(ULID())
    # Simulate redisvl doc structure for vector search results
    mock_vector_result_doc = MagicMock()
    mock_vector_result_doc.id = (
        f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{mock_vector_doc_id}"
    )
    mock_vector_result_doc.vector_distance = 0.5
    mock_vector_result_doc.prefix = "test_ns"  # Needed for filter_expression
    mock_vector_result_doc.key = "test_key"  # Needed for filter_expression

    async_store.vector_index.query = AsyncMock(return_value=[mock_vector_result_doc])

    # Mock client.json().get() to return a store doc
    expected_store_key = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{mock_vector_doc_id}"
    current_timestamp_search = int(datetime.now(timezone.utc).timestamp() * 1_000_000)
    # This mock_store_data will be returned by client.json().get() or pipeline.json().get()
    mock_store_data_search = {
        "prefix": "test_ns",
        "key": "test_key",
        "value": {"content": "data"},
        "created_at": current_timestamp_search,
        "updated_at": current_timestamp_search,
    }

    if async_store.cluster_mode:
        # Ensure the side_effect of client.json().get() returns the comprehensive mock_store_data_search
        async def cluster_json_get_side_effect(key):
            client.json_get_calls.append({"key": key})
            await asyncio.sleep(0)
            return mock_store_data_search

        cast(AsyncMock, client.json().get).side_effect = cluster_json_get_side_effect
    else:
        # For non-cluster, pipeline's json().get() will be called and then execute returns its result
        cast(
            MagicMock, cast(MagicMock, client.pipeline_instance.json).return_value.get
        ).return_value = mock_store_data_search
        client.pipeline_instance.execute = AsyncMock(
            return_value=[mock_store_data_search]
        )

    search_ops = [
        (
            0,
            SearchOp(
                namespace_prefix=("test_ns",), query="some query", limit=1, filter={}
            ),
        )
    ]
    results: list[Any] = [None] * 1  # Corrected typing
    await async_store._batch_search_ops(search_ops, results)

    if async_store.cluster_mode:
        assert len(client.json_get_calls) == 1
        assert {"key": expected_store_key} in client.json_get_calls
        if hasattr(client, "pipeline_instance") and hasattr(
            cast(MagicMock, client.pipeline_instance.json).return_value, "get"
        ):  # type: ignore[attr-defined]
            cast(
                MagicMock,
                cast(MagicMock, client.pipeline_instance.json).return_value.get,
            ).assert_not_called()
    else:  # Non-cluster
        assert len(client.pipeline_calls) > 0
        cast(
            MagicMock, cast(MagicMock, client.pipeline_instance.json).return_value.get
        ).assert_called_once_with(expected_store_key)
        assert (
            len(client.json_get_calls) == 0
        )  # Direct json().get() should not be called


@pytest.mark.parametrize("async_store", [False, True], indirect=True)
async def test_batch_list_namespaces_ops_behavior(async_store: AsyncRedisStore):
    """Test listing namespaces in _batch_list_namespaces_ops."""
    # Mock store_index.search to return documents with different prefixes
    mock_doc1 = MagicMock()
    mock_doc1.prefix = "test.documents.public"

    mock_doc2 = MagicMock()
    mock_doc2.prefix = "test.documents.private"

    mock_doc3 = MagicMock()
    mock_doc3.prefix = "test.images.public"

    mock_doc4 = MagicMock()
    mock_doc4.prefix = "prod.documents.public"

    # Set up the mock search result
    mock_search_result = MagicMock()
    mock_search_result.docs = [mock_doc1, mock_doc2, mock_doc3, mock_doc4]
    async_store.store_index.search = AsyncMock(return_value=mock_search_result)

    # Create list operations
    list_ops = [
        (
            0,
            ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ),
        (1, ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0)),
    ]

    results: list[Any] = [None] * 2
    await async_store._batch_list_namespaces_ops(list_ops, results)

    # Verify results
    # First operation should list all namespaces
    assert len(results[0]) == 4
    assert ("test", "documents", "public") in results[0]
    assert ("test", "documents", "private") in results[0]
    assert ("test", "images", "public") in results[0]
    assert ("prod", "documents", "public") in results[0]

    # Second operation should only return namespaces up to depth 2
    assert len(results[1]) == 3
    assert all(len(ns) <= 2 for ns in results[1])
    assert ("test", "documents") in results[1]
    assert ("test", "images") in results[1]
    assert ("prod", "documents") in results[1]

    # Verify that store_index.search was called with the correct parameters
    # It should be called twice, once for each operation
    assert async_store.store_index.search.call_count == 2

    # The behavior should be the same in both cluster and non-cluster modes
    # since _batch_list_namespaces_ops doesn't have special cluster mode handling
