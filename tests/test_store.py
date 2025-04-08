"""Tests for RedisStore."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Optional
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from redis import Redis
from redis.exceptions import ResponseError

from langgraph.store.redis import RedisStore
from tests.embed_test_utils import CharacterEmbeddings

TTL_SECONDS = 2
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    """Create test embeddings for vector search."""
    return CharacterEmbeddings(dims=4)


@pytest.fixture(scope="function")
def store(redis_url) -> Iterator[RedisStore]:
    """Fixture to create a Redis store with TTL support."""
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    with RedisStore.from_conn_string(redis_url, ttl=ttl_config) as store:
        store.setup()  # Initialize indices
        store.start_ttl_sweeper()
        yield store
        store.stop_ttl_sweeper()


@pytest.fixture(scope="function", params=["vector", "halfvec"])
def vector_store(
    request, redis_url, fake_embeddings: CharacterEmbeddings
) -> Iterator[RedisStore]:
    """Fixture to create a Redis store with vector search capabilities."""
    vector_type = request.param
    distance_type = "cosine"  # Other options: "l2", "inner_product"

    # Include fields parameter in index_config
    index_config = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "distance_type": distance_type,
        "fields": ["text"],  # Field to embed
    }

    ttl_config = {"default_ttl": 2, "refresh_on_read": True}

    # Create a unique index name for each test run
    unique_id = str(uuid4())[:8]

    # Use different Redis prefix for vector store tests to avoid conflicts
    with RedisStore.from_conn_string(
        redis_url, index=index_config, ttl=ttl_config
    ) as store:
        store.setup()  # Initialize indices
        store.start_ttl_sweeper()
        yield store
        store.stop_ttl_sweeper()


def test_batch_order(store: RedisStore) -> None:
    """Test that operations are executed in the correct order."""
    # Setup test data
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = store.batch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None  # Put operation returns None
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert len(results[3]) > 0  # Should contain at least our test namespaces
    assert results[4] is None  # Non-existent key returns None

    # Test reordered operations
    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = store.batch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) >= 2  # Should find at least our two test items
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert len(results_reordered[2]) > 0
    assert results_reordered[3] is None  # Put operation returns None
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


def test_batch_put_ops(store: RedisStore) -> None:
    """Test batch operations with multiple puts."""
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),  # Delete operation
    ]

    results = store.batch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)

    # Verify the puts worked
    item1 = store.get(("test",), "key1")
    item2 = store.get(("test",), "key2")
    item3 = store.get(("test",), "key3")

    assert item1 and item1.value == {"data": "value1"}
    assert item2 and item2.value == {"data": "value2"}
    assert item3 is None


def test_batch_search_ops(store: RedisStore) -> None:
    """Test batch operations with search operations."""
    # Setup test data
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops = [
        SearchOp(namespace_prefix=("test",), filter={"tag": "a"}, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    # First search should find items with tag "a"
    assert len(results[0]) == 2
    assert all(item.value["tag"] == "a" for item in results[0])

    # Second search should return first 2 items (depends on sorting which could be arbitrary)
    assert len(results[1]) == 2

    # Third search should only find items in test/foo namespace
    assert len(results[2]) == 1
    assert results[2][0].namespace == ("test", "foo")


def test_batch_list_namespaces_ops(store: RedisStore) -> None:
    """Test batch operations with list namespaces operations."""
    # Setup test data with various namespaces
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
        (("prod", "documents", "public"), "doc3", {"content": "prod doc"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops = [
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0),
        ListNamespacesOp(
            match_conditions=[MatchCondition("suffix", "public")],
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    # First operation should list all namespaces
    assert len(results[0]) >= len(test_data)

    # Second operation should only return namespaces up to depth 2
    assert all(len(ns) <= 2 for ns in results[1])

    # Third operation should only return namespaces ending with "public"
    assert all(ns[-1] == "public" for ns in results[2])


def test_list_namespaces(store: RedisStore) -> None:
    """Test listing namespaces with various filters."""
    # Create test data with various namespaces
    test_namespaces = [
        ("test", "documents", "public"),
        ("test", "documents", "private"),
        ("test", "images", "public"),
        ("test", "images", "private"),
        ("prod", "documents", "public"),
        ("prod", "documents", "private"),
    ]

    # Insert test data
    for namespace in test_namespaces:
        store.put(namespace, "dummy", {"content": "dummy"})

    # Test listing with various filters
    all_namespaces = store.list_namespaces()
    assert len(all_namespaces) >= len(test_namespaces)

    # Test prefix filtering
    test_prefix_namespaces = store.list_namespaces(prefix=["test"])
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)

    # Test suffix filtering
    public_namespaces = store.list_namespaces(suffix=["public"])
    assert len(public_namespaces) == 3
    assert all(ns[-1] == "public" for ns in public_namespaces)

    # Test max depth
    depth_2_namespaces = store.list_namespaces(max_depth=2)
    assert all(len(ns) <= 2 for ns in depth_2_namespaces)

    # Test pagination
    paginated_namespaces = store.list_namespaces(limit=3)
    assert len(paginated_namespaces) == 3

    # Cleanup
    for namespace in test_namespaces:
        store.delete(namespace, "dummy")


def test_vector_search(vector_store: RedisStore) -> None:
    """Test vector search functionality."""
    # Insert documents with text that can be embedded
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]

    for key, value in docs:
        vector_store.put(("test",), key, value)

    # Search with query
    results = vector_store.search(("test",), query="longer text")
    assert len(results) >= 2

    # Doc2 and doc3 should be closer matches to "longer text"
    doc_keys = [r.key for r in results]
    assert "doc2" in doc_keys
    assert "doc3" in doc_keys


def test_vector_search_with_filters(vector_store: RedisStore) -> None:
    """Test vector search with additional filters."""
    # Insert test documents
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]

    for key, value in docs:
        vector_store.put(("test",), key, value)

    # Search for "apple" within red items
    results = vector_store.search(("test",), query="apple", filter={"color": "red"})
    assert len(results) >= 1
    # Doc1 should be the closest match for "apple" with color=red
    assert results[0].key == "doc1"

    # Search for "car" within red items
    results = vector_store.search(("test",), query="car", filter={"color": "red"})
    assert len(results) >= 1
    # Doc2 should be the closest match for "car" with color=red
    assert results[0].key == "doc2"


def test_vector_update_with_score_verification(vector_store: RedisStore) -> None:
    """Test that updating items properly updates their embeddings."""
    vector_store.put(("test",), "doc1", {"text": "zany zebra xylophone"})
    vector_store.put(("test",), "doc2", {"text": "something about dogs"})

    # Search for a term similar to doc1's content
    results_initial = vector_store.search(("test",), query="zany xylophone")
    assert len(results_initial) >= 1
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    # Update doc1 to be about dogs instead
    vector_store.put(("test",), "doc1", {"text": "new text about dogs"})

    # The original query should now match doc1 less strongly
    results_after = vector_store.search(("test",), query="zany xylophone")
    assert len(results_after) >= 1
    after_score = next((r.score for r in results_after if r.key == "doc1"), None)
    if after_score is not None:
        assert after_score < initial_score

    # A dog-related query should now match doc1 more strongly
    results_new = vector_store.search(("test",), query="dogs text")
    doc1_score = next((r.score for r in results_new if r.key == "doc1"), None)
    assert doc1_score is not None
    if after_score is not None:
        assert doc1_score > after_score


def test_basic_ops(store: RedisStore) -> None:
    """Test basic CRUD operations."""
    namespace = ("test", "documents")
    item_id = "doc1"
    item_value = {"title": "Test Document", "content": "Hello, World!"}

    store.put(namespace, item_id, item_value)
    item = store.get(namespace, item_id)

    assert item
    assert item.namespace == namespace
    assert item.key == item_id
    assert item.value == item_value

    # Test update
    updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
    store.put(namespace, item_id, updated_value)
    updated_item = store.get(namespace, item_id)

    assert updated_item.value == updated_value
    assert updated_item.updated_at > item.updated_at

    # Test get from non-existent namespace
    different_namespace = ("test", "other_documents")
    item_in_different_namespace = store.get(different_namespace, item_id)
    assert item_in_different_namespace is None

    # Test delete
    store.delete(namespace, item_id)
    deleted_item = store.get(namespace, item_id)
    assert deleted_item is None


def test_search(store: RedisStore) -> None:
    """Test search functionality."""
    # Create test data
    test_data = [
        (
            ("test", "docs"),
            "doc1",
            {"title": "First Doc", "author": "Alice", "tags": ["important"]},
        ),
        (
            ("test", "docs"),
            "doc2",
            {"title": "Second Doc", "author": "Bob", "tags": ["draft"]},
        ),
        (
            ("test", "images"),
            "img1",
            {"title": "Image 1", "author": "Alice", "tags": ["final"]},
        ),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    # Test basic search
    all_items = store.search(["test"])
    assert len(all_items) == 3

    # Test namespace filtering
    docs_items = store.search(["test", "docs"])
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)

    # Test value filtering
    alice_items = store.search(["test"], filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(item.value["author"] == "Alice" for item in alice_items)

    # Test pagination
    paginated_items = store.search(["test"], limit=2)
    assert len(paginated_items) == 2

    offset_items = store.search(["test"], offset=2)
    assert len(offset_items) == 1

    # Cleanup
    for namespace, key, _ in test_data:
        store.delete(namespace, key)


def test_large_batches(store: RedisStore) -> None:
    """Test handling large numbers of operations."""
    # Reduce number of operations for stability
    N = 20
    ops = []

    # Add many put operations
    for i in range(N):
        ops.append(
            PutOp(
                namespace=("test", f"batch{i // 10}"),
                key=f"key{i}",
                value={"data": f"value{i}"},
            )
        )

    # Execute puts first to make sure data exists before querying
    put_results = store.batch(ops)
    assert len(put_results) == N
    assert all(result is None for result in put_results)

    # Create operations for gets, search, and list
    get_ops = []

    # Add get operations
    for i in range(0, N, 5):
        get_ops.append(
            GetOp(
                namespace=("test", f"batch{i // 10}"),
                key=f"key{i}",
            )
        )

    # Add search operations
    for i in range(0, N, 10):
        get_ops.append(
            SearchOp(
                namespace_prefix=("test", f"batch{i // 10}"),
                filter=None,
                limit=5,
                offset=0,
            )
        )

    # Add list namespaces operations
    get_ops.append(
        ListNamespacesOp(match_conditions=None, max_depth=2, limit=20, offset=0)
    )

    # Execute get, search, and list operations
    get_results = store.batch(get_ops)
    expected_results_len = N // 5 + N // 10 + 1
    assert len(get_results) == expected_results_len

    # Verify gets (they should return Items)
    for i in range(N // 5):
        result = get_results[i]
        assert isinstance(result, Item)
        assert result.value["data"] == f"value{i * 5}"

    # Verify searches (they should return lists)
    for i in range(N // 5, N // 5 + N // 10):
        assert isinstance(get_results[i], list)

    # Verify list namespaces (it should return a list)
    assert isinstance(get_results[-1], list)


def test_store_ttl(store: RedisStore) -> None:
    """Test TTL functionality in Redis store."""
    # Assumes a TTL of TTL_MINUTES
    ns = ("foo",)

    # Store an item with TTL
    store.put(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,
    )

    # Check item exists and refresh TTL
    res = store.get(ns, key="item1", refresh_ttl=True)
    assert res is not None

    # Search for the item with refresh
    results = store.search(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1

    # Do one more get without refreshing TTL
    res = store.get(ns, key="item1", refresh_ttl=False)
    assert res is not None

    # Wait for the TTL to expire
    time.sleep(TTL_SECONDS + 0.5)

    # Force a sweep to remove expired items
    store.sweep_ttl()

    # Verify item is gone due to TTL expiration
    res = store.search(ns, query="bar", refresh_ttl=False)
    assert len(res) == 0


def test_redis_store_client_info(redis_url: str, monkeypatch) -> None:
    """Test that RedisStore sets client info correctly."""
    from redis import Redis as NativeRedis
    from langgraph.checkpoint.redis.version import __full_lib_name__
    
    # Create a direct Redis client to bypass RedisVL validation
    client = NativeRedis.from_url(redis_url)
    
    try:
        # Create a mock to track if client_setinfo was called with our library name
        client_info_called = False
        original_client_setinfo = NativeRedis.client_setinfo
        
        def mock_client_setinfo(self, key, value):
            nonlocal client_info_called
            # We only track calls with our full lib name
            if key == "LIB-NAME" and __full_lib_name__ in value:
                client_info_called = True
            return original_client_setinfo(self, key, value)
        
        # Apply the mock
        monkeypatch.setattr(NativeRedis, "client_setinfo", mock_client_setinfo)
        
        # Test client info setting by creating store directly
        store = RedisStore(client)
        store.set_client_info()
        
        # Verify client_setinfo was called with our library info
        assert client_info_called, "client_setinfo was not called with our library name"
    finally:
        client.close()
        client.connection_pool.disconnect()


def test_redis_store_client_info_fallback(redis_url: str, monkeypatch) -> None:
    """Test that RedisStore falls back to echo when client_setinfo is not available."""
    from redis import Redis as NativeRedis
    from langgraph.checkpoint.redis.version import __full_lib_name__
    
    # Create a direct Redis client to bypass RedisVL validation
    client = NativeRedis.from_url(redis_url)
    
    try:
        # Track if echo was called
        echo_called = False
        original_echo = NativeRedis.echo
        
        # Remove client_setinfo to simulate older Redis version
        def mock_client_setinfo(self, key, value):
            raise ResponseError("ERR unknown command")
        
        def mock_echo(self, message):
            nonlocal echo_called
            # We only want to track our library's echo calls
            if __full_lib_name__ in message:
                echo_called = True
            return original_echo(self, message)
        
        # Apply the mocks
        monkeypatch.setattr(NativeRedis, "client_setinfo", mock_client_setinfo)
        monkeypatch.setattr(NativeRedis, "echo", mock_echo)
        
        # Test client info setting by creating store directly
        store = RedisStore(client)
        store.set_client_info()
        
        # Verify echo was called as fallback
        assert echo_called, "echo was not called as fallback when client_setinfo failed"
    finally:
        client.close()
        client.connection_pool.disconnect()


def test_redis_store_graceful_failure(redis_url: str, monkeypatch) -> None:
    """Test graceful failure when both client_setinfo and echo fail."""
    from redis import Redis as NativeRedis
    from redis.exceptions import ResponseError
    
    # Create a direct Redis client to bypass RedisVL validation
    client = NativeRedis.from_url(redis_url)
    
    try:
        # Simulate failures for both methods
        def mock_client_setinfo(self, key, value):
            raise ResponseError("ERR unknown command")
        
        def mock_echo(self, message):
            raise ResponseError("ERR broken connection")
        
        # Apply the mocks
        monkeypatch.setattr(NativeRedis, "client_setinfo", mock_client_setinfo)
        monkeypatch.setattr(NativeRedis, "echo", mock_echo)
        
        # Should not raise any exceptions when both methods fail
        try:
            # Test client info setting by creating store directly
            store = RedisStore(client)
            store.set_client_info()
        except Exception as e:
            assert False, f"set_client_info did not handle failure gracefully: {e}"
    finally:
        client.close()
        client.connection_pool.disconnect()
