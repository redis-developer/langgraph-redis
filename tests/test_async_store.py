"""Tests for AsyncRedisStore."""

import asyncio
import time
from typing import AsyncIterator
from uuid import uuid4

import pytest
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)

from langgraph.store.redis import AsyncRedisStore
from tests.embed_test_utils import CharacterEmbeddings

TTL_SECONDS = 2
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    """Create test embeddings for vector search."""
    return CharacterEmbeddings(dims=4)


@pytest.fixture(scope="function")
async def store(redis_url) -> AsyncIterator[AsyncRedisStore]:
    """Create an async Redis store with TTL enabled."""
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    async with AsyncRedisStore.from_conn_string(redis_url, ttl=ttl_config) as store:
        await store.setup()  # Initialize indices
        await store.start_ttl_sweeper()
        yield store
        await store.stop_ttl_sweeper()


@pytest.fixture(scope="function", params=["vector", "halfvec"])
async def vector_store(
    request, redis_url, fake_embeddings: CharacterEmbeddings
) -> AsyncIterator[AsyncRedisStore]:
    """Create an async Redis store with vector search capabilities."""
    vector_type = request.param
    distance_type = "cosine"

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
    async with AsyncRedisStore.from_conn_string(
        redis_url, index=index_config, ttl=ttl_config
    ) as store:
        await store.setup()  # Initialize indices
        await store.start_ttl_sweeper()
        yield store
        await store.stop_ttl_sweeper()


@pytest.mark.asyncio
async def test_basic_ops(store: AsyncRedisStore) -> None:
    """Test basic CRUD operations with async store."""
    namespace = ("test", "documents")
    item_id = "doc1"
    item_value = {"title": "Test Document", "content": "Hello, World!"}

    await store.aput(namespace, item_id, item_value)
    item = await store.aget(namespace, item_id)

    assert item
    assert item.namespace == namespace
    assert item.key == item_id
    assert item.value == item_value

    # Test update
    updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
    await store.aput(namespace, item_id, updated_value)
    updated_item = await store.aget(namespace, item_id)

    assert updated_item.value == updated_value
    assert updated_item.updated_at > item.updated_at

    # Test non-existent namespace
    different_namespace = ("test", "other_documents")
    item_in_different_namespace = await store.aget(different_namespace, item_id)
    assert item_in_different_namespace is None

    # Test delete
    await store.adelete(namespace, item_id)
    deleted_item = await store.aget(namespace, item_id)
    assert deleted_item is None


@pytest.mark.asyncio
async def test_search(store: AsyncRedisStore) -> None:
    """Test search functionality with async store."""
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
        await store.aput(namespace, key, value)

    # Test basic search
    all_items = await store.asearch(["test"])
    assert len(all_items) == 3

    # Test namespace filtering
    docs_items = await store.asearch(["test", "docs"])
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)

    # Test value filtering
    alice_items = await store.asearch(["test"], filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(item.value["author"] == "Alice" for item in alice_items)

    # Test pagination
    paginated_items = await store.asearch(["test"], limit=2)
    assert len(paginated_items) == 2

    offset_items = await store.asearch(["test"], offset=2)
    assert len(offset_items) == 1

    # Cleanup
    for namespace, key, _ in test_data:
        await store.adelete(namespace, key)


@pytest.mark.asyncio
async def test_batch_put_ops(store: AsyncRedisStore) -> None:
    """Test batch put operations with async store."""
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),  # Delete operation
    ]

    results = await store.abatch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)

    # Verify the puts worked
    item1 = await store.aget(("test",), "key1")
    item2 = await store.aget(("test",), "key2")
    item3 = await store.aget(("test",), "key3")

    assert item1 and item1.value == {"data": "value1"}
    assert item2 and item2.value == {"data": "value2"}
    assert item3 is None


@pytest.mark.asyncio
async def test_batch_search_ops(store: AsyncRedisStore) -> None:
    """Test batch search operations with async store."""
    # Setup test data
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    ops = [
        SearchOp(namespace_prefix=("test",), filter={"tag": "a"}, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]

    results = await store.abatch(ops)
    assert len(results) == 3

    # First search should find items with tag "a"
    assert len(results[0]) == 2
    assert all(item.value["tag"] == "a" for item in results[0])

    # Second search should return first 2 items
    assert len(results[1]) == 2

    # Third search should only find items in test/foo namespace
    assert len(results[2]) == 1
    assert results[2][0].namespace == ("test", "foo")


@pytest.mark.asyncio
async def test_batch_list_namespaces_ops(store: AsyncRedisStore) -> None:
    """Test batch list namespaces operations with async store."""
    # Setup test data with various namespaces
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
        (("prod", "documents", "public"), "doc3", {"content": "prod doc"}),
    ]
    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    ops = [
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0),
    ]

    results = await store.abatch(ops)
    assert len(results) == 2

    # First operation should list all namespaces
    assert len(results[0]) >= len(test_data)

    # Second operation should only return namespaces up to depth 2
    assert all(len(ns) <= 2 for ns in results[1])


@pytest.mark.asyncio
async def test_list_namespaces(store: AsyncRedisStore) -> None:
    """Test listing namespaces with async store."""
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
        await store.aput(namespace, "dummy", {"content": "dummy"})

    # Test listing with various filters
    all_namespaces = await store.alist_namespaces()
    assert len(all_namespaces) >= len(test_namespaces)

    # Test prefix filtering
    test_prefix_namespaces = await store.alist_namespaces(prefix=["test"])
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)

    # Test suffix filtering
    public_namespaces = await store.alist_namespaces(suffix=["public"])
    assert len(public_namespaces) == 3
    assert all(ns[-1] == "public" for ns in public_namespaces)

    # Test max depth
    depth_2_namespaces = await store.alist_namespaces(max_depth=2)
    assert all(len(ns) <= 2 for ns in depth_2_namespaces)

    # Test pagination
    paginated_namespaces = await store.alist_namespaces(limit=3)
    assert len(paginated_namespaces) == 3

    # Cleanup
    for namespace in test_namespaces:
        await store.adelete(namespace, "dummy")


@pytest.mark.asyncio
async def test_batch_order(store: AsyncRedisStore) -> None:
    """Test batch operations with async store.

    This test focuses on verifying that multiple operations can be executed
    successfully in a batch, rather than testing strict sequential ordering.
    """
    namespace = ("test", "batch")

    # First, put multiple items in a batch
    put_ops = [
        PutOp(namespace=namespace, key=f"key{i}", value={"data": f"value{i}"})
        for i in range(5)
    ]

    # Execute the batch of puts
    put_results = await store.abatch(put_ops)
    assert len(put_results) == 5
    assert all(result is None for result in put_results)

    # Then get multiple items in a batch
    get_ops = [GetOp(namespace=namespace, key=f"key{i}") for i in range(5)]

    # Execute the batch of gets
    get_results = await store.abatch(get_ops)
    assert len(get_results) == 5

    # Verify all items were retrieved correctly
    for i, result in enumerate(get_results):
        assert isinstance(result, Item)
        assert result.key == f"key{i}"
        assert result.value == {"data": f"value{i}"}

    # Create additional items individually
    namespace2 = ("test", "batch_mixed")
    await store.aput(namespace2, "item1", {"category": "fruit", "name": "apple"})
    await store.aput(namespace2, "item2", {"category": "fruit", "name": "banana"})
    await store.aput(namespace2, "item3", {"category": "vegetable", "name": "carrot"})

    # Now search for items in a separate operation
    fruit_items = await store.asearch(namespace2, filter={"category": "fruit"})
    assert isinstance(fruit_items, list)
    assert len(fruit_items) == 2
    assert all(item.value["category"] == "fruit" for item in fruit_items)

    # Cleanup - delete all the items we created
    for i in range(5):
        await store.adelete(namespace, f"key{i}")
    await store.adelete(namespace2, "item1")
    await store.adelete(namespace2, "item2")
    await store.adelete(namespace2, "item3")


@pytest.mark.asyncio
async def test_vector_search(vector_store: AsyncRedisStore) -> None:
    """Test vector search functionality with async store."""
    # Insert documents with text that can be embedded
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
    ]

    for key, value in docs:
        await vector_store.aput(("test",), key, value)

    # Search with query
    results = await vector_store.asearch(("test",), query="longer text")
    assert len(results) >= 2

    # Doc2 and doc3 should be closer matches to "longer text"
    doc_keys = [r.key for r in results]
    assert "doc2" in doc_keys
    assert "doc3" in doc_keys


@pytest.mark.asyncio
async def test_vector_update_with_score_verification(
    vector_store: AsyncRedisStore,
) -> None:
    """Test that updating items properly updates their embeddings with async store."""
    await vector_store.aput(("test",), "doc1", {"text": "zany zebra xylophone"})
    await vector_store.aput(("test",), "doc2", {"text": "something about dogs"})

    # Search for a term similar to doc1's content
    results_initial = await vector_store.asearch(("test",), query="zany xylophone")
    assert len(results_initial) >= 1
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    # Update doc1 to be about dogs instead
    await vector_store.aput(("test",), "doc1", {"text": "new text about dogs"})

    # The original query should now match doc1 less strongly
    results_after = await vector_store.asearch(("test",), query="zany xylophone")
    assert len(results_after) >= 1
    after_score = next((r.score for r in results_after if r.key == "doc1"), None)
    if after_score is not None:
        assert after_score < initial_score

    # A dog-related query should now match doc1 more strongly
    results_new = await vector_store.asearch(("test",), query="dogs text")
    doc1_score = next((r.score for r in results_new if r.key == "doc1"), None)
    assert doc1_score is not None
    if after_score is not None:
        assert doc1_score > after_score


@pytest.mark.asyncio
async def test_large_batches(store: AsyncRedisStore) -> None:
    """Test large batch operations with async store."""
    # Reduce number of operations for stability
    N = 20  # Smaller number for async test to avoid timeouts
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
    put_results = await store.abatch(ops)
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
    get_results = await store.abatch(get_ops)
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


@pytest.mark.asyncio
async def test_store_ttl(store: AsyncRedisStore) -> None:
    """Test TTL functionality in async Redis store."""
    # Assumes a TTL of TTL_MINUTES
    ns = ("foo",)

    # Store an item with TTL
    await store.aput(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,
    )

    # Check item exists and refresh TTL
    res = await store.aget(ns, key="item1", refresh_ttl=True)
    assert res is not None

    # Search for the item with refresh
    results = await store.asearch(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1

    # Do one more get without refreshing TTL
    res = await store.aget(ns, key="item1", refresh_ttl=False)
    assert res is not None

    # Wait for the TTL to expire
    await asyncio.sleep(TTL_SECONDS + 0.5)

    # Force a sweep to remove expired items
    await store.sweep_ttl()

    # Verify item is gone due to TTL expiration
    res = await store.asearch(ns, query="bar", refresh_ttl=False)
    assert len(res) == 0


@pytest.mark.asyncio
async def test_async_store_with_memory_persistence(redis_url: str) -> None:
    """Test basic persistence operations with Redis.

    This test verifies that data persists in Redis after
    creating a new store connection.
    """
    # Create a unique namespace for this test
    namespace = ("test", "persistence", str(uuid4()))
    key = "persisted_item"
    value = {"data": "persist_me", "timestamp": time.time()}

    # First store instance - write data
    async with AsyncRedisStore.from_conn_string(redis_url) as store1:
        await store1.setup()
        await store1.aput(namespace, key, value)

        # Verify the data was written
        item = await store1.aget(namespace, key)
        assert item is not None
        # Use approximate comparison for floating point values
        assert item.value["data"] == value["data"]
        assert abs(item.value["timestamp"] - value["timestamp"]) < 0.001

    # Second store instance - verify data persisted
    async with AsyncRedisStore.from_conn_string(redis_url) as store2:
        await store2.setup()

        # Read the item with the new store instance
        persisted_item = await store2.aget(namespace, key)
        assert persisted_item is not None
        # Use approximate comparison for floating point values
        assert persisted_item.value["data"] == value["data"]
        assert abs(persisted_item.value["timestamp"] - value["timestamp"]) < 0.001

        # Cleanup
        await store2.adelete(namespace, key)


@pytest.mark.asyncio
async def test_async_redis_store_client_info(redis_url: str, monkeypatch) -> None:
    """Test that AsyncRedisStore sets client info correctly."""
    from redis.asyncio import Redis

    from langgraph.checkpoint.redis.version import __full_lib_name__

    # Track if client_setinfo was called with the right parameters
    client_info_called = False

    # Store the original method
    original_client_setinfo = Redis.client_setinfo

    # Create a mock function for client_setinfo
    async def mock_client_setinfo(self, key, value):
        nonlocal client_info_called
        # Track calls with our full lib name
        if key == "LIB-NAME" and value == __full_lib_name__:
            client_info_called = True
        # Call original method to ensure normal function
        return await original_client_setinfo(self, key, value)

    # Apply the mock
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)

    # Test client info setting when creating a new async store
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()

    # Verify client_setinfo was called with our library info
    assert client_info_called, "client_setinfo was not called with our library name"


@pytest.mark.asyncio
async def test_async_redis_store_client_info_fallback(
    redis_url: str, monkeypatch
) -> None:
    """Test that AsyncRedisStore falls back to echo when client_setinfo is not available."""
    from redis.asyncio import Redis
    from redis.exceptions import ResponseError

    from langgraph.checkpoint.redis.version import __full_lib_name__

    # Remove client_setinfo to simulate older Redis version
    async def mock_client_setinfo(self, key, value):
        raise ResponseError("ERR unknown command")

    # Track if echo was called as fallback
    echo_called = False
    original_echo = Redis.echo

    # Create mock for echo
    async def mock_echo(self, message):
        nonlocal echo_called
        echo_called = True
        assert message == __full_lib_name__
        return await original_echo(self, message)

    # Apply the mocks
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    monkeypatch.setattr(Redis, "echo", mock_echo)

    # Test client info setting with fallback
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()

    # Verify echo was called as fallback
    assert (
        echo_called
    ), "echo was not called as fallback when client_setinfo failed in AsyncRedisStore"


@pytest.mark.asyncio
async def test_async_redis_store_graceful_failure(redis_url: str, monkeypatch) -> None:
    """Test that async store client info setting fails gracefully when all methods fail."""
    from redis.asyncio import Redis
    from redis.exceptions import ResponseError

    # Create a patch for the RedisVL validation to avoid it using echo
    from redisvl.redis.connection import RedisConnectionFactory

    original_validate = RedisConnectionFactory.validate_async_redis

    # Create a replacement validation function that doesn't use echo
    async def mock_validate(redis_client, lib_name=None):
        return redis_client

    # Apply the validation mock first to prevent echo from being called by RedisVL
    monkeypatch.setattr(RedisConnectionFactory, "validate_async_redis", mock_validate)

    # Simulate failures for both methods
    async def mock_client_setinfo(self, key, value):
        raise ResponseError("ERR unknown command")

    async def mock_echo(self, message):
        raise ResponseError("ERR connection broken")

    # Apply the Redis mocks
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    monkeypatch.setattr(Redis, "echo", mock_echo)

    # Should not raise any exceptions when both methods fail
    try:
        async with AsyncRedisStore.from_conn_string(redis_url) as store:
            await store.setup()
    except Exception as e:
        assert False, f"aset_client_info did not handle failure gracefully: {e}"
