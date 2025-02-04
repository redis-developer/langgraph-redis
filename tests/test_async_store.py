"""Tests for AsyncRedisStore."""

from typing import AsyncGenerator, Sequence, cast

import pytest
from redis.asyncio import Redis

from langgraph.store.base import (
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
)
from langgraph.store.redis import AsyncRedisStore
from tests.conftest import DEFAULT_REDIS_URI, VECTOR_TYPES
from tests.embed_test_utils import AsyncCharacterEmbeddings


@pytest.fixture(autouse=True)
async def clear_test_redis() -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(DEFAULT_REDIS_URI)
    try:
        await client.flushall()
    finally:
        await client.aclose()  # type: ignore[attr-defined]
        await client.connection_pool.disconnect()


@pytest.fixture
async def store() -> AsyncGenerator[AsyncRedisStore, None]:
    """Fixture providing configured AsyncRedisStore.

    Uses proper async cleanup and connection handling.
    """
    store = None
    try:
        async with AsyncRedisStore.from_conn_string(DEFAULT_REDIS_URI) as astore:
            await astore.setup()
            store = astore
            yield store
    finally:
        if store:
            if store._owns_client:
                await store._redis.aclose()  # type: ignore[attr-defined]
                await store._redis.connection_pool.disconnect()


@pytest.fixture
def fake_embeddings() -> AsyncCharacterEmbeddings:
    """Provide a simple embeddings implementation for testing."""
    return AsyncCharacterEmbeddings(dims=4)


@pytest.mark.asyncio
async def test_basic_ops(store: AsyncRedisStore) -> None:
    """Test basic store operations: put, get, delete with namespace handling."""

    # Test basic put and get
    await store.aput(("test",), "key1", {"data": "value1"})
    item = await store.aget(("test",), "key1")
    assert item is not None
    assert item.value["data"] == "value1"

    # Test update
    await store.aput(("test",), "key1", {"data": "updated"})
    updated = await store.aget(("test",), "key1")
    assert updated is not None
    assert updated.value["data"] == "updated"
    assert updated.updated_at > item.updated_at

    # Test delete
    await store.adelete(("test",), "key1")
    deleted = await store.aget(("test",), "key1")
    assert deleted is None

    # Test namespace isolation
    await store.aput(("test", "ns1"), "key1", {"data": "ns1"})
    await store.aput(("test", "ns2"), "key1", {"data": "ns2"})

    ns1_item = await store.aget(("test", "ns1"), "key1")
    ns2_item = await store.aget(("test", "ns2"), "key1")
    assert ns1_item is not None
    assert ns2_item is not None
    assert ns1_item.value["data"] == "ns1"
    assert ns2_item.value["data"] == "ns2"


@pytest.mark.asyncio
async def test_search(store: AsyncRedisStore) -> None:
    """Test search operations using async store."""

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

    # Store test data
    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    # Test basic search
    all_items = await store.asearch(tuple(["test"]))
    assert len(all_items) == 3

    # Test namespace filtering
    docs_items = await store.asearch(tuple(["test", "docs"]))
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)

    # Test value filtering
    alice_items = await store.asearch(tuple(["test"]), filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(item.value["author"] == "Alice" for item in alice_items)

    # Test pagination
    paginated_items = await store.asearch(tuple(["test"]), limit=2)
    assert len(paginated_items) == 2

    offset_items = await store.asearch(tuple(["test"]), offset=2)
    assert len(offset_items) == 1

    # Cleanup
    for namespace, key, _ in test_data:
        await store.adelete(namespace, key)


@pytest.mark.asyncio
async def test_batch_put_ops(store: AsyncRedisStore) -> None:
    """Test batch PUT operations with async store."""
    ops: list[Op] = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = await store.abatch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)

    search_results = await store.asearch(("test",), limit=10)
    assert len(search_results) == 2


@pytest.mark.asyncio
async def test_batch_search_ops(store: AsyncRedisStore) -> None:
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    ops: list[Op] = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]

    results = await store.abatch(ops)
    assert len(results) == 3

    if isinstance(results[0], list):
        assert len(results[0]) >= 2  # Should find all test documents
    else:
        raise AssertionError(
            "Expected results[0] to be a list, got None or incompatible type"
        )

    if isinstance(results[1], list):
        assert len(results[1]) == 2  # Limited to 2 results
    else:
        raise AssertionError(
            "Expected results[1] to be a list, got None or incompatible type"
        )

    if isinstance(results[2], list):
        assert len(results[2]) == 1  # Only foo namespace
    else:
        raise AssertionError(
            "Expected results[2] to be a list, got None or incompatible type"
        )


@pytest.mark.asyncio
async def test_batch_list_namespaces_ops(store: AsyncRedisStore) -> None:
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
    ]
    for namespace, key, value in test_data:
        await store.aput(namespace, key, value)

    ops: list[Op] = [
        ListNamespacesOp(match_conditions=(), max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=(), max_depth=2, limit=10, offset=0),
        ListNamespacesOp(
            match_conditions=(MatchCondition("suffix", ("public",)),),
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = await store.abatch(ops)

    namespaces = cast(list[tuple[str, ...]], results[0])
    assert len(namespaces) == len(test_data)

    namespaces_depth = cast(list[tuple[str, ...]], results[1])
    assert all(len(ns) <= 2 for ns in namespaces_depth)

    namespaces_public = cast(list[tuple[str, ...]], results[2])
    assert all(ns[-1] == "public" for ns in namespaces_public)


@pytest.mark.asyncio
async def test_list_namespaces(store: AsyncRedisStore) -> None:
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
    assert len(all_namespaces) == len(test_namespaces)

    # Test prefix filtering
    test_prefix_namespaces = await store.alist_namespaces(prefix=tuple(["test"]))
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)

    # Test suffix filtering
    public_namespaces = await store.alist_namespaces(suffix=tuple(["public"]))
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
    await store.aput(("test", "foo"), "key1", {"data": "value1"})
    await store.aput(("test", "bar"), "key2", {"data": "value2"})

    ops: list[Op] = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=(), max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = await store.abatch(ops)
    assert len(results) == 5

    item = cast(Item, results[0])
    assert isinstance(item, Item)
    assert item.value == {"data": "value1"}
    assert item.key == "key1"

    assert results[1] is None

    search_results = cast(Sequence[SearchItem], results[2])
    assert len(search_results) == 1
    assert search_results[0].value == {"data": "value1"}

    namespaces = cast(list[tuple[str, ...]], results[3])
    assert len(namespaces) >= 2
    assert ("test", "foo") in namespaces
    assert ("test", "bar") in namespaces

    assert results[4] is None


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [*[(vt, dt) for vt in VECTOR_TYPES for dt in ["cosine", "inner_product", "l2"]]],
)
@pytest.mark.asyncio
async def test_vector_search(
    fake_embeddings: AsyncCharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    index_config: IndexConfig = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "text_fields": ["text"],
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
    }

    async with AsyncRedisStore.from_conn_string(
        DEFAULT_REDIS_URI, index=index_config
    ) as store:
        await store.setup()

        docs = [
            ("doc1", {"text": "short text"}),
            ("doc2", {"text": "longer text document"}),
            ("doc3", {"text": "longest text document here"}),
        ]

        for key, value in docs:
            await store.aput(("test",), key, value)

        results = await store.asearch(("test",), query="long text")
        assert len(results) > 0

        doc_order = [r.key for r in results]
        assert "doc2" in doc_order
        assert "doc3" in doc_order

        results = await store.asearch(("test",), query="short text")
        assert len(results) > 0
        assert results[0].key == "doc1"


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [*[(vt, dt) for vt in VECTOR_TYPES for dt in ["cosine", "inner_product", "l2"]]],
)
@pytest.mark.asyncio
async def test_vector_update_with_score_verification(
    fake_embeddings: AsyncCharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """Test that updating items properly updates their embeddings and scores."""
    index_config: IndexConfig = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "text_fields": ["text"],
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
    }

    async with AsyncRedisStore.from_conn_string(
        DEFAULT_REDIS_URI, index=index_config
    ) as store:
        await store.setup()

        # Add initial documents
        await store.aput(("test",), "doc1", {"text": "zany zebra Xerxes"})
        await store.aput(("test",), "doc2", {"text": "something about dogs"})
        await store.aput(("test",), "doc3", {"text": "text about birds"})

        # Search for zebra content and verify initial scores
        results_initial = await store.asearch(("test",), query="Zany Xerxes")
        assert len(results_initial) > 0
        assert results_initial[0].key == "doc1"
        assert results_initial[0].score is not None
        initial_score = results_initial[0].score

        # Update doc1 to be about dogs instead of zebras
        await store.aput(("test",), "doc1", {"text": "new text about dogs"})

        # After updating content to be about dogs instead of zebras,
        # searching for the original zebra content should give a much lower score
        results_after = await store.asearch(("test",), query="Zany Xerxes")
        # The doc may not even be in top results anymore since content changed
        after_doc = next((r for r in results_after if r.key == "doc1"), None)
        assert after_doc is None or (
            after_doc.score is not None and after_doc.score < initial_score
        )

        # When searching for dog content, doc1 should now score highly
        results_new = await store.asearch(("test",), query="new text about dogs")
        doc1_new = next((r for r in results_new if r.key == "doc1"), None)
        assert doc1_new is not None and doc1_new.score is not None
        if after_doc is not None and after_doc.score is not None:
            assert doc1_new.score > after_doc.score

        # Don't index this one
        await store.aput(
            ("test",), "doc4", {"text": "new text about dogs"}, index=False
        )
        results_new = await store.asearch(
            ("test",), query="new text about dogs", limit=3
        )
        assert not any(r.key == "doc4" for r in results_new)


@pytest.mark.asyncio
async def test_large_batches(store: AsyncRedisStore) -> None:
    N = 100  # less important that we are performant here
    M = 10

    for m in range(M):
        for i in range(N):
            # First put operation
            await store.aput(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
                value={"foo": "bar" + str(i)},
            )

            # Get operation
            await store.aget(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
            )

            # List namespaces operation
            await store.alist_namespaces(
                prefix=None,
                max_depth=m + 1,
            )

            # Search operation
            await store.asearch(
                ("test",),
            )

            # Second put operation
            await store.aput(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
                value={"foo": "bar" + str(i)},
            )

            # Delete operation
            await store.adelete(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
            )
