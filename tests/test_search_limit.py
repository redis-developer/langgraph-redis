"""Tests for RedisStore search limits."""

from __future__ import annotations

import pytest

from langgraph.store.redis import RedisStore


@pytest.fixture(scope="function")
def store(redis_url) -> RedisStore:
    """Fixture to create a Redis store."""
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()  # Initialize indices
        yield store


def test_search_with_larger_limit(store: RedisStore) -> None:
    """Test search with limit > 10."""
    # Create 15 test documents
    for i in range(15):
        store.put(("test_namespace",), f"key{i}", {"data": f"value{i}", "index": i})

    # Search with a limit of 15
    results = store.search(("test_namespace",), limit=15)

    # Should return all 15 results
    assert len(results) == 15, f"Expected 15 results, got {len(results)}"

    # Verify we have all the items
    result_keys = {item.key for item in results}
    expected_keys = {f"key{i}" for i in range(15)}
    assert result_keys == expected_keys


def test_vector_search_with_larger_limit(redis_url) -> None:
    """Test vector search with limit > 10."""
    from tests.embed_test_utils import CharacterEmbeddings

    # Create vector store with embeddings
    embeddings = CharacterEmbeddings(dims=4)
    index_config = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "distance_type": "cosine",
        "fields": ["text"],
    }

    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()

        # Create 15 test documents
        for i in range(15):
            # Create documents with slightly different texts
            store.put(
                ("test_namespace",), f"key{i}", {"text": f"sample text {i}", "index": i}
            )

        # Search with a limit of 15
        results = store.search(("test_namespace",), query="sample", limit=15)

        # Should return all 15 results
        assert len(results) == 15, f"Expected 15 results, got {len(results)}"

        # Verify we have all the items
        result_keys = {item.key for item in results}
        expected_keys = {f"key{i}" for i in range(15)}
        assert result_keys == expected_keys
