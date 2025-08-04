"""Tests for AsyncRedisStore search limits."""

from __future__ import annotations

import pytest
import pytest_asyncio

from langgraph.store.redis import AsyncRedisStore


@pytest_asyncio.fixture(scope="function")
async def async_store(redis_url) -> AsyncRedisStore:
    """Fixture to create an AsyncRedisStore."""
    async with AsyncRedisStore(redis_url) as store:
        await store.setup()  # Initialize indices
        yield store


@pytest.mark.asyncio
async def test_async_search_with_larger_limit(async_store: AsyncRedisStore) -> None:
    """Test async search with limit > 10."""
    # Create 15 test documents
    for i in range(15):
        await async_store.aput(
            ("test_namespace",), f"key{i}", {"data": f"value{i}", "index": i}
        )

    # Search with a limit of 15
    results = await async_store.asearch(("test_namespace",), limit=15)

    # Should return all 15 results
    assert len(results) == 15, f"Expected 15 results, got {len(results)}"

    # Verify we have all the items
    result_keys = {item.key for item in results}
    expected_keys = {f"key{i}" for i in range(15)}
    assert result_keys == expected_keys


@pytest.mark.asyncio
async def test_async_vector_search_with_larger_limit(redis_url) -> None:
    """Test async vector search with limit > 10."""
    from tests.embed_test_utils import CharacterEmbeddings

    # Create vector store with embeddings
    embeddings = CharacterEmbeddings(dims=4)
    index_config = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "distance_type": "cosine",
        "fields": ["text"],
    }

    async with AsyncRedisStore(redis_url, index=index_config) as store:
        await store.setup()

        # Verify indices exist, retry if needed (handles race with flushall)
        retries = 3
        for i in range(retries):
            try:
                # Try to access the index to verify it exists
                await store._redis.ft(store.store_index.schema.index.name).info()
                if hasattr(store, "vector_index") and store.vector_index:
                    await store._redis.ft(store.vector_index.schema.index.name).info()
                break
            except Exception:
                if i < retries - 1:
                    import asyncio

                    await asyncio.sleep(0.1)
                    await store.setup()  # Re-setup if index was deleted
                else:
                    raise

        # Create 15 test documents
        for i in range(15):
            # Create documents with slightly different texts
            await store.aput(
                ("test_namespace",), f"key{i}", {"text": f"sample text {i}", "index": i}
            )

        # Search with a limit of 15
        results = await store.asearch(("test_namespace",), query="sample", limit=15)

        # Should return all 15 results
        assert len(results) == 15, f"Expected 15 results, got {len(results)}"

        # Verify we have all the items
        result_keys = {item.key for item in results}
        expected_keys = {f"key{i}" for i in range(15)}
        assert result_keys == expected_keys
