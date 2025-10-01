"""Tests for issue #96: AttributeError with AsyncRedisStore in cluster mode.

Issue: https://github.com/redis-developer/langgraph-redis/issues/96

The issue was caused by redisvl's get_protocol_version() accessing nodes_manager
attribute on ClusterPipeline objects which don't always have this attribute.

Fixed in redisvl 0.9.0 via issue #365.

This test suite focuses on AsyncRedisStore as that was where the issue was
originally reported. The issue occurred in redisvl's SearchIndex.load() method
which is used by both sync and async stores, so testing async coverage is
sufficient.
"""

import pytest

from langgraph.store.redis import AsyncRedisStore


@pytest.mark.asyncio
async def test_async_store_batch_put_no_attribute_error(redis_url: str) -> None:
    """Test that AsyncRedisStore batch put operations don't raise AttributeError.

    This is the primary test for issue #96 which was originally reported
    with AsyncRedisStore.
    """
    store = AsyncRedisStore(redis_url, cluster_mode=False)
    await store.setup()

    try:
        namespace = ("test", "issue_96_async")

        # Put multiple items to trigger batch operations
        items = [
            (f"async_item_{i}", {"data": f"async_value_{i}", "index": i})
            for i in range(10)
        ]

        for key, value in items:
            await store.aput(namespace, key, value)

        # Verify items were stored correctly
        retrieved = await store.aget(namespace, "async_item_0")
        assert retrieved is not None
        assert retrieved.value["data"] == "async_value_0"

    finally:
        # Cleanup
        for key, _ in items:
            await store.adelete(namespace, key)
        await store._redis.aclose()


@pytest.mark.asyncio
async def test_async_store_cluster_mode_batch_put(redis_url: str) -> None:
    """Test AsyncRedisStore with cluster_mode=True for batch operations.

    This is the exact scenario from issue #96 - using AsyncRedisStore with
    cluster mode enabled, which should trigger the code path that was causing
    the AttributeError about nodes_manager.
    """
    store = AsyncRedisStore(redis_url, cluster_mode=True)
    await store.setup()

    try:
        namespace = ("test", "issue_96_async_cluster")

        # Put multiple items to trigger batch operations
        items = [
            (
                f"async_cluster_item_{i}",
                {"data": f"async_cluster_value_{i}", "index": i},
            )
            for i in range(10)
        ]

        # This was raising AttributeError: 'ClusterPipeline' object has no attribute 'nodes_manager'
        # Should work now with redisvl 0.9.0
        for key, value in items:
            await store.aput(namespace, key, value)

        # Verify items were stored
        retrieved = await store.aget(namespace, "async_cluster_item_0")
        assert retrieved is not None
        assert retrieved.value["data"] == "async_cluster_value_0"

    finally:
        # Cleanup
        for key, _ in items:
            await store.adelete(namespace, key)
        await store._redis.aclose()


@pytest.mark.asyncio
async def test_async_store_large_batch_cluster_mode(redis_url: str) -> None:
    """Test AsyncRedisStore with larger batch to stress test the fix.

    This ensures the fix works with more substantial batch operations.
    """
    store = AsyncRedisStore(redis_url, cluster_mode=True)
    await store.setup()

    try:
        namespace = ("test", "issue_96_large_batch")

        # Put a larger batch of items
        items = [
            (f"large_batch_item_{i}", {"data": f"large_batch_value_{i}", "index": i})
            for i in range(50)
        ]

        # This should handle larger batches without AttributeError
        for key, value in items:
            await store.aput(namespace, key, value)

        # Verify some items were stored
        retrieved_first = await store.aget(namespace, "large_batch_item_0")
        assert retrieved_first is not None
        assert retrieved_first.value["index"] == 0

        retrieved_last = await store.aget(namespace, "large_batch_item_49")
        assert retrieved_last is not None
        assert retrieved_last.value["index"] == 49

    finally:
        # Cleanup
        for key, _ in items:
            await store.adelete(namespace, key)
        await store._redis.aclose()


@pytest.mark.asyncio
async def test_async_store_update_operations_cluster_mode(redis_url: str) -> None:
    """Test AsyncRedisStore update operations in cluster mode.

    Updates trigger both delete and insert operations in batch_put_ops,
    exercising the code path that was problematic in issue #96.
    """
    store = AsyncRedisStore(redis_url, cluster_mode=True)
    await store.setup()

    try:
        namespace = ("test", "issue_96_updates")

        # Initial put
        await store.aput(namespace, "update_test", {"version": 1, "data": "initial"})

        # Update the same key - this triggers delete + insert in batch_put_ops
        await store.aput(namespace, "update_test", {"version": 2, "data": "updated"})

        # Verify the update worked
        retrieved = await store.aget(namespace, "update_test")
        assert retrieved is not None
        assert retrieved.value["version"] == 2
        assert retrieved.value["data"] == "updated"

    finally:
        # Cleanup
        await store.adelete(namespace, "update_test")
        await store._redis.aclose()
