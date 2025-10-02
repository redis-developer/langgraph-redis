"""Tests for issue #99: Support customization of store prefixes.

Issue: https://github.com/redis-developer/langgraph-redis/issues/99

Feature request to allow customization of DEFAULT_STORE_PREFIX and
DEFAULT_STORE_VECTOR_PREFIX to enable multiple isolated stores in the same Redis instance.
"""

import pytest

from langgraph.store.redis import AsyncRedisStore, RedisStore


def test_default_store_prefix_is_store(redis_url: str) -> None:
    """Test that default store prefix is 'store'."""
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()

        # Default prefix should be "store"
        assert store.store_prefix == "store"
        assert store.vector_prefix == "store_vectors"

        # Verify keys use default prefix
        namespace = ("test", "default_prefix")
        store.put(namespace, "key1", {"data": "value1"})

        # Check that the key was created with default prefix
        keys = list(store._redis.scan_iter("store:*"))
        assert len(keys) > 0
        assert any(b"store:" in key for key in keys)


def test_custom_store_prefix_sync(redis_url: str) -> None:
    """Test RedisStore with custom store prefix."""
    custom_prefix = "myapp_store"
    custom_vector_prefix = "myapp_vectors"

    with RedisStore.from_conn_string(
        redis_url, store_prefix=custom_prefix, vector_prefix=custom_vector_prefix
    ) as store:
        store.setup()

        # Verify custom prefixes are set
        assert store.store_prefix == custom_prefix
        assert store.vector_prefix == custom_vector_prefix

        namespace = ("test", "custom_prefix")
        store.put(namespace, "key1", {"data": "value1"})

        # Verify keys use custom prefix
        keys = list(store._redis.scan_iter(f"{custom_prefix}:*"))
        assert len(keys) > 0

        # Verify default prefix is NOT used
        default_keys = list(store._redis.scan_iter("store:*"))
        # Filter out any keys from other tests
        default_keys = [k for k in default_keys if b"custom_prefix" in k]
        assert len(default_keys) == 0

        # Verify data can be retrieved
        item = store.get(namespace, "key1")
        assert item is not None
        assert item.value["data"] == "value1"


def test_custom_prefix_isolation(redis_url: str) -> None:
    """Test that different prefixes create isolated stores."""
    namespace = ("test", "isolation")

    # Store 1 with prefix "app1"
    with RedisStore.from_conn_string(redis_url, store_prefix="app1") as store1:
        store1.setup()
        store1.put(namespace, "shared_key", {"app": "app1", "value": "from_app1"})

    # Store 2 with prefix "app2"
    with RedisStore.from_conn_string(redis_url, store_prefix="app2") as store2:
        store2.setup()
        store2.put(namespace, "shared_key", {"app": "app2", "value": "from_app2"})

    # Verify isolation - each store should only see its own data
    with RedisStore.from_conn_string(redis_url, store_prefix="app1") as store1:
        store1.setup()
        item1 = store1.get(namespace, "shared_key")
        assert item1 is not None
        assert item1.value["app"] == "app1"
        assert item1.value["value"] == "from_app1"

    with RedisStore.from_conn_string(redis_url, store_prefix="app2") as store2:
        store2.setup()
        item2 = store2.get(namespace, "shared_key")
        assert item2 is not None
        assert item2.value["app"] == "app2"
        assert item2.value["value"] == "from_app2"


@pytest.mark.asyncio
async def test_custom_store_prefix_async(redis_url: str) -> None:
    """Test AsyncRedisStore with custom store prefix."""
    custom_prefix = "async_store"
    custom_vector_prefix = "async_vectors"

    store = AsyncRedisStore(
        redis_url, store_prefix=custom_prefix, vector_prefix=custom_vector_prefix
    )
    await store.setup()

    try:
        # Verify custom prefixes are set
        assert store.store_prefix == custom_prefix
        assert store.vector_prefix == custom_vector_prefix

        namespace = ("test", "async_custom")
        await store.aput(namespace, "key1", {"data": "async_value"})

        # Verify keys use custom prefix
        keys = []
        async for key in store._redis.scan_iter(f"{custom_prefix}:*"):
            keys.append(key)
        assert len(keys) > 0

        # Verify data can be retrieved
        item = await store.aget(namespace, "key1")
        assert item is not None
        assert item.value["data"] == "async_value"
    finally:
        await store._redis.aclose()


@pytest.mark.asyncio
async def test_custom_vector_prefix_async(redis_url: str) -> None:
    """Test that custom vector prefix is set correctly."""
    custom_vector_prefix = "custom_vectors"

    store = AsyncRedisStore(
        redis_url,
        store_prefix="custom_store",
        vector_prefix=custom_vector_prefix,
    )
    await store.setup()

    try:
        # Verify custom vector prefix is set
        assert store.vector_prefix == custom_vector_prefix
        assert store.store_prefix == "custom_store"

        # Verify the vector index name uses custom prefix
        if hasattr(store, "vector_index"):
            assert custom_vector_prefix == store.vector_index.schema.index.name
    finally:
        await store._redis.aclose()


def test_custom_prefix_with_special_characters(redis_url: str) -> None:
    """Test that custom prefixes work with various characters."""
    custom_prefix = "app-v1.2.3_store"

    with RedisStore.from_conn_string(redis_url, store_prefix=custom_prefix) as store:
        store.setup()

        namespace = ("test", "special_chars")
        store.put(namespace, "key1", {"data": "value1"})

        # Verify the custom prefix is preserved
        keys = list(store._redis.scan_iter(f"{custom_prefix}:*"))
        assert len(keys) > 0

        item = store.get(namespace, "key1")
        assert item is not None
        assert item.value["data"] == "value1"


def test_index_names_reflect_custom_prefix(redis_url: str) -> None:
    """Test that Redis search index names reflect custom prefixes."""
    custom_prefix = "myindex"
    custom_vector_prefix = "myvectors"

    with RedisStore.from_conn_string(
        redis_url, store_prefix=custom_prefix, vector_prefix=custom_vector_prefix
    ) as store:
        store.setup()

        # Index names should incorporate the custom prefix
        # This prevents index name collisions when using multiple stores
        assert custom_prefix in store.store_index.schema.index.name

        # If vector index exists, it should also use custom prefix
        if hasattr(store, "vector_index"):
            assert custom_vector_prefix in store.vector_index.schema.index.name


@pytest.mark.asyncio
async def test_put_and_get_with_custom_prefix(redis_url: str) -> None:
    """Test that put and get work correctly with custom prefix."""
    custom_prefix = "ns_test"

    store = AsyncRedisStore(redis_url, store_prefix=custom_prefix)
    await store.setup()

    try:
        # Add items to different namespaces
        await store.aput(("app", "users"), "user1", {"name": "Alice"})
        await store.aput(("app", "products"), "prod1", {"name": "Widget"})

        # Retrieve items - should work with custom prefix
        user = await store.aget(("app", "users"), "user1")
        assert user is not None
        assert user.value["name"] == "Alice"

        product = await store.aget(("app", "products"), "prod1")
        assert product is not None
        assert product.value["name"] == "Widget"

        # Verify keys use custom prefix
        keys = []
        async for key in store._redis.scan_iter(f"{custom_prefix}:*"):
            keys.append(key)
        assert len(keys) == 2
    finally:
        await store._redis.aclose()


def test_backward_compatibility_no_prefix_specified(redis_url: str) -> None:
    """Test that not specifying prefix uses defaults (backward compatibility)."""
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()

        # Should use defaults
        assert store.store_prefix == "store"
        assert store.vector_prefix == "store_vectors"

        namespace = ("test", "backward_compat")
        store.put(namespace, "key1", {"data": "value1"})

        item = store.get(namespace, "key1")
        assert item is not None
        assert item.value["data"] == "value1"
