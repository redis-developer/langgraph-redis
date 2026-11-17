"""Test for issue #237 - get() must find every item that search() finds.

``get()`` built its namespace filter with redisvl's ``Text`` filter, which
turned a double quote in a namespace label into a space, so ``get()`` returned
``None`` for items that ``put()`` had stored and ``search()`` could find.
"""

import pytest

from langgraph.store.redis import AsyncRedisStore, RedisStore

ITEMS = [
    pytest.param(("issue237", 'q"t'), "k", id="double-quote-in-namespace"),
]


@pytest.mark.parametrize("namespace,key", ITEMS)
def test_sync_store_get_update_delete(redis_url, namespace, key) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(namespace, key, {"v": 1})
        assert store.get(namespace, key).value == {"v": 1}

        store.put(namespace, key, {"v": 2})
        assert store.get(namespace, key).value == {"v": 2}
        assert [item.value for item in store.search(namespace)] == [{"v": 2}]

        store.delete(namespace, key)
        assert store.get(namespace, key) is None
        assert store.search(namespace) == []


@pytest.mark.parametrize("namespace,key", ITEMS)
async def test_async_store_get_update_delete(redis_url, namespace, key) -> None:
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()
        await store.aput(namespace, key, {"v": 1})
        assert (await store.aget(namespace, key)).value == {"v": 1}

        await store.aput(namespace, key, {"v": 2})
        assert (await store.aget(namespace, key)).value == {"v": 2}
        assert [item.value for item in await store.asearch(namespace)] == [{"v": 2}]

        await store.adelete(namespace, key)
        assert await store.aget(namespace, key) is None
        assert await store.asearch(namespace) == []


def test_get_stays_within_the_exact_namespace(redis_url) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(("issue237", "parent", "child"), "k", {"v": 1})

        assert store.get(("issue237", "parent"), "k") is None
