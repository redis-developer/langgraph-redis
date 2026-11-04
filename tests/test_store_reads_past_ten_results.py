"""RedisStore / AsyncRedisStore reads that must see every matching document.

RediSearch returns 10 documents unless a query asks for more. #31 fixed this for
search(); these tests cover list_namespaces() and batched gets, which run their own
queries against the same index.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.store.base import GetOp, PutOp

from langgraph.store.redis import AsyncRedisStore, RedisStore


@pytest.fixture
def store(redis_url: str):
    with RedisStore.from_conn_string(redis_url) as s:
        s.setup()
        yield s


@pytest.fixture
async def astore(redis_url: str):
    async with AsyncRedisStore.from_conn_string(redis_url) as s:
        await s.setup()
        yield s


def _user_namespaces(n: int) -> list[tuple[str, ...]]:
    return [("memories", f"user{i:04d}") for i in range(n)]


# list_namespaces


def test_list_namespaces_returns_more_than_ten(store: RedisStore) -> None:
    namespaces = _user_namespaces(15)
    for ns in namespaces:
        store.put(ns, "profile", {"name": ns[-1]})

    assert store.list_namespaces(prefix=("memories",)) == sorted(namespaces)
    assert store.list_namespaces() == sorted(namespaces)


@pytest.mark.parametrize("count", [1000, 1005])
def test_list_namespaces_across_result_pages(store: RedisStore, count: int) -> None:
    namespaces = _user_namespaces(count)
    store.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])

    assert store.list_namespaces(limit=2000) == sorted(namespaces)


def test_list_namespaces_limit_and_offset_over_all_namespaces(
    store: RedisStore,
) -> None:
    namespaces = _user_namespaces(15)
    for ns in namespaces:
        store.put(ns, "profile", {"name": ns[-1]})

    assert store.list_namespaces(limit=3, offset=12) == sorted(namespaces)[12:15]


def test_list_namespaces_under_ten_documents(store: RedisStore) -> None:
    """Control: fewer than 10 stored items were always listed in full."""
    namespaces = _user_namespaces(5)
    for ns in namespaces:
        store.put(ns, "profile", {"name": ns[-1]})

    assert store.list_namespaces(prefix=("memories",)) == sorted(namespaces)


@pytest.mark.asyncio
async def test_alist_namespaces_returns_more_than_ten(
    astore: AsyncRedisStore,
) -> None:
    namespaces = _user_namespaces(15)
    for ns in namespaces:
        await astore.aput(ns, "profile", {"name": ns[-1]})

    assert await astore.alist_namespaces(prefix=("memories",)) == sorted(namespaces)
    assert await astore.alist_namespaces() == sorted(namespaces)


@pytest.mark.asyncio
async def test_alist_namespaces_across_result_pages(astore: AsyncRedisStore) -> None:
    namespaces = _user_namespaces(1005)
    await astore.abatch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])

    assert await astore.alist_namespaces(limit=2000) == sorted(namespaces)


@pytest.mark.asyncio
async def test_alist_namespaces_under_ten_documents(astore: AsyncRedisStore) -> None:
    """Control: fewer than 10 stored items were always listed in full."""
    namespaces = _user_namespaces(5)
    for ns in namespaces:
        await astore.aput(ns, "profile", {"name": ns[-1]})

    assert await astore.alist_namespaces(prefix=("memories",)) == sorted(namespaces)


# batched gets


def test_batch_get_more_than_ten_keys_in_one_namespace(store: RedisStore) -> None:
    ns = ("docs",)
    for i in range(15):
        store.put(ns, f"k{i:02d}", {"i": i})

    results = store.batch([GetOp(ns, f"k{i:02d}") for i in range(15)])

    assert [r.value["i"] if r else None for r in results] == list(range(15))


def test_batch_get_across_result_pages(store: RedisStore) -> None:
    ns = ("docs",)
    store.batch([PutOp(ns, f"k{i:04d}", {"i": i}) for i in range(1005)])

    results = store.batch([GetOp(ns, f"k{i:04d}") for i in range(1005)])

    assert [r.value["i"] if r else None for r in results] == list(range(1005))


def test_batch_get_returns_none_only_for_missing_keys(store: RedisStore) -> None:
    """Control: a small batch finds the stored keys and returns None for the rest."""
    ns = ("docs",)
    for i in range(3):
        store.put(ns, f"k{i:02d}", {"i": i})

    results = store.batch([GetOp(ns, f"k{i:02d}") for i in range(5)])

    assert [r.value["i"] if r else None for r in results] == [0, 1, 2, None, None]


def test_single_gets_of_more_than_ten_keys(store: RedisStore) -> None:
    """Control: one get() per key finds every item."""
    ns = ("docs",)
    for i in range(15):
        store.put(ns, f"k{i:02d}", {"i": i})

    assert [store.get(ns, f"k{i:02d}").value["i"] for i in range(15)] == list(range(15))


@pytest.mark.asyncio
async def test_concurrent_aget_more_than_ten_keys(astore: AsyncRedisStore) -> None:
    """Concurrent aget() calls are batched by AsyncBatchedBaseStore into one GET query."""
    ns = ("docs",)
    for i in range(15):
        await astore.aput(ns, f"k{i:02d}", {"i": i})

    results = await asyncio.gather(*(astore.aget(ns, f"k{i:02d}") for i in range(15)))

    assert [r.value["i"] if r else None for r in results] == list(range(15))


@pytest.mark.asyncio
async def test_abatch_get_more_than_ten_keys(astore: AsyncRedisStore) -> None:
    ns = ("docs",)
    for i in range(15):
        await astore.aput(ns, f"k{i:02d}", {"i": i})

    results = await astore.abatch([GetOp(ns, f"k{i:02d}") for i in range(15)])

    assert [r.value["i"] if r else None for r in results] == list(range(15))


@pytest.mark.asyncio
async def test_sequential_aget_more_than_ten_keys(astore: AsyncRedisStore) -> None:
    """Control: one awaited aget() at a time finds every item."""
    ns = ("docs",)
    for i in range(15):
        await astore.aput(ns, f"k{i:02d}", {"i": i})

    results = [await astore.aget(ns, f"k{i:02d}") for i in range(15)]

    assert [r.value["i"] for r in results] == list(range(15))


# the index changing between reads


def _delete_after_first_search(
    monkeypatch: pytest.MonkeyPatch, store: RedisStore
) -> list:
    """After the store's first index search returns, delete the first document it returned.

    This makes the index change between the reads of a single store call. The deleted
    document is appended to the returned list.
    """
    deleted: list = []
    search = store.store_index.search

    def search_then_delete(*args, **kwargs):
        res = search(*args, **kwargs)
        if not deleted:
            doc_id = res.docs[0].id
            deleted.append(store._redis.json().get(doc_id))
            store._redis.delete(doc_id)
        return res

    monkeypatch.setattr(store.store_index, "search", search_then_delete)
    return deleted


def _adelete_after_first_search(
    monkeypatch: pytest.MonkeyPatch, astore: AsyncRedisStore
) -> list:
    """Async version of _delete_after_first_search."""
    deleted: list = []
    search = astore.store_index.search

    async def search_then_delete(*args, **kwargs):
        res = await search(*args, **kwargs)
        if not deleted:
            doc_id = res.docs[0].id
            deleted.append(await astore._redis.json().get(doc_id))
            await astore._redis.delete(doc_id)
        return res

    monkeypatch.setattr(astore.store_index, "search", search_then_delete)
    return deleted


def test_list_namespaces_with_a_delete_between_reads(
    store: RedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(1005)
    store.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
    deleted = _delete_after_first_search(monkeypatch, store)

    result = store.list_namespaces(limit=2000)

    # Every namespace that existed for the whole call is listed.
    gone = ("memories", deleted[0]["value"]["name"])
    assert set(namespaces) - {gone} <= set(result)


@pytest.mark.asyncio
async def test_alist_namespaces_with_a_delete_between_reads(
    astore: AsyncRedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(1005)
    await astore.abatch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
    deleted = _adelete_after_first_search(monkeypatch, astore)

    result = await astore.alist_namespaces(limit=2000)

    # Every namespace that existed for the whole call is listed.
    gone = ("memories", deleted[0]["value"]["name"])
    assert set(namespaces) - {gone} <= set(result)


def test_batch_get_with_a_delete_between_reads(
    store: RedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ns = ("docs",)
    store.batch([PutOp(ns, f"k{i:04d}", {"i": i}) for i in range(1005)])
    deleted = _delete_after_first_search(monkeypatch, store)

    results = store.batch([GetOp(ns, f"k{i:04d}") for i in range(1005)])

    # Every key that existed for the whole call is found.
    gone = int(deleted[0]["key"][1:])
    found = [r.value["i"] if r else None for r in results]
    assert [v for i, v in enumerate(found) if i != gone] == [
        i for i in range(1005) if i != gone
    ]


@pytest.mark.asyncio
async def test_abatch_get_with_a_delete_between_reads(
    astore: AsyncRedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ns = ("docs",)
    await astore.abatch([PutOp(ns, f"k{i:04d}", {"i": i}) for i in range(1005)])
    deleted = _adelete_after_first_search(monkeypatch, astore)

    results = await astore.abatch([GetOp(ns, f"k{i:04d}") for i in range(1005)])

    # Every key that existed for the whole call is found.
    gone = int(deleted[0]["key"][1:])
    found = [r.value["i"] if r else None for r in results]
    assert [v for i, v in enumerate(found) if i != gone] == [
        i for i in range(1005) if i != gone
    ]
