"""RedisStore / AsyncRedisStore reads that must see every matching document.

RediSearch returns 10 documents unless a query asks for more. #31 fixed this for
search(); these tests cover list_namespaces(), which pages through its query sorted
on a unique field, and batched gets, which ask for one result per key.
"""

from __future__ import annotations

import asyncio
import copy
import time
from contextlib import contextmanager

import pytest
from langgraph.store.base import GetOp, PutOp
from redis import Redis
from redis.exceptions import ResponseError
from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex

from langgraph.store.redis import (
    AsyncRedisStore,
    RedisStore,
    StoreIndexNotReadyError,
    StoreSetupRequiredError,
)
from langgraph.store.redis.base import STORE_SORT_FIELD


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


# indexes and documents from before the sort field


def _wait_for_indexing(client, index_name: str) -> None:
    for _ in range(200):
        info = client.ft(index_name).info()
        if str(info.get("indexing", "0")) in ("0", "0.0"):
            return
        time.sleep(0.05)
    raise AssertionError(f"{index_name} still indexing")


def _old_schema(store) -> dict:
    """The store schema as it was before STORE_SORT_FIELD."""
    schema = copy.deepcopy(store.store_schema)
    schema["fields"] = [f for f in schema["fields"] if f["name"] != STORE_SORT_FIELD]
    return schema


def _make_pre_sort_field(store: RedisStore) -> list[str]:
    """Leave the store's index and documents as an older version wrote them.

    Drops the index (keeping the documents), removes the sort field from every
    document and recreates the index from the old schema. Returns the keys.
    """
    client = store._redis
    keys = sorted(k.decode() for k in client.scan_iter(match=f"{store.store_prefix}:*"))
    client.ft(store.store_index.name).dropindex(delete_documents=False)
    for key in keys:
        client.json().delete(key, f"$.{STORE_SORT_FIELD}")
    SearchIndex.from_dict(_old_schema(store), redis_client=client).create()
    _wait_for_indexing(client, store.store_index.name)
    return keys


def _sort_field_values(client, keys: list[str]) -> dict[str, object]:
    return {k: client.json().get(k, f"$.{STORE_SORT_FIELD}") for k in keys}


def test_setup_adds_the_sort_field_to_an_older_index_and_its_documents(
    redis_url: str,
) -> None:
    with RedisStore.from_conn_string(redis_url) as old:
        old.setup()
        namespaces = _user_namespaces(1005)
        old.batch([PutOp(ns, "profile", {"name": ns[-1]}, ttl=60) for ns in namespaces])
        keys = _make_pre_sort_field(old)
        assert set(map(str, _sort_field_values(old._redis, keys).values())) == {"[]"}
        ttls_before = {k: old._redis.ttl(k) for k in keys}

    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        attributes = store.store_index.info()["attributes"]
        assert any(STORE_SORT_FIELD in [str(v) for v in a] for a in attributes)
        values = _sort_field_values(store._redis, keys)
        # Each document's sort field is the id in its own key.
        assert values == {k: [k.split(":", 1)[1]] for k in keys}
        # The backfill left each document's expiry in place.
        assert all(store._redis.ttl(k) > 0 for k in keys)
        assert all(store._redis.ttl(k) <= ttls_before[k] for k in keys)

        assert store.list_namespaces(limit=2000) == sorted(namespaces)
        results = store.batch([GetOp(ns, "profile") for ns in namespaces])
        assert [r.value["name"] for r in results] == [ns[-1] for ns in namespaces]

        # A second setup finds nothing left to do.
        store.setup()
        assert _sort_field_values(store._redis, keys) == values


@pytest.mark.asyncio
async def test_async_setup_adds_the_sort_field_to_an_older_index_and_its_documents(
    redis_url: str,
) -> None:
    namespaces = _user_namespaces(1005)
    with RedisStore.from_conn_string(redis_url) as old:
        old.setup()
        old.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
        keys = _make_pre_sort_field(old)

    async with AsyncRedisStore.from_conn_string(redis_url) as astore:
        await astore.setup()
        client = astore._redis
        values = {k: await client.json().get(k, f"$.{STORE_SORT_FIELD}") for k in keys}
        assert values == {k: [k.split(":", 1)[1]] for k in keys}

        assert await astore.alist_namespaces(limit=2000) == sorted(namespaces)
        results = await astore.abatch([GetOp(ns, "profile") for ns in namespaces])
        assert [r.value["name"] for r in results] == [ns[-1] for ns in namespaces]

        await astore.setup()


def _index_has_sort_field(client, index_name: str) -> bool:
    attributes = client.ft(index_name).info()["attributes"]
    return any(
        STORE_SORT_FIELD in [v.decode() if isinstance(v, bytes) else v for v in a]
        for a in attributes
    )


def _older_store_with(redis_url: str, namespaces) -> list[str]:
    with RedisStore.from_conn_string(redis_url) as old:
        old.setup()
        old.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
        return _make_pre_sort_field(old)


def test_list_namespaces_on_an_older_index_requires_setup(redis_url: str) -> None:
    namespaces = _user_namespaces(1005)
    keys = _older_store_with(redis_url, namespaces)

    # Constructed directly: from_conn_string() is not involved, and nothing here
    # calls setup() until the test does.
    client = Redis.from_url(redis_url)
    try:
        store = RedisStore(client)
        index_name = store.store_index.name
        assert not _index_has_sort_field(client, index_name)

        with pytest.raises(StoreSetupRequiredError, match=r"call setup\(\)") as exc:
            store.list_namespaces(limit=2000)
        assert index_name in str(exc.value)
        # The read changed nothing.
        assert not _index_has_sort_field(client, index_name)
        assert set(map(str, _sort_field_values(client, keys).values())) == {"[]"}

        # Gets do not use the sort field and work before setup().
        results = store.batch([GetOp(ns, "profile") for ns in namespaces])
        assert [r.value["name"] for r in results] == [ns[-1] for ns in namespaces]

        store.setup()
        assert _index_has_sort_field(client, index_name)
        assert store.list_namespaces(limit=2000) == sorted(namespaces)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_alist_namespaces_on_an_older_index_requires_setup(
    redis_url: str,
) -> None:
    namespaces = _user_namespaces(1005)
    keys = _older_store_with(redis_url, namespaces)

    # The direct constructor: from_conn_string() would call setup() itself.
    async with AsyncRedisStore(redis_url) as astore:
        index_name = astore.store_index.name
        client = Redis.from_url(redis_url)
        try:
            assert not _index_has_sort_field(client, index_name)

            with pytest.raises(StoreSetupRequiredError, match=r"call setup\(\)"):
                await astore.alist_namespaces(limit=2000)
            assert not _index_has_sort_field(client, index_name)
            assert set(map(str, _sort_field_values(client, keys).values())) == {"[]"}

            results = await astore.abatch([GetOp(ns, "profile") for ns in namespaces])
            assert [r.value["name"] for r in results] == [ns[-1] for ns in namespaces]

            await astore.setup()
            assert _index_has_sort_field(client, index_name)
            assert await astore.alist_namespaces(limit=2000) == sorted(namespaces)
        finally:
            client.close()


def test_setup_when_another_process_added_the_field_first(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with RedisStore.from_conn_string(redis_url) as old:
        old.setup()
        old.batch([PutOp(("a",), f"k{i}", {"i": i}) for i in range(5)])
        _make_pre_sort_field(old)

    with RedisStore.from_conn_string(redis_url) as store:
        info = store.store_index.info
        calls: list[int] = []

        def stale_info():
            # The first look predates the other process's FT.ALTER.
            result = info()
            if not calls:
                store._redis.ft(store.store_index.name).alter_schema_add(
                    [store._sort_field_definition()]
                )
            calls.append(1)
            return result

        monkeypatch.setattr(store.store_index, "info", stale_info)
        store.setup()
        assert len(store.list_namespaces()) == 1


def test_documents_without_the_sort_field_are_still_read(store: RedisStore) -> None:
    """A document an older version writes after the upgrade has no sort field.

    Sorted reads return it after the others; it is not dropped.
    """
    namespaces = _user_namespaces(1005)
    store.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
    late = [k.decode() for k in store._redis.scan_iter(match="store:*")][:3]
    for key in late:
        store._redis.json().delete(key, f"$.{STORE_SORT_FIELD}")

    assert store.list_namespaces(limit=2000) == sorted(namespaces)
    results = store.batch([GetOp(ns, "profile") for ns in namespaces])
    assert all(r is not None for r in results)


def test_new_documents_carry_their_id_as_the_sort_field(store: RedisStore) -> None:
    store.put(("a",), "k", {"v": 1})
    (key,) = [k.decode() for k in store._redis.scan_iter(match="store:*")]
    assert store._redis.json().get(key, f"$.{STORE_SORT_FIELD}") == [
        key.split(":", 1)[1]
    ]


def _aggregate_also_returning(store: RedisStore, monkeypatch, extra_key: str) -> None:
    """Make the backfill's first search also report ``extra_key``."""
    search = store._redis.ft(store.store_index.name)
    calls: list[int] = []

    class _Search:
        def __getattr__(self, name):
            return getattr(search, name)

        def aggregate(self, request):
            res = search.aggregate(request)
            if not calls:
                res.rows.append(["__key", extra_key])
            calls.append(1)
            return res

    monkeypatch.setattr(store._redis, "ft", lambda name: _Search())


def test_setup_skips_a_document_deleted_during_the_backfill(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with RedisStore.from_conn_string(redis_url) as old:
        old.setup()
        old.batch([PutOp(("a",), f"k{i}", {"i": i}) for i in range(5)])
        _make_pre_sort_field(old)

    with RedisStore.from_conn_string(redis_url) as store:
        # Found by the search, gone before its field is set.
        _aggregate_also_returning(store, monkeypatch, "store:01DELETEDBEFORESET00000")
        store.setup()
        assert len(store.list_namespaces()) == 1


def test_setup_raises_when_the_backfill_fails_on_a_document_that_exists(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store._redis.set("store:01NOTAJSONDOCUMENT000000", "x")
        _aggregate_also_returning(store, monkeypatch, "store:01NOTAJSONDOCUMENT000000")
        with pytest.raises(ResponseError):
            store.setup()


# batched gets: one query per chunk of keys


def test_batched_gets_send_one_query_per_chunk_of_distinct_keys(
    store: RedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.batch([PutOp(("docs",), f"k{i:04d}", {"i": i}) for i in range(1005)])
    store.put(("notes",), "k0000", {"i": -1})

    sizes: list[int] = []
    query = store.store_index.query

    def counting_query(q):
        sizes.append(q._num_results)
        return query(q)

    def no_paging(*args, **kwargs):
        raise AssertionError("gets query by key and do not page")

    monkeypatch.setattr(store.store_index, "query", counting_query)
    monkeypatch.setattr(store.store_index, "paginate", no_paging)

    assert store.get(("docs",), "k0000").value == {"i": 0}
    assert sizes == [1]

    sizes.clear()
    ops = [GetOp(("docs",), f"k{i:04d}") for i in range(1005)]
    ops += [GetOp(("docs",), "k0000"), GetOp(("notes",), "k0000")]
    results = store.batch(ops)
    # docs: 1005 distinct keys in chunks of 1000 and 5; notes: one key.
    assert sorted(sizes) == [1, 5, 1000]
    assert [r.value["i"] for r in results] == list(range(1005)) + [0, -1]


@pytest.mark.asyncio
async def test_abatch_gets_send_one_query_per_chunk_of_distinct_keys(
    astore: AsyncRedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await astore.abatch([PutOp(("docs",), f"k{i:04d}", {"i": i}) for i in range(1005)])
    await astore.aput(("notes",), "k0000", {"i": -1})

    sizes: list[int] = []
    query = astore.store_index.query

    async def counting_query(q):
        sizes.append(q._num_results)
        return await query(q)

    def no_paging(*args, **kwargs):
        raise AssertionError("gets query by key and do not page")

    monkeypatch.setattr(astore.store_index, "query", counting_query)
    monkeypatch.setattr(astore.store_index, "paginate", no_paging)

    assert (await astore.aget(("docs",), "k0000")).value == {"i": 0}
    assert sizes == [1]

    sizes.clear()
    ops = [GetOp(("docs",), f"k{i:04d}") for i in range(1005)]
    ops += [GetOp(("docs",), "k0000"), GetOp(("notes",), "k0000")]
    results = await astore.abatch(ops)
    assert sorted(sizes) == [1, 5, 1000]
    assert [r.value["i"] for r in results] == list(range(1005)) + [0, -1]


# setup() waits for indexing up to a deadline


def _sort_field_ids(keys: list[str]) -> dict[str, list[str]]:
    return {k: [k.split(":", 1)[1]] for k in keys}


def test_setup_raises_an_actionable_error_when_indexing_outlasts_the_deadline(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _older_store_with(redis_url, _user_namespaces(5))

    with RedisStore.from_conn_string(redis_url) as store:
        store.index_ready_timeout = 0.2
        monkeypatch.setattr(store, "_still_indexing", lambda info: True)
        started = time.monotonic()
        with pytest.raises(StoreIndexNotReadyError, match=r"Retry setup\(\)") as exc:
            store.setup()
        # It waited for the deadline, and not much longer.
        assert 0.2 <= time.monotonic() - started < 5
        assert store.store_index.name in str(exc.value)

        # Once indexing has finished, retrying setup() completes the migration.
        monkeypatch.undo()
        store.setup()
        assert _sort_field_values(store._redis, keys) == _sort_field_ids(keys)


@pytest.mark.asyncio
async def test_async_setup_raises_an_actionable_error_when_indexing_outlasts_the_deadline(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = _older_store_with(redis_url, _user_namespaces(5))

    async with AsyncRedisStore(redis_url) as astore:
        astore.index_ready_timeout = 0.2
        monkeypatch.setattr(astore, "_still_indexing", lambda info: True)
        started = time.monotonic()
        with pytest.raises(StoreIndexNotReadyError, match=r"Retry setup\(\)") as exc:
            await astore.setup()
        # It waited for the deadline, and not much longer.
        assert 0.2 <= time.monotonic() - started < 5
        assert astore.store_index.name in str(exc.value)

        monkeypatch.undo()
        await astore.setup()
        client = astore._redis
        values = {k: await client.json().get(k, f"$.{STORE_SORT_FIELD}") for k in keys}
        assert values == _sort_field_ids(keys)


def test_list_namespaces_while_the_index_is_still_indexing(
    store: RedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field is there but the rescan is not finished, as when another process
    is still in setup(): reading now could skip documents, so the read says so."""
    store.batch([PutOp(ns, "profile", {"n": 1}) for ns in _user_namespaces(15)])
    client = store._redis
    reader = RedisStore(client)
    assert _index_has_sort_field(client, reader.store_index.name)
    monkeypatch.setattr(reader, "_still_indexing", lambda info: True)
    with pytest.raises(StoreIndexNotReadyError, match=r"Retry setup\(\)"):
        reader.list_namespaces()
    monkeypatch.undo()
    assert reader.list_namespaces() == sorted(_user_namespaces(15))


@pytest.mark.asyncio
async def test_alist_namespaces_while_the_index_is_still_indexing(
    redis_url: str, astore: AsyncRedisStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await astore.abatch([PutOp(ns, "profile", {"n": 1}) for ns in _user_namespaces(15)])
    async with AsyncRedisStore(redis_url) as reader:
        monkeypatch.setattr(reader, "_still_indexing", lambda info: True)
        with pytest.raises(StoreIndexNotReadyError, match=r"Retry setup\(\)"):
            await reader.alist_namespaces()
        monkeypatch.undo()
        assert await reader.alist_namespaces() == sorted(_user_namespaces(15))


# setup() in cluster mode sets the field key by key, not in a pipeline


def _forbid_pipeline(store, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_pipeline(*args, **kwargs):
        raise AssertionError("cluster mode sets the field key by key")

    monkeypatch.setattr(store._redis, "pipeline", no_pipeline)


def _async_aggregate_also_returning(
    astore: AsyncRedisStore, monkeypatch: pytest.MonkeyPatch, extra_key: str
) -> None:
    """Make the async backfill's first search also report ``extra_key``."""
    ft = astore._redis.ft
    calls: list[int] = []

    def patched_ft(name):
        search = ft(name)

        class _Search:
            def __getattr__(self, attr):
                return getattr(search, attr)

            async def aggregate(self, request):
                res = await search.aggregate(request)
                if not calls:
                    res.rows.append(["__key", extra_key])
                calls.append(1)
                return res

        return _Search()

    monkeypatch.setattr(astore._redis, "ft", patched_ft)


def test_cluster_mode_setup_fills_in_missing_ids(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(1005)
    keys = _older_store_with(redis_url, namespaces)

    client = Redis.from_url(redis_url)
    try:
        store = RedisStore(client, cluster_mode=True)
        _forbid_pipeline(store, monkeypatch)
        store.setup()
        assert _sort_field_values(client, keys) == _sort_field_ids(keys)
        assert store.list_namespaces(limit=2000) == sorted(namespaces)
    finally:
        client.close()


def test_cluster_mode_setup_skips_a_document_deleted_during_the_backfill(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(5)
    keys = _older_store_with(redis_url, namespaces)

    client = Redis.from_url(redis_url)
    try:
        store = RedisStore(client, cluster_mode=True)
        _forbid_pipeline(store, monkeypatch)
        _aggregate_also_returning(store, monkeypatch, "store:01DELETEDBEFORESET00000")
        store.setup()
        assert _sort_field_values(client, keys) == _sort_field_ids(keys)
        assert store.list_namespaces() == sorted(namespaces)
    finally:
        client.close()


def test_cluster_mode_setup_raises_when_the_backfill_fails_on_a_document_that_exists(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = Redis.from_url(redis_url)
    try:
        store = RedisStore(client, cluster_mode=True)
        store.setup()
        client.set("store:01NOTAJSONDOCUMENT000000", "x")
        _forbid_pipeline(store, monkeypatch)
        _aggregate_also_returning(store, monkeypatch, "store:01NOTAJSONDOCUMENT000000")
        with pytest.raises(ResponseError):
            store.setup()
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_cluster_mode_setup_fills_in_missing_ids(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(1005)
    keys = _older_store_with(redis_url, namespaces)

    async with AsyncRedisStore(redis_url, cluster_mode=True) as astore:
        _forbid_pipeline(astore, monkeypatch)
        await astore.setup()
        client = astore._redis
        values = {k: await client.json().get(k, f"$.{STORE_SORT_FIELD}") for k in keys}
        assert values == _sort_field_ids(keys)
        assert await astore.alist_namespaces(limit=2000) == sorted(namespaces)


@pytest.mark.asyncio
async def test_async_cluster_mode_setup_skips_a_document_deleted_during_the_backfill(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespaces = _user_namespaces(5)
    keys = _older_store_with(redis_url, namespaces)

    async with AsyncRedisStore(redis_url, cluster_mode=True) as astore:
        _forbid_pipeline(astore, monkeypatch)
        _async_aggregate_also_returning(
            astore, monkeypatch, "store:01DELETEDBEFORESET00000"
        )
        await astore.setup()
        client = astore._redis
        values = {k: await client.json().get(k, f"$.{STORE_SORT_FIELD}") for k in keys}
        assert values == _sort_field_ids(keys)
        assert await astore.alist_namespaces() == sorted(namespaces)


@pytest.mark.asyncio
async def test_async_cluster_mode_setup_raises_when_the_backfill_fails_on_a_document_that_exists(
    redis_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with AsyncRedisStore(redis_url, cluster_mode=True) as astore:
        await astore.setup()
        await astore._redis.set("store:01NOTAJSONDOCUMENT000000", "x")
        _forbid_pipeline(astore, monkeypatch)
        _async_aggregate_also_returning(
            astore, monkeypatch, "store:01NOTAJSONDOCUMENT000000"
        )
        with pytest.raises(ResponseError):
            await astore.setup()


# a server with a lowered search-max-search-results


@contextmanager
def _max_search_results(redis_url: str, value: int):
    """Lower RediSearch's MAXSEARCHRESULTS (search-max-search-results) for a block."""
    client = Redis.from_url(redis_url)
    config = client.execute_command("FT.CONFIG", "GET", "MAXSEARCHRESULTS")
    if isinstance(config, dict):
        (original,) = config.values()
    else:
        ((_, original),) = config
    client.execute_command("FT.CONFIG", "SET", "MAXSEARCHRESULTS", value)
    try:
        yield
    finally:
        client.execute_command("FT.CONFIG", "SET", "MAXSEARCHRESULTS", original)
        client.close()


def test_reads_with_max_search_results_below_the_page_size(
    redis_url: str, store: RedisStore
) -> None:
    """A page or key chunk larger than the server allows is refused, not cut short."""
    namespaces = _user_namespaces(250)
    store.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
    store.batch([PutOp(("docs",), f"k{i:04d}", {"i": i}) for i in range(250)])

    with _max_search_results(redis_url, 100):
        with pytest.raises(RedisSearchError, match="exceeds maximum"):
            store.list_namespaces(limit=2000)
        with pytest.raises(RedisSearchError, match="exceeds maximum"):
            store.batch([GetOp(("docs",), f"k{i:04d}") for i in range(250)])
        # A batch with no more keys per namespace than the limit still works.
        results = store.batch([GetOp(("docs",), f"k{i:04d}") for i in range(100)])
        assert [r.value["i"] for r in results] == list(range(100))


def test_list_namespaces_sees_at_most_max_search_results_documents(
    redis_url: str, store: RedisStore
) -> None:
    """Documented limit: an offset page past search-max-search-results comes back
    empty, so list_namespaces sees only the first that many documents (in sort-field
    order). The default limit is 1000000."""
    namespaces = _user_namespaces(1005)
    store.batch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])

    with _max_search_results(redis_url, 1000):
        listed = store.list_namespaces(limit=2000)
    assert len(listed) == 1000
    assert set(listed) < set(namespaces)
    assert store.list_namespaces(limit=2000) == sorted(namespaces)


@pytest.mark.asyncio
async def test_async_reads_with_max_search_results_below_the_page_size(
    redis_url: str, astore: AsyncRedisStore
) -> None:
    namespaces = _user_namespaces(250)
    await astore.abatch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])
    await astore.abatch([PutOp(("docs",), f"k{i:04d}", {"i": i}) for i in range(250)])

    with _max_search_results(redis_url, 100):
        with pytest.raises(RedisSearchError, match="exceeds maximum"):
            await astore.alist_namespaces(limit=2000)
        with pytest.raises(RedisSearchError, match="exceeds maximum"):
            await astore.abatch([GetOp(("docs",), f"k{i:04d}") for i in range(250)])
        results = await astore.abatch(
            [GetOp(("docs",), f"k{i:04d}") for i in range(100)]
        )
        assert [r.value["i"] for r in results] == list(range(100))


@pytest.mark.asyncio
async def test_alist_namespaces_sees_at_most_max_search_results_documents(
    redis_url: str, astore: AsyncRedisStore
) -> None:
    namespaces = _user_namespaces(1005)
    await astore.abatch([PutOp(ns, "profile", {"name": ns[-1]}) for ns in namespaces])

    with _max_search_results(redis_url, 1000):
        listed = await astore.alist_namespaces(limit=2000)
    assert len(listed) == 1000
    assert set(listed) < set(namespaces)
    assert await astore.alist_namespaces(limit=2000) == sorted(namespaces)
