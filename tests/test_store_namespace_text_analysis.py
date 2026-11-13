"""Store namespaces whose labels RediSearch would otherwise analyse away.

The store indexes each item's namespace as the TEXT field ``prefix``. With the
index defaults, RediSearch drops default stopwords (``a``, ``the``, ``is``,
``of``, ...) both when indexing and when querying, and stems terms, so:

* an item in a stopword namespace such as ``("a",)`` is never indexed under
  its namespace, so ``get`` returns ``None``;
* the stopword clause vanishes from the query ``@prefix:(a) @key:{k}``, which
  becomes ``@key:{k}`` — the pre-delete in ``put``/``delete`` then removes the
  item with that key from *every* namespace;
* ``("running",)`` and ``("runs",)`` share the stem ``run`` and match each other.

The store index is now created with ``STOPWORDS 0`` and ``NOSTEM`` on ``prefix``,
and ``setup()`` rebuilds an index created without them, keeping its documents.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Iterator
from unittest.mock import patch

import pytest
from langgraph.store.base import GetOp
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redisvl.exceptions import RedisSearchError
from redisvl.index import SearchIndex
from testcontainers.redis import RedisContainer

from langgraph.store.redis import AsyncRedisStore, RedisStore
from tests.embed_test_utils import CharacterEmbeddings

STOPWORDS = ["a", "an", "the", "is", "of", "and", "to", "in"]


@pytest.fixture(scope="module")
def redis8_container() -> Iterator[RedisContainer]:
    """A Redis 8 container of this module's own.

    Named apart from the session-scoped ``redis_container``/``redis_url``
    fixtures: overriding those here would tear down the shared container for
    every module that runs after this one."""
    container = RedisContainer("redis:8")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="module")
def own_redis_url(redis8_container: RedisContainer) -> str:
    host = redis8_container.get_container_host_ip()
    port = redis8_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


@pytest.fixture(autouse=True)
def _clean_redis(own_redis_url: str) -> Iterator[None]:
    client = Redis.from_url(own_redis_url)
    client.flushall()
    yield
    client.close()


@pytest.fixture
def store(own_redis_url: str) -> Iterator[RedisStore]:
    with RedisStore.from_conn_string(own_redis_url) as s:
        s.setup()
        yield s


@pytest.fixture
async def astore(own_redis_url: str) -> AsyncIterator[AsyncRedisStore]:
    async with AsyncRedisStore.from_conn_string(own_redis_url) as s:
        await s.setup()
        yield s


def _vector_index_config() -> dict[str, Any]:
    embeddings = CharacterEmbeddings(dims=4)
    return {
        "dims": embeddings.dims,
        "embed": embeddings,
        "distance_type": "cosine",
        "fields": ["text"],
    }


def _create_legacy_store_index(own_redis_url: str) -> None:
    """Create the ``store`` index as releases before this fix did: default
    stopwords and stemming on ``prefix``."""
    SearchIndex.from_dict(
        {
            "index": {"name": "store", "prefix": "store:", "storage_type": "json"},
            "fields": [
                {"name": "prefix", "type": "text"},
                {"name": "key", "type": "tag"},
                {"name": "created_at", "type": "numeric"},
                {"name": "updated_at", "type": "numeric"},
                {"name": "ttl_minutes", "type": "numeric"},
                {"name": "expires_at", "type": "numeric"},
            ],
        },
        redis_url=own_redis_url,
    ).create(overwrite=True)


def _drop_store_index_definition(own_redis_url: str) -> None:
    """FT.DROPINDEX without DD, as another process upgrading would."""
    client = Redis.from_url(own_redis_url)
    try:
        client.execute_command("FT.DROPINDEX", "store")
    finally:
        client.close()


def _store_index_info(own_redis_url: str) -> dict[str, Any]:
    client = Redis.from_url(own_redis_url, decode_responses=True)
    try:
        raw = client.execute_command("FT.INFO", "store")
    finally:
        client.close()
    return dict(zip(raw[::2], raw[1::2]))


def _prefix_attribute(info: dict[str, Any]) -> list[str]:
    return next(attr for attr in info["attributes"] if attr[3] == "prefix")


# --------------------------------------------------------------------------- sync


@pytest.mark.parametrize("label", STOPWORDS)
def test_get_returns_item_in_stopword_namespace(store: RedisStore, label: str) -> None:
    store.put((label,), "k0", {"i": 0})

    item = store.get((label,), "k0")
    assert item is not None
    assert item.value == {"i": 0}
    assert item.namespace == (label,)

    [batched] = store.batch([GetOp(namespace=(label,), key="k0")])
    assert batched is not None and batched.value == {"i": 0}


def test_put_in_stopword_namespace_keeps_same_key_in_other_namespaces(
    store: RedisStore,
) -> None:
    store.put(("users",), "k", {"owner": "users"})
    store.put(("docs", "v1"), "k", {"owner": "docs"})

    store.put(("a",), "k", {"owner": "a"})
    store.put(("the",), "k", {"owner": "the"})

    assert store.get(("users",), "k").value == {"owner": "users"}
    assert store.get(("docs", "v1"), "k").value == {"owner": "docs"}
    assert store.get(("a",), "k").value == {"owner": "a"}
    assert store.get(("the",), "k").value == {"owner": "the"}


def test_delete_in_stopword_namespace_removes_only_that_item(store: RedisStore) -> None:
    store.put(("users",), "k", {"owner": "users"})
    store.put(("a",), "k", {"owner": "a"})

    store.delete(("a",), "k")

    assert store.get(("a",), "k") is None
    assert store.get(("users",), "k").value == {"owner": "users"}


def test_overwrite_in_stopword_namespace_leaves_one_item(store: RedisStore) -> None:
    store.put(("the",), "k", {"v": 1})
    store.put(("the",), "k", {"v": 2})

    results = store.search(("the",))
    assert [(r.key, r.value) for r in results] == [("k", {"v": 2})]


def test_stemmed_namespace_labels_stay_distinct(store: RedisStore) -> None:
    store.put(("runs",), "k", {"ns": "runs"})
    store.put(("running",), "k", {"ns": "running"})

    assert store.get(("runs",), "k").value == {"ns": "runs"}
    assert store.get(("running",), "k").value == {"ns": "running"}
    assert store.get(("run",), "k") is None


def test_search_and_list_namespaces_match_stopword_labels(store: RedisStore) -> None:
    store.put(("the", "x"), "k1", {"v": 1})
    store.put(("user", "of"), "k2", {"v": 2})

    assert [r.key for r in store.search(("the",))] == ["k1"]
    assert store.list_namespaces(prefix=("the",)) == [("the", "x")]
    assert store.list_namespaces(suffix=("of",)) == [("user", "of")]


def test_vector_search_in_stopword_namespace(own_redis_url: str) -> None:
    with RedisStore.from_conn_string(own_redis_url, index=_vector_index_config()) as s:
        s.setup()
        s.put(("the",), "doc1", {"text": "hello world"})
        s.put(("other",), "doc2", {"text": "hello world"})

        results = s.search(("the",), query="hello")
        assert [(r.namespace, r.key) for r in results] == [(("the",), "doc1")]


def test_setup_migrates_legacy_index_and_keeps_documents(own_redis_url: str) -> None:
    _create_legacy_store_index(own_redis_url)
    with RedisStore.from_conn_string(own_redis_url) as s:
        # Written under the legacy index, before this version's setup() runs.
        # ("a",) goes first: under the legacy index its pre-delete would
        # remove k0 from every namespace.
        s.put(("a",), "k0", {"owner": "a"})
        s.put(("users",), "k0", {"owner": "users"})
        assert s.get(("a",), "k0") is None

        s.setup()

        assert s.get(("a",), "k0").value == {"owner": "a"}
        assert s.get(("users",), "k0").value == {"owner": "users"}

    info = _store_index_info(own_redis_url)
    assert info["stopwords_list"] == []
    assert "NOSTEM" in _prefix_attribute(info)
    assert int(info["num_docs"]) == 2


def test_setup_leaves_current_index_in_place(
    own_redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    with RedisStore.from_conn_string(own_redis_url) as s:
        s.setup()
        s.put(("a",), "k0", {"i": 0})
        with (
            patch.object(s.store_index, "delete", wraps=s.store_index.delete) as delete,
            caplog.at_level(logging.WARNING),
        ):
            s.setup()
        delete.assert_not_called()
        assert "Rebuilding search index" not in caplog.text
        assert s.get(("a",), "k0").value == {"i": 0}


def test_setup_recovers_when_another_process_drops_the_legacy_index_first(
    own_redis_url: str,
) -> None:
    """Two processes upgrading at once both see the legacy index; the other one
    drops it between this one's check and its own FT.DROPINDEX, which then fails."""
    _create_legacy_store_index(own_redis_url)
    with RedisStore.from_conn_string(own_redis_url) as s:
        s.put(("a",), "k0", {"owner": "a"})
        real_delete = s.store_index.delete

        def dropped_by_other_process_first(drop: bool = True) -> None:
            _drop_store_index_definition(own_redis_url)
            real_delete(drop=drop)  # raises: the index is already gone

        with patch.object(
            s.store_index, "delete", side_effect=dropped_by_other_process_first
        ):
            s.setup()

        assert s.get(("a",), "k0").value == {"owner": "a"}
    assert _store_index_info(own_redis_url)["stopwords_list"] == []


def test_concurrent_setup_on_legacy_index_succeeds_in_every_worker(
    own_redis_url: str,
) -> None:
    _create_legacy_store_index(own_redis_url)
    with RedisStore.from_conn_string(own_redis_url) as seed:
        for i in range(200):
            seed.put(("a", str(i)), "k", {"i": i})

    all_started = threading.Barrier(4)

    def start_worker() -> None:
        with RedisStore.from_conn_string(own_redis_url) as worker:
            all_started.wait()
            worker.setup()
            # Every worker, not just the one that rebuilt, sees every item.
            assert int(_store_index_info(own_redis_url)["num_docs"]) == 200
            assert worker.get(("a", "199"), "k").value == {"i": 199}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(start_worker) for _ in range(4)]:
            future.result()  # re-raises any setup() error

    info = _store_index_info(own_redis_url)
    assert info["stopwords_list"] == []
    assert int(info["num_docs"]) == 200


def test_setup_raises_connection_error_while_waiting_for_indexing(
    own_redis_url: str,
) -> None:
    """A Redis that becomes unreachable during setup() is reported, not mistaken
    for another process's rebuild and waited out."""
    with RedisStore.from_conn_string(own_redis_url) as s:
        lost = RedisSearchError("Error while fetching store index info")
        lost.__cause__ = RedisConnectionError("Connection refused")
        started = time.monotonic()
        with patch.object(s.store_index, "info", side_effect=lost):
            with pytest.raises(RedisConnectionError):
                s.setup()
        assert time.monotonic() - started < 5


# -------------------------------------------------------------------------- async


@pytest.mark.parametrize("label", STOPWORDS)
async def test_async_get_returns_item_in_stopword_namespace(
    astore: AsyncRedisStore, label: str
) -> None:
    await astore.aput((label,), "k0", {"i": 0})

    item = await astore.aget((label,), "k0")
    assert item is not None
    assert item.value == {"i": 0}
    assert item.namespace == (label,)

    [batched] = await astore.abatch([GetOp(namespace=(label,), key="k0")])
    assert batched is not None and batched.value == {"i": 0}


async def test_async_put_in_stopword_namespace_keeps_same_key_in_other_namespaces(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(("users",), "k", {"owner": "users"})
    await astore.aput(("docs", "v1"), "k", {"owner": "docs"})

    await astore.aput(("a",), "k", {"owner": "a"})
    await astore.aput(("the",), "k", {"owner": "the"})

    assert (await astore.aget(("users",), "k")).value == {"owner": "users"}
    assert (await astore.aget(("docs", "v1"), "k")).value == {"owner": "docs"}
    assert (await astore.aget(("a",), "k")).value == {"owner": "a"}
    assert (await astore.aget(("the",), "k")).value == {"owner": "the"}


async def test_async_delete_in_stopword_namespace_removes_only_that_item(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(("users",), "k", {"owner": "users"})
    await astore.aput(("a",), "k", {"owner": "a"})

    await astore.adelete(("a",), "k")

    assert await astore.aget(("a",), "k") is None
    assert (await astore.aget(("users",), "k")).value == {"owner": "users"}


async def test_async_overwrite_in_stopword_namespace_leaves_one_item(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(("the",), "k", {"v": 1})
    await astore.aput(("the",), "k", {"v": 2})

    results = await astore.asearch(("the",))
    assert [(r.key, r.value) for r in results] == [("k", {"v": 2})]


async def test_async_stemmed_namespace_labels_stay_distinct(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(("runs",), "k", {"ns": "runs"})
    await astore.aput(("running",), "k", {"ns": "running"})

    assert (await astore.aget(("runs",), "k")).value == {"ns": "runs"}
    assert (await astore.aget(("running",), "k")).value == {"ns": "running"}
    assert await astore.aget(("run",), "k") is None


async def test_async_search_and_list_namespaces_match_stopword_labels(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(("the", "x"), "k1", {"v": 1})
    await astore.aput(("user", "of"), "k2", {"v": 2})

    assert [r.key for r in await astore.asearch(("the",))] == ["k1"]
    assert await astore.alist_namespaces(prefix=("the",)) == [("the", "x")]
    assert await astore.alist_namespaces(suffix=("of",)) == [("user", "of")]


async def test_async_vector_search_in_stopword_namespace(own_redis_url: str) -> None:
    async with AsyncRedisStore.from_conn_string(
        own_redis_url, index=_vector_index_config()
    ) as s:
        await s.setup()
        await s.aput(("the",), "doc1", {"text": "hello world"})
        await s.aput(("other",), "doc2", {"text": "hello world"})

        results = await s.asearch(("the",), query="hello")
        assert [(r.namespace, r.key) for r in results] == [(("the",), "doc1")]


async def test_async_setup_migrates_legacy_index_and_keeps_documents(
    own_redis_url: str,
) -> None:
    _create_legacy_store_index(own_redis_url)
    # Not from_conn_string(): that runs setup() on entry.
    async with AsyncRedisStore(own_redis_url) as s:
        # Written under the legacy index, before this version's setup() runs.
        # ("a",) goes first: under the legacy index its pre-delete would
        # remove k0 from every namespace.
        await s.aput(("a",), "k0", {"owner": "a"})
        await s.aput(("users",), "k0", {"owner": "users"})
        assert await s.aget(("a",), "k0") is None

        await s.setup()

        assert (await s.aget(("a",), "k0")).value == {"owner": "a"}
        assert (await s.aget(("users",), "k0")).value == {"owner": "users"}

    info = _store_index_info(own_redis_url)
    assert info["stopwords_list"] == []
    assert "NOSTEM" in _prefix_attribute(info)
    assert int(info["num_docs"]) == 2


async def test_async_setup_leaves_current_index_in_place(
    own_redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    async with AsyncRedisStore.from_conn_string(own_redis_url) as s:
        await s.aput(("a",), "k0", {"i": 0})
        with (
            patch.object(s.store_index, "delete", wraps=s.store_index.delete) as delete,
            caplog.at_level(logging.WARNING),
        ):
            await s.setup()
        delete.assert_not_called()
        assert "Rebuilding search index" not in caplog.text
        assert (await s.aget(("a",), "k0")).value == {"i": 0}


async def test_async_setup_recovers_when_another_process_drops_the_legacy_index_first(
    own_redis_url: str,
) -> None:
    _create_legacy_store_index(own_redis_url)
    async with AsyncRedisStore(own_redis_url) as s:
        await s.aput(("a",), "k0", {"owner": "a"})
        real_delete = s.store_index.delete

        async def dropped_by_other_process_first(drop: bool = True) -> None:
            _drop_store_index_definition(own_redis_url)
            await real_delete(drop=drop)  # raises: the index is already gone

        with patch.object(
            s.store_index, "delete", side_effect=dropped_by_other_process_first
        ):
            await s.setup()

        assert (await s.aget(("a",), "k0")).value == {"owner": "a"}
    assert _store_index_info(own_redis_url)["stopwords_list"] == []


async def test_async_concurrent_setup_on_legacy_index_succeeds_in_every_worker(
    own_redis_url: str,
) -> None:
    _create_legacy_store_index(own_redis_url)
    async with AsyncRedisStore(own_redis_url) as seed:
        for i in range(200):
            await seed.aput(("a", str(i)), "k", {"i": i})

    async def start_worker() -> None:
        async with AsyncRedisStore(own_redis_url) as worker:
            await worker.setup()
            # Every worker, not just the one that rebuilt, sees every item.
            assert int(_store_index_info(own_redis_url)["num_docs"]) == 200
            assert (await worker.aget(("a", "199"), "k")).value == {"i": 199}

    await asyncio.gather(*(start_worker() for _ in range(4)))

    info = _store_index_info(own_redis_url)
    assert info["stopwords_list"] == []
    assert int(info["num_docs"]) == 200


async def test_async_setup_raises_connection_error_while_waiting_for_indexing(
    own_redis_url: str,
) -> None:
    async with AsyncRedisStore(own_redis_url) as s:
        lost = RedisSearchError("Error while fetching store index info")
        lost.__cause__ = RedisConnectionError("Connection refused")
        started = time.monotonic()
        with patch.object(s.store_index, "info", side_effect=lost):
            with pytest.raises(RedisConnectionError):
                await s.setup()
        assert time.monotonic() - started < 5
