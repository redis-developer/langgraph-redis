"""Stopword namespaces must round-trip without affecting other namespaces."""

from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

import pytest
from langgraph.store.base import Item

from langgraph.store.redis import AsyncRedisStore, RedisStore
from tests.embed_test_utils import CharacterEmbeddings

WRITE_CASES = [
    pytest.param(False, False, id="put"),
    pytest.param(True, False, id="update"),
    pytest.param(False, True, id="delete_missing"),
    pytest.param(True, True, id="delete_existing"),
]


def _config(vector: bool = True) -> dict[str, Any]:
    prefix = f"namespace_stopwords_{uuid4().hex}"
    config: dict[str, Any] = {
        "store_prefix": prefix,
        "vector_prefix": prefix + "_vectors",
    }
    if vector:
        config["index"] = {
            "dims": 4,
            "embed": CharacterEmbeddings(dims=4),
            "fields": ["text"],
        }
    return config


def _assert_item(
    item: Item | None,
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
) -> None:
    assert item is not None
    assert item.namespace == namespace
    assert item.key == key
    assert item.value == value


@pytest.fixture
def vector() -> bool:
    """Use vectors by default; round-trip tests also parameterize plain stores."""
    return True


@pytest.fixture
def store(redis_url: str, vector: bool) -> Iterator[RedisStore]:
    with RedisStore.from_conn_string(redis_url, **_config(vector)) as store:
        store.setup()
        yield store


@pytest.fixture
async def astore(redis_url: str, vector: bool) -> AsyncIterator[AsyncRedisStore]:
    # The async context manager calls setup() before yielding the store.
    async with AsyncRedisStore.from_conn_string(redis_url, **_config(vector)) as store:
        yield store


@pytest.mark.parametrize("vector", [False, True])
def test_sync_stopword_namespace_round_trip(store: RedisStore, vector: bool) -> None:
    namespace = ("in",)
    value = {"text": "hello", "tag": "x"}
    store.put(namespace, "k", value)
    _assert_item(store.get(namespace, "k"), namespace, "k", value)
    assert [i.value for i in store.search(namespace, filter={"tag": "x"})] == [value]
    assert store.list_namespaces(prefix=namespace) == [namespace]
    if vector:
        assert [i.value for i in store.search(namespace, query="hello")] == [value]

    nested_namespace = ("tenant", "in")
    store.put(nested_namespace, "nested", {"text": "nested"})
    assert store.list_namespaces(suffix=("in",)) == [namespace, nested_namespace]


@pytest.mark.parametrize("vector", [False, True])
async def test_async_stopword_namespace_round_trip(
    astore: AsyncRedisStore, vector: bool
) -> None:
    namespace = ("in",)
    value = {"text": "hello", "tag": "x"}
    await astore.aput(namespace, "k", value)
    _assert_item(await astore.aget(namespace, "k"), namespace, "k", value)
    assert [i.value for i in await astore.asearch(namespace, filter={"tag": "x"})] == [
        value
    ]
    assert await astore.alist_namespaces(prefix=namespace) == [namespace]
    if vector:
        assert [i.value for i in await astore.asearch(namespace, query="hello")] == [
            value
        ]

    nested_namespace = ("tenant", "in")
    await astore.aput(nested_namespace, "nested", {"text": "nested"})
    assert await astore.alist_namespaces(suffix=("in",)) == [
        namespace,
        nested_namespace,
    ]


@pytest.mark.parametrize(("existing", "delete"), WRITE_CASES)
def test_sync_stopword_writes_preserve_other_namespaces(
    store: RedisStore, existing: bool, delete: bool
) -> None:
    if existing:
        store.put(("in",), "shared", {"text": "old"})
    control = {"text": "control"}
    store.put(("users",), "shared", control)
    if delete:
        store.delete(("in",), "shared")
    else:
        store.put(("in",), "shared", {"text": "new"})
    _assert_item(store.get(("users",), "shared"), ("users",), "shared", control)
    assert [i.value for i in store.search(("users",), query="control")] == [control]
    if delete:
        assert store.get(("in",), "shared") is None
        assert store.search(("in",), query="old") == []
    else:
        assert [i.value for i in store.search(("in",), query="new")] == [
            {"text": "new"}
        ]


@pytest.mark.parametrize(("existing", "delete"), WRITE_CASES)
async def test_async_stopword_writes_preserve_other_namespaces(
    astore: AsyncRedisStore, existing: bool, delete: bool
) -> None:
    if existing:
        await astore.aput(("in",), "shared", {"text": "old"})
    control = {"text": "control"}
    await astore.aput(("users",), "shared", control)
    if delete:
        await astore.adelete(("in",), "shared")
    else:
        await astore.aput(("in",), "shared", {"text": "new"})
    _assert_item(await astore.aget(("users",), "shared"), ("users",), "shared", control)
    assert [i.value for i in await astore.asearch(("users",), query="control")] == [
        control
    ]
    if delete:
        assert await astore.aget(("in",), "shared") is None
        assert await astore.asearch(("in",), query="old") == []
    else:
        assert [i.value for i in await astore.asearch(("in",), query="new")] == [
            {"text": "new"}
        ]
