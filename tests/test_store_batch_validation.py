"""RedisStore / AsyncRedisStore check every op in a batch before any op runs.

A batch deletes the old documents of each written key op by op and writes the new
ones at the end. An op whose key or namespace the store's search queries can't
match would fail part-way through, or match other items, so it is rejected with
ValueError before the batch sends anything to Redis: a rejected batch leaves every
stored item as it was.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.store.base import (
    GetOp,
    InvalidNamespaceError,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)

from langgraph.store.redis import AsyncRedisStore, RedisStore

NS = ("users", "u1")
ORIGINAL = {"v": "original"}
BAD = {"v": "bad"}

NOTHING_TO_INDEX = "has nothing Redis can index"
CONTROL_CHARACTER = "contains the control character"
SYNTAX_ERROR = "is a Redis search syntax error"
NOT_A_STRING = "Store keys must be strings"


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


def _stored(store: RedisStore) -> list[tuple[str, dict]]:
    return [(item.key, item.value) for item in store.search(NS)]


async def _astored(astore: AsyncRedisStore) -> list[tuple[str, dict]]:
    return [(item.key, item.value) for item in await astore.asearch(NS)]


# The op that comes first in the batch, on a key that is already stored.
FIRST_OPS = [
    pytest.param(PutOp(NS, "valid", {"v": "new"}), id="update"),
    pytest.param(PutOp(NS, "valid", None), id="delete"),
]

# Keys a put or delete can't match, with the error each one raises.
REJECTED_KEYS = [
    # Redis indexes no tag for these.
    pytest.param("", NOTHING_TO_INDEX, id="empty"),
    pytest.param(" ", NOTHING_TO_INDEX, id="space"),
    pytest.param(",", NOTHING_TO_INDEX, id="comma"),
    pytest.param(" , ", NOTHING_TO_INDEX, id="comma-and-spaces"),
    pytest.param("\t", NOTHING_TO_INDEX, id="tab"),
    pytest.param("\n", NOTHING_TO_INDEX, id="newline"),
    pytest.param("\r", NOTHING_TO_INDEX, id="carriage-return"),
    pytest.param("\x0b", NOTHING_TO_INDEX, id="vertical-tab"),
    pytest.param("\x0c", NOTHING_TO_INDEX, id="form-feed"),
    # The query for these misses the key, or is a syntax error.
    pytest.param("\x00", CONTROL_CHARACTER, id="nul"),
    pytest.param("a\x01b", CONTROL_CHARACTER, id="inner-U+0001"),
    pytest.param("a\x1f", CONTROL_CHARACTER, id="trailing-U+001F"),
    pytest.param("\x7fa", CONTROL_CHARACTER, id="leading-DEL"),
    pytest.param("_", SYNTAX_ERROR, id="underscore"),
    # The query drops the tab, line feed, CR, VT or FF around the "_".
    pytest.param("_\t", SYNTAX_ERROR, id="underscore-tab"),
    pytest.param("\t_", SYNTAX_ERROR, id="tab-underscore"),
    pytest.param("_\n", SYNTAX_ERROR, id="underscore-newline"),
    pytest.param("\n_", SYNTAX_ERROR, id="newline-underscore"),
    pytest.param("_\r", SYNTAX_ERROR, id="underscore-carriage-return"),
    pytest.param("\x0b_\x0c", SYNTAX_ERROR, id="vertical-tab-underscore-form-feed"),
    pytest.param("\t\n_\t", SYNTAX_ERROR, id="mixed-whitespace-underscore"),
    pytest.param(7, NOT_A_STRING, id="int"),
    pytest.param(b"k", NOT_A_STRING, id="bytes"),
]

# The same keys without the errors, for gets.
UNSTORABLE_KEYS = [pytest.param(p.values[0], id=p.id) for p in REJECTED_KEYS]

# Ops whose namespace the store can't query, with the error each one raises.
REJECTED_NAMESPACE_OPS = [
    pytest.param(PutOp((), "x", BAD), "Namespace cannot be empty", id="put-empty"),
    pytest.param(GetOp((), "x"), "Namespace cannot be empty", id="get-empty"),
    # ("users.u1",) would match the item stored under ("users", "u1").
    pytest.param(PutOp(("users.u1",), "valid", BAD), "periods", id="put-period"),
    pytest.param(PutOp(("users.u1",), "valid", None), "periods", id="delete-period"),
    pytest.param(GetOp(("users.u1",), "valid"), "periods", id="get-period"),
    pytest.param(SearchOp(("users.u1",)), "periods", id="search-period"),
    pytest.param(
        ListNamespacesOp(match_conditions=(MatchCondition("prefix", ("users.u1",)),)),
        "periods",
        id="list-namespaces-period",
    ),
    pytest.param(PutOp(("users", ""), "k", BAD), "empty strings", id="put-empty-label"),
    pytest.param(SearchOp(("users", "")), "empty strings", id="search-empty-label"),
    pytest.param(PutOp(("users", 1), "k", BAD), "must be strings", id="put-int-label"),
    pytest.param(PutOp(("users", "u\x00"), "k", BAD), "control", id="put-nul-label"),
    pytest.param(PutOp(("users", "a\tb"), "k", None), "control", id="delete-tab-label"),
    pytest.param(GetOp(("users", "a\nb"), "k"), "control", id="get-newline-label"),
    pytest.param(SearchOp(("users", "\x7f")), "control", id="search-del-label"),
]

# Calls that AsyncBatchedBaseStore could merge into the same abatch() as an aput()
# made at the same time, with the error each one raises.
INVALID_CALLS = [
    pytest.param(lambda s: s.aput(NS, " ", BAD), ValueError, id="aput-space-key"),
    pytest.param(lambda s: s.aput(NS, 7, BAD), ValueError, id="aput-int-key"),
    pytest.param(lambda s: s.adelete(NS, 7), ValueError, id="adelete-int-key"),
    pytest.param(
        lambda s: s.adelete(("users.u1",), "valid"),
        InvalidNamespaceError,
        id="adelete-period",
    ),
    pytest.param(
        lambda s: s.aget(("users.u1",), "valid"),
        InvalidNamespaceError,
        id="aget-period",
    ),
    pytest.param(
        lambda s: s.asearch(("users", "a\nb")),
        InvalidNamespaceError,
        id="asearch-control-character",
    ),
    pytest.param(
        lambda s: s.alist_namespaces(prefix=("users.u1",)),
        InvalidNamespaceError,
        id="alist-namespaces-period",
    ),
]

# Unusual keys and labels that Redis indexes and finds, so they stay allowed.
ACCEPTED = [
    pytest.param(NS, "first name", id="key-with-space"),
    pytest.param(NS, "_id", id="key-starting-with-underscore"),
    pytest.param(NS, "\xa0", id="key-no-break-space"),
    pytest.param(("users", "first name"), "k", id="label-with-space"),
    pytest.param(("users", "café"), "k", id="label-non-ascii"),
]


@pytest.mark.parametrize("value", [BAD, None], ids=["put", "delete"])
@pytest.mark.parametrize("key, error", REJECTED_KEYS)
@pytest.mark.parametrize("first_op", FIRST_OPS)
def test_rejected_key_leaves_stored_value_intact(
    store: RedisStore, first_op: PutOp, key: object, error: str, value: dict | None
) -> None:
    store.put(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=error):
        store.batch([first_op, PutOp(NS, key, value)])

    assert _stored(store) == [("valid", ORIGINAL)]


@pytest.mark.parametrize("rejected_op, error", REJECTED_NAMESPACE_OPS)
@pytest.mark.parametrize("first_op", FIRST_OPS)
def test_rejected_namespace_leaves_stored_value_intact(
    store: RedisStore, first_op: PutOp, rejected_op: object, error: str
) -> None:
    store.put(NS, "valid", ORIGINAL)

    with pytest.raises(InvalidNamespaceError, match=error):
        store.batch([first_op, rejected_op])

    assert _stored(store) == [("valid", ORIGINAL)]


@pytest.mark.parametrize("key", UNSTORABLE_KEYS)
def test_get_of_rejected_key_finds_nothing(store: RedisStore, key: object) -> None:
    store.put(NS, "valid", ORIGINAL)

    assert store.get(NS, key) is None
    missing, found = store.batch([GetOp(NS, key), GetOp(NS, "valid")])

    assert missing is None
    assert found is not None and found.value == ORIGINAL


def test_put_with_empty_key_raises(store: RedisStore) -> None:
    with pytest.raises(ValueError, match=NOTHING_TO_INDEX):
        store.put(NS, "", BAD)

    assert _stored(store) == []


def test_delete_with_empty_key_raises(store: RedisStore) -> None:
    store.put(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=NOTHING_TO_INDEX):
        store.delete(NS, "")

    assert _stored(store) == [("valid", ORIGINAL)]


def test_valid_batch_still_writes_and_deletes(store: RedisStore) -> None:
    store.put(NS, "updated", ORIGINAL)
    store.put(NS, "deleted", ORIGINAL)

    results = store.batch(
        [
            PutOp(NS, "updated", {"v": "new"}),
            PutOp(NS, "added", {"v": "added"}),
            PutOp(NS, "deleted", None),
        ]
    )

    assert results == [None, None, None]
    assert sorted(_stored(store)) == [
        ("added", {"v": "added"}),
        ("updated", {"v": "new"}),
    ]


def test_whole_store_search_and_listing_still_work(store: RedisStore) -> None:
    store.put(NS, "valid", ORIGINAL)

    found, namespaces = store.batch([SearchOp(()), ListNamespacesOp()])

    assert [(item.namespace, item.key) for item in found] == [(NS, "valid")]
    assert namespaces == [NS]


@pytest.mark.parametrize("namespace, key", ACCEPTED)
def test_unusual_keys_and_labels_still_work(
    store: RedisStore, namespace: tuple[str, ...], key: str
) -> None:
    store.put(namespace, key, ORIGINAL)
    item = store.get(namespace, key)
    assert item is not None and item.value == ORIGINAL

    store.delete(namespace, key)
    assert store.get(namespace, key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [BAD, None], ids=["put", "delete"])
@pytest.mark.parametrize("key, error", REJECTED_KEYS)
@pytest.mark.parametrize("first_op", FIRST_OPS)
async def test_rejected_key_leaves_stored_value_intact_async(
    astore: AsyncRedisStore,
    first_op: PutOp,
    key: object,
    error: str,
    value: dict | None,
) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=error):
        await astore.abatch([first_op, PutOp(NS, key, value)])

    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_op, error", REJECTED_NAMESPACE_OPS)
@pytest.mark.parametrize("first_op", FIRST_OPS)
async def test_rejected_namespace_leaves_stored_value_intact_async(
    astore: AsyncRedisStore, first_op: PutOp, rejected_op: object, error: str
) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    with pytest.raises(InvalidNamespaceError, match=error):
        await astore.abatch([first_op, rejected_op])

    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_call, error", INVALID_CALLS)
async def test_invalid_call_does_not_fail_a_concurrent_aput(
    astore: AsyncRedisStore, invalid_call, error: type[Exception]
) -> None:
    # Started together, the two calls would be merged into one abatch().
    results = await asyncio.gather(
        astore.aput(NS, "valid", ORIGINAL),
        invalid_call(astore),
        return_exceptions=True,
    )

    assert results[0] is None
    assert isinstance(results[1], error)
    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", UNSTORABLE_KEYS)
async def test_aget_of_rejected_key_does_not_fail_a_concurrent_aput(
    astore: AsyncRedisStore, key: object
) -> None:
    # Started together, the two calls are merged into one abatch().
    results = await asyncio.gather(
        astore.aput(NS, "valid", ORIGINAL),
        astore.aget(NS, key),
        return_exceptions=True,
    )

    assert results == [None, None]
    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
async def test_aput_with_empty_key_raises(astore: AsyncRedisStore) -> None:
    with pytest.raises(ValueError, match=NOTHING_TO_INDEX):
        await astore.aput(NS, "", BAD)

    assert await _astored(astore) == []


@pytest.mark.asyncio
async def test_adelete_with_empty_key_raises(astore: AsyncRedisStore) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=NOTHING_TO_INDEX):
        await astore.adelete(NS, "")

    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
async def test_valid_abatch_still_writes_and_deletes(astore: AsyncRedisStore) -> None:
    await astore.aput(NS, "updated", ORIGINAL)
    await astore.aput(NS, "deleted", ORIGINAL)

    results = await astore.abatch(
        [
            PutOp(NS, "updated", {"v": "new"}),
            PutOp(NS, "added", {"v": "added"}),
            PutOp(NS, "deleted", None),
        ]
    )

    assert results == [None, None, None]
    assert sorted(await _astored(astore)) == [
        ("added", {"v": "added"}),
        ("updated", {"v": "new"}),
    ]


@pytest.mark.asyncio
async def test_whole_store_search_and_listing_still_work_async(
    astore: AsyncRedisStore,
) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    found = await astore.asearch(())
    namespaces = await astore.alist_namespaces(prefix=("users",))

    assert [(item.namespace, item.key) for item in found] == [(NS, "valid")]
    assert namespaces == [NS]


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace, key", ACCEPTED)
async def test_unusual_keys_and_labels_still_work_async(
    astore: AsyncRedisStore, namespace: tuple[str, ...], key: str
) -> None:
    await astore.aput(namespace, key, ORIGINAL)
    item = await astore.aget(namespace, key)
    assert item is not None and item.value == ORIGINAL

    await astore.adelete(namespace, key)
    assert await astore.aget(namespace, key) is None
