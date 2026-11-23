"""RedisStore / AsyncRedisStore batches with an empty key or a malformed document.

Redis doesn't index an empty tag, so a put or delete of "" can't be written or
found. A batch writes op by op and deletes each key's old documents first, so it
is checked before any op reaches Redis: a rejected batch leaves every stored item
as it was.
"""

from __future__ import annotations

import pytest
from langgraph.store.base import PutOp

from langgraph.store.redis import AsyncRedisStore, RedisStore

NS = ("users", "u1")
ORIGINAL = {"v": "original"}
EMPTY_KEY_ERROR = "Store keys cannot be empty strings"


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


# The op that comes first in the batch, on a key that is already stored.
VALID_OPS = [
    pytest.param(PutOp(NS, "valid", {"v": "new"}), id="update"),
    pytest.param(PutOp(NS, "valid", None), id="delete"),
]

# The op that follows it, with the error the batch must raise.
REJECTED_OPS = [
    pytest.param(PutOp(NS, "", {"v": "bad"}), EMPTY_KEY_ERROR, id="empty-key-put"),
    pytest.param(PutOp(NS, "", None), EMPTY_KEY_ERROR, id="empty-key-delete"),
    # RedisVL's schema validation: the key field must be a string.
    pytest.param(PutOp(NS, 7, {"v": "bad"}), "valid string", id="non-string-key-put"),
]


def _stored(store: RedisStore) -> list[tuple[str, dict]]:
    return [(item.key, item.value) for item in store.search(NS)]


async def _astored(astore: AsyncRedisStore) -> list[tuple[str, dict]]:
    return [(item.key, item.value) for item in await astore.asearch(NS)]


@pytest.mark.parametrize("rejected_op, error", REJECTED_OPS)
@pytest.mark.parametrize("valid_op", VALID_OPS)
def test_rejected_batch_leaves_stored_value_intact(
    store: RedisStore, valid_op: PutOp, rejected_op: PutOp, error: str
) -> None:
    store.put(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=error):
        store.batch([valid_op, rejected_op])

    assert _stored(store) == [("valid", ORIGINAL)]


def test_put_with_empty_key_raises(store: RedisStore) -> None:
    with pytest.raises(ValueError, match=EMPTY_KEY_ERROR):
        store.put(NS, "", {"v": "bad"})

    assert _stored(store) == []


def test_delete_with_empty_key_raises(store: RedisStore) -> None:
    store.put(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=EMPTY_KEY_ERROR):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_op, error", REJECTED_OPS)
@pytest.mark.parametrize("valid_op", VALID_OPS)
async def test_rejected_abatch_leaves_stored_value_intact(
    astore: AsyncRedisStore, valid_op: PutOp, rejected_op: PutOp, error: str
) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=error):
        await astore.abatch([valid_op, rejected_op])

    assert await _astored(astore) == [("valid", ORIGINAL)]


@pytest.mark.asyncio
async def test_aput_with_empty_key_raises(astore: AsyncRedisStore) -> None:
    with pytest.raises(ValueError, match=EMPTY_KEY_ERROR):
        await astore.aput(NS, "", {"v": "bad"})

    assert await _astored(astore) == []


@pytest.mark.asyncio
async def test_adelete_with_empty_key_raises(astore: AsyncRedisStore) -> None:
    await astore.aput(NS, "valid", ORIGINAL)

    with pytest.raises(ValueError, match=EMPTY_KEY_ERROR):
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
