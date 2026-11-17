"""Test for issue #237 - get() must find every item that search() finds.

``get()`` built its query with redisvl's ``Text`` and ``Tag`` filters. ``Text``
turned a double quote in a namespace label into a space, and ``Tag`` left a
newline in a key unescaped, so ``get()`` returned ``None`` for items that
``put()`` had stored and ``search()`` could find.

``delete()`` and the update path of ``put()`` had the same problem with keys,
and with ``|`` and backticks too: ``delete()`` left the item in place or removed
a different key, and ``put()`` kept the old document next to the new one or
removed a different key.
"""

import pytest
from langgraph.store.base import GetOp

from langgraph.store.redis import AsyncRedisStore, RedisStore
from langgraph.store.redis._redisvl_compat import _tag_escaper

ITEMS = [
    pytest.param(("issue237", 'q"t'), "k", id="double-quote-in-namespace"),
    pytest.param(("issue237", "newline"), "k\n1", id="newline-in-key"),
    pytest.param(("issue237", "carriage-return"), "k\r1", id="carriage-return-in-key"),
    pytest.param(("issue237", "tab"), "k\t1", id="tab-in-key"),
    pytest.param(("issue237", "vertical-tab"), "k\x0b1", id="vertical-tab-in-key"),
    pytest.param(("issue237", "form-feed"), "k\x0c1", id="form-feed-in-key"),
    pytest.param(("issue237", "pipe"), "k|1", id="pipe-in-key"),
    pytest.param(("issue237", "backtick"), "k`1", id="backtick-in-key"),
    # Whitespace at the ends of a key stays unescaped; see _redisvl_compat.
    pytest.param(("issue237", "leading"), "\tk", id="leading-tab-in-key"),
    pytest.param(("issue237", "trailing"), "k\n", id="trailing-newline-in-key"),
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


def test_batched_get_with_several_keys(redis_url) -> None:
    namespace = ("issue237", "batch")
    keys = ["plain", "new\nline", "pi|pe"]
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        for key in keys:
            store.put(namespace, key, {"key": key})

        items = store.batch([GetOp(namespace, key) for key in keys])

        assert [item.value["key"] for item in items] == keys


def test_get_stays_within_the_exact_namespace(redis_url) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(("issue237", "parent", "child"), "k", {"v": 1})

        assert store.get(("issue237", "parent"), "k") is None


# The five ASCII whitespace characters, other than the space, that the query
# parser reads as a term break inside a tag value.
WHITESPACE = [
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param("\r", id="carriage-return"),
    pytest.param("\x0b", id="vertical-tab"),
    pytest.param("\x0c", id="form-feed"),
]


@pytest.mark.parametrize(
    "template,expected",
    [
        pytest.param("{ws}key", "{ws}key", id="at-start-stays-raw"),
        pytest.param("k{ws}ey", "k\\{ws}ey", id="in-middle-is-escaped"),
        pytest.param("key{ws}", "key{ws}", id="at-end-stays-raw"),
    ],
)
@pytest.mark.parametrize("ws", WHITESPACE)
def test_tag_escaper_whitespace_by_position(ws, template, expected) -> None:
    assert _tag_escaper.escape(template.format(ws=ws)) == expected.format(ws=ws)


# A key, and another key in the same namespace that the key's tag query could
# reach if a character in the first one were left unescaped.
COLLIDING_KEYS = [
    pytest.param("a|b", "a", id="pipe-vs-left-alternative"),
    pytest.param("a|b", "b", id="pipe-vs-right-alternative"),
    pytest.param("a`b", "a b", id="backtick-vs-space"),
    pytest.param("a\tb", "a b", id="tab-vs-space"),
    pytest.param("a\nb", "a b", id="newline-vs-space"),
    pytest.param("a\rb", "a b", id="carriage-return-vs-space"),
    pytest.param("a\x0bb", "a b", id="vertical-tab-vs-space"),
    pytest.param("a\x0cb", "a b", id="form-feed-vs-space"),
    pytest.param("ab*", "abc", id="asterisk-vs-prefix-match"),
    pytest.param("a?", "a", id="question-mark-vs-shorter-key"),
]


def _expected_after(operation, key, other) -> list:
    if operation == "delete":
        return [(other, other)]
    return sorted([(key, "updated"), (other, other)])


@pytest.mark.parametrize("operation", ["update", "delete"])
@pytest.mark.parametrize("key,other", COLLIDING_KEYS)
def test_sync_store_mutation_leaves_colliding_key_alone(
    redis_url, key, other, operation
) -> None:
    namespace = ("issue237", "isolation")
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(namespace, key, {"v": key})
        store.put(namespace, other, {"v": other})

        if operation == "update":
            store.put(namespace, key, {"v": "updated"})
        else:
            store.delete(namespace, key)

        results = store.search(namespace)
        stored = sorted((item.key, item.value["v"]) for item in results)
        assert stored == _expected_after(operation, key, other)


@pytest.mark.parametrize("operation", ["update", "delete"])
@pytest.mark.parametrize("key,other", COLLIDING_KEYS)
async def test_async_store_mutation_leaves_colliding_key_alone(
    redis_url, key, other, operation
) -> None:
    namespace = ("issue237", "isolation")
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()
        await store.aput(namespace, key, {"v": key})
        await store.aput(namespace, other, {"v": other})

        if operation == "update":
            await store.aput(namespace, key, {"v": "updated"})
        else:
            await store.adelete(namespace, key)

        results = await store.asearch(namespace)
        stored = sorted((item.key, item.value["v"]) for item in results)
        assert stored == _expected_after(operation, key, other)
