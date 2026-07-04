"""Test for issue #196 — store delete must escape RediSearch-special key characters.

The delete branch of ``_prepare_batch_PUT_queries`` / ``_aprepare_batch_PUT_queries``
interpolated the raw item key into the FT.SEARCH query, so ``delete()`` /
``adelete()`` raised ``RedisSearchError: ... SEARCH_SYNTAX Syntax error ...`` for
any key containing characters RediSearch treats as query syntax (``-``, ``.``,
``#``, ...). ``put()`` and ``get()`` accept such keys — filename-derived keys like
``my-doc.md#0`` could be written but never deleted.
"""

from langgraph.store.redis import AsyncRedisStore, RedisStore

NAMESPACE = ("issue196",)

SPECIAL_KEYS = [
    "my-doc.md#0",  # filename-derived key from the issue report
    "a-b",  # '-' is the negation operator
    "a.b",  # '.' is a token separator
    "a#1",  # '#' breaks tag parsing
    "user@example.com",  # '@' starts a field filter
]


def test_sync_store_delete_key_with_special_chars(redis_url: str) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        for key in SPECIAL_KEYS:
            store.put(NAMESPACE, key, {"text": "hello"})
            assert store.get(NAMESPACE, key) is not None
            store.delete(NAMESPACE, key)  # raised RedisSearchError before the fix
            assert store.get(NAMESPACE, key) is None


async def test_async_store_adelete_key_with_special_chars(redis_url: str) -> None:
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()
        for key in SPECIAL_KEYS:
            await store.aput(NAMESPACE, key, {"text": "hello"})
            assert await store.aget(NAMESPACE, key) is not None
            await store.adelete(
                NAMESPACE, key
            )  # raised RedisSearchError before the fix
            assert await store.aget(NAMESPACE, key) is None
