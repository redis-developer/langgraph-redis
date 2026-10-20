"""Each store wraps its own kind of client in its own kind of redisvl index.

BaseRedisStore.__init__ used to build sync ``SearchIndex`` objects on whatever
client it was given — for AsyncRedisStore that is an ``redis.asyncio.Redis`` —
and the async subclass then replaced them with ``AsyncSearchIndex``. redisvl 0.28
started validating the client type in ``SearchIndex.__init__``, so constructing
an AsyncRedisStore with an index config raised "SearchIndex requires a sync
Redis client" before the subclass ever ran. The base now builds the schemas and
defers to ``create_indexes()``; these tests pin the kind of index each store
ends up with. No Redis round-trip is needed: construction is what broke.
"""

from __future__ import annotations

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex, SearchIndex

from langgraph.checkpoint.redis.util import unwrap_connection_errors
from langgraph.store.redis import AsyncRedisStore, RedisStore
from tests.embed_test_utils import CharacterEmbeddings

_INDEX = {"dims": 4, "embed": CharacterEmbeddings(dims=4), "fields": ["text"]}


@pytest.mark.asyncio
async def test_async_store_builds_async_indexes_only(redis_url: str) -> None:
    store = AsyncRedisStore(
        redis_url, index=_INDEX, store_prefix="kind_a", vector_prefix="kind_av"
    )
    assert type(store.store_index) is AsyncSearchIndex
    assert type(store.vector_index) is AsyncSearchIndex
    # The schemas the indexes were built from carry THIS instance's prefixes,
    # not the module-level defaults.
    assert store.store_index.name == "kind_a"
    assert store.vector_index.name == "kind_av"
    assert store.vector_schema is not None
    assert store.vector_schema["fields"][3]["attrs"]["dims"] == 4


@pytest.mark.asyncio
async def test_async_store_without_index_has_no_vector_index(redis_url: str) -> None:
    store = AsyncRedisStore(redis_url, store_prefix="kind_b")
    assert type(store.store_index) is AsyncSearchIndex
    assert store.vector_schema is None
    assert not hasattr(store, "vector_index")


def test_sync_store_builds_sync_indexes(redis_url: str) -> None:
    store = RedisStore(
        redis_url, index=_INDEX, store_prefix="kind_c", vector_prefix="kind_cv"
    )
    assert type(store.store_index) is SearchIndex
    assert type(store.vector_index) is SearchIndex
    assert store.vector_index.name == "kind_cv"


@pytest.mark.asyncio
async def test_async_store_keeps_json_vector_storage(redis_url: str) -> None:
    """The async store has always written vectors as JSON documents, and its
    index storage type was hardcoded "json" before the schemas moved to the
    base. Honouring index["vector_storage_type"]="hash" here would build a
    hash index over JSON data — it would index nothing — and would break
    existing async users who passed "hash" and got JSON anyway."""
    store = AsyncRedisStore(
        redis_url,
        index={**_INDEX, "vector_storage_type": "hash"},  # type: ignore[typeddict-unknown-key]
        store_prefix="kind_f",
        vector_prefix="kind_fv",
    )
    assert store.vector_schema is not None
    assert store.vector_schema["index"]["storage_type"] == "json"


def test_sync_store_still_honours_the_configured_vector_storage(
    redis_url: str,
) -> None:
    """Unchanged by the async pin above: the sync store has a hash write path
    (`_batch_put_ops` buffers the vector), so "hash" means hash there."""
    store = RedisStore(
        redis_url,
        index={**_INDEX, "vector_storage_type": "hash"},  # type: ignore[typeddict-unknown-key]
        store_prefix="kind_g",
        vector_prefix="kind_gv",
    )
    assert store.vector_schema is not None
    assert store.vector_schema["index"]["storage_type"] == "hash"


@pytest.mark.asyncio
async def test_async_store_setup_on_unreachable_redis_raises_connection_error() -> None:
    """redisvl >= 0.28 wraps the connection failure inside create() in a
    RedisSearchError; setup() keeps surfacing redis-py's ConnectionError."""
    store = AsyncRedisStore("redis://nonexistent:6379", store_prefix="kind_d")
    with pytest.raises(RedisConnectionError):
        await store.setup()


def test_sync_store_setup_on_unreachable_redis_raises_connection_error() -> None:
    # The sync store already probes the server in __init__ (cluster detection),
    # so construction and setup both sit inside the expectation.
    with pytest.raises(RedisConnectionError):
        RedisStore(
            Redis.from_url("redis://nonexistent:6379"), store_prefix="kind_e"
        ).setup()


def test_unwrap_connection_errors_only_unwraps_connection_causes() -> None:
    with pytest.raises(RedisConnectionError):
        with unwrap_connection_errors():
            try:
                raise RedisConnectionError("down")
            except RedisConnectionError as e:
                raise RedisSearchError("wrapped") from e
    # Any other search error — or a search error with another cause — passes through.
    with pytest.raises(RedisSearchError):
        with unwrap_connection_errors():
            try:
                raise ValueError("bad schema")
            except ValueError as e:
                raise RedisSearchError("wrapped") from e
    with pytest.raises(RedisSearchError):
        with unwrap_connection_errors():
            raise RedisSearchError("plain")
