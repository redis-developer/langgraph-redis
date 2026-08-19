"""
Regression tests for orphaned RediSearch index entries caused by best-effort
TTL application.

Root cause:
`_apply_ttl_to_keys` (and the equivalent bespoke calls in key_registry.py,
shallow.py, ashallow.py) applied EXPIRE to each checkpoint-related key
individually, each wrapped in a single try/except that only logged a warning
on failure (see PR #174). A single transient failure (e.g. a MOVED error on
a Redis Enterprise proxy) permanently left that one key without a TTL. Since
checkpoint and checkpoint_write documents are RediSearch-indexed JSON keys, a
key that never got its TTL applied never expires and never leaves the search
index — an orphaned index entry.

These tests verify:
1. A single transient EXPIRE failure no longer permanently strands a key
   without a TTL (retry recovers it).
2. After natural expiry, RediSearch has zero indexed entries left over for
   the expired thread (the actual "leftover index" regression).
3. The `checkpoint_latest:*` pointer key's TTL stays in sync with the
   checkpoint it points to under `refresh_on_read`.
"""

import time
from typing import Any
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from redis.exceptions import ResponseError
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.base import aexpire_with_retry, expire_with_retry


def _make_checkpoint() -> Any:
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": ["test"]}, step=1
    )
    checkpoint["channel_values"]["messages"] = ["test"]
    return checkpoint


def _flaky_once(real_method: Any) -> Any:
    """Wrap a bound Redis method so its first call raises, subsequent calls pass through."""
    calls = {"count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ResponseError("MOVED 12345 host:port")
        return real_method(*args, **kwargs)

    return wrapper


def _aflaky_once(real_method: Any) -> Any:
    calls = {"count": 0}

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ResponseError("MOVED 12345 host:port")
        return await real_method(*args, **kwargs)

    return wrapper


# ── Unit-level retry helper tests ───────────────────────────────────────────


def test_expire_with_retry_recovers_from_one_transient_failure(
    redis_url: str,
) -> None:
    """A single transient EXPIRE failure is retried and ultimately succeeds."""
    from redis import Redis

    client = Redis.from_url(redis_url)
    try:
        key = f"ttl-retry-test-{uuid4()}"
        client.set(key, "value")

        real_expire = client.expire
        client.expire = _flaky_once(real_expire)  # type: ignore[method-assign]
        try:
            result = expire_with_retry(client, key, 60)
        finally:
            client.expire = real_expire  # type: ignore[method-assign]

        assert result is True
        ttl = client.ttl(key)
        assert ttl > 0, f"Key should have a TTL after retry succeeded, got {ttl}"
    finally:
        client.close()


@pytest.mark.asyncio
async def test_aexpire_with_retry_recovers_from_one_transient_failure(
    redis_url: str,
) -> None:
    """Async: a single transient EXPIRE failure is retried and ultimately succeeds."""
    from redis.asyncio import Redis as AsyncRedis

    client = AsyncRedis.from_url(redis_url)
    try:
        key = f"ttl-retry-test-async-{uuid4()}"
        await client.set(key, "value")

        real_expire = client.expire
        client.expire = _aflaky_once(real_expire)  # type: ignore[method-assign]
        try:
            result = await aexpire_with_retry(client, key, 60)
        finally:
            client.expire = real_expire  # type: ignore[method-assign]

        assert result is True
        ttl = await client.ttl(key)
        assert ttl > 0, f"Key should have a TTL after retry succeeded, got {ttl}"
    finally:
        await client.aclose()


def test_expire_with_retry_gives_up_without_raising(redis_url: str) -> None:
    """Best-effort contract: exhausted retries log and return False, never raise."""
    from redis import Redis

    client = Redis.from_url(redis_url)
    try:
        key = f"ttl-retry-exhausted-{uuid4()}"
        client.set(key, "value")

        def always_fails(*args: Any, **kwargs: Any) -> Any:
            raise ResponseError("MOVED 12345 host:port")

        real_expire = client.expire
        client.expire = always_fails  # type: ignore[method-assign]
        try:
            result = expire_with_retry(client, key, 60, attempts=2, base_delay=0.01)
        finally:
            client.expire = real_expire  # type: ignore[method-assign]

        assert result is False
    finally:
        client.close()


# ── Integration: writes/checkpoints survive a transient EXPIRE failure ─────


def test_write_key_gets_ttl_despite_one_transient_expire_failure(
    redis_url: str,
) -> None:
    """
    Reproduces the orphaned-index bug directly: before the fix, a single
    failed EXPIRE permanently left a checkpoint_write key without a TTL,
    so it never expired and never left the RediSearch index.
    """
    with RedisSaver.from_conn_string(redis_url, ttl={"default_ttl": 1}) as saver:
        saver.setup()
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = _make_checkpoint()

        saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

        real_expire = saver._redis.expire
        saver._redis.expire = _flaky_once(real_expire)  # type: ignore[method-assign]
        try:
            saver.put_writes(saved_config, [("messages", "write-1")], "task-1")
        finally:
            saver._redis.expire = real_expire  # type: ignore[method-assign]

        results = saver.checkpoint_writes_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        assert len(results.docs) > 0, "Write should have been created"
        for doc in results.docs:
            ttl = saver._redis.ttl(doc.id)
            assert ttl > 0, (
                f"Write key {doc.id} should have a TTL despite one transient "
                f"EXPIRE failure — otherwise it never expires and lingers in "
                f"the search index forever (got ttl={ttl})"
            )


@pytest.mark.asyncio
async def test_async_write_key_gets_ttl_despite_one_transient_expire_failure(
    redis_url: str,
) -> None:
    """Async counterpart of the write-key retry regression test."""
    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": 1}
    ) as saver:
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = _make_checkpoint()

        saved_config = await saver.aput(
            config, checkpoint, {"source": "test", "step": 1}, {}
        )

        real_expire = saver._redis.expire
        saver._redis.expire = _aflaky_once(real_expire)  # type: ignore[method-assign]
        try:
            await saver.aput_writes(saved_config, [("messages", "write-1")], "task-1")
        finally:
            saver._redis.expire = real_expire  # type: ignore[method-assign]

        results = await saver.checkpoint_writes_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        assert len(results.docs) > 0, "Write should have been created"
        for doc in results.docs:
            ttl = await saver._redis.ttl(doc.id)
            assert ttl > 0, (
                f"Write key {doc.id} should have a TTL despite one transient "
                f"EXPIRE failure (got ttl={ttl})"
            )


# ── Integration: search index has no orphans after natural expiry ──────────


def test_search_index_has_no_orphans_after_expiry(redis_url: str) -> None:
    """
    No existing test asserted FT.SEARCH state after expiry — only key
    existence / get_tuple(). This closes that gap: once a checkpoint and
    its writes naturally expire, both indexes must report zero hits.
    """
    with RedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": 0.02}  # ~1 second
    ) as saver:
        saver.setup()
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        checkpoint = _make_checkpoint()

        saved_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})
        saver.put_writes(saved_config, [("messages", "write-1")], "task-1")

        checkpoint_hits = saver.checkpoints_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        write_hits = saver.checkpoint_writes_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        assert len(checkpoint_hits.docs) > 0
        assert len(write_hits.docs) > 0

        time.sleep(3)

        checkpoint_hits_after = saver.checkpoints_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        write_hits_after = saver.checkpoint_writes_index.search(
            FilterQuery(Tag("thread_id") == thread_id)
        )
        assert len(checkpoint_hits_after.docs) == 0, (
            "Checkpoint document should have left the search index after "
            "expiry, but is still indexed"
        )
        assert len(write_hits_after.docs) == 0, (
            "Write document(s) should have left the search index after "
            "expiry, but are still indexed"
        )


# ── Integration: pointer key TTL stays in sync with the checkpoint ─────────


def test_latest_pointer_ttl_refreshed_alongside_checkpoint(redis_url: str) -> None:
    """
    The checkpoint_latest:* pointer key must be refreshed alongside the
    checkpoint it points to when refresh_on_read is enabled, otherwise it
    can expire independently of the (still-alive) data it references.
    """
    with RedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": 2, "refresh_on_read": True}
    ) as saver:
        saver.setup()
        thread_id = str(uuid4())
        checkpoint_ns = ""
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = _make_checkpoint()

        saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

        pointer_key = saver._make_redis_checkpoint_latest_key(thread_id, checkpoint_ns)
        initial_pointer_ttl = saver._redis.ttl(pointer_key)
        assert initial_pointer_ttl > 0

        time.sleep(1.5)

        # Read via the latest-checkpoint pointer (no explicit checkpoint_id).
        result = saver.get_tuple(config)
        assert result is not None

        refreshed_pointer_ttl = saver._redis.ttl(pointer_key)
        assert refreshed_pointer_ttl > initial_pointer_ttl - 1, (
            "Pointer key TTL should be refreshed on read alongside the "
            f"checkpoint it points to (initial={initial_pointer_ttl}, "
            f"after read={refreshed_pointer_ttl})"
        )


@pytest.mark.asyncio
async def test_async_latest_pointer_ttl_refreshed_alongside_checkpoint(
    redis_url: str,
) -> None:
    """Async counterpart of the pointer-TTL-sync regression test."""
    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": 2, "refresh_on_read": True}
    ) as saver:
        thread_id = str(uuid4())
        checkpoint_ns = ""
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        }
        checkpoint = _make_checkpoint()

        await saver.aput(config, checkpoint, {"source": "test", "step": 1}, {})

        pointer_key = saver._make_redis_checkpoint_latest_key(thread_id, checkpoint_ns)
        initial_pointer_ttl = await saver._redis.ttl(pointer_key)
        assert initial_pointer_ttl > 0

        time.sleep(1.5)

        result = await saver.aget_tuple(config)
        assert result is not None

        refreshed_pointer_ttl = await saver._redis.ttl(pointer_key)
        assert refreshed_pointer_ttl > initial_pointer_ttl - 1, (
            "Pointer key TTL should be refreshed on read alongside the "
            f"checkpoint it points to (initial={initial_pointer_ttl}, "
            f"after read={refreshed_pointer_ttl})"
        )
