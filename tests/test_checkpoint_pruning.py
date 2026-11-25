"""Integration tests for checkpoint pruning, retention, and input validation.

Tests cover:
- keep_last=1  (keep only the latest checkpoint, default)
- keep_last=N  (interrupt-safe window for multi-tool chains)
- keep_last=0  (remove all checkpoints)
- per-namespace isolation (subgraph chains pruned independently)
- empty thread is a no-op
- other threads are untouched
- associated writes are removed for evicted checkpoints
- sync (RedisSaver.prune) and async (AsyncRedisSaver.aprune) variants
- shallow savers: keep_last>=1 is no-op, keep_last=0 deletes all
- invalid strategies raise without changing stored data (issue #238)
- incomplete checkpoint and write searches are rejected before deleting thread data
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from redis import Redis
from redisvl.query import CountQuery, FilterQuery
from redisvl.query.filter import Tag
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from langgraph.checkpoint.redis.util import safely_decode

INVALID_STRATEGIES = ("keep_last", "Delete", "KEEP_LATEST", "zzz", "", "keep_latest ")
PRUNE_SEARCH_CASES = (
    # Strategy, keep_last, retained per namespace, offset from the complete limit.
    pytest.param("keep_latest", None, 1, -1, id="reject-partial-retention"),
    pytest.param("delete", None, 0, -1, id="reject-partial-deletion"),
    pytest.param("delete", 2, 2, -1, id="reject-partial-override"),
    pytest.param("delete", None, 0, 1, id="above-limit"),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ulid() -> str:
    """Return a time-ordered ULID string, with a small sleep to guarantee ordering."""
    time.sleep(0.002)  # 2 ms gap — enough for distinct millisecond timestamps
    return str(ULID())


def _make_checkpoint(cp_id: str, channel_value: str = "data") -> Checkpoint:
    return Checkpoint(
        v=1,
        id=cp_id,
        ts="2024-01-01T00:00:00Z",
        channel_values={"messages": [channel_value]},
        channel_versions={"messages": "1"},
        versions_seen={"agent": {"messages": "1"}},
        pending_sends=[],
        tasks=[],
    )


def _config(thread_id: str, cp_id: str, ns: str = "") -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ns,
            "checkpoint_id": cp_id,
        }
    }


def _thread_data(client: Redis, thread_id: str) -> dict[bytes, Optional[bytes]]:
    """Snapshot checkpoints, writes, registries, and pointers without changing TTLs."""
    return {key: client.dump(key) for key in client.scan_iter(match=f"*{thread_id}*")}


@pytest.fixture
def saver(client: Redis) -> Iterator[RedisSaver]:
    """Reuse the Redis client for seeding and inspecting persisted checkpoints."""
    with RedisSaver.from_conn_string(redis_client=client) as saver:
        saver.setup()
        yield saver


def _put_checkpoint_with_writes(
    saver: RedisSaver, config: RunnableConfig, write_count: int = 1
) -> None:
    """Seed the shared storage format used by both pruning APIs."""
    saver.put(
        config,
        _make_checkpoint(config["configurable"]["checkpoint_id"]),
        CheckpointMetadata(source="input", step=0, writes={}),
        {"messages": "1"},
    )
    for write in range(write_count):
        saver.put_writes(config, [("messages", f"write-{write}")], f"task-{write}")


def _get_search_config(client: Redis, option: str) -> str:
    config = dict(client.execute_command("FT.CONFIG", "GET", option))
    return safely_decode(next(iter(config.values()))).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("evicted_ns,retained_ns", [("", "child:0"), ("child:0", "")])
async def test_prune_preserves_writes_for_same_id_in_another_namespace(
    saver: RedisSaver,
    client: Redis,
    prune: Callable[..., Awaitable[None]],
    evicted_ns: str,
    retained_ns: str,
) -> None:
    thread_id = f"prune-namespaces-{_make_ulid()}"
    shared_id, latest_id = _make_ulid(), _make_ulid()
    evicted = _config(thread_id, shared_id, evicted_ns)
    retained = _config(thread_id, shared_id, retained_ns)
    latest = _config(thread_id, latest_id, evicted_ns)
    for config in (evicted, retained, latest):
        _put_checkpoint_with_writes(saver, config)
    before = _thread_data(client, thread_id)
    assert saver._key_registry is not None
    evicted_keys = (
        saver._make_redis_checkpoint_key(thread_id, evicted_ns, shared_id).encode(),
        saver._make_redis_checkpoint_writes_key(
            thread_id, evicted_ns, shared_id, "task-0", 0
        ).encode(),
        saver._key_registry.make_write_keys_zset_key(
            thread_id, evicted_ns, shared_id
        ).encode(),
    )

    await prune([thread_id])

    assert saver.get_tuple(evicted) is None
    for config in (retained, latest):
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.pending_writes == [("task-0", "messages", "write-0")]
    assert _thread_data(client, thread_id) == {
        key: value for key, value in before.items() if key not in evicted_keys
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["return", "unreadable"])
@pytest.mark.parametrize(
    ("strategy", "retained_per_ns"), [("keep_latest", 1), ("delete", 0)]
)
async def test_prune_does_not_depend_on_search_timeout_configuration(
    saver: RedisSaver,
    client: Redis,
    prune: Callable[..., Awaitable[None]],
    policy: str,
    strategy: str,
    retained_per_ns: int,
) -> None:
    """Prune needs no FT.CONFIG access and never changes the timeout policy.

    Redis Cloud and Redis Software do not support FT.CONFIG, and ACL users
    without @admin cannot run it.
    """
    thread_id = f"prune-timeout-policy-{_make_ulid()}"
    checkpoints = [_config(thread_id, _make_ulid()) for _ in range(3)]
    for config in checkpoints:
        _put_checkpoint_with_writes(saver, config, write_count=2)
    before = _thread_data(client, thread_id)
    original = _get_search_config(client, "ON_TIMEOUT")
    try:
        if policy == "unreadable":
            client.execute_command("ACL", "SETUSER", "default", "-ft.config")
        else:
            client.execute_command("FT.CONFIG", "SET", "ON_TIMEOUT", policy)

        await prune([thread_id], strategy=strategy)
    finally:
        if policy == "unreadable":
            client.execute_command("ACL", "SETUSER", "default", "+ft.config")
        policy_after_prune = _get_search_config(client, "ON_TIMEOUT")
        client.execute_command("FT.CONFIG", "SET", "ON_TIMEOUT", original)

    assert _thread_data(client, thread_id) == _expected_pruned_data(
        before, checkpoints, retained_per_ns
    )
    # The saver must never change deployment-wide configuration itself.
    assert policy_after_prune == (original if policy == "unreadable" else policy)


@contextmanager
def _forced_search_timeout(client: Redis, mode: Optional[str]) -> Iterator[None]:
    """Force Search timeouts for one test, then restore the server's settings."""
    if mode is None:
        yield
        return
    if mode == "search-timeout":  # FT.CONFIG works on Redis Stack and Redis 8.
        original: dict[str, Any] = {
            option: _get_search_config(client, option)
            for option in ("TIMEOUT", "ON_TIMEOUT")
        }
        settings: dict[str, Any] = {"TIMEOUT": 1, "ON_TIMEOUT": "RETURN"}

        def apply(name: str, value: Any) -> None:
            client.execute_command("FT.CONFIG", "SET", name, value)

    else:  # foreground-cap: Redis 8.10+ caps every query's timeout when search-workers is 0.
        names = (
            "search-workers",
            "search-_max-foreground-timeout-limit",
            "search-on-timeout",
        )
        original = {name: client.config_get(name).get(name) for name in names}
        if original["search-_max-foreground-timeout-limit"] is None:
            pytest.skip("search-_max-foreground-timeout-limit requires Redis 8.10+")
        settings = dict(zip(names, (0, 1, "return")))

        def apply(name: str, value: Any) -> None:
            client.config_set(name, value)

    try:
        for name, value in settings.items():
            apply(name, value)
        yield
    finally:
        for name, value in original.items():
            apply(name, value)


def _put_checkpoint_with_bulk_writes(
    saver: RedisSaver, config: RunnableConfig, count: int
) -> None:
    """Seed a checkpoint and ``count`` writes in one pipelined call."""
    checkpoint_id = config["configurable"]["checkpoint_id"]
    saver.put(
        config,
        _make_checkpoint(checkpoint_id),
        CheckpointMetadata(source="input", step=0, writes={}),
        {"messages": "1"},
    )
    saver.put_writes(config, [("messages", "v")] * count, "task")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout_mode", "write_count", "decode_responses"),
    [
        # 2,500 writes span three 1,000-row cursor batches.
        pytest.param(None, 2_500, False, id="no-timeout-bytes"),
        pytest.param(None, 2_500, True, id="no-timeout-str"),
        # 50,000 writes reliably exceed a 1 ms timeout on every tested version.
        pytest.param("search-timeout", 50_000, False, id="search-timeout"),
        pytest.param("foreground-cap", 50_000, False, id="foreground-cap"),
    ],
)
@pytest.mark.parametrize("protocol", [2, 3], ids=["resp2", "resp3"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_prune_reads_every_match_through_its_cursor(
    redis_url: str,
    client: Redis,
    saver: RedisSaver,
    use_async: bool,
    protocol: int,
    timeout_mode: Optional[str],
    write_count: int,
    decode_responses: bool,
) -> None:
    """Prune sees every checkpoint and write, even when a single search times out.

    A timed-out FT.SEARCH under ON_TIMEOUT=RETURN reports a partial total that
    matches its returned count, and Redis 8.10+ caps even ``TIMEOUT 0``.
    """
    thread_id = f"prune-cursor-{_make_ulid()}"
    evicted, retained = [], []
    # A large evicted checkpoint, a small one in a subgraph namespace, and a
    # newer retained checkpoint in each namespace.
    for ns, count in (("", write_count), ("parent:1|child:2", 3)):
        old, new = (_config(thread_id, _make_ulid(), ns) for _ in range(2))
        _put_checkpoint_with_bulk_writes(saver, old, count)
        _put_checkpoint_with_bulk_writes(saver, new, 1)
        evicted.append(old["configurable"]["checkpoint_id"])
        retained.append(new)

    with _forced_search_timeout(client, timeout_mode):
        if timeout_mode:
            # Precondition: a single search over the large checkpoint is cut short.
            # Count only (LIMIT 0 0): Redis Stack caps LIMIT at 10,000 rows.
            matched = saver.checkpoint_writes_index.query(
                CountQuery(
                    filter_expression=(Tag("thread_id") == thread_id)
                    & (Tag("checkpoint_id") == evicted[0])
                )
            )
            if matched == write_count:
                pytest.skip("Search finished within 1 ms; cannot exercise a timeout")
        connection_args = {"protocol": protocol, "decode_responses": decode_responses}
        if use_async:
            async with AsyncRedisSaver.from_conn_string(
                redis_url, connection_args=connection_args
            ) as async_pruner:
                await async_pruner.aprune([thread_id], max_results=write_count)
        else:
            with RedisSaver.from_conn_string(
                redis_url, connection_args=connection_args
            ) as pruner:
                pruner.setup()
                pruner.prune([thread_id], max_results=write_count)

    for cp_id in evicted:
        assert list(client.scan_iter(match=f"*{cp_id}*", count=10_000)) == []
    for config in retained:
        checkpoint = saver.get_tuple(config)
        assert checkpoint is not None
        assert checkpoint.pending_writes == [("task", "messages", "v")]


@pytest.fixture(params=["checkpoints", "writes"])
def prune_search_data(
    request: pytest.FixtureRequest, saver: RedisSaver
) -> tuple[str, list[RunnableConfig], int]:
    """Truncate either the global namespace inventory or the last write query."""
    thread_id = f"prune-search-{_make_ulid()}"
    namespaces = (
        [""] * 4 if request.param == "writes" else ["", "", "child:0", "child:0"]
    )
    write_counts = [5, 1, 1, 1] if request.param == "writes" else [1] * 4
    checkpoints = [_config(thread_id, _make_ulid(), ns) for ns in namespaces]
    for config, write_count in zip(checkpoints, write_counts):
        _put_checkpoint_with_writes(saver, config, write_count)
    return thread_id, checkpoints, 5 if request.param == "writes" else 4


@pytest.fixture(params=["sync", "async"])
async def prune(
    request: pytest.FixtureRequest, redis_url: str
) -> AsyncIterator[Callable[..., Awaitable[None]]]:
    """Run the same assertions against either pruning API."""
    if request.param == "async":
        async with AsyncRedisSaver.from_conn_string(
            redis_url, connection_args={"protocol": 2}
        ) as async_saver:
            yield async_saver.aprune
    else:
        with RedisSaver.from_conn_string(
            redis_url, connection_args={"protocol": 2}
        ) as sync_saver:
            sync_saver.setup()

            async def prune_sync(thread_ids: Sequence[str], **kwargs: Any) -> None:
                sync_saver.prune(thread_ids, **kwargs)

            yield prune_sync


@pytest.fixture
async def prune_call(
    request: pytest.FixtureRequest, redis_url: str
) -> AsyncIterator[Callable[..., Awaitable[None]]]:
    saver_cls = request.param
    if saver_cls in (AsyncRedisSaver, AsyncShallowRedisSaver):
        async with saver_cls.from_conn_string(redis_url) as saver:
            yield saver.aprune
    else:
        with saver_cls.from_conn_string(redis_url) as saver:
            saver.setup()

            async def prune_sync(thread_ids: Sequence[str], **kwargs: Any) -> None:
                saver.prune(thread_ids, **kwargs)

            yield prune_sync


def _expected_pruned_data(
    before: dict[bytes, Optional[bytes]],
    checkpoints: list[RunnableConfig],
    retained_per_ns: int,
) -> dict[bytes, Optional[bytes]]:
    if retained_per_ns == 0:
        return {}
    configs = [config["configurable"] for config in checkpoints]
    evicted_ids = []
    for ns in {config["checkpoint_ns"] for config in configs}:
        ids = [
            config["checkpoint_id"].encode()
            for config in configs
            if config["checkpoint_ns"] == ns
        ]
        evicted_ids.extend(ids[:-retained_per_ns])
    return {
        key: value
        for key, value in before.items()
        if not any(cp_id in key for cp_id in evicted_ids)
    }


def _seed_prune_checkpoint(
    saver: RedisSaver | ShallowRedisSaver, thread_id: str, writes: int = 1
) -> None:
    checkpoint_id = _make_ulid()
    config = _config(thread_id, checkpoint_id)
    saver.put(
        config,
        _make_checkpoint(checkpoint_id),
        CheckpointMetadata(source="input", step=0, writes={}),
        {"messages": "1"},
    )
    for write in range(writes):
        saver.put_writes(config, [("messages", f"write-{write}")], f"task-{write}")


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "keep_last"),
    [("keep_latest", None), ("keep_latest", 1), ("delete", 1)],
)
async def test_aprune_keep_last_1_removes_older_checkpoints(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """aprune(keep_last=1) retains only the single most-recent checkpoint."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"prune-keep-1-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
        for cp_id in cp_ids:
            await saver.aput(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        await saver.aprune([thread_id], strategy=strategy, keep_last=keep_last)

        # Only the latest should survive
        result = await saver.aget_tuple(_config(thread_id, cp_ids[-1]))
        assert result is not None, "Latest checkpoint must be retained"

        # Older two must be gone
        for old_cp_id in cp_ids[:-1]:
            result = await saver.aget_tuple(_config(thread_id, old_cp_id))
            assert result is None, f"Checkpoint {old_cp_id} should have been pruned"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["keep_latest", "delete"])
async def test_aprune_keep_last_n_retains_window(redis_url: str, strategy: str) -> None:
    """aprune(keep_last=N) retains the N most-recent checkpoints."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"prune-keep-n-{_make_ulid()}"
        N = 3

        cp_ids = [_make_ulid() for _ in range(5)]
        for cp_id in cp_ids:
            await saver.aput(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        await saver.aprune([thread_id], strategy=strategy, keep_last=N)

        # Last N should survive
        for cp_id in cp_ids[-N:]:
            result = await saver.aget_tuple(_config(thread_id, cp_id))
            assert result is not None, f"Checkpoint {cp_id} should be retained"

        # Earlier ones must be evicted
        for cp_id in cp_ids[:-N]:
            result = await saver.aget_tuple(_config(thread_id, cp_id))
            assert result is None, f"Checkpoint {cp_id} should be pruned"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "keep_last"), [("keep_latest", 0), ("delete", None)]
)
async def test_aprune_keep_last_0_removes_all_checkpoints(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """aprune(keep_last=0) removes every checkpoint for the thread."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"prune-delete-{_make_ulid()}"
        cp_ids = [_make_ulid() for _ in range(4)]

        for cp_id in cp_ids:
            await saver.aput(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        await saver.aprune([thread_id], strategy=strategy, keep_last=keep_last)

        for cp_id in cp_ids:
            result = await saver.aget_tuple(_config(thread_id, cp_id))
            assert result is None, f"Checkpoint {cp_id} should have been deleted"


@pytest.mark.asyncio
async def test_aprune_per_namespace_isolation(redis_url: str) -> None:
    """keep_last is applied independently per namespace, not globally.

    A thread with a root namespace and a subgraph namespace should have
    keep_last applied to each chain separately — not ranked together.
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"prune-ns-{_make_ulid()}"
        root_ns = ""
        sub_ns = "subgraph:0"

        # 3 checkpoints in root namespace, 3 in subgraph namespace
        root_ids = [_make_ulid() for _ in range(3)]
        sub_ids = [_make_ulid() for _ in range(3)]

        for cp_id in root_ids:
            await saver.aput(
                config=_config(thread_id, cp_id, root_ns),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )
        for cp_id in sub_ids:
            await saver.aput(
                config=_config(thread_id, cp_id, sub_ns),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        # Prune to keep_last=1 — should keep the latest in EACH namespace
        await saver.aprune([thread_id], keep_last=1)

        # Latest in root namespace must survive
        assert (
            await saver.aget_tuple(_config(thread_id, root_ids[-1], root_ns))
            is not None
        )
        # Older root checkpoints evicted
        for cp_id in root_ids[:-1]:
            assert await saver.aget_tuple(_config(thread_id, cp_id, root_ns)) is None

        # Latest in subgraph namespace must also survive
        assert (
            await saver.aget_tuple(_config(thread_id, sub_ids[-1], sub_ns)) is not None
        )
        # Older subgraph checkpoints evicted
        for cp_id in sub_ids[:-1]:
            assert await saver.aget_tuple(_config(thread_id, cp_id, sub_ns)) is None


@pytest.mark.asyncio
async def test_aprune_empty_thread_is_noop(redis_url: str) -> None:
    """aprune on a thread with no checkpoints completes without error."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.aprune([f"nonexistent-{_make_ulid()}"])


@pytest.mark.asyncio
async def test_aprune_does_not_affect_other_threads(redis_url: str) -> None:
    """Pruning one thread must leave a different thread's checkpoints intact."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        target_thread = f"target-{_make_ulid()}"
        other_thread = f"other-{_make_ulid()}"

        for thread_id in (target_thread, other_thread):
            for _ in range(3):
                cp_id = _make_ulid()
                await saver.aput(
                    config=_config(thread_id, cp_id),
                    checkpoint=_make_checkpoint(cp_id),
                    metadata=CheckpointMetadata(source="input", step=0, writes={}),
                    new_versions={"messages": "1"},
                )

        await saver.aprune([target_thread], keep_last=1)

        other_checkpoints = [
            c
            async for c in saver.alist(
                {"configurable": {"thread_id": other_thread, "checkpoint_ns": ""}}
            )
        ]
        assert len(other_checkpoints) == 3, "Other thread must be untouched"


@pytest.mark.asyncio
async def test_aprune_removes_associated_writes(redis_url: str) -> None:
    """Writes for evicted checkpoints are deleted; writes for retained ones survive."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"prune-writes-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(2)]
        for i, cp_id in enumerate(cp_ids):
            cfg = _config(thread_id, cp_id)
            await saver.aput(
                config=cfg,
                checkpoint=_make_checkpoint(cp_id, f"step-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )
            await saver.aput_writes(
                config=cfg,
                writes=[("messages", f"write-{i}")],
                task_id=f"task-{i}",
            )

        await saver.aprune([thread_id], keep_last=1)

        # Latest checkpoint must survive
        latest_tuple = await saver.aget_tuple(_config(thread_id, cp_ids[-1]))
        assert latest_tuple is not None, "Latest checkpoint must survive"

        # Evicted checkpoint must be gone
        evicted_tuple = await saver.aget_tuple(_config(thread_id, cp_ids[0]))
        assert evicted_tuple is None, "Evicted checkpoint must be gone"

        # Evicted checkpoint's writes must be gone from the index
        evicted_results = await saver.checkpoint_writes_index.search(
            FilterQuery(Tag("checkpoint_id") == cp_ids[0])
        )
        assert len(evicted_results.docs) == 0, "Evicted write must be removed"

        # Retained checkpoint's writes must still exist
        retained_results = await saver.checkpoint_writes_index.search(
            FilterQuery(Tag("checkpoint_id") == cp_ids[-1])
        )
        assert len(retained_results.docs) > 0, "Retained writes must still exist"


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "keep_last"),
    [("keep_latest", None), ("keep_latest", 1), ("delete", 1)],
)
def test_prune_keep_last_1_sync(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """Sync prune(keep_last=1) removes older checkpoints."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-prune-{_make_ulid()}"
        cp_ids = [_make_ulid() for _ in range(3)]

        for cp_id in cp_ids:
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], strategy=strategy, keep_last=keep_last)

        assert saver.get_tuple(_config(thread_id, cp_ids[-1])) is not None
        for old_cp_id in cp_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, old_cp_id)) is None


@pytest.mark.parametrize("strategy", ["keep_latest", "delete"])
def test_prune_keep_last_n_sync(redis_url: str, strategy: str) -> None:
    """Sync prune(keep_last=N) retains the N most-recent checkpoints."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-keep-last-{_make_ulid()}"
        N = 2
        cp_ids = [_make_ulid() for _ in range(4)]

        for cp_id in cp_ids:
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], strategy=strategy, keep_last=N)

        for cp_id in cp_ids[-N:]:
            assert saver.get_tuple(_config(thread_id, cp_id)) is not None

        for cp_id in cp_ids[:-N]:
            assert saver.get_tuple(_config(thread_id, cp_id)) is None


@pytest.mark.parametrize(
    ("strategy", "keep_last"), [("keep_latest", 0), ("delete", None)]
)
def test_prune_keep_last_0_sync(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """Sync prune(keep_last=0) removes all checkpoints."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-delete-{_make_ulid()}"
        cp_ids = [_make_ulid() for _ in range(3)]

        for cp_id in cp_ids:
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], strategy=strategy, keep_last=keep_last)

        for cp_id in cp_ids:
            assert saver.get_tuple(_config(thread_id, cp_id)) is None


# ---------------------------------------------------------------------------
# Shallow saver tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "keep_last"),
    [("keep_latest", None), ("keep_latest", 1), ("delete", 1)],
)
async def test_ashallow_aprune_keep_last_1_is_noop(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """aprune(keep_last>=1) is a no-op for AsyncShallowRedisSaver."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"shallow-noop-{_make_ulid()}"
        cp_id = _make_ulid()

        await saver.aput(
            config=_config(thread_id, cp_id),
            checkpoint=_make_checkpoint(cp_id),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        await saver.aprune([thread_id], strategy=strategy, keep_last=keep_last)

        # Checkpoint must still be there
        result = await saver.aget_tuple(_config(thread_id, cp_id))
        assert result is not None, "Shallow checkpoint must survive keep_last=1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "keep_last"), [("keep_latest", 0), ("delete", None)]
)
async def test_ashallow_aprune_keep_last_0_deletes_all(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """aprune(keep_last=0) removes all checkpoints for AsyncShallowRedisSaver."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"shallow-delete-{_make_ulid()}"
        cp_id = _make_ulid()

        await saver.aput(
            config=_config(thread_id, cp_id),
            checkpoint=_make_checkpoint(cp_id),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        await saver.aprune([thread_id], strategy=strategy, keep_last=keep_last)

        result = await saver.aget_tuple(_config(thread_id, cp_id))
        assert result is None, "Shallow checkpoint must be deleted with keep_last=0"


@pytest.mark.parametrize(
    ("strategy", "keep_last"),
    [("keep_latest", None), ("keep_latest", 1), ("delete", 1)],
)
def test_shallow_prune_keep_last_1_is_noop(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """prune(keep_last>=1) is a no-op for ShallowRedisSaver."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-shallow-noop-{_make_ulid()}"
        cp_id = _make_ulid()

        saver.put(
            config=_config(thread_id, cp_id),
            checkpoint=_make_checkpoint(cp_id),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        saver.prune([thread_id], strategy=strategy, keep_last=keep_last)

        result = saver.get_tuple(_config(thread_id, cp_id))
        assert result is not None, "Shallow checkpoint must survive keep_last=1"


@pytest.mark.parametrize(
    ("strategy", "keep_last"), [("keep_latest", 0), ("delete", None)]
)
def test_shallow_prune_keep_last_0_deletes_all(
    redis_url: str, strategy: str, keep_last: Optional[int]
) -> None:
    """prune(keep_last=0) removes all checkpoints for ShallowRedisSaver."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-shallow-delete-{_make_ulid()}"
        cp_id = _make_ulid()

        saver.put(
            config=_config(thread_id, cp_id),
            checkpoint=_make_checkpoint(cp_id),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        saver.prune([thread_id], strategy=strategy, keep_last=keep_last)

        result = saver.get_tuple(_config(thread_id, cp_id))
        assert result is None, "Shallow checkpoint must be deleted with keep_last=0"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy", "keep_last", "retained_per_ns", "limit_offset"), PRUNE_SEARCH_CASES
)
async def test_prune_requires_complete_search_results(
    client: Redis,
    prune: Callable[..., Awaitable[None]],
    prune_search_data: tuple[str, list[RunnableConfig], int],
    strategy: str,
    keep_last: Optional[int],
    retained_per_ns: int,
    limit_offset: int,
) -> None:
    thread_id, checkpoints, complete_limit = prune_search_data
    before = _thread_data(client, thread_id)
    assert before
    kwargs = dict(
        strategy=strategy,
        keep_last=keep_last,
        max_results=complete_limit + limit_offset,
    )
    if limit_offset < 0:
        with pytest.raises(ValueError, match="max_results") as error:
            await prune([thread_id], **kwargs)
        assert thread_id in str(error.value)
        assert _thread_data(client, thread_id) == before
        kwargs["max_results"] = complete_limit
    await prune([thread_id], **kwargs)
    assert _thread_data(client, thread_id) == _expected_pruned_data(
        before, checkpoints, retained_per_ns
    )


@pytest.mark.parametrize(
    "saver_cls", [RedisSaver, ShallowRedisSaver], ids=["full", "shallow"]
)
@pytest.mark.parametrize("strategy", INVALID_STRATEGIES)
@pytest.mark.parametrize(
    "keep_last", [None, 0, 2], ids=["implicit", "delete", "keep-two"]
)
def test_prune_rejects_invalid_strategy_without_changing_data(
    redis_url: str,
    client: Redis,
    saver_cls: type[RedisSaver] | type[ShallowRedisSaver],
    strategy: str,
    keep_last: Optional[int],
) -> None:
    """Issue #238: reject unsupported strategies, including with an override."""
    with saver_cls.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"invalid-strategy-{_make_ulid()}"
        for _ in range(3):
            cp_id = _make_ulid()
            config = _config(thread_id, cp_id)
            saver.put(
                config,
                _make_checkpoint(cp_id),
                CheckpointMetadata(source="input", step=0, writes={}),
                {"messages": "1"},
            )
            saver.put_writes(config, [("messages", "write")], f"task-{cp_id}")
        before = _thread_data(client, thread_id)
        assert before

        with pytest.raises(ValueError) as error:
            saver.prune([thread_id], strategy=strategy, keep_last=keep_last)

        assert "strategy must be 'keep_latest' or 'delete'" in str(error.value)
        assert repr(strategy) in str(error.value)
        assert _thread_data(client, thread_id) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "saver_cls", [AsyncRedisSaver, AsyncShallowRedisSaver], ids=["full", "shallow"]
)
@pytest.mark.parametrize("strategy", INVALID_STRATEGIES)
@pytest.mark.parametrize(
    "keep_last", [None, 0, 2], ids=["implicit", "delete", "keep-two"]
)
async def test_aprune_rejects_invalid_strategy_without_changing_data(
    redis_url: str,
    client: Redis,
    saver_cls: type[AsyncRedisSaver] | type[AsyncShallowRedisSaver],
    strategy: str,
    keep_last: Optional[int],
) -> None:
    """Issue #238: invalid async pruning must leave all thread data intact."""
    async with saver_cls.from_conn_string(redis_url) as saver:
        await saver.asetup()
        thread_id = f"invalid-strategy-{_make_ulid()}"
        for _ in range(3):
            cp_id = _make_ulid()
            config = _config(thread_id, cp_id)
            await saver.aput(
                config,
                _make_checkpoint(cp_id),
                CheckpointMetadata(source="input", step=0, writes={}),
                {"messages": "1"},
            )
            await saver.aput_writes(config, [("messages", "write")], f"task-{cp_id}")
        before = _thread_data(client, thread_id)
        assert before

        with pytest.raises(ValueError) as error:
            await saver.aprune([thread_id], strategy=strategy, keep_last=keep_last)

        assert "strategy must be 'keep_latest' or 'delete'" in str(error.value)
        assert repr(strategy) in str(error.value)
        assert _thread_data(client, thread_id) == before


@pytest.mark.parametrize(
    ("prune_call", "writer_cls"),
    [
        (RedisSaver, RedisSaver),
        (AsyncRedisSaver, RedisSaver),
        (ShallowRedisSaver, ShallowRedisSaver),
        (AsyncShallowRedisSaver, ShallowRedisSaver),
    ],
    indirect=["prune_call"],
    ids=["sync", "async", "shallow", "async-shallow"],
)
@pytest.mark.parametrize(
    ("thread_ids", "kwargs", "invalid_arg"),
    [
        ("ab", {"strategy": "delete"}, "thread_ids"),
        (["ab"], {"keep_last": 0.5}, "keep_last"),
    ],
    ids=["bare-thread-id", "fractional-keep-last"],
)
async def test_prune_rejects_invalid_arguments_without_deleting_checkpoints(
    redis_url: str,
    client: Redis,
    prune_call: Callable[..., Awaitable[None]],
    writer_cls: type[RedisSaver] | type[ShallowRedisSaver],
    thread_ids: Sequence[str],
    kwargs: dict[str, Any],
    invalid_arg: str,
) -> None:
    with writer_cls.from_conn_string(redis_url) as saver:
        saver.setup()
        for thread_id in ("ab", "a", "b"):
            _seed_prune_checkpoint(saver, thread_id)

    before = _thread_data(client, "")
    assert before
    with pytest.raises(ValueError, match=invalid_arg):
        await prune_call(thread_ids, **kwargs)
    assert _thread_data(client, "") == before


@pytest.mark.parametrize(
    "prune_call", [RedisSaver, AsyncRedisSaver], indirect=True, ids=["sync", "async"]
)
@pytest.mark.parametrize(
    "write_counts", [(1, 1, 1), (3, 1)], ids=["checkpoint-search", "write-search"]
)
async def test_prune_rejects_truncated_search_before_deleting_thread_data(
    redis_url: str,
    client: Redis,
    prune_call: Callable[..., Awaitable[None]],
    write_counts: tuple[int, ...],
) -> None:
    thread_id = f"truncated-prune-{_make_ulid()}"
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        for write_count in write_counts:
            _seed_prune_checkpoint(saver, thread_id, writes=write_count)

    before = _thread_data(client, thread_id)
    assert before
    with pytest.raises(ValueError, match=r"search returned 2 of 3.*max_results=2"):
        await prune_call([thread_id], strategy="delete", max_results=2)
    assert _thread_data(client, thread_id) == before

    await prune_call([thread_id], strategy="delete", max_results=3)
    assert not _thread_data(client, thread_id)


def test_prune_negative_keep_last_raises(redis_url: str) -> None:
    """prune(keep_last=-1) must raise ValueError immediately."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        with pytest.raises(ValueError, match="keep_last"):
            saver.prune(["any-thread"], keep_last=-1)


@pytest.mark.asyncio
async def test_aprune_negative_keep_last_raises(redis_url: str) -> None:
    """aprune(keep_last=-1) must raise ValueError immediately.

    Args:
        redis_url (str): Redis Connection URL
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        with pytest.raises(ValueError, match="keep_last"):
            await saver.aprune(["any-thread"], keep_last=-1)


# ---------------------------------------------------------------------------
# Sync coverage
# ---------------------------------------------------------------------------


def test_prune_per_namespace_isolation_sync(redis_url: str) -> None:
    """Sync prune: keep_last applied independently per namespace."""
    with RedisSaver.from_conn_string(redis_url=redis_url) as saver:
        saver.setup()
        thread_id = f"sync-ns-{_make_ulid()}"
        root_ids = [_make_ulid() for _ in range(3)]
        sub_ids = [_make_ulid() for _ in range(3)]

        for cp_id in root_ids:
            saver.put(
                config=_config(thread_id, cp_id, ""),
                checkpoint=_make_checkpoint(cp_id=cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )
        for cp_id in sub_ids:
            saver.put(
                config=_config(thread_id, cp_id, "subgraph:0"),
                checkpoint=_make_checkpoint(cp_id=cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], keep_last=1)

        assert saver.get_tuple(_config(thread_id, root_ids[-1], "")) is not None
        for cp_id in root_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, cp_id, "")) is None

        assert (
            saver.get_tuple(_config(thread_id, sub_ids[-1], "subgraph:0")) is not None
        )
        for cp_id in sub_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, cp_id, "subgraph:0")) is None


def test_prune_removes_associated_writes_sync(redis_url: str) -> None:
    """Sync prune: writes for evicted checkpoints deleted; retained writes survive."""
    with RedisSaver.from_conn_string(redis_url=redis_url) as saver:
        saver.setup()
        thread_id = f"sync-writes-{_make_ulid()}"
        cp_ids = [_make_ulid() for _ in range(2)]

        for i, cp_id in enumerate(cp_ids):
            cfg = _config(thread_id, cp_id)
            saver.put(
                config=cfg,
                checkpoint=_make_checkpoint(cp_id, f"step-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )
            saver.put_writes(
                config=cfg,
                writes=[("messages", f"write-{i}")],
                task_id=f"task-{i}",
            )

        saver.prune([thread_id], keep_last=1)

        assert saver.get_tuple(_config(thread_id, cp_ids[-1])) is not None
        assert saver.get_tuple(_config(thread_id, cp_ids[0])) is None

        evicted = saver.checkpoint_writes_index.search(
            FilterQuery(Tag("checkpoint_id") == cp_ids[0])
        )
        assert len(evicted.docs) == 0, "Evicted writes must be removed"

        retained = saver.checkpoint_writes_index.search(
            FilterQuery(Tag("checkpoint_id") == cp_ids[-1])
        )
        assert len(retained.docs) > 0, "Retained writes must survive"


def test_prune_empty_thread_is_noop_sync(redis_url: str) -> None:
    """Sync prune on a thread with no checkpoints completes without error."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saver.prune([f"nonexistent-{_make_ulid()}"])


def test_prune_does_not_affect_other_threads_sync(redis_url: str) -> None:
    """Sync prune: pruning one thread leaves another thread's checkpoints intact."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        target = f"sync-target-{_make_ulid()}"
        other = f"sync-other-{_make_ulid()}"

        for thread_id in (target, other):
            for _ in range(3):
                cp_id = _make_ulid()
                saver.put(
                    config=_config(thread_id, cp_id),
                    checkpoint=_make_checkpoint(cp_id),
                    metadata=CheckpointMetadata(source="input", step=0, writes={}),
                    new_versions={"messages": "1"},
                )

        saver.prune([target], keep_last=1)

        other_checkpoints = list(
            saver.list({"configurable": {"thread_id": other, "checkpoint_ns": ""}})
        )
        assert len(other_checkpoints) == 3, "Other thread must be untouched"


def test_prune_multiple_thread_ids_sync(redis_url: str) -> None:
    """Sync prune: pruning multiple thread_ids in one call works correctly."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        threads = [f"multi-{_make_ulid()}" for _ in range(2)]
        all_cp_ids = {}

        for thread_id in threads:
            cp_ids = [_make_ulid() for _ in range(3)]
            all_cp_ids[thread_id] = cp_ids
            for cp_id in cp_ids:
                saver.put(
                    config=_config(thread_id, cp_id),
                    checkpoint=_make_checkpoint(cp_id),
                    metadata=CheckpointMetadata(source="input", step=0, writes={}),
                    new_versions={"messages": "1"},
                )

        saver.prune(threads, keep_last=1)

        for thread_id, cp_ids in all_cp_ids.items():
            assert saver.get_tuple(_config(thread_id, cp_ids[-1])) is not None
            for old_id in cp_ids[:-1]:
                assert saver.get_tuple(_config(thread_id, old_id)) is None


def test_prune_keep_last_greater_than_total_is_noop_sync(redis_url: str) -> None:
    """Sync prune: keep_last > total checkpoints should not delete anything."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-noop-{_make_ulid()}"
        cp_ids = [_make_ulid() for _ in range(2)]

        for cp_id in cp_ids:
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], keep_last=100)  # way more then 2 stored

        for cp_id in cp_ids:
            assert (
                saver.get_tuple(_config(thread_id, cp_id)) is not None
            ), f"All checkpoints must survive when keep_last > total: {cp_id}"


def test_prune_then_continue_sync(redis_url: str) -> None:
    """Sync: after pruning, putting new checkpoints continues working correctly."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-continue-{_make_ulid()}"

        # Write some checkpoints, then prune them all
        for _ in range(10):
            cp_id = _make_ulid()
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id),
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], keep_last=0)

        # Now write a fresh checkpoint — saver must still work
        new_cp_id = _make_ulid()
        saver.put(
            config=_config(thread_id, new_cp_id),
            checkpoint=_make_checkpoint(new_cp_id),
            metadata=CheckpointMetadata(source="input", step=10, writes={}),
            new_versions={"messages": "2"},
        )

        result = saver.get_tuple(_config(thread_id, new_cp_id))
        assert result is not None, "New checkpoint after prune must be retrievable"


# ---------------------------------------------------------------------------
# Key registry (write_keys_zset) cleanup
# ---------------------------------------------------------------------------


def test_prune_cleans_write_keys_zset_for_evicted_checkpoints(redis_url: str) -> None:
    """prune() must delete write_keys_zset entries for evicted checkpoints.

    The key registry maintains a sorted set per checkpoint that tracks all
    write keys for that checkpoint. Evicting a checkpoint without cleaning
    its zset leaks the sorted set in Redis indefinitely.
    """
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"sync-zset-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(2)]
        for i, cp_id in enumerate(cp_ids):
            cfg = _config(thread_id, cp_id)
            saver.put(
                config=cfg,
                checkpoint=_make_checkpoint(cp_id, f"step-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )
            # put_writes populates the key registry zset for this checkpoint
            saver.put_writes(
                config=cfg,
                writes=[("messages", f"write-{i}")],
                task_id=f"task-{i}",
            )

        # Both zsets must exist before pruning
        assert (
            saver._key_registry is not None
        ), "Key registry must be initialised by setup()"
        evicted_zset = saver._key_registry.make_write_keys_zset_key(
            thread_id, "", cp_ids[0]
        )
        retained_zset = saver._key_registry.make_write_keys_zset_key(
            thread_id, "", cp_ids[-1]
        )
        assert saver._redis.exists(evicted_zset), "Evicted zset must exist before prune"
        assert saver._redis.exists(
            retained_zset
        ), "Retained zset must exist before prune"

        saver.prune([thread_id], keep_last=1)

        # Evicted checkpoint's zset must be gone
        assert not saver._redis.exists(
            evicted_zset
        ), "write_keys_zset for evicted checkpoint must be deleted"
        # Retained checkpoint's zset must still be there
        assert saver._redis.exists(
            retained_zset
        ), "write_keys_zset for retained checkpoint must survive"


# ---------------------------------------------------------------------------
# checkpoint_latest pointer cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prune_keep_last_0_cleans_checkpoint_latest(redis_url: str) -> None:
    """aprune(keep_last=0) must delete checkpoint_latest pointer keys.

    Args:
        redis_url (str): Redis Client
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"latest-ptr-{_make_ulid()}"
        from langgraph.checkpoint.redis.util import (
            to_storage_safe_id,
            to_storage_safe_str,
        )

        cp_id = _make_ulid()
        await saver.aput(
            config=_config(thread_id, cp_id),
            checkpoint=_make_checkpoint(cp_id),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        safe_tid = to_storage_safe_id(thread_id)
        safe_ns = to_storage_safe_str("")  # root namespace -> "__empty__"
        pointer_key = f"checkpoint_latest:{safe_tid}:{safe_ns}"

        # Pointer must exist before pruning
        assert await saver._redis.exists(pointer_key)

        await saver.aprune([thread_id], keep_last=0)

        # Pointer must be gone after full eviction
        assert not await saver._redis.exists(
            pointer_key
        ), "checkpoint_latest pointer must be cleaned up when keep_last=0"
