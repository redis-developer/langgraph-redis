"""Integration tests for issue #159 — aprune() / prune() with keep_last strategy.

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
"""

import time

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag


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


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aprune_keep_last_1_removes_older_checkpoints(redis_url: str) -> None:
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

        await saver.aprune([thread_id], keep_last=1)

        # Only the latest should survive
        result = await saver.aget_tuple(_config(thread_id, cp_ids[-1]))
        assert result is not None, "Latest checkpoint must be retained"

        # Older two must be gone
        for old_cp_id in cp_ids[:-1]:
            result = await saver.aget_tuple(_config(thread_id, old_cp_id))
            assert result is None, f"Checkpoint {old_cp_id} should have been pruned"


@pytest.mark.asyncio
async def test_aprune_keep_last_n_retains_window(redis_url: str) -> None:
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

        await saver.aprune([thread_id], keep_last=N)

        # Last N should survive
        for cp_id in cp_ids[-N:]:
            result = await saver.aget_tuple(_config(thread_id, cp_id))
            assert result is not None, f"Checkpoint {cp_id} should be retained"

        # Earlier ones must be evicted
        for cp_id in cp_ids[:-N]:
            result = await saver.aget_tuple(_config(thread_id, cp_id))
            assert result is None, f"Checkpoint {cp_id} should be pruned"


@pytest.mark.asyncio
async def test_aprune_keep_last_0_removes_all_checkpoints(redis_url: str) -> None:
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

        await saver.aprune([thread_id], keep_last=0)

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
        assert await saver.aget_tuple(_config(thread_id, root_ids[-1], root_ns)) is not None
        # Older root checkpoints evicted
        for cp_id in root_ids[:-1]:
            assert await saver.aget_tuple(_config(thread_id, cp_id, root_ns)) is None

        # Latest in subgraph namespace must also survive
        assert await saver.aget_tuple(_config(thread_id, sub_ids[-1], sub_ns)) is not None
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


def test_prune_keep_last_1_sync(redis_url: str) -> None:
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

        saver.prune([thread_id], keep_last=1)

        assert saver.get_tuple(_config(thread_id, cp_ids[-1])) is not None
        for old_cp_id in cp_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, old_cp_id)) is None


def test_prune_keep_last_n_sync(redis_url: str) -> None:
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

        saver.prune([thread_id], keep_last=N)

        for cp_id in cp_ids[-N:]:
            assert saver.get_tuple(_config(thread_id, cp_id)) is not None

        for cp_id in cp_ids[:-N]:
            assert saver.get_tuple(_config(thread_id, cp_id)) is None


def test_prune_keep_last_0_sync(redis_url: str) -> None:
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

        saver.prune([thread_id], keep_last=0)

        for cp_id in cp_ids:
            assert saver.get_tuple(_config(thread_id, cp_id)) is None


# ---------------------------------------------------------------------------
# Shallow saver tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ashallow_aprune_keep_last_1_is_noop(redis_url: str) -> None:
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

        await saver.aprune([thread_id], keep_last=1)

        # Checkpoint must still be there
        result = await saver.aget_tuple(_config(thread_id, cp_id))
        assert result is not None, "Shallow checkpoint must survive keep_last=1"


@pytest.mark.asyncio
async def test_ashallow_aprune_keep_last_0_deletes_all(redis_url: str) -> None:
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

        await saver.aprune([thread_id], keep_last=0)

        result = await saver.aget_tuple(_config(thread_id, cp_id))
        assert result is None, "Shallow checkpoint must be deleted with keep_last=0"


def test_shallow_prune_keep_last_1_is_noop(redis_url: str) -> None:
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

        saver.prune([thread_id], keep_last=1)

        result = saver.get_tuple(_config(thread_id, cp_id))
        assert result is not None, "Shallow checkpoint must survive keep_last=1"


def test_shallow_prune_keep_last_0_deletes_all(redis_url: str) -> None:
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

        saver.prune([thread_id], keep_last=0)

        result = saver.get_tuple(_config(thread_id, cp_id))
        assert result is None, "Shallow checkpoint must be deleted with keep_last=0"
