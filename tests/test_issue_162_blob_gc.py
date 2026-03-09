"""Integration tests for issue #162 — blob GC after prune().

These tests prove that:
- No separate checkpoint_blob:* keys are ever created (channel values are inline)
- prune() / aprune() leaves no orphaned keys
- Evicted checkpoints' channel data is removed with the checkpoint document
- Shallow savers also use inline storage

The "orphaned blob" problem described in #162 does not exist because all four
saver implementations store channel values inline within each checkpoint
document.  When a checkpoint is deleted, its inline data goes with it.
"""

import time
from contextlib import asynccontextmanager, contextmanager

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from redis import Redis
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.base import (
    CHECKPOINT_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
)
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ulid() -> str:
    """Return a time-ordered ULID string, with a small sleep to guarantee ordering."""
    time.sleep(0.002)
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


@contextmanager
def _saver(redis_url: str):
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        yield saver


@asynccontextmanager
async def _async_saver(redis_url: str):
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        yield saver


def _scan_keys(redis_url: str, pattern: str) -> list[str]:
    """Scan for keys matching pattern using a separate decode_responses client."""
    client = Redis.from_url(redis_url, decode_responses=True)
    try:
        return list(client.scan_iter(match=pattern))
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Test 1: No blob keys created during checkpoint lifecycle
# ---------------------------------------------------------------------------


def test_no_blob_keys_created_during_checkpoint_lifecycle(redis_url: str) -> None:
    """No checkpoint_blob:* keys are ever created — channel values are inline."""
    with _saver(redis_url) as saver:
        thread_id = f"blob-lifecycle-{_make_ulid()}"

        for i in range(3):
            cp_id = _make_ulid()
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id, f"step-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )

            # After each put, assert zero blob keys exist for this thread
            blob_keys = _scan_keys(redis_url, f"checkpoint_blob:*{thread_id}*")
            assert (
                len(blob_keys) == 0
            ), f"Expected 0 blob keys after step {i}, found {len(blob_keys)}: {blob_keys}"


# ---------------------------------------------------------------------------
# Test 2: prune leaves no orphaned keys (sync)
# ---------------------------------------------------------------------------


def test_prune_leaves_no_orphaned_keys(redis_url: str) -> None:
    """After prune(keep_last=1), only the kept checkpoint's keys remain."""
    with _saver(redis_url) as saver:
        thread_id = f"prune-orphan-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
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

        # Zero blob keys
        blob_keys = _scan_keys(redis_url, f"checkpoint_blob:*{thread_id}*")
        assert len(blob_keys) == 0, f"Unexpected blob keys: {blob_keys}"

        # Only the kept checkpoint survives
        assert saver.get_tuple(_config(thread_id, cp_ids[-1])) is not None
        for old_id in cp_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, old_id)) is None


# ---------------------------------------------------------------------------
# Test 3: aprune leaves no orphaned keys (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aprune_leaves_no_orphaned_keys(redis_url: str) -> None:
    """After aprune(keep_last=1), only the kept checkpoint's keys remain."""
    async with _async_saver(redis_url) as saver:
        thread_id = f"aprune-orphan-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
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

        # Zero blob keys
        blob_keys = _scan_keys(redis_url, f"checkpoint_blob:*{thread_id}*")
        assert len(blob_keys) == 0, f"Unexpected blob keys: {blob_keys}"

        # Only the kept checkpoint survives
        assert await saver.aget_tuple(_config(thread_id, cp_ids[-1])) is not None
        for old_id in cp_ids[:-1]:
            assert await saver.aget_tuple(_config(thread_id, old_id)) is None


# ---------------------------------------------------------------------------
# Test 4: prune keep_last=0 leaves no checkpoint keys (sync)
# ---------------------------------------------------------------------------


def test_prune_keep_last_0_leaves_no_checkpoint_keys(redis_url: str) -> None:
    """prune(keep_last=0) removes all checkpoint-related keys."""
    with _saver(redis_url) as saver:
        thread_id = f"prune-zero-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
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

        saver.prune([thread_id], keep_last=0)

        # All thread-related keys should be gone
        all_thread_keys = _scan_keys(redis_url, f"*{thread_id}*")
        # Filter to only checkpoint/write/blob prefixed keys (ignore write_keys_zset)
        checkpoint_keys = [
            k
            for k in all_thread_keys
            if k.startswith(CHECKPOINT_PREFIX + ":")
            or k.startswith(CHECKPOINT_WRITE_PREFIX + ":")
            or k.startswith("checkpoint_blob:")
            or k.startswith("checkpoint_latest:")
        ]
        assert len(checkpoint_keys) == 0, (
            f"Expected 0 checkpoint-related keys after keep_last=0, "
            f"found {len(checkpoint_keys)}: {checkpoint_keys}"
        )


# ---------------------------------------------------------------------------
# Test 5: aprune keep_last=0 leaves no checkpoint keys (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aprune_keep_last_0_leaves_no_checkpoint_keys(redis_url: str) -> None:
    """aprune(keep_last=0) removes all checkpoint-related keys."""
    async with _async_saver(redis_url) as saver:
        thread_id = f"aprune-zero-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
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

        await saver.aprune([thread_id], keep_last=0)

        # All thread-related keys should be gone
        all_thread_keys = _scan_keys(redis_url, f"*{thread_id}*")
        checkpoint_keys = [
            k
            for k in all_thread_keys
            if k.startswith(CHECKPOINT_PREFIX + ":")
            or k.startswith(CHECKPOINT_WRITE_PREFIX + ":")
            or k.startswith("checkpoint_blob:")
            or k.startswith("checkpoint_latest:")
        ]
        assert len(checkpoint_keys) == 0, (
            f"Expected 0 checkpoint-related keys after keep_last=0, "
            f"found {len(checkpoint_keys)}: {checkpoint_keys}"
        )


# ---------------------------------------------------------------------------
# Test 6: prune removes channel data with checkpoint
# ---------------------------------------------------------------------------


def test_prune_channel_data_removed_with_checkpoint(redis_url: str) -> None:
    """Evicted checkpoints' channel data is gone; kept checkpoint's data is intact."""
    with _saver(redis_url) as saver:
        thread_id = f"prune-channel-{_make_ulid()}"

        cp_ids = [_make_ulid() for _ in range(3)]
        for i, cp_id in enumerate(cp_ids):
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id, f"value-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )

        saver.prune([thread_id], keep_last=1)

        # Evicted checkpoints are gone
        for old_id in cp_ids[:-1]:
            assert saver.get_tuple(_config(thread_id, old_id)) is None

        # Kept checkpoint has its channel data intact
        kept = saver.get_tuple(_config(thread_id, cp_ids[-1]))
        assert kept is not None
        assert "messages" in kept.checkpoint["channel_values"]
        assert kept.checkpoint["channel_values"]["messages"] == ["value-2"]


# ---------------------------------------------------------------------------
# Test 7: shallow saver also uses inline storage (no blob keys)
# ---------------------------------------------------------------------------


def test_shallow_no_blob_keys_created(redis_url: str) -> None:
    """ShallowRedisSaver stores channel values inline — no blob keys created."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"shallow-blob-{_make_ulid()}"

        for i in range(3):
            cp_id = _make_ulid()
            saver.put(
                config=_config(thread_id, cp_id),
                checkpoint=_make_checkpoint(cp_id, f"step-{i}"),
                metadata=CheckpointMetadata(source="input", step=i, writes={}),
                new_versions={"messages": "1"},
            )

            blob_keys = _scan_keys(redis_url, f"checkpoint_blob:*{thread_id}*")
            assert (
                len(blob_keys) == 0
            ), f"Expected 0 blob keys after step {i}, found {len(blob_keys)}"

        # Verify the latest checkpoint has inline channel data
        result = saver.get_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert result is not None
        assert "messages" in result.checkpoint["channel_values"]
