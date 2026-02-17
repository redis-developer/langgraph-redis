"""Integration tests for AsyncShallowRedisSaver.adelete_thread."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.util import to_storage_safe_id, to_storage_safe_str


def _expected_write_keys(
    *,
    saver: AsyncShallowRedisSaver,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    writes: List[Tuple[str, Any]],
) -> List[str]:
    """Compute the concrete Redis keys created by aput_writes."""
    keys: List[str] = []
    for enum_idx, (channel, _value) in enumerate(writes):
        idx = WRITES_IDX_MAP.get(channel, enum_idx)
        keys.append(
            saver._make_redis_checkpoint_writes_key_cached(  # noqa: SLF001
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
        )
    return keys


@pytest.mark.asyncio
async def test_adelete_thread_cleans_shallow_checkpoints_writes_and_registry(
    redis_url: str, async_client
) -> None:
    thread_id = "test-ashallow-adelete-thread"
    other_thread_id = "test-ashallow-adelete-thread-other"

    # Two namespaces to simulate subgraph usage in shallow mode.
    namespaces = ["", "inner"]

    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        created: Dict[str, Dict[str, Any]] = {}

        for checkpoint_ns in namespaces:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            }
            checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
            metadata: CheckpointMetadata = {"source": "input", "step": 1, "writes": {}}

            saved_config = await saver.aput(config, checkpoint, metadata, {})
            checkpoint_id = saved_config["configurable"]["checkpoint_id"]

            # Create a couple writes and record expected keys.
            writes = [("channel1", "value1"), ("channel2", "value2")]
            task_id = f"task-{checkpoint_ns or 'root'}"
            await saver.aput_writes(saved_config, writes, task_id)

            checkpoint_key = (
                saver._make_shallow_redis_checkpoint_key_cached(  # noqa: SLF001
                    thread_id, checkpoint_ns
                )
            )
            zset_key = (
                f"write_keys_zset:{to_storage_safe_id(thread_id)}:"
                f"{to_storage_safe_str(checkpoint_ns)}:shallow"
            )
            write_keys = _expected_write_keys(
                saver=saver,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                writes=writes,
            )

            created[checkpoint_ns] = {
                "saved_config": saved_config,
                "checkpoint_key": checkpoint_key,
                "zset_key": zset_key,
                "write_keys": write_keys,
            }

        # Also create a checkpoint for a different thread that must not be deleted.
        other_config: RunnableConfig = {
            "configurable": {"thread_id": other_thread_id, "checkpoint_ns": ""}
        }
        other_checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        other_saved_config = await saver.aput(
            other_config,
            other_checkpoint,
            {"source": "input", "step": 1, "writes": {}},
            {},
        )
        other_checkpoint_key = (
            saver._make_shallow_redis_checkpoint_key_cached(  # noqa: SLF001
                other_thread_id, ""
            )
        )

        # Assert keys exist before deletion (direct key checks; avoids index lag).
        assert await async_client.exists(other_checkpoint_key) == 1
        assert await saver.aget_tuple(other_saved_config) is not None

        for checkpoint_ns in namespaces:
            checkpoint_key = created[checkpoint_ns]["checkpoint_key"]
            zset_key = created[checkpoint_ns]["zset_key"]
            write_keys = created[checkpoint_ns]["write_keys"]

            assert await async_client.exists(checkpoint_key) == 1
            assert await async_client.exists(zset_key) == 1
            assert await async_client.zcard(zset_key) == len(write_keys)

            for key in write_keys:
                assert await async_client.exists(key) == 1

        # Delete everything for thread_id.
        await saver.adelete_thread(thread_id)

        # The other thread should still exist.
        assert await async_client.exists(other_checkpoint_key) == 1
        assert await saver.aget_tuple(other_saved_config) is not None

        # Keys for thread_id should be gone.
        for checkpoint_ns in namespaces:
            saved_config = created[checkpoint_ns]["saved_config"]
            checkpoint_key = created[checkpoint_ns]["checkpoint_key"]
            zset_key = created[checkpoint_ns]["zset_key"]
            write_keys = created[checkpoint_ns]["write_keys"]

            assert await saver.aget_tuple(saved_config) is None
            assert await async_client.exists(checkpoint_key) == 0
            assert await async_client.exists(zset_key) == 0

            for key in write_keys:
                assert await async_client.exists(key) == 0
