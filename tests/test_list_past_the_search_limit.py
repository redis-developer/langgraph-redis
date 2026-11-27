"""list() without a limit returns every matching checkpoint (#260).

FT.SEARCH returns at most MAXSEARCHRESULTS documents (10,000 on Redis Stack, where
a larger LIMIT raises), and the savers asked for at most 10,000 (100 in the async
shallow saver), so `get_state_history` and listings across threads silently lost
the oldest checkpoints. Past that the savers read the matches through an
FT.AGGREGATE cursor.
"""

import uuid
from typing import Any, List

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointTuple,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# Past 10,000: Redis Stack's default MAXSEARCHRESULTS and the savers' old cap
COUNT = 10_050


def _ids(tuples: List[CheckpointTuple]) -> List[str]:
    return [t.config["configurable"]["checkpoint_id"] for t in tuples]


def _next_checkpoint(previous: Checkpoint, step: int) -> Checkpoint:
    checkpoint = create_checkpoint(previous, None, step)
    checkpoint["channel_values"] = {"step": step}
    return checkpoint


def _write_history(saver: RedisSaver, thread_id: str) -> List[str]:
    """COUNT checkpoints on one thread, a pending write on the oldest; their ids."""
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    ids = []
    for step in range(COUNT):
        checkpoint = _next_checkpoint(checkpoint, step)
        config = saver.put(config, checkpoint, {"source": "loop", "step": step}, {})
        ids.append(config["configurable"]["checkpoint_id"])
        if step == 0:
            saver.put_writes(config, [("channel", "value")], "task-1")
    return ids


async def _awrite_history(saver: AsyncRedisSaver, thread_id: str) -> List[str]:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    ids = []
    for step in range(COUNT):
        checkpoint = _next_checkpoint(checkpoint, step)
        config = await saver.aput(
            config, checkpoint, {"source": "loop", "step": step}, {}
        )
        ids.append(config["configurable"]["checkpoint_id"])
        if step == 0:
            await saver.aput_writes(config, [("channel", "value")], "task-1")
    return ids


def _assert_history(tuples: List[CheckpointTuple], ids: List[str]) -> None:
    assert _ids(tuples) == ids[::-1]
    # The oldest checkpoints come from the cursor read, with all their data
    oldest, second = tuples[-1], tuples[-2]
    assert oldest.metadata["step"] == 0
    assert oldest.checkpoint["channel_values"] == {"step": 0}
    assert oldest.parent_config is None
    assert [(w[1], w[2]) for w in oldest.pending_writes or []] == [("channel", "value")]
    assert second.parent_config is not None
    assert second.parent_config["configurable"]["checkpoint_id"] == ids[0]


def test_list_returns_every_checkpoint_of_a_long_thread(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"long-{uuid.uuid4()}"
        ids = _write_history(saver, thread_id)
        thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        _assert_history(list(saver.list(thread)), ids)
        assert len(list(saver.list(None, filter={"source": "loop"}))) == COUNT
        # A limit above the server's FT.SEARCH maximum doesn't raise
        assert _ids(list(saver.list(thread, limit=COUNT - 20))) == ids[:19:-1]
        assert _ids(list(saver.list(thread, limit=5))) == ids[:-6:-1]


async def test_alist_returns_every_checkpoint_of_a_long_thread(
    redis_url: str,
) -> None:
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = f"long-{uuid.uuid4()}"
        ids = await _awrite_history(saver, thread_id)
        thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        _assert_history([t async for t in saver.alist(thread)], ids)
        assert len([t async for t in saver.alist(None, filter={"source": "loop"})]) == (
            COUNT
        )
        assert _ids([t async for t in saver.alist(thread, limit=COUNT - 20)]) == (
            ids[:19:-1]
        )
        assert _ids([t async for t in saver.alist(thread, limit=5)]) == ids[:-6:-1]


def _thread_ids(count: int) -> List[str]:
    run = uuid.uuid4().hex[:8]
    return [f"thread-{run}-{i:05d}" for i in range(count)]


def _shallow_put_args(thread_id: str, step: int) -> Any:
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    checkpoint = _next_checkpoint(empty_checkpoint(), step)
    return config, checkpoint, {"source": "loop", "step": step}, {}


def _assert_threads(tuples: List[CheckpointTuple], thread_ids: List[str]) -> None:
    listed = [t.config["configurable"]["thread_id"] for t in tuples]
    assert sorted(listed) == thread_ids
    assert _ids(tuples) == sorted(_ids(tuples), reverse=True)
    steps = {
        t.config["configurable"]["thread_id"]: t.checkpoint["channel_values"]
        for t in tuples
    }
    assert steps[thread_ids[0]] == {"step": 0}


def test_shallow_list_returns_the_checkpoint_of_every_thread(redis_url: str) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_ids = _thread_ids(COUNT)
        for step, thread_id in enumerate(thread_ids):
            saver.put(*_shallow_put_args(thread_id, step))

        _assert_threads(list(saver.list(None)), thread_ids)
        assert len(list(saver.list(None, limit=COUNT - 20))) == COUNT - 20


async def test_async_shallow_alist_returns_the_checkpoint_of_every_thread(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        thread_ids = _thread_ids(COUNT)
        for step, thread_id in enumerate(thread_ids):
            await saver.aput(*_shallow_put_args(thread_id, step))

        _assert_threads([t async for t in saver.alist(None)], thread_ids)
        assert len([t async for t in saver.alist(None, limit=COUNT - 20)]) == (
            COUNT - 20
        )


@pytest.mark.parametrize("count", [99, 101, 150])
async def test_async_shallow_alist_without_limit_returns_more_than_100(
    redis_url: str, count: int
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        thread_ids = _thread_ids(count)
        for step, thread_id in enumerate(thread_ids):
            await saver.aput(*_shallow_put_args(thread_id, step))

        _assert_threads([t async for t in saver.alist(None)], thread_ids)
