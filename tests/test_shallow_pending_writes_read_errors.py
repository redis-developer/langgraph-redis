"""A failed pending-writes read must raise, not come back as "no pending writes".

AsyncShallowRedisSaver._aload_pending_writes caught every exception and returned [],
so aget_tuple answered a Redis error with the checkpoint and pending_writes=[] - the
same answer as a checkpoint with nothing pending. LangGraph uses a checkpoint's pending
writes on resume, so tasks that had already finished would run again. The sync
ShallowRedisSaver raises on the same failure.

The failures here are real Redis errors (WRONGTYPE) on keys the saver reads.
"""

import uuid

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from redis import Redis
from redis.exceptions import ResponseError

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

WRITES = [("messages", "done"), ("counter", 1), ("notes", "x")]
METADATA: CheckpointMetadata = {"source": "loop", "step": 1, "writes": {}}


def _thread_config(prefix: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": f"{prefix}-{uuid.uuid4()}",
            "checkpoint_ns": "",
        }
    }


def _break_one_write(redis_url: str, thread_id: str) -> None:
    """Replace one stored write with a plain string, so JSON.GET on it fails."""
    client = Redis.from_url(redis_url)
    try:
        keys = sorted(client.keys(f"checkpoint_write:{thread_id}:*"))
        assert len(keys) == len(WRITES)
        client.delete(keys[1])
        client.set(keys[1], "not json")
    finally:
        client.close()


def _break_registry(redis_url: str, registry_key: str) -> None:
    """Replace the write registry (a sorted set) with a plain string."""
    client = Redis.from_url(redis_url)
    try:
        assert client.zcard(registry_key) == len(WRITES)
        client.delete(registry_key)
        client.set(registry_key, "not a sorted set")
    finally:
        client.close()


async def _aseed(saver: AsyncShallowRedisSaver, config: RunnableConfig) -> None:
    saved = await saver.aput(config, empty_checkpoint(), METADATA, {})
    await saver.aput_writes(saved, WRITES, task_id="task-a")


def _seed(saver: ShallowRedisSaver, config: RunnableConfig) -> None:
    saved = saver.put(config, empty_checkpoint(), METADATA, {})
    saver.put_writes(saved, WRITES, task_id="task-a")


@pytest.mark.asyncio
async def test_ashallow_aget_tuple_raises_when_a_pending_write_cannot_be_read(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread_config("ashallow-write-error")
        await _aseed(saver, config)
        _break_one_write(redis_url, config["configurable"]["thread_id"])

        with pytest.raises(ResponseError):
            await saver.aget_tuple(config)


@pytest.mark.asyncio
async def test_ashallow_aget_tuple_raises_when_write_registry_cannot_be_read(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread_config("ashallow-registry-error")
        await _aseed(saver, config)
        thread_id = config["configurable"]["thread_id"]
        _break_registry(redis_url, f"write_keys_zset:{thread_id}:__empty__:shallow")

        with pytest.raises(ResponseError):
            await saver.aget_tuple(config)


@pytest.mark.asyncio
async def test_ashallow_aget_tuple_returns_pending_writes(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread_config("ashallow-writes")
        await _aseed(saver, config)

        result = await saver.aget_tuple(config)

        assert result is not None
        assert result.pending_writes == [
            ("task-a", channel, value) for channel, value in WRITES
        ]


@pytest.mark.asyncio
async def test_ashallow_aget_tuple_without_writes_has_no_pending_writes(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread_config("ashallow-no-writes")
        await saver.aput(config, empty_checkpoint(), METADATA, {})

        result = await saver.aget_tuple(config)

        assert result is not None
        assert result.pending_writes == []


def test_shallow_get_tuple_raises_when_a_pending_write_cannot_be_read(
    redis_url: str,
) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread_config("shallow-write-error")
        _seed(saver, config)
        _break_one_write(redis_url, config["configurable"]["thread_id"])

        with pytest.raises(ResponseError):
            saver.get_tuple(config)


def test_shallow_get_tuple_raises_when_write_registry_cannot_be_read(
    redis_url: str,
) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread_config("shallow-registry-error")
        _seed(saver, config)
        thread_id = config["configurable"]["thread_id"]
        _break_registry(redis_url, f"write_registry:{thread_id}:__empty__:shallow")

        with pytest.raises(ResponseError):
            saver.get_tuple(config)
