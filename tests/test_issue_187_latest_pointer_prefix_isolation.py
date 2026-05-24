"""Regression tests for latest checkpoint pointers with custom prefixes."""

from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver
from langgraph.checkpoint.redis.util import to_storage_safe_id, to_storage_safe_str


@pytest.fixture
def thread_id() -> str:
    """Generate a unique thread ID."""
    return f"latest_pointer_{uuid4()}"


@pytest.fixture
def latest_config(thread_id: str) -> RunnableConfig:
    """Create a latest-checkpoint config without a checkpoint ID."""
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


@pytest.fixture
def metadata() -> CheckpointMetadata:
    """Create simple checkpoint metadata."""
    return {"source": "input", "step": 1, "writes": {}}


def _checkpoint(owner: str) -> Checkpoint:
    checkpoint_id = str(uuid4())
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"owner": owner},
        "channel_versions": {"owner": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


def _global_latest_pointer_key(config: RunnableConfig) -> str:
    configurable = config["configurable"]
    return (
        "checkpoint_latest:"
        f"{to_storage_safe_id(configurable['thread_id'])}:"
        f"{to_storage_safe_str(configurable.get('checkpoint_ns', ''))}"
    )


def test_latest_pointer_key_uses_checkpoint_prefix(redis_url: str) -> None:
    """Test that latest-pointer key formatting includes the checkpoint prefix.

    Expected behaviour: the default key shape is preserved while custom
    checkpoint prefixes produce isolated latest-pointer keys.
    """
    with RedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="env_checkpoint"
    ) as saver:
        assert (
            saver._make_redis_checkpoint_latest_key("t1", "")
            == "env_checkpoint_latest:t1:__empty__"
        )


def test_sync_latest_pointer_uses_custom_prefix(
    redis_url: str,
    latest_config: RunnableConfig,
    metadata: CheckpointMetadata,
) -> None:
    """Test that sync savers write latest pointers under custom prefixes.

    Expected behaviour: the prefixed latest pointer exists and the bare
    checkpoint_latest namespace stays empty.
    """
    with RedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="env_checkpoint"
    ) as saver:
        saver.setup()
        checkpoint = _checkpoint("env")
        saver.put(latest_config, checkpoint, metadata, {})

        configurable = latest_config["configurable"]
        latest_pointer_key = saver._make_redis_checkpoint_latest_key(
            configurable["thread_id"],
            configurable["checkpoint_ns"],
        )
        checkpoint_key = saver._make_redis_checkpoint_key(
            configurable["thread_id"],
            configurable["checkpoint_ns"],
            checkpoint["id"],
        )
        # The custom-prefixed latest pointer should resolve to this checkpoint.
        assert saver._redis.get(latest_pointer_key) == checkpoint_key.encode()
        # The legacy global latest pointer must not be written for custom prefixes.
        assert not saver._redis.exists(_global_latest_pointer_key(latest_config))


def test_sync_latest_lookup_is_prefix_isolated(
    redis_url: str,
    latest_config: RunnableConfig,
    metadata: CheckpointMetadata,
) -> None:
    """Test that sync prefixed savers sharing Redis isolate latest lookup.

    Expected behaviour: each saver reads its own latest checkpoint.
    """
    with RedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="env_a_checkpoint"
    ) as saver_a:
        saver_a.setup()

        with RedisSaver.from_conn_string(
            redis_url, checkpoint_prefix="env_b_checkpoint"
        ) as saver_b:
            saver_b.setup()

            checkpoint_a = _checkpoint("A")
            checkpoint_b = _checkpoint("B")
            saver_a.put(latest_config, checkpoint_a, metadata, {})
            saver_b.put(latest_config, checkpoint_b, metadata, {})

            tuple_a = saver_a.get_tuple(latest_config)
            assert tuple_a.checkpoint["id"] == checkpoint_a["id"]
            assert tuple_a.checkpoint["channel_values"]["owner"] == "A"

            tuple_b = saver_b.get_tuple(latest_config)
            assert tuple_b.checkpoint["id"] == checkpoint_b["id"]
            assert tuple_b.checkpoint["channel_values"]["owner"] == "B"


@pytest.mark.asyncio
async def test_async_latest_pointer_uses_custom_prefix(
    redis_url: str,
    latest_config: RunnableConfig,
    metadata: CheckpointMetadata,
) -> None:
    """Test that async savers write latest pointers under custom prefixes.

    Expected behaviour: the prefixed latest pointer exists and the bare
    checkpoint_latest namespace stays empty.
    """
    async with AsyncRedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="async_env_checkpoint"
    ) as saver:
        await saver.setup()
        checkpoint = _checkpoint("env")
        await saver.aput(latest_config, checkpoint, metadata, {})

        configurable = latest_config["configurable"]
        latest_pointer_key = saver._make_redis_checkpoint_latest_key(
            configurable["thread_id"],
            configurable["checkpoint_ns"],
        )
        checkpoint_key = saver._make_redis_checkpoint_key(
            configurable["thread_id"],
            configurable["checkpoint_ns"],
            checkpoint["id"],
        )
        # The custom-prefixed latest pointer should resolve to this checkpoint.
        assert await saver._redis.get(latest_pointer_key) == checkpoint_key.encode()
        # The legacy global latest pointer must not be written for custom prefixes.
        assert not await saver._redis.exists(_global_latest_pointer_key(latest_config))


@pytest.mark.asyncio
async def test_async_latest_lookup_is_prefix_isolated(
    redis_url: str,
    latest_config: RunnableConfig,
    metadata: CheckpointMetadata,
) -> None:
    """Test that async prefixed savers sharing Redis isolate latest lookup.

    Expected behaviour: each saver reads its own latest checkpoint.
    """
    async with AsyncRedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="async_env_a_checkpoint"
    ) as saver_a:
        await saver_a.setup()

        async with AsyncRedisSaver.from_conn_string(
            redis_url, checkpoint_prefix="async_env_b_checkpoint"
        ) as saver_b:
            await saver_b.setup()

            checkpoint_a = _checkpoint("A")
            checkpoint_b = _checkpoint("B")
            await saver_a.aput(latest_config, checkpoint_a, metadata, {})
            await saver_b.aput(latest_config, checkpoint_b, metadata, {})

            tuple_a = await saver_a.aget_tuple(latest_config)
            assert tuple_a.checkpoint["id"] == checkpoint_a["id"]
            assert tuple_a.checkpoint["channel_values"]["owner"] == "A"

            tuple_b = await saver_b.aget_tuple(latest_config)
            assert tuple_b.checkpoint["id"] == checkpoint_b["id"]
            assert tuple_b.checkpoint["channel_values"]["owner"] == "B"
