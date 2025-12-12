"""Tests for issue #125: Support customization of checkpoint prefixes.

Issue: https://github.com/redis-developer/langgraph-redis/issues/125

Feature request to allow customization of checkpoint_prefix, checkpoint_blob_prefix,
and checkpoint_write_prefix to enable multiple isolated checkpoint savers in the same
Redis instance.
"""

from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver


@pytest.fixture
def checkpoint_id() -> str:
    """Generate a unique checkpoint ID."""
    return str(uuid4())


@pytest.fixture
def thread_id() -> str:
    """Generate a unique thread ID."""
    return f"test_{uuid4()}"


@pytest.fixture
def simple_checkpoint(checkpoint_id: str) -> Checkpoint:
    """Create a simple checkpoint."""
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_values": {"data": "test"},
        "channel_versions": {"data": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


@pytest.fixture
def simple_metadata() -> CheckpointMetadata:
    """Create simple checkpoint metadata."""
    return {"source": "input", "step": 1, "writes": {}}


@pytest.fixture
def config(thread_id: str, checkpoint_id: str) -> RunnableConfig:
    """Create a checkpoint config."""
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


def test_default_checkpoint_prefix_is_checkpoint(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
) -> None:
    """Test that default checkpoint prefix is 'checkpoint'."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()

        assert saver._checkpoint_prefix == "checkpoint"
        assert saver._checkpoint_blob_prefix == "checkpoint_blob"
        assert saver._checkpoint_write_prefix == "checkpoint_write"

        saver.put(config, simple_checkpoint, simple_metadata, {})

        keys = list(saver._redis.scan_iter("*"))
        assert len(keys) > 0
        prefixes = {k.split(b":")[0] for k in keys}
        assert prefixes == {b"checkpoint", b"checkpoint_latest"}


def test_backward_compatibility_no_prefix_specified(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    checkpoint_id: str,
) -> None:
    """Test backward compatibility when no prefix is specified."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()

        saver.put(config, simple_checkpoint, simple_metadata, {})

        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint_id


def test_custom_checkpoint_prefix_sync_all_components(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    thread_id: str,
    checkpoint_id: str,
) -> None:
    """Test RedisSaver with custom checkpoint, blob, and write prefixes."""
    custom_checkpoint_prefix = "myapp_checkpoint"
    custom_blob_prefix = "myapp_checkpoint_blob"
    custom_write_prefix = "myapp_checkpoint_write"

    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix=custom_checkpoint_prefix,
        checkpoint_blob_prefix=custom_blob_prefix,
        checkpoint_write_prefix=custom_write_prefix,
    ) as saver:
        saver.setup()

        assert saver._checkpoint_prefix == custom_checkpoint_prefix
        assert saver._checkpoint_blob_prefix == custom_blob_prefix
        assert saver._checkpoint_write_prefix == custom_write_prefix

        saver.put(config, simple_checkpoint, simple_metadata, {})

        # Verify that the custom checkpoint prefix is used
        keys = list(saver._redis.scan_iter("*"))
        prefixes = {k.split(b":")[0] for k in keys}
        assert len(keys) > 0
        assert prefixes == {
            custom_checkpoint_prefix.encode(),
            b"checkpoint_latest",
        }

        # Test write prefix by adding writes
        saver.put_writes(
            config,
            writes=[("channel1", "value1"), ("channel2", "value2")],
            task_id="task_1",
        )

        # Verify that the custom write prefix is used
        all_keys = list(saver._redis.scan_iter("*"))
        write_keys = [k for k in all_keys if k.startswith(custom_write_prefix.encode())]
        assert len(write_keys) == 2, f"Expected 2 write keys, found {len(write_keys)}"

        # Verify write keys have correct structure
        for key in write_keys:
            key_str = key.decode()
            assert key_str.startswith(custom_write_prefix)

        # Verify blob key generation
        blob_key = saver._make_redis_checkpoint_blob_key(
            thread_id, "", "test_channel", "v1"
        )
        assert blob_key.startswith(custom_blob_prefix)

        # Verify data can be retrieved
        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint_id


def test_custom_checkpoint_prefix_isolation_sync(
    redis_url: str,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    checkpoint_id: str,
) -> None:
    """Test that different checkpoint prefixes create isolated savers."""
    # Saver 1 with prefix "app1_checkpoint"
    with RedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="app1_checkpoint"
    ) as saver1:
        saver1.setup()

        # Saver 2 with prefix "app2_checkpoint"
        with RedisSaver.from_conn_string(
                redis_url, checkpoint_prefix="app2_checkpoint"
        ) as saver2:
            saver2.setup()

            checkpoint1: Checkpoint = {
                "v": 1,
                "id": checkpoint_id,
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"app": "app1", "value": "from_app1"},
                "channel_versions": {"app": "1"},
                "versions_seen": {"agent": {"app": "1"}},
                "pending_sends": [],
            }
            saver1.put(config, checkpoint1, simple_metadata, {})

            checkpoint2: Checkpoint = {
                "v": 1,
                "id": checkpoint_id,
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"app": "app2", "value": "from_app2"},
                "channel_versions": {"app": "1"},
                "versions_seen": {"agent": {"app": "1"}},
                "pending_sends": [],
            }
            saver2.put(config, checkpoint2, simple_metadata, {})

            retrieved1 = saver1.get_tuple(config)
            assert retrieved1 is not None
            assert retrieved1.checkpoint["channel_values"]["app"] == "app1"
            assert retrieved1.checkpoint["channel_values"]["value"] == "from_app1"

            retrieved2 = saver2.get_tuple(config)
            assert retrieved2 is not None
            assert retrieved2.checkpoint["channel_values"]["app"] == "app2"
            assert retrieved2.checkpoint["channel_values"]["value"] == "from_app2"


def test_custom_checkpoint_prefix_with_special_characters(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    checkpoint_id: str,
) -> None:
    """Test custom prefixes with special characters."""
    custom_prefix = "my-app_v2_checkpoint"

    with RedisSaver.from_conn_string(
        redis_url, checkpoint_prefix=custom_prefix
    ) as saver:
        saver.setup()

        saver.put(config, simple_checkpoint, simple_metadata, {})

        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint_id


def test_custom_checkpoint_prefix_with_ttl(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    thread_id: str,
    checkpoint_id: str,
) -> None:
    """Test custom prefixes work correctly with TTL."""
    custom_prefix = "ttl_test_checkpoint"

    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix=custom_prefix,
        ttl={"default_ttl": 60},
    ) as saver:
        saver.setup()

        saver.put(config, simple_checkpoint, simple_metadata, {})

        # Verify checkpoint was created with custom prefix
        keys = list(saver._redis.scan_iter(f"{custom_prefix}:*"))
        assert len(keys) > 0

        # Verify TTL is set on the checkpoint key
        checkpoint_key = saver._make_redis_checkpoint_key(thread_id, "", checkpoint_id)
        ttl = saver._redis.ttl(checkpoint_key)
        assert ttl > 0, "TTL should be set on checkpoint"


def test_custom_checkpoint_prefix_delete_thread(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    thread_id: str,
) -> None:
    """Test delete_thread works correctly with custom prefixes."""
    custom_checkpoint_prefix = "delete_test_checkpoint"
    custom_write_prefix = "delete_test_write"

    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix=custom_checkpoint_prefix,
        checkpoint_write_prefix=custom_write_prefix,
    ) as saver:
        saver.setup()

        saver.put(config, simple_checkpoint, simple_metadata, {})
        saver.put_writes(config, writes=[("channel1", "value1")], task_id="task_1")

        # Verify keys exist
        checkpoint_keys = list(saver._redis.scan_iter(f"{custom_checkpoint_prefix}:*"))
        write_keys = list(saver._redis.scan_iter(f"{custom_write_prefix}:*"))
        assert len(checkpoint_keys) > 0
        assert len(write_keys) > 0

        # Delete thread
        saver.delete_thread(thread_id)

        # Verify all keys are deleted
        checkpoint_keys_after = list(saver._redis.scan_iter(f"{custom_checkpoint_prefix}:*"))
        write_keys_after = list(saver._redis.scan_iter(f"{custom_write_prefix}:*"))
        assert len(checkpoint_keys_after) == 0
        assert len(write_keys_after) == 0


def test_custom_write_prefix_isolation(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
) -> None:
    """Test that different write prefixes create isolated writes."""
    # Saver 1 with write prefix "app1_write"
    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix="app1_checkpoint",
        checkpoint_write_prefix="app1_write",
    ) as saver1:
        saver1.setup()
        saver1.put(config, simple_checkpoint, simple_metadata, {})
        saver1.put_writes(config, writes=[("channel1", "app1_value")], task_id="task_1")

    # Saver 2 with write prefix "app2_write"
    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix="app2_checkpoint",
        checkpoint_write_prefix="app2_write",
    ) as saver2:
        saver2.setup()
        saver2.put(config, simple_checkpoint, simple_metadata, {})
        saver2.put_writes(config, writes=[("channel1", "app2_value")], task_id="task_1")

    # Verify isolation - each saver should only see its own writes
    with RedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix="app1_checkpoint",
        checkpoint_write_prefix="app1_write",
    ) as saver1:
        saver1.setup()
        app1_write_keys = list(saver1._redis.scan_iter("app1_write:*"))
        app2_write_keys = list(saver1._redis.scan_iter("app2_write:*"))
        assert len(app1_write_keys) > 0
        assert len(app2_write_keys) > 0

        retrieved = saver1.get_tuple(config)
        assert retrieved is not None


@pytest.mark.asyncio
async def test_custom_checkpoint_prefix_async_all_components(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    thread_id: str,
    checkpoint_id: str,
) -> None:
    """Test AsyncRedisSaver with custom checkpoint, blob, and write prefixes."""
    custom_checkpoint_prefix = "async_myapp_checkpoint"
    custom_blob_prefix = "async_myapp_checkpoint_blob"
    custom_write_prefix = "async_myapp_checkpoint_write"

    async with AsyncRedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix=custom_checkpoint_prefix,
        checkpoint_blob_prefix=custom_blob_prefix,
        checkpoint_write_prefix=custom_write_prefix,
    ) as saver:
        await saver.setup()

        assert saver._checkpoint_prefix == custom_checkpoint_prefix
        assert saver._checkpoint_blob_prefix == custom_blob_prefix
        assert saver._checkpoint_write_prefix == custom_write_prefix

        await saver.aput(config, simple_checkpoint, simple_metadata, {})

        # Verify that the custom checkpoint prefix is used
        keys = [k async for k in saver._redis.scan_iter("*")]
        prefixes = {k.split(b":")[0] for k in keys}
        assert len(keys) > 0
        assert prefixes == {
            custom_checkpoint_prefix.encode(),
            b"checkpoint_latest",
        }

        # Test write prefix by adding writes
        await saver.aput_writes(
            config,
            writes=[("channel1", "value1"), ("channel2", "value2")],
            task_id="task_1",
        )

        # Verify that the custom write prefix is used
        all_keys = [k async for k in saver._redis.scan_iter("*")]
        write_keys = [k for k in all_keys if k.startswith(custom_write_prefix.encode())]
        assert len(write_keys) == 2, f"Expected 2 write keys, found {len(write_keys)}"

        # Verify write keys have correct structure
        for key in write_keys:
            key_str = key.decode()
            assert key_str.startswith(custom_write_prefix)

        # Verify blob key generation
        blob_key = saver._make_redis_checkpoint_blob_key(
            thread_id, "", "test_channel", "v1"
        )
        assert blob_key.startswith(custom_blob_prefix)

        # Verify data can be retrieved
        retrieved = await saver.aget_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint["id"] == checkpoint_id


@pytest.mark.asyncio
async def test_custom_checkpoint_prefix_isolation_async(
    redis_url: str,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    checkpoint_id: str,
) -> None:
    """Test that different checkpoint prefixes create isolated async savers."""
    # Saver 1 with prefix "async_app1_checkpoint"
    async with AsyncRedisSaver.from_conn_string(
        redis_url, checkpoint_prefix="async_app1_checkpoint"
    ) as saver1:
        await saver1.setup()

        # Saver 2 with prefix "async_app2_checkpoint"
        async with AsyncRedisSaver.from_conn_string(
                redis_url, checkpoint_prefix="async_app2_checkpoint"
        ) as saver2:
            await saver2.setup()

            checkpoint1: Checkpoint = {
                "v": 1,
                "id": checkpoint_id,
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"app": "app1", "value": "from_app1"},
                "channel_versions": {"app": "1"},
                "versions_seen": {"agent": {"app": "1"}},
                "pending_sends": [],
            }
            await saver1.aput(config, checkpoint1, simple_metadata, {})


            checkpoint2: Checkpoint = {
                "v": 1,
                "id": checkpoint_id,
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {"app": "app2", "value": "from_app2"},
                "channel_versions": {"app": "1"},
                "versions_seen": {"agent": {"app": "1"}},
                "pending_sends": [],
            }
            await saver2.aput(config, checkpoint2, simple_metadata, {})

            retrieved1 = await saver1.aget_tuple(config)
            assert retrieved1 is not None
            assert retrieved1.checkpoint["channel_values"]["app"] == "app1"
            assert retrieved1.checkpoint["channel_values"]["value"] == "from_app1"

            retrieved2 = await saver2.aget_tuple(config)
            assert retrieved2 is not None
            assert retrieved2.checkpoint["channel_values"]["app"] == "app2"
            assert retrieved2.checkpoint["channel_values"]["value"] == "from_app2"


@pytest.mark.asyncio
async def test_custom_checkpoint_prefix_async_delete_thread(
    redis_url: str,
    simple_checkpoint: Checkpoint,
    simple_metadata: CheckpointMetadata,
    config: RunnableConfig,
    thread_id: str,
) -> None:
    """Test delete_thread works correctly with custom prefixes in async mode."""
    custom_checkpoint_prefix = "async_delete_test_checkpoint"
    custom_write_prefix = "async_delete_test_write"

    async with AsyncRedisSaver.from_conn_string(
        redis_url,
        checkpoint_prefix=custom_checkpoint_prefix,
        checkpoint_write_prefix=custom_write_prefix,
    ) as saver:
        await saver.setup()

        await saver.aput(config, simple_checkpoint, simple_metadata, {})
        await saver.aput_writes(config, writes=[("channel1", "value1")], task_id="task_1")

        # Verify keys exist
        checkpoint_keys = [k async for k in saver._redis.scan_iter(f"{custom_checkpoint_prefix}:*")]
        write_keys = [k async for k in saver._redis.scan_iter(f"{custom_write_prefix}:*")]
        assert len(checkpoint_keys) > 0
        assert len(write_keys) > 0

        # Delete thread
        await saver.adelete_thread(thread_id)

        # Verify all keys are deleted
        checkpoint_keys_after = [k async for k in saver._redis.scan_iter(f"{custom_checkpoint_prefix}:*")]
        write_keys_after = [k async for k in saver._redis.scan_iter(f"{custom_write_prefix}:*")]
        assert len(checkpoint_keys_after) == 0
        assert len(write_keys_after) == 0
