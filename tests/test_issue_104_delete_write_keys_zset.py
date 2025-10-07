"""Test for issue #104 - delete_thread should clean up write_keys_zset keys.

When delete_thread is called, it should remove all related keys including:
- checkpoint keys
- checkpoint_latest pointers
- blob keys
- write keys
- write_keys_zset (registry) keys

The issue reports that write_keys_zset keys are not being deleted properly.
"""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.util import to_storage_safe_id, to_storage_safe_str


def test_delete_thread_cleans_write_keys_zset(redis_url, client):
    """Test that delete_thread removes write_keys_zset keys created by put_writes."""
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        # Create a checkpoint with writes
        thread_id = "test-thread-zset-cleanup"
        checkpoint_ns = ""  # Empty namespace as reported in issue
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": "1",
            }
        }

        checkpoint = Checkpoint(
            v=1,
            id="1",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": ["Test"]},
            channel_versions={"messages": "1"},
            versions_seen={"agent": {"messages": "1"}},
            pending_sends=[],
            tasks=[],
        )

        # Store checkpoint
        checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        # Add writes which should create write_keys_zset entries
        checkpointer.put_writes(
            config=config,
            writes=[("messages", "Write 1"), ("messages", "Write 2")],
            task_id="task-1",
        )

        # Construct the expected write_keys_zset key
        # Format: write_keys_zset:thread_id:checkpoint_ns:checkpoint_id
        safe_thread_id = to_storage_safe_id(thread_id)
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        safe_checkpoint_id = to_storage_safe_id("1")

        zset_key = f"write_keys_zset:{safe_thread_id}:{safe_checkpoint_ns}:{safe_checkpoint_id}"

        # Verify the write_keys_zset key exists
        zset_exists_before = client.exists(zset_key)
        assert (
            zset_exists_before == 1
        ), f"write_keys_zset key should exist before delete: {zset_key}"

        # Get the count of items in the zset
        zset_count = client.zcard(zset_key)
        assert (
            zset_count == 2
        ), f"write_keys_zset should have 2 entries, got {zset_count}"

        # Delete the thread
        checkpointer.delete_thread(thread_id)

        # Verify checkpoint is deleted
        result = checkpointer.get_tuple(config)
        assert result is None, "Checkpoint should be deleted"

        # Verify write_keys_zset key is also deleted (THIS IS THE BUG)
        zset_exists_after = client.exists(zset_key)
        assert (
            zset_exists_after == 0
        ), f"write_keys_zset key should be deleted: {zset_key}"


def test_delete_thread_cleans_multiple_write_keys_zsets(redis_url, client):
    """Test delete_thread with multiple checkpoints and namespaces."""
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        thread_id = "test-thread-multi-zset"

        # Create checkpoints with different namespaces
        checkpoints_data = [
            ("", "1"),
            ("", "2"),
            ("ns1", "3"),
            ("ns2", "4"),
        ]

        zset_keys = []

        for checkpoint_ns, checkpoint_id in checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

            checkpoint = Checkpoint(
                v=1,
                id=checkpoint_id,
                ts=f"2024-01-01T00:00:0{checkpoint_id}Z",
                channel_values={"messages": ["Test"]},
                channel_versions={"messages": "1"},
                versions_seen={"agent": {"messages": "1"}},
                pending_sends=[],
                tasks=[],
            )

            checkpointer.put(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source="input", step=0, writes={}),
                new_versions={"messages": "1"},
            )

            # Add writes
            checkpointer.put_writes(
                config=config,
                writes=[("messages", f"Write for {checkpoint_id}")],
                task_id=f"task-{checkpoint_id}",
            )

            # Track the zset key
            safe_thread_id = to_storage_safe_id(thread_id)
            safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
            safe_checkpoint_id = to_storage_safe_id(checkpoint_id)
            zset_key = f"write_keys_zset:{safe_thread_id}:{safe_checkpoint_ns}:{safe_checkpoint_id}"
            zset_keys.append(zset_key)

        # Verify all zset keys exist
        for zset_key in zset_keys:
            assert client.exists(zset_key) == 1, f"zset key should exist: {zset_key}"

        # Delete the thread
        checkpointer.delete_thread(thread_id)

        # Verify all zset keys are deleted
        for zset_key in zset_keys:
            assert (
                client.exists(zset_key) == 0
            ), f"zset key should be deleted: {zset_key}"


@pytest.mark.asyncio
async def test_adelete_thread_cleans_write_keys_zset(redis_url, async_client):
    """Test that adelete_thread removes write_keys_zset keys (async version)."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        # Create a checkpoint with writes
        thread_id = "test-thread-zset-cleanup-async"
        checkpoint_ns = ""
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": "1",
            }
        }

        checkpoint = Checkpoint(
            v=1,
            id="1",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": ["Test"]},
            channel_versions={"messages": "1"},
            versions_seen={"agent": {"messages": "1"}},
            pending_sends=[],
            tasks=[],
        )

        # Store checkpoint
        await checkpointer.aput(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        # Add writes
        await checkpointer.aput_writes(
            config=config,
            writes=[("messages", "Write 1"), ("messages", "Write 2")],
            task_id="task-1",
        )

        # Construct the expected write_keys_zset key
        safe_thread_id = to_storage_safe_id(thread_id)
        safe_checkpoint_ns = to_storage_safe_str(checkpoint_ns)
        safe_checkpoint_id = to_storage_safe_id("1")

        zset_key = f"write_keys_zset:{safe_thread_id}:{safe_checkpoint_ns}:{safe_checkpoint_id}"

        # Verify the write_keys_zset key exists
        zset_exists_before = await async_client.exists(zset_key)
        assert (
            zset_exists_before == 1
        ), f"write_keys_zset key should exist before delete: {zset_key}"

        # Delete the thread
        await checkpointer.adelete_thread(thread_id)

        # Verify checkpoint is deleted
        result = await checkpointer.aget_tuple(config)
        assert result is None

        # Verify write_keys_zset key is also deleted
        zset_exists_after = await async_client.exists(zset_key)
        assert (
            zset_exists_after == 0
        ), f"write_keys_zset key should be deleted: {zset_key}"


def test_delete_thread_with_only_thread_id(redis_url, client):
    """Test the exact scenario from issue #104: only using thread_id."""
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        # User only provides thread_id (as mentioned in the issue)
        # checkpoint_ns will default to empty string when not provided
        thread_id = "simple-thread-id"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",  # Explicitly set to empty string like in the issue
            }
        }

        checkpoint = Checkpoint(
            v=1,
            id="auto-generated-id",
            ts="2024-01-01T00:00:00Z",
            channel_values={"messages": ["Test"]},
            channel_versions={"messages": "1"},
            versions_seen={"agent": {"messages": "1"}},
            pending_sends=[],
            tasks=[],
        )

        # Store checkpoint
        result_config = checkpointer.put(
            config=config,
            checkpoint=checkpoint,
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        # Extract the actual checkpoint_id that was used
        actual_checkpoint_id = result_config["configurable"]["checkpoint_id"]
        actual_checkpoint_ns = result_config["configurable"].get("checkpoint_ns", "")

        # Add writes
        checkpointer.put_writes(
            config=result_config,
            writes=[("messages", "Some write")],
            task_id="task-1",
        )

        # Construct the expected write_keys_zset key with empty namespace
        safe_thread_id = to_storage_safe_id(thread_id)
        safe_checkpoint_ns = to_storage_safe_str(actual_checkpoint_ns)
        safe_checkpoint_id = to_storage_safe_id(actual_checkpoint_id)

        zset_key = f"write_keys_zset:{safe_thread_id}:{safe_checkpoint_ns}:{safe_checkpoint_id}"

        # Verify zset key exists
        assert client.exists(zset_key) == 1, f"zset key should exist: {zset_key}"

        # Delete using only thread_id (as user does in issue)
        checkpointer.delete_thread(thread_id)

        # Verify zset key is deleted
        assert client.exists(zset_key) == 0, f"zset key should be deleted: {zset_key}"
