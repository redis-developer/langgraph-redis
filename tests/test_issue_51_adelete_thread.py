"""Test for issue #51 - adelete_thread implementation."""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


@pytest.mark.asyncio
async def test_adelete_thread_implemented(redis_url):
    """Test that adelete_thread method is now implemented in AsyncRedisSaver."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        # Create a checkpoint
        thread_id = "test-thread-to-delete"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
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

        # Verify checkpoint exists
        result = await checkpointer.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "1"

        # Delete the thread
        await checkpointer.adelete_thread(thread_id)

        # Verify checkpoint is deleted
        result = await checkpointer.aget_tuple(config)
        assert result is None


def test_delete_thread_implemented(redis_url):
    """Test that delete_thread method is now implemented in RedisSaver."""
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()  # Initialize Redis indices

        # Create a checkpoint
        thread_id = "test-thread-to-delete-sync"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
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

        # Verify checkpoint exists
        result = checkpointer.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == "1"

        # Delete the thread
        checkpointer.delete_thread(thread_id)

        # Verify checkpoint is deleted
        result = checkpointer.get_tuple(config)
        assert result is None


@pytest.mark.asyncio
async def test_adelete_thread_comprehensive(redis_url):
    """Comprehensive test for adelete_thread with multiple checkpoints and namespaces."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        thread_id = "test-thread-comprehensive"
        other_thread_id = "other-thread"

        # Create multiple checkpoints for the thread
        checkpoints_data = [
            ("", "1", {"messages": ["First"]}, "input", 0),
            ("", "2", {"messages": ["Second"]}, "output", 1),
            ("ns1", "3", {"messages": ["Third"]}, "input", 0),
            ("ns2", "4", {"messages": ["Fourth"]}, "output", 1),
        ]

        # Also create checkpoints for another thread that should not be deleted
        other_checkpoints_data = [
            ("", "5", {"messages": ["Other1"]}, "input", 0),
            ("ns1", "6", {"messages": ["Other2"]}, "output", 1),
        ]

        # Store all checkpoints
        for ns, cp_id, channel_values, source, step in checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }

            checkpoint = Checkpoint(
                v=1,
                id=cp_id,
                ts=f"2024-01-01T00:00:0{cp_id}Z",
                channel_values=channel_values,
                channel_versions={"messages": "1"},
                versions_seen={"agent": {"messages": "1"}},
                pending_sends=[],
                tasks=[],
            )

            await checkpointer.aput(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source=source, step=step, writes={}),
                new_versions={"messages": "1"},
            )

            # Also add some writes
            await checkpointer.aput_writes(
                config=config,
                writes=[("messages", f"Write for {cp_id}")],
                task_id=f"task-{cp_id}",
            )

        # Store checkpoints for other thread
        for ns, cp_id, channel_values, source, step in other_checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": other_thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }

            checkpoint = Checkpoint(
                v=1,
                id=cp_id,
                ts=f"2024-01-01T00:00:0{cp_id}Z",
                channel_values=channel_values,
                channel_versions={"messages": "1"},
                versions_seen={"agent": {"messages": "1"}},
                pending_sends=[],
                tasks=[],
            )

            await checkpointer.aput(
                config=config,
                checkpoint=checkpoint,
                metadata=CheckpointMetadata(source=source, step=step, writes={}),
                new_versions={"messages": "1"},
            )

        # Verify all checkpoints exist
        for ns, cp_id, _, _, _ in checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }
            result = await checkpointer.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == cp_id

        # Verify other thread checkpoints exist
        for ns, cp_id, _, _, _ in other_checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": other_thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }
            result = await checkpointer.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == cp_id

        # Delete the thread
        await checkpointer.adelete_thread(thread_id)

        # Verify all checkpoints for the thread are deleted
        for ns, cp_id, _, _, _ in checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }
            result = await checkpointer.aget_tuple(config)
            assert result is None

        # Verify other thread checkpoints still exist
        for ns, cp_id, _, _, _ in other_checkpoints_data:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": other_thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": cp_id,
                }
            }
            result = await checkpointer.aget_tuple(config)
            assert result is not None
            assert result.checkpoint["id"] == cp_id
