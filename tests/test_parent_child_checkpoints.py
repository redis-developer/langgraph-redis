"""Test parent-child checkpoint relationships in Redis implementation.

This test verifies that the Redis implementation correctly handles parent-child
checkpoint relationships like the reference Postgres and MongoDB implementations.
"""

import uuid

import pytest
from redis import Redis

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def create_test_checkpoint(checkpoint_id: str, messages=None):
    """Create a test checkpoint with the given ID."""
    return {
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00",
        "v": 1,
        "channel_values": {"messages": messages or []},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_parent_child_checkpoint_sync(redis_url: str):
    """Test that sync implementation stores parent-child relationships correctly."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"test-parent-child-{uuid.uuid4()}"

        # Save parent checkpoint
        parent_id = str(uuid.uuid4())
        parent_checkpoint = create_test_checkpoint(parent_id, ["message1"])

        parent_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        returned_config = saver.put(
            parent_config,
            parent_checkpoint,
            {"source": "parent"},
            parent_checkpoint["channel_versions"],
        )

        assert returned_config["configurable"]["checkpoint_id"] == parent_id

        # Save child checkpoint with parent reference
        child_id = str(uuid.uuid4())
        child_checkpoint = create_test_checkpoint(child_id, ["message1", "message2"])

        child_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": parent_id,  # Reference to parent
            }
        }

        returned_config = saver.put(
            child_config,
            child_checkpoint,
            {"source": "child"},
            child_checkpoint["channel_versions"],
        )

        assert returned_config["configurable"]["checkpoint_id"] == child_id

        # List checkpoints - should have both
        checkpoints = list(
            saver.list(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                }
            )
        )

        assert len(checkpoints) == 2

        # Find parent and child
        parent_tuple = None
        child_tuple = None
        for cp in checkpoints:
            if cp.checkpoint["id"] == parent_id:
                parent_tuple = cp
            elif cp.checkpoint["id"] == child_id:
                child_tuple = cp

        assert parent_tuple is not None
        assert child_tuple is not None

        # Verify parent has no parent
        assert parent_tuple.parent_config is None

        # Verify child has parent reference
        assert child_tuple.parent_config is not None
        assert child_tuple.parent_config["configurable"]["checkpoint_id"] == parent_id


@pytest.mark.asyncio
async def test_parent_child_checkpoint_async(redis_url: str):
    """Test that async implementation stores parent-child relationships correctly."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        thread_id = f"test-parent-child-async-{uuid.uuid4()}"

        # Save parent checkpoint
        parent_id = str(uuid.uuid4())
        parent_checkpoint = create_test_checkpoint(parent_id, ["message1"])

        parent_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        returned_config = await saver.aput(
            parent_config,
            parent_checkpoint,
            {"source": "parent"},
            parent_checkpoint["channel_versions"],
        )

        assert returned_config["configurable"]["checkpoint_id"] == parent_id

        # Save child checkpoint with parent reference
        child_id = str(uuid.uuid4())
        child_checkpoint = create_test_checkpoint(child_id, ["message1", "message2"])

        child_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": parent_id,  # Reference to parent
            }
        }

        returned_config = await saver.aput(
            child_config,
            child_checkpoint,
            {"source": "child"},
            child_checkpoint["channel_versions"],
        )

        assert returned_config["configurable"]["checkpoint_id"] == child_id

        # List checkpoints - should have both
        checkpoints = []
        async for cp in saver.alist(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }
        ):
            checkpoints.append(cp)

        assert len(checkpoints) == 2

        # Find parent and child
        parent_tuple = None
        child_tuple = None
        for cp in checkpoints:
            if cp.checkpoint["id"] == parent_id:
                parent_tuple = cp
            elif cp.checkpoint["id"] == child_id:
                child_tuple = cp

        assert parent_tuple is not None
        assert child_tuple is not None

        # Verify parent has no parent
        assert parent_tuple.parent_config is None

        # Verify child has parent reference
        assert child_tuple.parent_config is not None
        assert child_tuple.parent_config["configurable"]["checkpoint_id"] == parent_id


def test_multiple_checkpoints_stored_separately(redis_url: str):
    """Test that Redis stores multiple checkpoints separately, not updating in place."""
    # Clear Redis first
    redis_client = Redis.from_url(redis_url)
    redis_client.flushall()
    redis_client.close()

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"test-multiple-{uuid.uuid4()}"

        # Save 3 checkpoints in sequence
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = str(uuid.uuid4())
            checkpoint_ids.append(checkpoint_id)

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }

            # For checkpoints after the first, reference the previous as parent
            if i > 0:
                config["configurable"]["checkpoint_id"] = checkpoint_ids[i - 1]

            checkpoint = create_test_checkpoint(
                checkpoint_id, [f"message{j}" for j in range(i + 1)]
            )

            saver.put(
                config,
                checkpoint,
                {"source": f"step_{i}", "step": i},
                checkpoint["channel_versions"],
            )

        # List all checkpoints
        checkpoints = list(
            saver.list(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                }
            )
        )

        # Should have all 3 checkpoints
        assert len(checkpoints) == 3

        # Verify all checkpoint IDs are present
        found_ids = {cp.checkpoint["id"] for cp in checkpoints}
        assert found_ids == set(checkpoint_ids)

        # Verify parent relationships
        for cp in checkpoints:
            idx = checkpoint_ids.index(cp.checkpoint["id"])
            if idx == 0:
                # First checkpoint has no parent
                assert cp.parent_config is None
            else:
                # Others have previous checkpoint as parent
                assert cp.parent_config is not None
                assert (
                    cp.parent_config["configurable"]["checkpoint_id"]
                    == checkpoint_ids[idx - 1]
                )
