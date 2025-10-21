"""Test for issue #106: alist should sort by checkpoint ID DESC."""

import asyncio
import time
from typing import AsyncGenerator, Generator

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


@pytest.fixture
async def async_saver(redis_url: str) -> AsyncGenerator[AsyncRedisSaver, None]:
    """Async saver fixture."""
    saver = AsyncRedisSaver(redis_url)
    await saver.asetup()
    yield saver


@pytest.fixture
def sync_saver(redis_url: str) -> Generator[RedisSaver, None, None]:
    """Sync saver fixture."""
    saver = RedisSaver(redis_url)
    saver.setup()
    yield saver


@pytest.mark.asyncio
async def test_alist_sorts_by_checkpoint_id_desc(async_saver: AsyncRedisSaver) -> None:
    """Test that alist returns checkpoints sorted by checkpoint ID in descending order.

    This is a reproducer for issue #106: when listing checkpoints, they should be
    sorted by checkpoint ID (which embeds timestamp via ULID) in descending order,
    so that the most recent checkpoints appear first. This allows users to efficiently
    find crashed/unfinished sessions after restart.
    """
    thread_id = "test-thread-sort"
    checkpoint_ns = ""

    # Create multiple checkpoints with increasing timestamps
    # We'll use explicit checkpoint IDs with different timestamps to ensure ordering
    checkpoint_ids = []

    # Create 5 checkpoints with small delays between them to ensure different timestamps
    for i in range(5):
        # Create a checkpoint with a unique ULID
        checkpoint_id = str(ULID())
        checkpoint_ids.append(checkpoint_id)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        checkpoint: Checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": i,
            "writes": {},
        }

        await async_saver.aput(config, checkpoint, metadata, {})

        # Small delay to ensure different ULID timestamps
        # ULID has millisecond precision, so we need to wait at least 1ms
        await asyncio.sleep(0.01)

    # Now list all checkpoints for this thread
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }

    listed_checkpoints = []
    async for checkpoint_tuple in async_saver.alist(config):
        listed_checkpoints.append(
            checkpoint_tuple.config["configurable"]["checkpoint_id"]
        )

    # Verify we got all checkpoints
    assert (
        len(listed_checkpoints) == 5
    ), f"Expected 5 checkpoints, got {len(listed_checkpoints)}"

    # Verify they are sorted in descending order (most recent first)
    # Since we created them in chronological order, the last one created should be first
    # checkpoint_ids[4] should appear first, then checkpoint_ids[3], etc.
    expected_order = checkpoint_ids[::-1]  # Reverse the list

    assert listed_checkpoints == expected_order, (
        f"Checkpoints are not sorted in descending order by checkpoint ID.\n"
        f"Expected: {expected_order}\n"
        f"Got:      {listed_checkpoints}"
    )


@pytest.mark.asyncio
async def test_alist_sorts_multiple_threads(async_saver: AsyncRedisSaver) -> None:
    """Test that alist sorts correctly when filtering by thread_id."""
    # Create checkpoints for two different threads
    thread1_ids = []
    thread2_ids = []

    # Thread 1: Create 3 checkpoints
    for i in range(3):
        checkpoint_id = str(ULID())
        thread1_ids.append(checkpoint_id)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1-sort",
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id

        metadata: CheckpointMetadata = {"source": "test", "step": i, "writes": {}}
        await async_saver.aput(config, checkpoint, metadata, {})
        await asyncio.sleep(0.01)

    # Thread 2: Create 3 checkpoints (interleaved with thread 1)
    for i in range(3):
        checkpoint_id = str(ULID())
        thread2_ids.append(checkpoint_id)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2-sort",
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint: Checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id

        metadata: CheckpointMetadata = {"source": "test", "step": i, "writes": {}}
        await async_saver.aput(config, checkpoint, metadata, {})
        await asyncio.sleep(0.01)

    # List checkpoints for thread 1
    config1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1-sort",
            "checkpoint_ns": "",
        }
    }

    thread1_listed = []
    async for checkpoint_tuple in async_saver.alist(config1):
        thread1_listed.append(checkpoint_tuple.config["configurable"]["checkpoint_id"])

    # Verify thread 1 checkpoints are in descending order
    assert thread1_listed == thread1_ids[::-1], (
        f"Thread 1 checkpoints not sorted correctly.\n"
        f"Expected: {thread1_ids[::-1]}\n"
        f"Got:      {thread1_listed}"
    )

    # List checkpoints for thread 2
    config2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2-sort",
            "checkpoint_ns": "",
        }
    }

    thread2_listed = []
    async for checkpoint_tuple in async_saver.alist(config2):
        thread2_listed.append(checkpoint_tuple.config["configurable"]["checkpoint_id"])

    # Verify thread 2 checkpoints are in descending order
    assert thread2_listed == thread2_ids[::-1], (
        f"Thread 2 checkpoints not sorted correctly.\n"
        f"Expected: {thread2_ids[::-1]}\n"
        f"Got:      {thread2_listed}"
    )


def test_list_sorts_by_checkpoint_id_desc(sync_saver: RedisSaver) -> None:
    """Test that list (sync) returns checkpoints sorted by checkpoint ID in descending order.

    This is a sync version of the test for issue #106.
    """
    thread_id = "test-thread-sort-sync"
    checkpoint_ns = ""

    # Create multiple checkpoints with increasing timestamps
    checkpoint_ids = []

    # Create 5 checkpoints with small delays between them to ensure different timestamps
    for i in range(5):
        # Create a checkpoint with a unique ULID
        checkpoint_id = str(ULID())
        checkpoint_ids.append(checkpoint_id)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        checkpoint: Checkpoint = empty_checkpoint()
        checkpoint["id"] = checkpoint_id

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": i,
            "writes": {},
        }

        sync_saver.put(config, checkpoint, metadata, {})

        # Small delay to ensure different ULID timestamps
        time.sleep(0.01)

    # Now list all checkpoints for this thread
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }

    listed_checkpoints = []
    for checkpoint_tuple in sync_saver.list(config):
        listed_checkpoints.append(
            checkpoint_tuple.config["configurable"]["checkpoint_id"]
        )

    # Verify we got all checkpoints
    assert (
        len(listed_checkpoints) == 5
    ), f"Expected 5 checkpoints, got {len(listed_checkpoints)}"

    # Verify they are sorted in descending order (most recent first)
    expected_order = checkpoint_ids[::-1]  # Reverse the list

    assert listed_checkpoints == expected_order, (
        f"Checkpoints are not sorted in descending order by checkpoint ID.\n"
        f"Expected: {expected_order}\n"
        f"Got:      {listed_checkpoints}"
    )
