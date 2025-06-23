"""Test for AsyncRedisSaver aget_tuple checkpoint_id issue (GitHub issue #64)."""

import asyncio
import uuid
from typing import AsyncGenerator

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.redis.aio import AsyncRedisSaver


@pytest.fixture
async def saver(redis_url: str) -> AsyncGenerator[AsyncRedisSaver, None]:
    """Async saver fixture for this test."""
    saver = AsyncRedisSaver(redis_url)
    await saver.asetup()
    yield saver


@pytest.mark.asyncio
async def test_aget_tuple_returns_correct_checkpoint_id(saver: AsyncRedisSaver):
    """Test that aget_tuple returns the correct checkpoint_id when not specified in config.
    
    This test reproduces the issue described in GitHub issue #64 where AsyncRedisSaver
    aget_tuple was returning None for checkpoint_id while the sync version worked correctly.
    """
    # Create a unique thread ID
    thread_id = str(uuid.uuid4())
    
    # Config with only thread_id and checkpoint_ns (no checkpoint_id)
    runnable_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    
    # Put several checkpoints
    checkpoint_ids = []
    for run in range(3):
        checkpoint_id = str(run)
        checkpoint_ids.append(checkpoint_id)
        
        await saver.aput(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": "",
                }
            },
            empty_checkpoint(),
            {
                "source": "loop",
                "step": run,
                "writes": {},
            },
            {},
        )
        
        # Get the tuple using the config without checkpoint_id
        # This should return the latest checkpoint
        get_tuple = await saver.aget_tuple(runnable_config)
        
        # Verify the checkpoint_id is not None and matches the expected value
        assert get_tuple is not None, f"Expected checkpoint tuple, got None for run {run}"
        
        returned_checkpoint_id = get_tuple.config["configurable"]["checkpoint_id"]
        assert returned_checkpoint_id is not None, (
            f"Expected checkpoint_id to be set, got None for run {run}. "
            f"This indicates the bug where aget_tuple returns None for checkpoint_id."
        )
        
        # Since we're getting the latest checkpoint each time, it should be the current checkpoint_id
        assert returned_checkpoint_id == checkpoint_id, (
            f"Expected checkpoint_id {checkpoint_id}, got {returned_checkpoint_id} for run {run}"
        )


@pytest.mark.asyncio
async def test_aget_tuple_with_explicit_checkpoint_id(saver: AsyncRedisSaver):
    """Test that aget_tuple works correctly when checkpoint_id is explicitly provided."""
    # Create a unique thread ID
    thread_id = str(uuid.uuid4())
    
    # Put several checkpoints
    checkpoint_ids = []
    for run in range(3):
        checkpoint_id = str(run)
        checkpoint_ids.append(checkpoint_id)
        
        await saver.aput(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": "",
                }
            },
            empty_checkpoint(),
            {
                "source": "loop",
                "step": run,
                "writes": {},
            },
            {},
        )
    
    # Test retrieving each checkpoint by explicit checkpoint_id
    for checkpoint_id in checkpoint_ids:
        config_with_id: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": ""
            }
        }
        
        get_tuple = await saver.aget_tuple(config_with_id)
        
        assert get_tuple is not None, f"Expected checkpoint tuple, got None for checkpoint_id {checkpoint_id}"
        
        returned_checkpoint_id = get_tuple.config["configurable"]["checkpoint_id"]
        assert returned_checkpoint_id == checkpoint_id, (
            f"Expected checkpoint_id {checkpoint_id}, got {returned_checkpoint_id}"
        )


@pytest.mark.asyncio
async def test_aget_tuple_no_checkpoint_returns_none(saver: AsyncRedisSaver):
    """Test that aget_tuple returns None when no checkpoint exists for the thread."""
    # Use a thread ID that doesn't exist
    thread_id = str(uuid.uuid4())
    
    runnable_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    
    get_tuple = await saver.aget_tuple(runnable_config)
    assert get_tuple is None, "Expected None when no checkpoint exists for thread"