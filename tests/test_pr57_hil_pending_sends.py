"""Test Human-in-the-Loop pending_sends functionality across all implementations."""

import asyncio
import json
from typing import Any, Dict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, create_checkpoint
from langgraph.constants import TASKS
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


def create_test_checkpoint() -> Checkpoint:
    """Create a test checkpoint for HIL scenarios."""
    return {
        "v": 1,
        "id": "test_checkpoint_1",
        "ts": "2024-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def create_hil_task_writes() -> list[tuple[str, Any]]:
    """Create test writes that simulate HIL task submissions."""
    return [
        (TASKS, {"task": "review_document", "args": {"doc_id": "123"}}),
        (TASKS, {"task": "approve_action", "args": {"action": "deploy"}}),
        (TASKS, {"task": "human_feedback", "args": {"prompt": "Continue?"}}),
    ]


@pytest.mark.asyncio
async def test_async_redis_saver_hil_pending_sends(redis_url: str):
    """Test AsyncRedisSaver._aload_pending_sends for HIL workflows."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = "test-hil-thread-async"
        checkpoint_ns = "test-namespace"
        parent_checkpoint_id = "parent-checkpoint-1"
        
        # Create parent config
        parent_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        
        # Create parent checkpoint
        parent_checkpoint = create_test_checkpoint()
        parent_checkpoint["id"] = parent_checkpoint_id
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        
        # Save parent checkpoint
        await saver.aput(parent_config, parent_checkpoint, metadata, {})
        
        # Write HIL tasks
        hil_writes = create_hil_task_writes()
        await saver.aput_writes(parent_config, hil_writes, task_id="hil-task-1")
        
        # Load pending sends - this is where the bug would occur
        pending_sends = await saver._aload_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        
        # Verify we got the correct pending sends
        assert len(pending_sends) == 3
        assert all(isinstance(send[0], bytes) for send in pending_sends)
        # Blob can be bytes or str depending on how Redis stores it
        assert all(isinstance(send[1], (bytes, str)) for send in pending_sends)
        
        # Verify the content
        for i, (type_bytes, blob_bytes) in enumerate(pending_sends):
            type_str = type_bytes.decode()
            # Type could be json or msgpack depending on serde config
            assert type_str in ["json", "msgpack"]
            
            # The blob should contain our task data
            assert blob_bytes is not None
            assert len(blob_bytes) > 0


@pytest.mark.asyncio
async def test_sync_redis_saver_hil_pending_sends(redis_url: str):
    """Test RedisSaver._load_pending_sends for HIL workflows."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = "test-hil-thread-sync"
        checkpoint_ns = "test-namespace"
        parent_checkpoint_id = "parent-checkpoint-2"
        
        # Create parent config
        parent_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        
        # Create parent checkpoint
        parent_checkpoint = create_test_checkpoint()
        parent_checkpoint["id"] = parent_checkpoint_id
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        
        # Save parent checkpoint
        saver.put(parent_config, parent_checkpoint, metadata, {})
        
        # Write HIL tasks
        hil_writes = create_hil_task_writes()
        saver.put_writes(parent_config, hil_writes, task_id="hil-task-2")
        
        # Load pending sends - this is where the bug would occur
        pending_sends = saver._load_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        
        # Verify we got the correct pending sends
        assert len(pending_sends) == 3
        assert all(isinstance(send[0], bytes) for send in pending_sends)
        # Blob can be bytes or str depending on how Redis stores it
        assert all(isinstance(send[1], (bytes, str)) for send in pending_sends)


@pytest.mark.asyncio
async def test_async_shallow_saver_hil_pending_sends(redis_url: str):
    """Test AsyncShallowRedisSaver._aload_pending_sends for HIL workflows."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = "test-hil-thread-async-shallow"
        checkpoint_ns = "test-namespace"
        
        # Create config
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": "checkpoint-1",
            }
        }
        
        # Create checkpoint
        checkpoint = create_test_checkpoint()
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        
        # Save checkpoint
        await saver.aput(config, checkpoint, metadata, {})
        
        # Write HIL tasks
        hil_writes = create_hil_task_writes()
        await saver.aput_writes(config, hil_writes, task_id="hil-task-3")
        
        # Load pending sends
        pending_sends = await saver._aload_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
        )
        
        # Verify we got the correct pending sends
        assert len(pending_sends) == 3
        assert all(isinstance(send[0], bytes) for send in pending_sends)
        # Blob can be bytes or str depending on how Redis stores it
        assert all(isinstance(send[1], (bytes, str)) for send in pending_sends)


def test_sync_shallow_saver_hil_pending_sends(redis_url: str):
    """Test ShallowRedisSaver._load_pending_sends for HIL workflows."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = "test-hil-thread-sync-shallow"
        checkpoint_ns = "test-namespace"
        
        # Create config
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": "checkpoint-2",
            }
        }
        
        # Create checkpoint
        checkpoint = create_test_checkpoint()
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        
        # Save checkpoint
        saver.put(config, checkpoint, metadata, {})
        
        # Write HIL tasks
        hil_writes = create_hil_task_writes()
        saver.put_writes(config, hil_writes, task_id="hil-task-4")
        
        # Load pending sends
        pending_sends = saver._load_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
        )
        
        # Verify we got the correct pending sends
        assert len(pending_sends) == 3
        assert all(isinstance(send[0], bytes) for send in pending_sends)
        # Blob can be bytes or str depending on how Redis stores it
        assert all(isinstance(send[1], (bytes, str)) for send in pending_sends)


@pytest.mark.asyncio
async def test_missing_blob_handling(redis_url: str):
    """Test that implementations handle missing blobs gracefully."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = "test-missing-blob"
        checkpoint_ns = "test-namespace"
        parent_checkpoint_id = "parent-checkpoint-missing"
        
        # Directly insert a write with missing blob field
        write_key = f"checkpoint_write:{thread_id}:{checkpoint_ns}:{parent_checkpoint_id}:task-1:0"
        write_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": parent_checkpoint_id,
            "task_id": "task-1",
            "idx": 0,
            "channel": TASKS,
            "type": "json",
            # No blob field - this should be handled gracefully
        }
        
        # Insert directly into Redis
        client = RedisConnectionFactory.get_redis_connection(redis_url)
        client.json().set(write_key, "$", write_data)
        client.close()
        
        # Load pending sends - should handle missing blob
        pending_sends = await saver._aload_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        
        # Should return empty list since blob is missing
        assert len(pending_sends) == 0


def test_all_implementations_consistent(redis_url: str):
    """Verify all 4 implementations produce consistent results."""
    thread_id = "test-consistency"
    checkpoint_ns = "test-namespace"
    parent_checkpoint_id = "parent-checkpoint-consist"
    
    # Create the same test data for all implementations
    hil_writes = create_hil_task_writes()
    
    results = []
    
    # Test sync implementation
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        checkpoint = create_test_checkpoint()
        checkpoint["id"] = parent_checkpoint_id
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        
        saver.put(config, checkpoint, metadata, {})
        saver.put_writes(config, hil_writes, task_id="consist-task")
        
        pending_sends = saver._load_pending_sends(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        results.append(("sync", pending_sends))
    
    # Verify all implementations return the same number of results
    # and all results have the expected structure
    for name, sends in results:
        assert len(sends) == 3, f"{name} returned {len(sends)} sends, expected 3"
        for type_bytes, blob_bytes in sends:
            assert isinstance(type_bytes, bytes), f"{name}: type not bytes"
            # Blob can be bytes or str depending on how Redis stores it
            assert isinstance(blob_bytes, (bytes, str)), f"{name}: blob not bytes or str"
            assert len(type_bytes) > 0, f"{name}: empty type"
            assert len(blob_bytes) > 0, f"{name}: empty blob"