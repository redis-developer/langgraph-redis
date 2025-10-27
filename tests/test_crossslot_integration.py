"""Integration tests for CrossSlot error fix in checkpoint operations."""

from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langgraph.checkpoint.redis import RedisSaver


def test_checkpoint_operations_no_crossslot_errors(redis_url: str) -> None:
    """Test that checkpoint operations work without CrossSlot errors.

    This test verifies that the fix for using search indexes instead of keys()
    works correctly in a real Redis environment.
    """
    # Create a saver
    saver = RedisSaver(redis_url)
    saver.setup()

    # Create test data
    thread_id = "test-thread-crossslot"
    checkpoint_ns = "test-ns"

    # Create checkpoints with unique IDs
    checkpoint1 = create_checkpoint(empty_checkpoint(), {}, 1)
    checkpoint2 = create_checkpoint(checkpoint1, {"messages": ["hello"]}, 2)
    checkpoint3 = create_checkpoint(checkpoint2, {"messages": ["hello", "world"]}, 3)

    # Create metadata
    metadata1 = {"source": "input", "step": 1, "writes": {"task1": "value1"}}
    metadata2 = {"source": "loop", "step": 2, "writes": {"task2": "value2"}}
    metadata3 = {"source": "loop", "step": 3, "writes": {"task3": "value3"}}

    # Put checkpoints with writes
    config1 = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    config2 = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}
    config3 = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}

    # Put checkpoints first to get configs with checkpoint_ids
    saved_config1 = saver.put(config1, checkpoint1, metadata1, {})
    saved_config2 = saver.put(config2, checkpoint2, metadata2, {})
    saved_config3 = saver.put(config3, checkpoint3, metadata3, {})

    # Add some pending writes using saved configs
    saver.put_writes(
        saved_config1,
        [
            ("channel1", {"value": "data1"}),
            ("channel2", {"value": "data2"}),
        ],
        "task-1",
    )

    # Now test operations that previously used keys() and would fail in cluster mode

    # Test 1: Load pending writes (uses _load_pending_writes)
    # This should work without CrossSlot errors
    tuple1 = saver.get_tuple(saved_config1)
    assert tuple1 is not None
    # Verify pending writes were loaded
    assert len(tuple1.pending_writes) == 2
    pending_channels = [w[1] for w in tuple1.pending_writes]
    assert "channel1" in pending_channels
    assert "channel2" in pending_channels

    # Test 2: Get tuple with TTL (uses get_tuple which searches for blob and write keys)
    saver_with_ttl = RedisSaver(redis_url, ttl={"checkpoint": 3600})
    saver_with_ttl.setup()

    # Put a checkpoint with TTL
    config_ttl = {
        "configurable": {"thread_id": "ttl-thread", "checkpoint_ns": "ttl-ns"}
    }
    saver_with_ttl.put(config_ttl, checkpoint1, metadata1, {})

    # Get the checkpoint - this triggers TTL application which uses key searches
    tuple_ttl = saver_with_ttl.get_tuple(config_ttl)
    assert tuple_ttl is not None

    # Test 3: List checkpoints - this should work without CrossSlot errors
    # List returns all checkpoints
    checkpoints = list(saver.list(config1))
    assert len(checkpoints) >= 1

    # Find the checkpoint that has the pending writes (saved_config1)
    checkpoint_with_writes = None
    saved_checkpoint_id = saved_config1["configurable"]["checkpoint_id"]
    for checkpoint in checkpoints:
        if checkpoint.checkpoint["id"] == saved_checkpoint_id:
            checkpoint_with_writes = checkpoint
            break

    assert checkpoint_with_writes is not None
    assert len(checkpoint_with_writes.pending_writes) == 2

    # The important part is that all these operations work without CrossSlot errors
    # In a Redis cluster, the old keys() based approach would have failed by now


def test_subgraph_checkpoint_operations(redis_url: str) -> None:
    """Test checkpoint operations with subgraphs work without CrossSlot errors."""
    saver = RedisSaver(redis_url)
    saver.setup()

    # Create nested namespace checkpoints
    thread_id = "test-thread-subgraph"

    # Parent checkpoint
    parent_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }
    parent_checkpoint = empty_checkpoint()
    parent_metadata = {"source": "input", "step": 1}

    # Child checkpoint in subgraph
    child_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "subgraph1",
        }
    }
    child_checkpoint = create_checkpoint(parent_checkpoint, {"subgraph": "data"}, 1)
    child_metadata = {"source": "loop", "step": 1}

    # Grandchild checkpoint in nested subgraph
    grandchild_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "subgraph1:subgraph2",
        }
    }
    grandchild_checkpoint = create_checkpoint(child_checkpoint, {"nested": "data"}, 2)
    grandchild_metadata = {"source": "loop", "step": 2}

    # Put all checkpoints first to get saved configs
    saved_parent_config = saver.put(
        parent_config, parent_checkpoint, parent_metadata, {}
    )
    saved_child_config = saver.put(child_config, child_checkpoint, child_metadata, {})
    saved_grandchild_config = saver.put(
        grandchild_config, grandchild_checkpoint, grandchild_metadata, {}
    )

    # Put checkpoints with writes using saved configs
    saver.put_writes(
        saved_parent_config, [("parent_channel", {"parent": "data"})], "parent-task"
    )
    saver.put_writes(
        saved_child_config, [("child_channel", {"child": "data"})], "child-task"
    )
    saver.put_writes(
        saved_grandchild_config,
        [("grandchild_channel", {"grandchild": "data"})],
        "grandchild-task",
    )

    # Test loading checkpoints with pending writes from different namespaces
    parent_tuple = saver.get_tuple(parent_config)
    assert parent_tuple is not None

    child_tuple = saver.get_tuple(child_config)
    assert child_tuple is not None

    grandchild_tuple = saver.get_tuple(grandchild_config)
    assert grandchild_tuple is not None

    # List all checkpoints - should work without CrossSlot errors
    all_checkpoints = list(saver.list({"configurable": {"thread_id": thread_id}}))
    assert len(all_checkpoints) >= 3
