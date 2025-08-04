"""Integration tests for checkpoint metadata handling and special channel operations.

This file tests:
- Checkpoint metadata serialization with null character handling
- Special channel write operations (__start__, __end__, __error__, __interrupt__)
- TTL (Time To Live) configuration for checkpoint writes
- Checkpoint version number generation and edge cases
- Write operation dumps and structure validation
- Mixed regular and special channel operations
"""

from contextlib import contextmanager
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis import RedisSaver

# Use the shared redis_url fixture from conftest.py instead of creating our own


@contextmanager
def _saver(redis_url: str):
    """Create a checkpoint saver with proper setup and teardown."""
    saver = RedisSaver(redis_url)
    saver.setup()
    try:
        yield saver
    finally:
        # Don't flush the entire database - it removes indices
        pass


def test_load_blobs_method(redis_url: str) -> None:
    """Test _load_blobs method with various scenarios.

    This covers lines 297-299 in base.py
    """
    with _saver(redis_url) as saver:
        # Test 1: Empty blob_values
        result = saver._load_blobs({})
        assert result == {}

        # Test 2: None blob_values
        result = saver._load_blobs(None)  # type: ignore
        assert result == {}

        # Test 3: Blob values with different types
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        blob_values = {}

        for key, value in test_data.items():
            type_, blob = saver.serde.dumps_typed(value)
            blob_values[key] = {"type": type_, "blob": blob}

        # Add an "empty" type that should be filtered out
        blob_values["empty_key"] = {"type": "empty", "blob": b""}

        # Load blobs
        loaded = saver._load_blobs(blob_values)

        # Verify all non-empty values are loaded correctly
        assert len(loaded) == 3  # Should not include empty_key
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        assert loaded["list"] == [1, 2, 3]
        assert "empty_key" not in loaded


def test_metadata_conversion_methods(redis_url: str) -> None:
    """Test _load_metadata and _dump_metadata methods.

    This covers lines 338 and 351 in base.py
    """
    with _saver(redis_url) as saver:
        # Test 1: Simple metadata
        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {"task1": "value1"},
            "parents": {},
        }

        # Dump metadata
        dumped = saver._dump_metadata(metadata)
        assert isinstance(dumped, str)
        assert "\\u0000" not in dumped  # Null chars should be removed

        # Test 2: Metadata with null characters
        metadata_with_nulls: CheckpointMetadata = {
            "source": "test\x00with\x00nulls",
            "step": 2,
            "writes": {"key": "value\x00null"},
            "parents": {},
        }

        dumped_nulls = saver._dump_metadata(metadata_with_nulls)
        assert isinstance(dumped_nulls, str)
        assert "\x00" not in dumped_nulls
        assert "\\u0000" not in dumped_nulls

        # Test 3: Load metadata
        test_metadata_dict = {
            "source": "loaded",
            "step": 3,
            "writes": {"loaded": "data"},
            "parents": {"parent1": {"checkpoint_id": "123"}},
        }

        loaded = saver._load_metadata(test_metadata_dict)
        assert loaded["source"] == "loaded"
        assert loaded["step"] == 3
        assert loaded["writes"]["loaded"] == "data"
        assert "parent1" in loaded["parents"]


def test_get_next_version_more_cases(redis_url: str) -> None:
    """Test get_next_version with additional edge cases.

    This provides more coverage for the get_next_version method.
    """
    with _saver(redis_url) as saver:
        # Mock channel
        class MockChannel:
            pass

        channel = MockChannel()

        # Test with string version that has decimal part
        version_with_decimal = "00000000000000000000000000000042.9876543210"
        next_version = saver.get_next_version(version_with_decimal, channel)
        assert next_version.startswith("00000000000000000000000000000043.")

        # Test incrementing from 0
        version_zero = saver.get_next_version(None, channel)
        assert version_zero.startswith("00000000000000000000000000000001.")

        # Test with current as string "0"
        version_str_zero = saver.get_next_version("0", channel)
        assert version_str_zero.startswith("00000000000000000000000000000001.")


def test_put_writes_edge_cases(redis_url: str) -> None:
    """Test put_writes method with various edge cases.

    This covers more of lines 419-493 in base.py
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "edge-case-test",
                "checkpoint_ns": "test-ns",  # Test with namespace
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Test 1: Writes with WRITES_IDX_MAP channels (special channels)
        special_writes = [
            ("__start__", {"started": True}),
            ("__end__", {"completed": True}),
            ("__error__", {"error": "test error"}),
            ("__interrupt__", {"interrupted": True}),
        ]

        saver.put_writes(saved_config, special_writes, "special_task")

        # Verify writes were stored
        result = saver.get_tuple(saved_config)
        assert result is not None
        pending_writes = result.pending_writes

        # Find our special writes
        special_channels = {w[1] for w in pending_writes if w[0] == "special_task"}
        assert "__start__" in special_channels
        assert "__end__" in special_channels
        assert "__error__" in special_channels
        assert "__interrupt__" in special_channels

        # Test 2: Update existing writes (UPSERT case)
        updated_writes = [
            ("__start__", {"started": True, "timestamp": "2024-01-01"}),
        ]

        saver.put_writes(saved_config, updated_writes, "special_task")

        # Verify the write was updated
        result2 = saver.get_tuple(saved_config)
        assert result2 is not None

        # Test 3: Mixed special and regular channels
        mixed_writes = [
            ("regular_channel", "regular_value"),
            ("__error__", {"error": "another error"}),
            ("another_channel", {"data": "test"}),
        ]

        saver.put_writes(saved_config, mixed_writes, "mixed_task")

        # Test 4: Empty task_path (default parameter)
        path_writes = [("path_channel", "path_value")]
        saver.put_writes(
            saved_config, path_writes, "path_task", task_path="custom/path"
        )

        # Verify all writes
        final_result = saver.get_tuple(saved_config)
        assert final_result is not None
        assert len(final_result.pending_writes) > 0


@contextmanager
def _saver_with_ttl(redis_url: str, ttl_config: dict):
    """Create a checkpoint saver with TTL config."""
    saver = RedisSaver(redis_url, ttl=ttl_config)
    saver.setup()
    try:
        yield saver
    finally:
        # Don't flush the entire database - it removes indices
        pass


def test_put_writes_with_ttl(redis_url: str) -> None:
    """Test put_writes with TTL configuration.

    This tests TTL application in put_writes method.
    """
    # Create saver with TTL config
    ttl_config = {"default_ttl": 0.1}  # 6 seconds TTL
    with _saver_with_ttl(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Add writes - should have TTL applied
        writes = [("ttl_channel", "ttl_value")]
        saver.put_writes(saved_config, writes, "ttl_task")

        # Get the actual checkpoint ID from saved_config
        actual_checkpoint_id = saved_config["configurable"]["checkpoint_id"]

        # Check that TTL was applied to write keys using the actual checkpoint ID
        write_keys = list(
            saver._redis.scan_iter(
                match=f"checkpoint_write:{thread_id}:*:{actual_checkpoint_id}:ttl_task:*"
            )
        )

        assert len(write_keys) > 0

        # Verify TTL is set
        for key in write_keys:
            ttl = saver._redis.ttl(key)
            assert ttl > 0 and ttl <= 6


def test_dump_writes_method(redis_url: str) -> None:
    """Test _dump_writes method directly.

    This covers line 314 and the method implementation.
    """
    with _saver(redis_url) as saver:
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        checkpoint_id = "test-checkpoint"
        task_id = "test-task"

        # Test various write scenarios
        writes = [
            ("channel1", "simple_string"),
            ("channel2", {"complex": "object", "nested": {"data": 123}}),
            ("channel3", b"binary_data"),
            ("__error__", {"error": "test_error"}),  # Special channel
        ]

        # Call _dump_writes
        dumped = saver._dump_writes(
            thread_id, checkpoint_ns, checkpoint_id, task_id, writes
        )

        assert len(dumped) == 4

        # Verify structure of dumped writes
        for i, dumped_write in enumerate(dumped):
            assert "thread_id" in dumped_write
            assert "checkpoint_ns" in dumped_write
            assert "checkpoint_id" in dumped_write
            assert "task_id" in dumped_write
            assert "idx" in dumped_write
            assert "channel" in dumped_write
            assert "type" in dumped_write
            assert "blob" in dumped_write

            # Check special channel gets special index
            if writes[i][0] == "__error__":
                assert dumped_write["idx"] == WRITES_IDX_MAP["__error__"]
            else:
                assert dumped_write["idx"] == i


def test_get_next_version_edge_cases(redis_url: str) -> None:
    """Test get_next_version with edge cases.

    This covers line 360 and related logic.
    """
    with _saver(redis_url) as saver:
        # Mock channel
        class MockChannel:
            pass

        channel = MockChannel()

        # Test with integer current version
        version = saver.get_next_version(10, channel)  # type: ignore
        assert version.startswith("00000000000000000000000000000011.")

        # Test with very large integer
        large_version = saver.get_next_version(999999, channel)  # type: ignore
        assert large_version.startswith("00000000000000000000000001000000.")

        # Test version parsing from string with decimal
        existing_version = "00000000000000000000000000000005.1234567890123456"
        next_version = saver.get_next_version(existing_version, channel)
        assert next_version.startswith("00000000000000000000000000000006.")
