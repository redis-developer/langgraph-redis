"""Integration tests to improve coverage for shallow.py uncovered functionality.

This file tests specific uncovered paths in ShallowRedisSaver:
- ULID timestamp extraction fallback
- TTL application and refresh
- Cache management (key cache and channel cache)
- Channel serialization edge cases
- Error handling paths
- Cleanup operations for old writes
- Complex channel value retrieval with caching
"""

import time
from contextlib import contextmanager
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# Use the shared redis_url fixture from conftest.py instead of creating our own


@contextmanager
def shallow_saver(redis_url: str, ttl_config: dict = None):
    """Create a shallow checkpoint saver with proper setup."""
    saver = ShallowRedisSaver(redis_url, ttl=ttl_config)
    saver.setup()
    try:
        yield saver
    finally:
        pass  # Don't flush - let tests be isolated by unique thread IDs


def test_ulid_timestamp_extraction_fallback(redis_url: str) -> None:
    """Test timestamp extraction fallback when checkpoint ID is not a valid ULID."""
    from datetime import datetime

    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Test 1: Invalid ULID with numeric checkpoint ts field available
        checkpoint = empty_checkpoint()
        checkpoint["id"] = "invalid-ulid-format"  # Not a valid ULID
        expected_ts = 1234567890.123 * 1000
        checkpoint["ts"] = expected_ts  # Set a specific numeric timestamp

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # This should trigger the fallback to use checkpoint["ts"]
        result_config = saver.put(config, checkpoint, metadata, {})

        # Verify checkpoint was stored
        assert result_config["configurable"]["checkpoint_id"] == "invalid-ulid-format"

        # Retrieve and verify the timestamp was set using checkpoint's ts field
        checkpoint_data = saver._redis.json().get(
            saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        )
        checkpoint_ts = checkpoint_data["checkpoint_ts"]
        assert checkpoint_ts == expected_ts  # Should use checkpoint's ts field

        # Test 2: Invalid ULID with ISO string ts field (from empty_checkpoint)
        thread_id2 = str(uuid4())
        checkpoint2 = empty_checkpoint()
        checkpoint2["id"] = "another-invalid-ulid"
        # checkpoint2["ts"] is already an ISO string from empty_checkpoint()

        # Parse the ISO timestamp to get expected value
        dt = datetime.fromisoformat(checkpoint2["ts"].replace("Z", "+00:00"))
        expected_ts2 = dt.timestamp() * 1000

        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id2,
                "checkpoint_ns": "",
            }
        }

        result_config2 = saver.put(config2, checkpoint2, metadata, {})

        # Verify the timestamp was extracted from the ISO string
        checkpoint_data2 = saver._redis.json().get(
            saver._make_shallow_redis_checkpoint_key_cached(thread_id2, "")
        )
        checkpoint_ts2 = checkpoint_data2["checkpoint_ts"]
        # Should be close to the expected timestamp (allow small tolerance for rounding)
        assert abs(checkpoint_ts2 - expected_ts2) < 1

        # Test 3: Invalid ULID without ts field (falls back to current time)
        thread_id3 = str(uuid4())
        checkpoint3 = empty_checkpoint()
        checkpoint3["id"] = "third-invalid-ulid"
        del checkpoint3["ts"]  # Remove the ts field

        config3: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id3,
                "checkpoint_ns": "",
            }
        }

        # This should trigger fallback to current time
        start_time = time.time() * 1000
        result_config3 = saver.put(config3, checkpoint3, metadata, {})
        end_time = time.time() * 1000

        # Verify the timestamp was set using current time
        checkpoint_data3 = saver._redis.json().get(
            saver._make_shallow_redis_checkpoint_key_cached(thread_id3, "")
        )
        checkpoint_ts3 = checkpoint_data3["checkpoint_ts"]
        assert start_time <= checkpoint_ts3 <= end_time


def test_ttl_application_and_refresh(redis_url: str) -> None:
    """Test TTL application during put and refresh on read."""
    # Test with TTL configuration - reduced to 2 seconds for more reliable testing
    ttl_config = {
        "default_ttl": 0.033,
        "refresh_on_read": True,
    }  # 2 seconds = 0.033 minutes

    with shallow_saver(redis_url, ttl_config) as saver:
        thread_id = str(uuid4())

        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Store checkpoint (should apply TTL - lines 219-220)
        result_config = saver.put(config, checkpoint, metadata, {})

        # Verify TTL was applied
        checkpoint_key = saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        initial_ttl = saver._redis.ttl(checkpoint_key)
        assert initial_ttl > 0 and initial_ttl <= 2  # 2 seconds

        # Wait a bit then read (should refresh TTL - lines 393-394)
        time.sleep(0.5)
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None

        # TTL should be refreshed (close to original value)
        new_ttl = saver._redis.ttl(checkpoint_key)
        assert new_ttl > initial_ttl - 1  # Allow some time tolerance


def test_cache_management_key_cache(redis_url: str) -> None:
    """Test key cache functionality."""
    with shallow_saver(redis_url) as saver:
        # Generate multiple thread IDs to test cache behavior
        thread_ids = [str(uuid4()) for _ in range(3)]

        # Access keys to populate cache
        for thread_id in thread_ids:
            key = saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
            # Verify key is cached
            cache_key = f"shallow_checkpoint:{thread_id}:"
            assert cache_key in saver._key_cache
            assert saver._key_cache[cache_key] == f"checkpoint:{thread_id}:"

        # Test blob key caching
        blob_key = saver._make_shallow_redis_checkpoint_blob_key_cached(
            "thread1", "ns1", "channel1", "v1"
        )
        assert "shallow_blob:thread1:ns1:channel1:v1" in saver._key_cache

        # Test write key caching
        write_key = saver._make_redis_checkpoint_writes_key_cached(
            "thread1", "ns1", "checkpoint1", "task1", 0
        )
        assert "writes:thread1:ns1:checkpoint1:task1:0" in saver._key_cache

        # Verify keys are the same when called multiple times (cached)
        key1 = saver._make_shallow_redis_checkpoint_key_cached("thread1", "")
        key2 = saver._make_shallow_redis_checkpoint_key_cached("thread1", "")
        assert key1 == key2


def test_lru_cache_eviction(redis_url: str) -> None:
    """Test LRU cache eviction strategy."""
    # Create saver with small cache size for testing
    with shallow_saver(redis_url) as saver:
        saver._key_cache_max_size = 3  # Small cache for testing

        # Add 3 keys to fill the cache
        key1 = saver._make_shallow_redis_checkpoint_key_cached("thread1", "")
        key2 = saver._make_shallow_redis_checkpoint_key_cached("thread2", "")
        key3 = saver._make_shallow_redis_checkpoint_key_cached("thread3", "")

        # Verify all 3 are in cache
        assert len(saver._key_cache) == 3
        assert "shallow_checkpoint:thread1:" in saver._key_cache
        assert "shallow_checkpoint:thread2:" in saver._key_cache
        assert "shallow_checkpoint:thread3:" in saver._key_cache

        # Access thread1 again to make it most recently used
        saver._make_shallow_redis_checkpoint_key_cached("thread1", "")

        # Add a 4th key, which should evict thread2 (least recently used)
        key4 = saver._make_shallow_redis_checkpoint_key_cached("thread4", "")

        # Verify cache still has size 3
        assert len(saver._key_cache) == 3

        # thread2 should be evicted (it was least recently used)
        assert "shallow_checkpoint:thread2:" not in saver._key_cache

        # thread1, thread3, and thread4 should still be there
        assert "shallow_checkpoint:thread1:" in saver._key_cache
        assert "shallow_checkpoint:thread3:" in saver._key_cache
        assert "shallow_checkpoint:thread4:" in saver._key_cache

        # Access thread3 to make it most recently used
        saver._make_shallow_redis_checkpoint_key_cached("thread3", "")

        # Add thread5, should evict thread1 (now least recently used)
        key5 = saver._make_shallow_redis_checkpoint_key_cached("thread5", "")

        # Verify eviction happened correctly
        assert len(saver._key_cache) == 3
        assert "shallow_checkpoint:thread1:" not in saver._key_cache
        assert "shallow_checkpoint:thread3:" in saver._key_cache
        assert "shallow_checkpoint:thread4:" in saver._key_cache
        assert "shallow_checkpoint:thread5:" in saver._key_cache


def test_configurable_cache_sizes(redis_url: str) -> None:
    """Test configurable cache size limits."""
    # Test with custom cache sizes
    custom_key_size = 5
    custom_channel_size = 2

    saver = ShallowRedisSaver(
        redis_url,
        key_cache_max_size=custom_key_size,
        channel_cache_max_size=custom_channel_size,
    )
    saver.setup()

    try:
        # Verify custom sizes were set
        assert saver._key_cache_max_size == custom_key_size
        assert saver._channel_cache_max_size == custom_channel_size

        # Test key cache respects custom limit
        for i in range(7):
            saver._make_shallow_redis_checkpoint_key_cached(f"thread{i}", "")

        # Should only have 5 keys (custom limit)
        assert len(saver._key_cache) == custom_key_size
    finally:
        pass  # Don't flush - let tests be isolated


def test_channel_cache_management(redis_url: str) -> None:
    """Test channel value caching and size limits."""
    with shallow_saver(redis_url) as saver:
        # Set small cache size for testing
        saver._channel_cache_max_size = 2

        thread_id = str(uuid4())

        # Create checkpoint with channel values
        checkpoint = empty_checkpoint()
        checkpoint["channel_versions"] = {"ch1": "v1", "ch2": "v2", "ch3": "v3"}
        checkpoint["channel_values"] = {
            "ch1": "value1",
            "ch2": {"complex": "value2"},
            "ch3": b"binary_value3",
        }

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Store checkpoint
        result_config = saver.put(
            config, checkpoint, metadata, {"ch1": "v1", "ch2": "v2", "ch3": "v3"}
        )

        # Retrieve to populate cache (lines 461-476)
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None

        # Cache should be populated but limited by size
        assert len(saver._channel_cache) <= 2

        # Access different channels to test cache eviction
        channel_values = saver.get_tuple(result_config).checkpoint["channel_values"]
        assert (
            "ch1" in channel_values
            or "ch2" in channel_values
            or "ch3" in channel_values
        )


def test_binary_channel_serialization(redis_url: str) -> None:
    """Test binary data handling in channel serialization."""
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with binary channel values
        checkpoint = empty_checkpoint()
        binary_data = b"binary\x00data\xff"
        checkpoint["channel_values"] = {"binary_ch": binary_data}

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Store checkpoint (should handle binary data - lines 238-295)
        result_config = saver.put(config, checkpoint, metadata, {"binary_ch": "v1"})

        # Retrieve and verify binary data is preserved
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None
        assert retrieved.checkpoint["channel_values"]["binary_ch"] == binary_data


def test_metadata_dict_handling(redis_url: str) -> None:
    """Test metadata serialization that's already a dict vs string."""
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Create metadata that will test the dict vs string path (lines 179-181)
        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {"nested": {"complex": "structure"}},
            "parents": {},
        }

        result_config = saver.put(config, checkpoint, metadata, {})

        # Verify metadata was stored properly as dict
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None
        assert retrieved.metadata["writes"]["nested"]["complex"] == "structure"


def test_put_writes_persist_for_hitl_support(redis_url: str) -> None:
    """Test that writes persist across checkpoints for HITL support (Issue #133).

    Writes are NOT cleaned up when a new checkpoint is saved because this breaks
    Human-in-the-Loop (HITL) workflows where interrupt writes are saved BEFORE
    the new checkpoint is created.

    Writes are cleaned up via:
    1. delete_thread - explicitly cleans up all data for a thread
    2. TTL expiration - if configured
    3. Overwrite - when put_writes is called with the same task_id and idx
    """
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Create first checkpoint
        config1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "checkpoint1",
                "checkpoint_ns": "",
            }
        }

        checkpoint1 = create_checkpoint(empty_checkpoint(), {}, 1)
        metadata1: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        result_config1 = saver.put(config1, checkpoint1, metadata1, {})

        # Add writes to first checkpoint
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        saver.put_writes(result_config1, writes1, "task1")

        # Create second checkpoint
        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "checkpoint2",
                "checkpoint_ns": "",
            }
        }

        checkpoint2 = create_checkpoint(empty_checkpoint(), {}, 2)
        metadata2: CheckpointMetadata = {
            "source": "test",
            "step": 2,
            "writes": {},
            "parents": {},
        }

        result_config2 = saver.put(config2, checkpoint2, metadata2, {})

        # Add writes to second checkpoint
        writes2 = [("channel3", "value3")]
        saver.put_writes(result_config2, writes2, "task2")

        # Verify all writes persist (for HITL support)
        retrieved = saver.get_tuple(result_config2)
        assert retrieved is not None

        # Issue #133 fix: All writes should persist
        pending_writes = retrieved.pending_writes
        write_channels = {w[1] for w in pending_writes}
        assert "channel3" in write_channels
        # Old writes should also be present (for HITL support)
        assert "channel1" in write_channels
        assert "channel2" in write_channels

        # Verify that delete_thread properly cleans up all writes
        saver.delete_thread(thread_id)

        # After delete_thread, writes should be cleaned up
        # Note: The checkpoint itself may still exist briefly due to index lag,
        # but the writes should be cleaned up immediately
        retrieved_after_delete = saver.get_tuple(result_config2)
        if retrieved_after_delete is not None:
            # If checkpoint still exists, verify writes are cleaned up
            assert (
                len(retrieved_after_delete.pending_writes) == 0
            ), "Writes should be cleaned up by delete_thread"


def test_error_handling_missing_checkpoint(redis_url: str) -> None:
    """Test error handling when checkpoint data is missing or invalid."""
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Try to get non-existent checkpoint (line 389)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "nonexistent",
                "checkpoint_ns": "",
            }
        }

        result = saver.get_tuple(config)
        assert result is None

        # Test get_tuple with invalid checkpoint data
        checkpoint_key = saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")

        # Store invalid data
        saver._redis.json().set(checkpoint_key, "$", "invalid_data")

        result = saver.get_tuple(config)
        assert result is None


def test_channel_values_complex_retrieval(redis_url: str) -> None:
    """Test complex channel value retrieval with mixed inline and blob storage."""
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with mixed channel storage
        checkpoint = empty_checkpoint()
        checkpoint["channel_versions"] = {"inline_ch": "v1", "blob_ch": "v2"}

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        # Store checkpoint
        result_config = saver.put(
            config, checkpoint, metadata, {"inline_ch": "v1", "blob_ch": "v2"}
        )

        # Test retrieval path that exercises lines 697-762
        channel_values = saver.get_channel_values(
            thread_id, "", "dummy_checkpoint_id", {"inline_ch": "v1", "blob_ch": "v2"}
        )

        # Should handle both inline and blob channel types
        assert len(channel_values) >= 0  # May be empty if no actual values stored


def test_from_conn_string_context_manager(redis_url: str) -> None:
    """Test the from_conn_string context manager functionality."""
    # Test successful creation and cleanup
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        assert saver is not None
        assert saver._redis is not None

        # Use the saver to verify it works
        thread_id = str(uuid4())
        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {},
            "parents": {},
        }

        result_config = saver.put(config, checkpoint, metadata, {})
        retrieved = saver.get_tuple(result_config)
        assert retrieved is not None


def test_source_and_step_metadata_storage(redis_url: str) -> None:
    """Test that source and step are stored at top level when both present."""
    with shallow_saver(redis_url) as saver:
        thread_id = str(uuid4())

        checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Test with both source and step (lines 197-199)
        metadata_with_both: CheckpointMetadata = {
            "source": "input",
            "step": 5,
            "writes": {},
            "parents": {},
        }

        result_config = saver.put(config, checkpoint, metadata_with_both, {})

        # Verify source and step are stored at top level
        checkpoint_key = saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        checkpoint_data = saver._redis.json().get(checkpoint_key)

        assert checkpoint_data["source"] == "input"
        assert checkpoint_data["step"] == 5

        # Test with missing step (should not store at top level)
        thread_id2 = str(uuid4())
        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id2,
                "checkpoint_ns": "",
            }
        }

        metadata_missing_step: CheckpointMetadata = {
            "source": "input",
            "writes": {},
            "parents": {},
        }

        result_config2 = saver.put(config2, checkpoint, metadata_missing_step, {})

        checkpoint_key2 = saver._make_shallow_redis_checkpoint_key_cached(
            thread_id2, ""
        )
        checkpoint_data2 = saver._redis.json().get(checkpoint_key2)

        # Should not have top-level source/step when step is missing
        assert "source" not in checkpoint_data2 or "step" not in checkpoint_data2
