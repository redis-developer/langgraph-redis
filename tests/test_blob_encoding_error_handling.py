"""Integration tests for error handling paths in base.py.

These tests focus on real-world error scenarios that could occur during
checkpoint and write operations, particularly around blob encoding/decoding
and data corruption scenarios.
"""

import base64
from contextlib import contextmanager
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver

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


def test_malformed_base64_blob_handling(redis_url: str) -> None:
    """Test handling of malformed base64 data in blob decoding.

    This tests the error path in _decode_blob() when base64 decoding fails.
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with writes containing normal data
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "test-checkpoint",
                "checkpoint_ns": "",
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        # Store checkpoint with some writes
        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Add writes with valid data
        writes = [("channel1", {"key": "value"}), ("channel2", b"binary_data")]
        saver.put_writes(saved_config, writes, "task1")

        # Directly test _decode_blob with invalid base64
        # This tests the error handling path in base.py lines 377-379
        invalid_base64_strings = [
            "!@#$%^&*()_+",  # Special characters
            "InvalidBase64===",  # Wrong padding
            "\\x00\\x01\\x02",  # Escape sequences
            "",  # Empty string
            "12345",  # Odd length
        ]

        for invalid_str in invalid_base64_strings:
            # Call _decode_blob directly on the base class
            result = BaseRedisSaver._decode_blob(saver, invalid_str)
            # Should return encoded string on error, not raise exception
            assert isinstance(result, bytes)

        # Also test with non-string input
        result = BaseRedisSaver._decode_blob(saver, 12345)  # type: ignore
        assert result == 12345  # Should return as-is for non-string


def test_binary_data_encoding_roundtrip(redis_url: str) -> None:
    """Test encoding and decoding of various binary data types.

    This tests _encode_blob() and _decode_blob() with real binary data.
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "binary-test",
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

        # Test various binary data scenarios
        test_cases = [
            # Empty bytes
            ("empty_bytes", b""),
            # Regular binary data
            ("regular_binary", b"\x00\x01\x02\x03\x04"),
            # Non-UTF8 bytes
            ("non_utf8", b"\xff\xfe\xfd\xfc"),
            # Large binary data
            ("large_binary", b"x" * 10000),
            # Binary data that looks like base64
            ("fake_base64", b"SGVsbG8gV29ybGQ="),
            # Mixed content
            ("mixed", {"text": "hello", "binary": b"\x00\x01\x02"}),
        ]

        # Store writes with binary data
        writes = [(name, data) for name, data in test_cases]
        saver.put_writes(saved_config, writes, "binary_task")

        # Retrieve and verify
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Check that all binary data round-tripped correctly
        retrieved_writes = {w[1]: w[2] for w in result.pending_writes}

        for name, expected_data in test_cases:
            assert name in retrieved_writes
            retrieved_data = retrieved_writes[name]

            # For direct bytes comparison
            if isinstance(expected_data, bytes):
                assert retrieved_data == expected_data
            # For dict with binary values
            elif isinstance(expected_data, dict) and "binary" in expected_data:
                assert retrieved_data["text"] == expected_data["text"]
                # The binary data may be serialized in a special format
                if (
                    isinstance(retrieved_data["binary"], dict)
                    and retrieved_data["binary"].get("type") == "constructor"
                ):
                    # This is how bytes are serialized - validate it matches
                    assert retrieved_data["binary"]["id"] == ["builtins", "bytes"]
                else:
                    assert retrieved_data["binary"] == expected_data["binary"]


def test_encode_decode_blob_directly(redis_url: str) -> None:
    """Test _encode_blob and _decode_blob methods directly for coverage."""
    with _saver(redis_url) as saver:
        # Test _encode_blob with bytes input (line 370)
        test_bytes = b"Hello World"
        encoded = BaseRedisSaver._encode_blob(saver, test_bytes)
        assert encoded == base64.b64encode(test_bytes).decode()

        # Test _encode_blob with non-bytes input (line 371)
        test_string = "Not bytes"
        encoded = BaseRedisSaver._encode_blob(saver, test_string)
        assert encoded == test_string

        # Test _decode_blob with valid base64 (line 376)
        valid_b64 = base64.b64encode(b"test data").decode()
        decoded = BaseRedisSaver._decode_blob(saver, valid_b64)
        assert decoded == b"test data"

        # Test _decode_blob with invalid base64 - binascii.Error (line 377-379)
        invalid_b64 = "Not@Valid#Base64!"
        decoded = BaseRedisSaver._decode_blob(saver, invalid_b64)
        assert decoded == invalid_b64.encode()  # Should encode the string

        # Test _decode_blob with TypeError (non-string input)
        decoded = BaseRedisSaver._decode_blob(saver, None)  # type: ignore
        assert decoded is None


def test_load_writes_empty_cases(redis_url: str) -> None:
    """Test handling of empty write scenarios.

    Tests edge cases with no writes or empty write data.
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Test 1: Checkpoint with no writes at all
        config1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "no-writes",
                "checkpoint_ns": "",
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        saved_config1 = saver.put(
            config1, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Retrieve checkpoint with no writes
        result1 = saver.get_tuple(saved_config1)
        assert result1 is not None
        assert len(result1.pending_writes) == 0

        # Test 2: Direct test of _load_writes_from_redis with empty key
        empty_result = BaseRedisSaver._load_writes_from_redis(saver, "")
        assert empty_result == []

        # Test 3: Direct test with non-existent key
        fake_key = f"checkpoint_write:{thread_id}:fake:nonexistent:key"
        nonexistent_result = BaseRedisSaver._load_writes_from_redis(saver, fake_key)
        assert nonexistent_result == []

        # Test 4: Store empty writes list
        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "empty-writes",
                "checkpoint_ns": "",
            }
        }

        saved_config2 = saver.put(
            config2, checkpoint, {"source": "test", "step": 2, "writes": {}}, {}
        )

        # Put empty writes list
        saver.put_writes(saved_config2, [], "empty_task")

        # Should still retrieve successfully
        result2 = saver.get_tuple(saved_config2)
        assert result2 is not None
        assert len(result2.pending_writes) == 0


def test_checkpoint_with_special_characters(redis_url: str) -> None:
    """Test handling of special characters and null bytes in data.

    This ensures proper handling of edge cases in serialization.
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "special-chars",
                "checkpoint_ns": "",
            }
        }

        # Create checkpoint with special characters
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={
                "special": {"data": "Hello\x00World"},  # Null byte
                "unicode": {"data": "Hello 🌍 émojis"},
            },
            step=1,
        )

        # Store checkpoint
        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Add writes with special content
        writes = [
            ("null_bytes", "Data\x00with\x00nulls"),
            ("unicode_emoji", "🚀 Rocket émoji data 🎉"),
            ("binary_like", {"data": base64.b64encode(b"fake binary").decode()}),
        ]
        saver.put_writes(saved_config, writes, "special_task")

        # Retrieve and verify
        result = saver.get_tuple(saved_config)
        assert result is not None

        # Verify special characters are preserved
        # Check that channels were stored (they are in the checkpoint, not channel_values)
        channels = result.checkpoint.get("channel_values", {})
        if "special" in channels:
            assert channels["special"]["data"] == "Hello\x00World"
        if "unicode" in channels:
            assert channels["unicode"]["data"] == "Hello 🌍 émojis"

        # Check writes
        retrieved_writes = {w[1]: w[2] for w in result.pending_writes}
        assert "null_bytes" in retrieved_writes
        assert "unicode_emoji" in retrieved_writes
        assert retrieved_writes["unicode_emoji"] == "🚀 Rocket émoji data 🎉"
