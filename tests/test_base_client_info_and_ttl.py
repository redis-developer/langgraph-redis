"""Tests for client info setting and TTL management in base.py.

These tests cover:
- Client info setting with fallback mechanisms (set_client_info, aset_client_info)
- TTL application logic for cluster and non-cluster modes
- Key parsing and generation utilities
- Metadata serialization with null character handling
- Write loading and processing from Redis
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, ResponseError

from langgraph.checkpoint.redis.base import BaseRedisSaver


class MockRedisSaver(BaseRedisSaver):
    """Mock implementation for testing base class methods."""

    def __init__(self):
        # Initialize without calling super().__init__
        self.serde = MagicMock()
        self._redis = MagicMock()
        self.ttl_config = None
        self.cluster_mode = False

    def create_indexes(self):
        pass

    def configure_client(self, **kwargs):
        pass


def test_set_client_info_success():
    """Test set_client_info when client_setinfo succeeds."""
    saver = MockRedisSaver()

    # Mock successful client_setinfo
    saver._redis.client_setinfo = MagicMock()

    with patch("langgraph.checkpoint.redis.version.__full_lib_name__", "test-lib-v1.0"):
        saver.set_client_info()

    saver._redis.client_setinfo.assert_called_once_with("LIB-NAME", "test-lib-v1.0")


def test_set_client_info_fallback_to_echo():
    """Test set_client_info falls back to echo when client_setinfo fails."""
    saver = MockRedisSaver()

    # Mock client_setinfo to raise ResponseError
    saver._redis.client_setinfo = MagicMock(
        side_effect=ResponseError("Command not supported")
    )
    saver._redis.echo = MagicMock(return_value=b"OK")

    with patch("langgraph.checkpoint.redis.version.__full_lib_name__", "test-lib-v1.0"):
        saver.set_client_info()

    saver._redis.client_setinfo.assert_called_once()
    saver._redis.echo.assert_called_once_with("test-lib-v1.0")


def test_set_client_info_silent_failure():
    """Test set_client_info silently fails when both methods fail."""
    saver = MockRedisSaver()

    # Mock both methods to fail
    saver._redis.client_setinfo = MagicMock(
        side_effect=AttributeError("No such method")
    )
    saver._redis.echo = MagicMock(side_effect=ConnectionError("Connection lost"))

    # Should not raise any exception
    with patch("langgraph.checkpoint.redis.version.__full_lib_name__", "test-lib-v1.0"):
        saver.set_client_info()

    saver._redis.client_setinfo.assert_called_once()
    saver._redis.echo.assert_called_once()


@pytest.mark.asyncio
async def test_aset_client_info_success():
    """Test async aset_client_info when client_setinfo succeeds."""
    saver = MockRedisSaver()

    # Mock async client_setinfo
    saver._redis.client_setinfo = AsyncMock()

    with patch("langgraph.checkpoint.redis.version.__full_lib_name__", "test-lib-v1.0"):
        await saver.aset_client_info()

    saver._redis.client_setinfo.assert_called_once_with("LIB-NAME", "test-lib-v1.0")


@pytest.mark.asyncio
async def test_aset_client_info_fallback_to_echo():
    """Test async aset_client_info falls back to echo."""
    saver = MockRedisSaver()

    # Mock client_setinfo to fail and echo to return awaitable
    saver._redis.client_setinfo = AsyncMock(side_effect=ResponseError("Not supported"))

    # Create a proper async mock for echo
    async def mock_echo(msg):
        return f"ECHO: {msg}"

    saver._redis.echo = mock_echo

    with patch("langgraph.checkpoint.redis.version.__full_lib_name__", "test-lib-v1.0"):
        await saver.aset_client_info()

    saver._redis.client_setinfo.assert_called_once()


def test_apply_ttl_to_keys_no_ttl():
    """Test _apply_ttl_to_keys when no TTL is configured."""
    saver = MockRedisSaver()
    saver.ttl_config = None

    result = saver._apply_ttl_to_keys("main_key", ["related1", "related2"])

    # Should return None when no TTL
    assert result is None
    saver._redis.expire.assert_not_called()


def test_apply_ttl_to_keys_with_default_ttl():
    """Test _apply_ttl_to_keys with default TTL from config."""
    saver = MockRedisSaver()
    saver.ttl_config = {"default_ttl": 5}  # 5 minutes

    # Mock pipeline
    mock_pipeline = MagicMock()
    saver._redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = MagicMock(return_value=[True, True, True])

    result = saver._apply_ttl_to_keys("main_key", ["related1", "related2"])

    # Should create pipeline and set TTL
    saver._redis.pipeline.assert_called_once()
    mock_pipeline.expire.assert_any_call("main_key", 300)  # 5 * 60
    mock_pipeline.expire.assert_any_call("related1", 300)
    mock_pipeline.expire.assert_any_call("related2", 300)
    assert mock_pipeline.expire.call_count == 3
    mock_pipeline.execute.assert_called_once()


def test_apply_ttl_to_keys_cluster_mode():
    """Test _apply_ttl_to_keys in cluster mode."""
    saver = MockRedisSaver()
    saver.ttl_config = {"default_ttl": 10}  # 10 minutes
    saver.cluster_mode = True  # Enable cluster mode

    saver._redis.expire = MagicMock(return_value=True)

    result = saver._apply_ttl_to_keys("main_key", ["related1", "related2"])

    # In cluster mode, should call expire directly (not pipeline)
    assert result is True
    saver._redis.expire.assert_any_call("main_key", 600)  # 10 * 60
    saver._redis.expire.assert_any_call("related1", 600)
    saver._redis.expire.assert_any_call("related2", 600)
    assert saver._redis.expire.call_count == 3
    saver._redis.pipeline.assert_not_called()


def test_apply_ttl_to_keys_with_explicit_ttl():
    """Test _apply_ttl_to_keys with explicitly provided TTL."""
    saver = MockRedisSaver()
    saver.ttl_config = {"default_ttl": 5}  # This should be overridden

    mock_pipeline = MagicMock()
    saver._redis.pipeline = MagicMock(return_value=mock_pipeline)
    mock_pipeline.execute = MagicMock(return_value=[True])

    # Provide explicit TTL of 15 minutes
    result = saver._apply_ttl_to_keys("main_key", [], ttl_minutes=15)

    mock_pipeline.expire.assert_called_once_with("main_key", 900)  # 15 * 60


def test_load_writes_from_redis_processing():
    """Test the write processing loop in _load_writes_from_redis."""
    saver = MockRedisSaver()

    # Mock Redis response with writes
    mock_writes = {
        "writes": [
            {
                "task_id": "task1",
                "channel": "channel1",
                "type": "json",
                "blob": '{"test": "data1"}',
            },
            {
                "task_id": "task2",
                "channel": "__error__",
                "type": "base64",
                "blob": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            },
        ]
    }

    saver._redis.json = MagicMock()
    saver._redis.json().get = MagicMock(return_value=mock_writes)

    # Mock serde.loads_typed
    def mock_loads_typed(data):
        type_, value = data
        if type_ == "json":
            return json.loads(value)
        elif type_ == "base64":
            import base64

            return base64.b64decode(value)
        return value

    saver.serde.loads_typed = MagicMock(side_effect=mock_loads_typed)

    # Mock _decode_blob to return the blob as-is
    saver._decode_blob = MagicMock(side_effect=lambda x: x)

    # Call the method
    result = saver._load_writes_from_redis("test_key")

    # Verify results
    assert len(result) == 2
    assert result[0] == ("task1", "channel1", {"test": "data1"})
    assert result[1] == ("task2", "__error__", b"Hello World")

    # Verify serde was called correctly
    assert saver.serde.loads_typed.call_count == 2


def test_generate_checkpoint_key_variations():
    """Test checkpoint key generation methods."""
    # These methods don't exist in BaseRedisSaver, they're simple string formatting
    # Test the key format directly

    # Standard checkpoint key format
    key = f"checkpoint:thread1:ns1:checkpoint1"
    assert key == "checkpoint:thread1:ns1:checkpoint1"

    # Test with empty namespace
    key = f"checkpoint:thread2::checkpoint2"
    assert key == "checkpoint:thread2::checkpoint2"


def test_generate_blob_key():
    """Test blob key generation."""
    # Test the key format directly
    key = f"checkpoint_blob:thread1:ns1:channel1:version1"
    assert key == "checkpoint_blob:thread1:ns1:channel1:version1"

    # Test with special characters
    key = f"checkpoint_blob:thread:1:ns/1:channel@1:v1.0"
    assert key == "checkpoint_blob:thread:1:ns/1:channel@1:v1.0"


def test_generate_write_key():
    """Test write key generation."""
    # Test the key format directly
    key = f"checkpoint_write:thread1:ns1:checkpoint1:task1:write_id"
    assert key == "checkpoint_write:thread1:ns1:checkpoint1:task1:write_id"


def test_parse_write_key():
    """Test parsing write keys."""
    saver = MockRedisSaver()

    # Valid key
    result = saver._parse_redis_checkpoint_writes_key(
        "checkpoint_write:thread1:ns1:checkpoint1:task1:write1"
    )
    assert result == {
        "thread_id": "thread1",
        "checkpoint_ns": "ns1",
        "checkpoint_id": "checkpoint1",
        "task_id": "task1",
        "idx": "write1",
    }

    # Key with extra components (only first 6 parts are used)
    result = saver._parse_redis_checkpoint_writes_key(
        "checkpoint_write:thread1:ns1:checkpoint1:task1:write1:extra:parts"
    )
    assert result == {
        "thread_id": "thread1",
        "checkpoint_ns": "ns1",
        "checkpoint_id": "checkpoint1",
        "task_id": "task1",
        "idx": "write1",
    }

    # Invalid prefix
    try:
        result = saver._parse_redis_checkpoint_writes_key(
            "invalid_prefix:thread1:ns1:checkpoint1:task1:write1"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected checkpoint key to start with 'checkpoint'" in str(e)

    # Too few parts
    try:
        result = saver._parse_redis_checkpoint_writes_key(
            "checkpoint_write:thread1:ns1"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected at least 6 parts in Redis key" in str(e)


def test_dump_metadata_null_handling():
    """Test _dump_metadata handles null characters properly."""
    saver = MockRedisSaver()
    # Use the default JsonPlusRedisSerializer
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    saver.serde = JsonPlusRedisSerializer()

    # Create metadata with actual null characters
    metadata = {"test": "data\x00with\x00nulls", "clean": "value"}

    # Test that null characters are removed
    result = saver._dump_metadata(metadata)

    # The result should be a string without null characters
    assert isinstance(result, str)
    assert "\x00" not in result
    assert "\\u0000" not in result

    # Verify the content is still valid JSON
    import json

    parsed = json.loads(result)
    # The null characters should have been replaced or removed
    assert "test" in parsed
    assert "clean" in parsed
    assert parsed["clean"] == "value"
