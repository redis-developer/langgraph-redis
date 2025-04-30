"""Test Redis key decoding functionality to ensure it works with both decode_responses=True and False."""

import pytest
from redis import Redis
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.util import safely_decode


def test_safely_decode():
    """Test the safely_decode function with both bytes and strings."""
    # Test with bytes
    assert safely_decode(b"test_key") == "test_key"

    # Test with string
    assert safely_decode("test_key") == "test_key"


@pytest.fixture
def redis_client_decoded():
    """Redis client with decode_responses=True."""
    client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def redis_client_bytes():
    """Redis client with decode_responses=False (default)."""
    client = Redis.from_url("redis://localhost:6379", decode_responses=False)
    yield client
    client.close()


def test_redis_keys_with_decode_responses(redis_client_decoded, redis_client_bytes):
    """Test that redis.keys() behaves as expected with different decode_responses settings."""
    # Generate a unique key prefix for this test
    test_key_prefix = "test_decode_responses_"

    # Create some test keys
    for i in range(3):
        key = f"{test_key_prefix}{i}"
        redis_client_bytes.set(key, f"value{i}")

    try:
        # Test with decode_responses=False (returns bytes)
        keys_bytes = redis_client_bytes.keys(f"{test_key_prefix}*")
        assert all(isinstance(k, bytes) for k in keys_bytes)

        # Test with decode_responses=True (returns strings)
        keys_str = redis_client_decoded.keys(f"{test_key_prefix}*")
        assert all(isinstance(k, str) for k in keys_str)

        # Test that our safely_decode function works with both
        decoded_bytes = [safely_decode(k) for k in keys_bytes]
        decoded_str = [safely_decode(k) for k in keys_str]

        # Both should now be lists of strings
        assert all(isinstance(k, str) for k in decoded_bytes)
        assert all(isinstance(k, str) for k in decoded_str)

        # Both should contain the same keys
        assert sorted(decoded_bytes) == sorted(decoded_str)

    finally:
        # Clean up
        for i in range(3):
            redis_client_bytes.delete(f"{test_key_prefix}{i}")


def test_parse_redis_key_with_different_clients(
    redis_client_decoded, redis_client_bytes
):
    """Test that our _parse_redis_checkpoint_writes_key method works correctly."""
    # Create a test key using the format expected by the parser
    from langgraph.checkpoint.redis.base import (
        CHECKPOINT_WRITE_PREFIX,
        REDIS_KEY_SEPARATOR,
    )

    test_key = f"{CHECKPOINT_WRITE_PREFIX}{REDIS_KEY_SEPARATOR}thread1{REDIS_KEY_SEPARATOR}ns1{REDIS_KEY_SEPARATOR}cp1{REDIS_KEY_SEPARATOR}task1{REDIS_KEY_SEPARATOR}0"

    # Test parsing with bytes key (as would come from decode_responses=False)
    bytes_key = test_key.encode()
    parsed_bytes = BaseRedisSaver._parse_redis_checkpoint_writes_key(
        safely_decode(bytes_key)
    )

    # Test parsing with string key (as would come from decode_responses=True)
    parsed_str = BaseRedisSaver._parse_redis_checkpoint_writes_key(
        safely_decode(test_key)
    )

    # Both should produce the same result
    assert parsed_bytes == parsed_str
    assert parsed_bytes["thread_id"] == "thread1"
    assert parsed_bytes["checkpoint_ns"] == "ns1"
    assert parsed_bytes["checkpoint_id"] == "cp1"
    assert parsed_bytes["task_id"] == "task1"
    assert parsed_bytes["idx"] == "0"
