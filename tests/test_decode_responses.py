"""Tests for Redis key decoding functionality."""

import os
import time
import uuid
from typing import Any, Dict, Optional

import pytest
from redis import Redis

from langgraph.checkpoint.redis.util import safely_decode


def test_safely_decode_basic_types():
    """Test safely_decode function with basic type inputs."""
    # Test with bytes
    assert safely_decode(b"test") == "test"

    # Test with string
    assert safely_decode("test") == "test"

    # Test with None
    assert safely_decode(None) is None

    # Test with other types
    assert safely_decode(123) == 123
    assert safely_decode(1.23) == 1.23
    assert safely_decode(True) is True


def test_safely_decode_nested_structures():
    """Test safely_decode function with nested data structures."""
    # Test with dictionary
    assert safely_decode({b"key": b"value"}) == {"key": "value"}
    assert safely_decode({b"key1": b"value1", "key2": 123}) == {
        "key1": "value1",
        "key2": 123,
    }

    # Test with nested dictionary
    nested_dict = {b"outer": {b"inner": b"value"}}
    assert safely_decode(nested_dict) == {"outer": {"inner": "value"}}

    # Test with list
    assert safely_decode([b"item1", b"item2"]) == ["item1", "item2"]

    # Test with tuple
    assert safely_decode((b"item1", b"item2")) == ("item1", "item2")

    # Test with set
    decoded_set = safely_decode({b"item1", b"item2"})
    assert isinstance(decoded_set, set)
    assert "item1" in decoded_set
    assert "item2" in decoded_set

    # Test with complex nested structure
    complex_struct = {
        b"key1": [b"list_item1", {b"nested_key": b"nested_value"}],
        b"key2": (b"tuple_item", 123),
        b"key3": {b"set_item1", b"set_item2"},
    }
    decoded = safely_decode(complex_struct)
    assert decoded["key1"][0] == "list_item1"
    assert decoded["key1"][1]["nested_key"] == "nested_value"
    assert decoded["key2"][0] == "tuple_item"
    assert decoded["key2"][1] == 123
    assert isinstance(decoded["key3"], set)
    assert "set_item1" in decoded["key3"]
    assert "set_item2" in decoded["key3"]


@pytest.mark.parametrize("decode_responses", [True, False])
def test_safely_decode_with_redis(decode_responses: bool, redis_url):
    """Test safely_decode function with actual Redis responses using TestContainers."""
    r = Redis.from_url(redis_url, decode_responses=decode_responses)

    try:
        # Clean up before test to ensure a clean state
        r.delete("test:string")
        r.delete("test:hash")
        r.delete("test:list")
        r.delete("test:set")

        # Set up test data
        r.set("test:string", "value")
        r.hset("test:hash", mapping={"field1": "value1", "field2": "value2"})
        r.rpush("test:list", "item1", "item2", "item3")
        r.sadd("test:set", "member1", "member2")

        # Test string value
        string_val = r.get("test:string")
        decoded_string = safely_decode(string_val)
        assert decoded_string == "value"

        # Test hash value
        hash_val = r.hgetall("test:hash")
        decoded_hash = safely_decode(hash_val)
        assert decoded_hash == {"field1": "value1", "field2": "value2"}

        # Test list value
        list_val = r.lrange("test:list", 0, -1)
        decoded_list = safely_decode(list_val)
        assert decoded_list == ["item1", "item2", "item3"]

        # Test set value
        set_val = r.smembers("test:set")
        decoded_set = safely_decode(set_val)
        assert isinstance(decoded_set, set)
        assert "member1" in decoded_set
        assert "member2" in decoded_set

        # Test key fetching
        keys = r.keys("test:*")
        decoded_keys = safely_decode(keys)
        assert sorted(decoded_keys) == sorted(
            ["test:string", "test:hash", "test:list", "test:set"]
        )

    finally:
        # Clean up after test
        r.delete("test:string")
        r.delete("test:hash")
        r.delete("test:list")
        r.delete("test:set")
        r.close()


def test_safely_decode_unicode_error_handling():
    """Test safely_decode function with invalid UTF-8 bytes."""
    # Create bytes that will cause UnicodeDecodeError
    invalid_utf8 = b"\xff\xfe\xfd"

    # Should return the original bytes if it can't be decoded
    result = safely_decode(invalid_utf8)
    assert result == invalid_utf8

    # Test with mixed valid and invalid in a complex structure
    mixed = {
        b"valid": b"This is valid UTF-8",
        b"invalid": invalid_utf8,
        b"nested": [b"valid", invalid_utf8],
    }

    result = safely_decode(mixed)
    assert result["valid"] == "This is valid UTF-8"
    assert result["invalid"] == invalid_utf8
    assert result["nested"][0] == "valid"
    assert result["nested"][1] == invalid_utf8
