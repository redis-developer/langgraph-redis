"""
RediSearch versions below 2.10 don't support indexing and querying
empty strings, so we use a sentinel value to represent empty strings.
Because checkpoint queries are sorted by checkpoint_id, we use a UUID
that is lexicographically sortable. Typically, checkpoints that need
sentinel values are from the first run of the graph, so this should
generally be correct.

This module also includes utility functions for safely handling Redis responses,
including handling bytes vs string responses depending on how the Redis client is
configured with decode_responses.
"""

import uuid
from typing import Any, Optional

from ulid import ULID

EMPTY_STRING_SENTINEL = "__empty__"
EMPTY_ID_SENTINEL = "00000000-0000-0000-0000-000000000000"

# 100-nanosecond intervals between the UUID epoch (1582-10-15) and the Unix epoch.
_UUID_EPOCH_OFFSET = 0x01B21DD213814000


def checkpoint_id_timestamp(checkpoint_id: str) -> Optional[float]:
    """Return the timestamp a checkpoint id encodes, or None if it encodes none.

    This is the value stored as ``checkpoint_ts`` and compared by ``list(before=...)``,
    so both sides must come from here.

    A ULID returns ``ULID.timestamp``, the value the savers have always stored for one.
    LangGraph's own checkpoint ids are UUIDv6, whose 60-bit timestamp is returned in
    milliseconds since the Unix epoch, the unit already stored for non-ULID ids. Any
    other id (for example a uuid4) carries no time and returns None.
    """
    if not isinstance(checkpoint_id, str):
        return None
    try:
        return ULID.from_str(checkpoint_id).timestamp
    except ValueError:
        pass
    try:
        parsed = uuid.UUID(checkpoint_id)
    except ValueError:
        return None
    if parsed.version != 6:
        return None
    n = parsed.int
    ticks = ((n >> 96) << 28) | (((n >> 80) & 0xFFFF) << 12) | ((n >> 64) & 0x0FFF)
    return (ticks - _UUID_EPOCH_OFFSET) / 10_000


def to_storage_safe_str(value: str) -> str:
    """
    Prepare a value for storage in Redis as a string.

    Convert an empty string to a sentinel value, otherwise return the
    value as a string.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == "":
        return EMPTY_STRING_SENTINEL
    else:
        return str(value)


def from_storage_safe_str(value: str) -> str:
    """
    Convert a value from a sentinel value to an empty string if present,
    otherwise return the value unchanged.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == EMPTY_STRING_SENTINEL:
        return ""
    else:
        return value


def to_storage_safe_id(value: str) -> str:
    """
    Prepare a value for storage in Redis as an ID.

    Convert an empty string to a sentinel value for empty ID strings, otherwise
    return the value as a string.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == "":
        return EMPTY_ID_SENTINEL
    else:
        return str(value)


def from_storage_safe_id(value: str) -> str:
    """
    Convert a value from a sentinel value for empty ID strings to an empty
    ID string if present, otherwise return the value unchanged.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == EMPTY_ID_SENTINEL:
        return ""
    else:
        return value


def safely_decode(obj: Any) -> Any:
    """
    Safely decode Redis responses, handling both string and bytes types.

    This is especially useful when working with Redis clients configured with
    different decode_responses settings. It recursively processes nested
    data structures (dicts, lists, tuples, sets).

    Based on RedisVL's convert_bytes function (redisvl.redis.utils.convert_bytes)
    but implemented directly to avoid runtime import issues and ensure consistent
    behavior with sets and other data structures. See PR #34 and referenced
    implementation: https://github.com/redis/redis-vl-python/blob/9f22a9ad4c2166af6462b007833b456448714dd9/redisvl/redis/utils.py#L20

    Args:
        obj: The object to decode. Can be a string, bytes, or a nested structure
            containing strings/bytes (dict, list, tuple, set).

    Returns:
        The decoded object with all bytes converted to strings.
    """
    if obj is None:
        return None
    elif isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            # If decoding fails, return the original bytes
            return obj
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return {safely_decode(k): safely_decode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safely_decode(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(safely_decode(item) for item in obj)
    elif isinstance(obj, set):
        return {safely_decode(item) for item in obj}
    else:
        # For other types (int, float, bool, etc.), return as is
        return obj
