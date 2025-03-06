from typing import Any, Callable, Optional, TypeVar, Union

"""
RediSearch versions below 2.10 don't support indexing and querying
empty strings, so we use a sentinel value to represent empty strings.
Because checkpoint queries are sorted by checkpoint_id, we use a UUID
that is lexicographically sortable. Typically, checkpoints that need
sentinel values are from the first run of the graph, so this should
generally be correct.
"""
EMPTY_STRING_SENTINEL = "__empty__"
EMPTY_ID_SENTINEL = "00000000-0000-0000-0000-000000000000"


def to_storage_safe_str(value: str) -> str:
    """
    Convert any empty string to an empty string sentinel if found,
    otherwise return the value unchanged.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == "":
        return EMPTY_STRING_SENTINEL
    else:
        return value


def from_storage_safe_str(value: str) -> str:
    """
    Convert a value from an empty string sentinel to an empty string
    if found, otherwise return the value unchanged.

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
    Convert any empty ID string to an empty ID sentinel if found,
    otherwise return the value unchanged.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == "":
        return EMPTY_ID_SENTINEL
    else:
        return value


def from_storage_safe_id(value: str) -> str:
    """
    Convert a value from an empty ID sentinel to an empty ID
    if found, otherwise return the value unchanged.

    Args:
        value (str): The value to convert.

    Returns:
        str: The converted value.
    """
    if value == EMPTY_ID_SENTINEL:
        return ""
    else:
        return value


def storage_safe_get(
    doc: dict[str, Any], key: str, default: Any = None
) -> Optional[Any]:
    """
    Get a value from a Redis document or dictionary, using a sentinel
    value to represent empty strings.

    If the sentinel value is found, it is converted back to an empty string.

    Args:
        doc (dict[str, Any]): The document to get the value from.
        key (str): The key to get the value from.
        default (Any): The default value to return if the key is not found.
    Returns:
        Optional[Any]: None if the key is not found, or else the value from
                       the document or dictionary, with empty strings converted
                       to the empty string sentinel and the sentinel converted
                       back to an empty string.
    """
    try:
        # NOTE: The Document class that comes back from `search()` support
        # [key] access but not `get()` for some reason, so we use direct
        # key access with an exception guard.
        value = doc[key]
    except KeyError:
        value = None

    if value is None:
        return default

    return to_storage_safe_str(value)
