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
