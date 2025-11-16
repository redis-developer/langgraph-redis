"""
Test JsonPlusRedisSerializer compatibility with langgraph-checkpoint 3.0.

This test suite verifies that our custom serializer works with the new
checkpoint 3.0 API which changed the dumps_typed/loads_typed signatures.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.types import Interrupt


def test_dumps_typed_returns_bytes_not_string() -> None:
    """Test that dumps_typed returns tuple[str, bytes], not tuple[str, str]."""
    serializer = JsonPlusRedisSerializer()

    # Test with simple object
    obj = {"test": "data", "number": 42}
    type_str, data = serializer.dumps_typed(obj)

    # Verify signature: must return (str, bytes)
    assert isinstance(type_str, str), f"Expected str, got {type(type_str)}"
    assert isinstance(data, bytes), f"Expected bytes, got {type(data)}"


def test_loads_typed_accepts_bytes() -> None:
    """Test that loads_typed accepts tuple[str, bytes]."""
    serializer = JsonPlusRedisSerializer()

    obj = {"test": "data"}
    type_str, data_bytes = serializer.dumps_typed(obj)

    # loads_typed must accept (str, bytes)
    result = serializer.loads_typed((type_str, data_bytes))
    assert result == obj


def test_serialization_roundtrip_simple_objects() -> None:
    """Test serialization roundtrip for simple Python objects."""
    serializer = JsonPlusRedisSerializer()

    test_cases = [
        None,
        {"key": "value"},
        [1, 2, 3],
        "string",
        42,
        3.14,
        True,
        False,
    ]

    for obj in test_cases:
        type_str, data_bytes = serializer.dumps_typed(obj)
        result = serializer.loads_typed((type_str, data_bytes))
        assert result == obj, f"Roundtrip failed for {obj}"


def test_serialization_roundtrip_bytes() -> None:
    """Test serialization roundtrip for bytes objects."""
    serializer = JsonPlusRedisSerializer()

    test_bytes = b"hello world"
    type_str, data_bytes = serializer.dumps_typed(test_bytes)

    assert type_str == "bytes"
    assert isinstance(data_bytes, bytes)

    result = serializer.loads_typed((type_str, data_bytes))
    assert result == test_bytes
    assert isinstance(result, bytes)


def test_serialization_roundtrip_bytearray() -> None:
    """Test serialization roundtrip for bytearray objects."""
    serializer = JsonPlusRedisSerializer()

    test_bytearray = bytearray(b"hello world")
    type_str, data_bytes = serializer.dumps_typed(test_bytearray)

    assert type_str == "bytearray"
    assert isinstance(data_bytes, bytes)

    result = serializer.loads_typed((type_str, data_bytes))
    assert result == test_bytearray
    assert isinstance(result, bytearray)


def test_serialization_roundtrip_langchain_messages() -> None:
    """Test serialization roundtrip for LangChain message objects."""
    serializer = JsonPlusRedisSerializer()

    messages = [
        HumanMessage(content="Hello", id="human-1"),
        AIMessage(content="Hi there!", id="ai-1"),
    ]

    for msg in messages:
        type_str, data_bytes = serializer.dumps_typed(msg)

        assert isinstance(type_str, str)
        assert isinstance(data_bytes, bytes)

        result = serializer.loads_typed((type_str, data_bytes))

        # Should deserialize back to the same message type
        assert type(result) == type(msg)
        assert result.content == msg.content


def test_serialization_roundtrip_interrupt_objects() -> None:
    """Test serialization roundtrip for Interrupt objects (Issue #113)."""
    serializer = JsonPlusRedisSerializer()

    interrupt = Interrupt(value={"test": "data"}, resumable=True)

    type_str, data_bytes = serializer.dumps_typed(interrupt)

    assert isinstance(type_str, str)
    assert isinstance(data_bytes, bytes)

    result = serializer.loads_typed((type_str, data_bytes))

    # CRITICAL: Must deserialize back to Interrupt, not dict
    assert isinstance(result, Interrupt), (
        f"Expected Interrupt object, got {type(result)}. "
        f"This is the Issue #113 regression!"
    )
    assert result.value == {"test": "data"}
    assert result.resumable is True


def test_serialization_roundtrip_nested_interrupts() -> None:
    """Test serialization of nested Interrupt objects."""
    serializer = JsonPlusRedisSerializer()

    # Interrupt containing another Interrupt in value
    nested = Interrupt(
        value={"nested": Interrupt(value={"inner": "data"}, resumable=False)},
        resumable=True,
    )

    type_str, data_bytes = serializer.dumps_typed(nested)
    result = serializer.loads_typed((type_str, data_bytes))

    assert isinstance(result, Interrupt)
    assert isinstance(result.value["nested"], Interrupt)
    assert result.value["nested"].value == {"inner": "data"}


def test_serialization_roundtrip_list_of_interrupts() -> None:
    """Test serialization of lists containing Interrupt objects."""
    serializer = JsonPlusRedisSerializer()

    pending_sends = [
        ("__interrupt__", [Interrupt(value={"test": "data"}, resumable=False)]),
        ("messages", ["some message"]),
    ]

    type_str, data_bytes = serializer.dumps_typed(pending_sends)
    result = serializer.loads_typed((type_str, data_bytes))

    assert isinstance(result, list)
    assert len(result) == 2

    channel, value = result[0]
    assert channel == "__interrupt__"
    assert isinstance(value, list)
    assert len(value) == 1

    # CRITICAL: Must be Interrupt object, not dict (Issue #113)
    assert isinstance(value[0], Interrupt)
    assert value[0].value == {"test": "data"}
    assert value[0].resumable is False


def test_no_public_dumps_loads_methods() -> None:
    """Verify that dumps/loads are not part of the SerializerProtocol in 3.0."""
    from langgraph.checkpoint.serde.base import SerializerProtocol

    # SerializerProtocol should only have dumps_typed and loads_typed
    protocol_methods = [
        name for name in dir(SerializerProtocol) if not name.startswith("_")
    ]

    assert "dumps_typed" in protocol_methods
    assert "loads_typed" in protocol_methods
    # In 3.0, dumps and loads are NOT in the protocol
    assert "dumps" not in protocol_methods
    assert "loads" not in protocol_methods
