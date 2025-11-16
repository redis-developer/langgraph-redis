"""Test JsonPlusRedisSerializer uses orjson with default handler for LangChain objects.

This test validates the fix for the bug where JsonPlusRedisSerializer.dumps()
was not using the default parameter with orjson, causing TypeError when
serializing LangChain message objects like HumanMessage and AIMessage.

The fix ensures all LangChain Serializable objects are properly handled by
using orjson.dumps(obj, default=self._default) instead of plain orjson.dumps(obj).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


# Helper functions for backward compatibility with tests
def dumps_helper(serializer: JsonPlusRedisSerializer, obj):
    """Helper to simulate old dumps() method using dumps_typed().

    Returns the full (type_str, data_bytes) tuple to preserve type information.
    """
    return serializer.dumps_typed(obj)


def loads_helper(serializer: JsonPlusRedisSerializer, typed_data):
    """Helper to simulate old loads() method using loads_typed().

    Args:
        typed_data: Full (type_str, data_bytes) tuple from dumps_helper
    """
    return serializer.loads_typed(typed_data)


def test_serializer_uses_default_handler_for_messages():
    """Test that dumps() uses the default handler for LangChain message objects.

    Before the fix, this would raise:
        TypeError: Type is not JSON serializable: HumanMessage

    After the fix, messages are properly serialized via the _default handler.
    """
    serializer = JsonPlusRedisSerializer()

    # Test HumanMessage
    human_msg = HumanMessage(content="What is the weather?", id="msg-1")

    # This should NOT raise TypeError
    serialized_bytes = dumps_helper(serializer, human_msg)
    assert isinstance(serialized_bytes, bytes)

    # Deserialize and verify
    deserialized = loads_helper(serializer, serialized_bytes)
    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "What is the weather?"
    assert deserialized.id == "msg-1"


def test_serializer_handles_all_message_types():
    """Test that all LangChain message types are properly serialized.

    This ensures the fix works for all message subclasses, not just HumanMessage.
    """
    serializer = JsonPlusRedisSerializer()

    messages = [
        HumanMessage(content="Hello", id="human-1"),
        AIMessage(content="Hi there!", id="ai-1"),
        SystemMessage(content="You are a helpful assistant", id="sys-1"),
        ToolMessage(content="Tool result", tool_call_id="tool-1", id="tool-msg-1"),
    ]

    for msg in messages:
        # Serialize
        serialized = dumps_helper(serializer, msg)
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = loads_helper(serializer, serialized)

        # Verify type is preserved
        assert type(deserialized) == type(msg)
        assert deserialized.content == msg.content
        assert deserialized.id == msg.id


def test_serializer_handles_message_lists():
    """Test that lists of messages are properly serialized.

    This is a common pattern in LangGraph state where messages are stored as lists.
    """
    serializer = JsonPlusRedisSerializer()

    messages = [
        HumanMessage(content="What's 2+2?"),
        AIMessage(content="2+2 equals 4"),
        HumanMessage(content="Thanks!"),
    ]

    # Serialize the list
    serialized = dumps_helper(serializer, messages)
    assert isinstance(serialized, bytes)

    # Deserialize
    deserialized = loads_helper(serializer, serialized)

    # Verify structure
    assert isinstance(deserialized, list)
    assert len(deserialized) == 3
    assert all(isinstance(msg, (HumanMessage, AIMessage)) for msg in deserialized)
    assert deserialized[0].content == "What's 2+2?"
    assert deserialized[1].content == "2+2 equals 4"


def test_serializer_handles_nested_structures_with_messages():
    """Test that nested structures containing messages are properly serialized.

    This tests the scenario where messages are embedded in dicts or other structures.
    """
    serializer = JsonPlusRedisSerializer()

    state = {
        "messages": [
            HumanMessage(content="Query"),
            AIMessage(content="Response"),
        ],
        "metadata": {
            "step": 1,
            "last_message": HumanMessage(content="Latest"),
        },
    }

    # Serialize
    serialized = dumps_helper(serializer, state)
    assert isinstance(serialized, bytes)

    # Deserialize
    deserialized = loads_helper(serializer, serialized)

    # Verify structure
    assert "messages" in deserialized
    assert len(deserialized["messages"]) == 2
    assert isinstance(deserialized["messages"][0], HumanMessage)
    assert isinstance(deserialized["messages"][1], AIMessage)
    assert isinstance(deserialized["metadata"]["last_message"], HumanMessage)


def test_dumps_typed_with_messages():
    """Test that dumps_typed also properly handles messages.

    This tests the full serialization path used by Redis checkpointer.
    """
    serializer = JsonPlusRedisSerializer()

    msg = HumanMessage(content="Test message", id="test-123")

    # Use dumps_typed (what the checkpointer actually calls)
    type_str, blob = serializer.dumps_typed(msg)

    assert type_str == "json"
    # Checkpoint 3.0: dumps_typed now returns bytes, not str
    assert isinstance(blob, bytes)

    # Deserialize
    deserialized = serializer.loads_typed((type_str, blob))

    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "Test message"
    assert deserialized.id == "test-123"


def test_serializer_backwards_compatible():
    """Test that the fix doesn't break serialization of regular objects.

    Ensures that non-LangChain objects still serialize correctly.
    """
    serializer = JsonPlusRedisSerializer()

    test_cases = [
        "simple string",
        42,
        3.14,
        True,
        None,
        [1, 2, 3],
        {"key": "value"},
        {"nested": {"data": [1, 2, 3]}},
    ]

    for obj in test_cases:
        serialized = dumps_helper(serializer, obj)
        deserialized = loads_helper(serializer, serialized)
        assert deserialized == obj


def test_serializer_with_langchain_serialized_format():
    """Test that manually constructed LangChain serialized dicts are revived.

    This tests the _revive_if_needed functionality works with the new dumps() implementation.
    """
    serializer = JsonPlusRedisSerializer()

    # This is the format that LangChain objects serialize to
    message_dict = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "HumanMessage"],
        "kwargs": {
            "content": "Manually constructed message",
            "type": "human",
            "id": "manual-123",
        },
    }

    # Serialize and deserialize
    serialized = dumps_helper(serializer, message_dict)
    deserialized = loads_helper(serializer, serialized)

    # Should be revived as a HumanMessage
    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "Manually constructed message"
    assert deserialized.id == "manual-123"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
