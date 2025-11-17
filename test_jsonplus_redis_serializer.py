"""Standalone test to verify the JsonPlusRedisSerializer fix works.

This can be run directly without pytest infrastructure:
    python test_fix_standalone.py
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


def test_human_message_serialization():
    """Test that HumanMessage can be serialized without TypeError."""
    print("Testing HumanMessage serialization...")

    serializer = JsonPlusRedisSerializer()
    msg = HumanMessage(content="What is the weather?", id="msg-1")

    # Checkpoint 3.0: Use dumps_typed instead of dumps
    type_str, serialized = serializer.dumps_typed(msg)
    print(f"  ✓ Serialized to {len(serialized)} bytes (type: {type_str})")

    # Deserialize
    deserialized = serializer.loads_typed((type_str, serialized))
    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "What is the weather?"
    assert deserialized.id == "msg-1"
    print(f"  ✓ Deserialized correctly: {deserialized.content}")


def test_all_message_types():
    """Test all LangChain message types."""
    print("\nTesting all message types...")

    serializer = JsonPlusRedisSerializer()
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi!"),
        SystemMessage(content="System prompt"),
    ]

    for msg in messages:
        type_str, serialized = serializer.dumps_typed(msg)
        deserialized = serializer.loads_typed((type_str, serialized))
        assert type(deserialized) == type(msg)
        print(f"  ✓ {type(msg).__name__} works")


def test_message_list():
    """Test list of messages (common pattern in LangGraph)."""
    print("\nTesting message list...")

    serializer = JsonPlusRedisSerializer()
    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1"),
        HumanMessage(content="Question 2"),
    ]

    type_str, serialized = serializer.dumps_typed(messages)
    deserialized = serializer.loads_typed((type_str, serialized))

    assert isinstance(deserialized, list)
    assert len(deserialized) == 3
    assert all(isinstance(m, (HumanMessage, AIMessage)) for m in deserialized)
    print(f"  ✓ List of {len(deserialized)} messages works")


def test_nested_structure():
    """Test nested structure with messages (realistic LangGraph state)."""
    print("\nTesting nested structure with messages...")

    serializer = JsonPlusRedisSerializer()
    state = {
        "messages": [
            HumanMessage(content="Query"),
            AIMessage(content="Response"),
        ],
        "step": 1,
    }

    type_str, serialized = serializer.dumps_typed(state)
    deserialized = serializer.loads_typed((type_str, serialized))

    assert "messages" in deserialized
    assert len(deserialized["messages"]) == 2
    assert isinstance(deserialized["messages"][0], HumanMessage)
    assert isinstance(deserialized["messages"][1], AIMessage)
    print(f"  ✓ Nested structure works")


def test_dumps_typed():
    """Test dumps_typed (what checkpointer actually uses)."""
    print("\nTesting dumps_typed...")

    serializer = JsonPlusRedisSerializer()
    msg = HumanMessage(content="Test", id="test-123")

    type_str, blob = serializer.dumps_typed(msg)
    assert type_str == "json"
    # Checkpoint 3.0: blob is now bytes, not str
    assert isinstance(blob, bytes)
    print(f"  ✓ dumps_typed returns: type='{type_str}', blob={len(blob)} bytes")

    deserialized = serializer.loads_typed((type_str, blob))
    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "Test"
    print(f"  ✓ loads_typed works correctly")


def test_backwards_compatibility():
    """Test that regular objects still work."""
    print("\nTesting backwards compatibility...")

    serializer = JsonPlusRedisSerializer()
    test_cases = [
        ("string", "hello"),
        ("int", 42),
        ("dict", {"key": "value"}),
        ("list", [1, 2, 3]),
    ]

    for name, obj in test_cases:
        type_str, serialized = serializer.dumps_typed(obj)
        deserialized = serializer.loads_typed((type_str, serialized))
        assert deserialized == obj
        print(f"  ✓ {name} works")


def main():
    """Run all tests."""
    print("=" * 70)
    print("JsonPlusRedisSerializer Fix Validation")
    print("=" * 70)

    tests = [
        test_human_message_serialization,
        test_all_message_types,
        test_message_list,
        test_nested_structure,
        test_dumps_typed,
        test_backwards_compatibility,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
        return 0
    else:
        print(f"\n❌ {failed} TESTS FAILED - Fix may not be working")
        return 1


if __name__ == "__main__":
    exit(main())
