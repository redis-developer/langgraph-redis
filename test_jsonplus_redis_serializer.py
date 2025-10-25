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

    try:
        # This would raise TypeError before the fix
        serialized = serializer.dumps(msg)
        print(f"  ✓ Serialized to {len(serialized)} bytes")

        # Deserialize
        deserialized = serializer.loads(serialized)
        assert isinstance(deserialized, HumanMessage)
        assert deserialized.content == "What is the weather?"
        assert deserialized.id == "msg-1"
        print(f"  ✓ Deserialized correctly: {deserialized.content}")

        return True
    except TypeError as e:
        print(f"  ✗ FAILED: {e}")
        return False


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
        try:
            serialized = serializer.dumps(msg)
            deserialized = serializer.loads(serialized)
            assert type(deserialized) == type(msg)
            print(f"  ✓ {type(msg).__name__} works")
        except Exception as e:
            print(f"  ✗ {type(msg).__name__} FAILED: {e}")
            return False

    return True


def test_message_list():
    """Test list of messages (common pattern in LangGraph)."""
    print("\nTesting message list...")

    serializer = JsonPlusRedisSerializer()
    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1"),
        HumanMessage(content="Question 2"),
    ]

    try:
        serialized = serializer.dumps(messages)
        deserialized = serializer.loads(serialized)

        assert isinstance(deserialized, list)
        assert len(deserialized) == 3
        assert all(isinstance(m, (HumanMessage, AIMessage)) for m in deserialized)
        print(f"  ✓ List of {len(deserialized)} messages works")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


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

    try:
        serialized = serializer.dumps(state)
        deserialized = serializer.loads(serialized)

        assert "messages" in deserialized
        assert len(deserialized["messages"]) == 2
        assert isinstance(deserialized["messages"][0], HumanMessage)
        assert isinstance(deserialized["messages"][1], AIMessage)
        print(f"  ✓ Nested structure works")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_dumps_typed():
    """Test dumps_typed (what checkpointer actually uses)."""
    print("\nTesting dumps_typed...")

    serializer = JsonPlusRedisSerializer()
    msg = HumanMessage(content="Test", id="test-123")

    try:
        type_str, blob = serializer.dumps_typed(msg)
        assert type_str == "json"
        assert isinstance(blob, str)
        print(f"  ✓ dumps_typed returns: type='{type_str}', blob={len(blob)} chars")

        deserialized = serializer.loads_typed((type_str, blob))
        assert isinstance(deserialized, HumanMessage)
        assert deserialized.content == "Test"
        print(f"  ✓ loads_typed works correctly")

        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


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
        try:
            serialized = serializer.dumps(obj)
            deserialized = serializer.loads(serialized)
            assert deserialized == obj
            print(f"  ✓ {name} works")
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            return False

    return True


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

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)

    if all(results):
        print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix may not be working")
        return 1


if __name__ == "__main__":
    exit(main())
