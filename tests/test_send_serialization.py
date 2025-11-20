"""Test Send object serialization fix for issue #94.

This test validates that langgraph.types.Send objects are properly serialized
and deserialized by the JsonPlusRedisSerializer.

Before the fix, Send objects were not serialized correctly, which led to issues
with handling Interrupts - namely the user's response would not be treated as
the response to the Interrupt.

The issue occurs in `prepare_single_task` in pregel._algo.py where:
```
if not isinstance(packet, Send):
    logger.warning(
        f"Ignoring invalid packet type {type(packet)} in pending sends"
    )
    return.          <<<<< task not added
```

The fix adds custom serialization/deserialization for Send objects similar to
how Interrupt objects are handled.
"""

import pytest
from langgraph.types import Send

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


def test_send_object_serialization():
    """Test that Send objects are properly serialized and deserialized.

    Before the fix, Send objects would not serialize correctly, causing
    isinstance(packet, Send) checks to fail after deserialization.
    """
    serializer = JsonPlusRedisSerializer()

    # Create a Send object
    send_obj = Send(node="my_node", arg={"key": "value"})

    # Serialize
    type_str, blob = serializer.dumps_typed(send_obj)
    assert type_str == "json"
    assert isinstance(blob, bytes)

    # Deserialize
    deserialized = serializer.loads_typed((type_str, blob))

    # Critical check: the deserialized object must be an instance of Send
    assert isinstance(deserialized, Send), (
        f"Expected Send instance, got {type(deserialized)}. "
        "This will cause isinstance(packet, Send) checks to fail!"
    )
    assert deserialized.node == "my_node"
    assert deserialized.arg == {"key": "value"}
    assert deserialized == send_obj


def test_send_object_in_pending_sends_list():
    """Test that Send objects in pending_sends lists are properly handled.

    This simulates the scenario where Send objects are stored in checkpoint
    pending_sends and must be correctly deserialized for interrupt handling.
    """
    serializer = JsonPlusRedisSerializer()

    # Create multiple Send objects as they would appear in pending_sends
    pending_sends = [
        Send(node="node1", arg={"data": "first"}),
        Send(node="node2", arg={"data": "second"}),
        Send(node="node3", arg={"data": "third"}),
    ]

    # Serialize the list
    type_str, blob = serializer.dumps_typed(pending_sends)

    # Deserialize
    deserialized = serializer.loads_typed((type_str, blob))

    # Verify all items are still Send instances
    assert isinstance(deserialized, list)
    assert len(deserialized) == 3

    for i, send_obj in enumerate(deserialized):
        assert isinstance(
            send_obj, Send
        ), f"Item {i} is not a Send instance: {type(send_obj)}"

    assert deserialized[0].node == "node1"
    assert deserialized[1].node == "node2"
    assert deserialized[2].node == "node3"


def test_send_object_with_complex_args():
    """Test Send objects with complex nested arguments."""
    serializer = JsonPlusRedisSerializer()

    # Create Send with complex nested arg
    complex_arg = {
        "messages": ["msg1", "msg2"],
        "metadata": {
            "step": 1,
            "config": {
                "model": "gpt-4",
                "temperature": 0.7,
            },
        },
        "nested_list": [
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
        ],
    }

    send_obj = Send(node="processor", arg=complex_arg)

    # Serialize and deserialize
    type_str, blob = serializer.dumps_typed(send_obj)
    deserialized = serializer.loads_typed((type_str, blob))

    # Verify type and structure
    assert isinstance(deserialized, Send)
    assert deserialized.node == "processor"
    assert deserialized.arg == complex_arg
    assert deserialized.arg["metadata"]["config"]["model"] == "gpt-4"


def test_send_object_in_checkpoint_structure():
    """Test Send objects embedded in checkpoint-like structures.

    This simulates how Send objects appear in actual checkpoint data.
    """
    serializer = JsonPlusRedisSerializer()

    # Simulate checkpoint structure with pending_sends
    checkpoint_data = {
        "v": 1,
        "id": "checkpoint_1",
        "pending_sends": [
            Send(node="task1", arg={"task_data": "A"}),
            Send(node="task2", arg={"task_data": "B"}),
        ],
        "channel_values": {"messages": ["msg1", "msg2"]},
    }

    # Serialize and deserialize
    type_str, blob = serializer.dumps_typed(checkpoint_data)
    deserialized = serializer.loads_typed((type_str, blob))

    # Verify Send objects are preserved correctly
    assert "pending_sends" in deserialized
    assert len(deserialized["pending_sends"]) == 2

    for send_obj in deserialized["pending_sends"]:
        assert isinstance(
            send_obj, Send
        ), f"pending_sends contains non-Send object: {type(send_obj)}"


def test_send_object_equality_after_roundtrip():
    """Test that Send objects maintain equality after serialization roundtrip."""
    serializer = JsonPlusRedisSerializer()

    send1 = Send(node="test_node", arg={"value": 42})

    # Serialize and deserialize
    type_str, blob = serializer.dumps_typed(send1)
    send2 = serializer.loads_typed((type_str, blob))

    # Send objects should be equal
    assert send1 == send2

    # Test hash equality with hashable args
    send_hashable1 = Send(node="test", arg="hashable_string")
    type_str, blob = serializer.dumps_typed(send_hashable1)
    send_hashable2 = serializer.loads_typed((type_str, blob))
    assert hash(send_hashable1) == hash(send_hashable2)


def test_send_object_with_none_arg():
    """Test Send object with None as argument."""
    serializer = JsonPlusRedisSerializer()

    send_obj = Send(node="null_handler", arg=None)

    # Serialize and deserialize
    type_str, blob = serializer.dumps_typed(send_obj)
    deserialized = serializer.loads_typed((type_str, blob))

    assert isinstance(deserialized, Send)
    assert deserialized.node == "null_handler"
    assert deserialized.arg is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
