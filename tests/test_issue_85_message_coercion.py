"""Test to reproduce issue #85: MESSAGE_COERCION_FAILURE.

When sending a second message with the same thread_id, the error occurs:
"Message dict must contain 'role' and 'content' keys, got {'lc': 1, 'type': 'constructor', 'id': [...], 'kwargs': {'content': 'nihao1111', 'type': 'human', 'id': '...'}}"

This test reproduces the issue and validates the fix.
"""

import json
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver


def test_message_coercion_failure_second_message():
    """Test that second message with same thread_id doesn't cause MESSAGE_COERCION_FAILURE."""

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer
        checkpointer = RedisSaver(redis_url)
        checkpointer.setup()

        # Define the graph state
        class GraphState(TypedDict):
            messages: Annotated[list, add_messages]

        # Create a simple graph
        def chat_model(state: GraphState):
            """Simulate a chat model response."""
            return {"messages": [AIMessage(content="Hello! How can I help you?")]}

        # Build graph
        builder = StateGraph(GraphState)
        builder.add_node("chat", chat_model)
        builder.add_edge(START, "chat")

        # Compile with checkpointer
        graph = builder.compile(checkpointer=checkpointer)

        # Use same thread_id for multiple messages (this triggers the issue)
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # First message - this should work fine
        first_input = {"messages": [HumanMessage(content="Hello")]}
        first_result = None

        for node, _, event_data in graph.stream(
            input=first_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            first_result = event_data

        assert first_result is not None, "First message should process successfully"

        # Second message with same thread_id - this triggers the MESSAGE_COERCION_FAILURE
        second_input = {"messages": [HumanMessage(content="nihao1111")]}
        second_result = None

        # This should NOT raise an error about message dict format
        for node, _, event_data in graph.stream(
            input=second_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            second_result = event_data

        assert second_result is not None, "Second message should process successfully"

        # Verify the messages are properly stored and retrieved
        state = graph.get_state(config)
        assert len(state.values["messages"]) == 4  # 2 human, 2 AI messages

        # Verify all messages are proper message objects
        for msg in state.values["messages"]:
            assert isinstance(
                msg, BaseMessage
            ), f"Message should be BaseMessage, got {type(msg)}"
            assert hasattr(msg, "content"), "Message should have content attribute"

    finally:
        redis_container.stop()


def test_message_serialization_roundtrip():
    """Test that messages stored in LangChain format are properly deserialized."""

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer
        checkpointer = RedisSaver(redis_url)
        checkpointer.setup()

        thread_id = str(uuid4())

        # Create checkpoint with messages in the problematic format
        messages = [
            HumanMessage(
                content="what's the weather in sf",
                id="d7fc45f1-6c5d-4b8b-8b5d-2040d25e9ee4",
            ),
            AIMessage(content="It's sunny today!", id="ai-msg-123"),
        ]

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": messages},
            step=1,
        )

        checkpoint["channel_values"]["messages"] = messages

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Save checkpoint
        saved_config = checkpointer.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load checkpoint back
        loaded_checkpoint = checkpointer.get(saved_config)

        assert loaded_checkpoint is not None
        loaded_messages = loaded_checkpoint["channel_values"]["messages"]

        # Messages should be proper objects, not dicts with 'lc', 'type', 'constructor'
        for i, msg in enumerate(loaded_messages):
            assert isinstance(
                msg, BaseMessage
            ), f"Message {i} should be BaseMessage, got {type(msg)}"

            # If it's a dict, it shouldn't have the problematic structure
            if isinstance(msg, dict):
                assert (
                    "lc" not in msg or msg.get("lc") != 1
                ), "Message shouldn't be in LangChain serialized format"
                assert (
                    "role" in msg and "content" in msg
                ), "If dict, should have 'role' and 'content' keys"
            else:
                # Should be a proper message object
                assert hasattr(msg, "content")
                assert msg.content == messages[i].content

    finally:
        redis_container.stop()


def test_message_dict_format_causes_error():
    """Test that messages in problematic dict format are handled correctly.

    The issue occurs when messages are stored as:
    {'lc': 1, 'type': 'constructor', 'id': [...], 'kwargs': {'content': '...', 'type': 'human', ...}}

    This format should be properly deserialized to message objects.
    """

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer
        checkpointer = RedisSaver(redis_url)
        checkpointer.setup()

        # This is the exact format that causes the error
        problematic_message_dict = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "HumanMessage"],
            "kwargs": {
                "content": "nihao1111",
                "type": "human",
                "id": "28ef5b05-b571-4454-9753-10d194e52024",
            },
        }

        # Test serializer can handle this format
        from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

        serializer = JsonPlusRedisSerializer()

        # Convert to JSON and back (simulating storage/retrieval)
        json_str = json.dumps(problematic_message_dict)
        deserialized = serializer.loads(json_str.encode())

        # Should be a proper HumanMessage, not the dict
        assert isinstance(
            deserialized, HumanMessage
        ), f"Should deserialize to HumanMessage, got {type(deserialized)}"
        assert deserialized.content == "nihao1111"
        assert deserialized.id == "28ef5b05-b571-4454-9753-10d194e52024"

        # Now test in a real checkpoint scenario
        thread_id = str(uuid4())

        # Create a checkpoint that would contain this problematic format
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        # Manually set the message in the problematic format
        # (simulating what happens when it's stored incorrectly)
        checkpoint["channel_values"]["messages"] = [problematic_message_dict]

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Save checkpoint
        saved_config = checkpointer.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load checkpoint back
        loaded_checkpoint = checkpointer.get(saved_config)

        assert loaded_checkpoint is not None
        loaded_messages = loaded_checkpoint["channel_values"]["messages"]

        # Should have properly deserialized the message
        assert len(loaded_messages) == 1
        msg = loaded_messages[0]

        # The message should be properly deserialized
        if isinstance(msg, dict):
            # If still a dict, it should have the correct format for the application
            assert (
                "role" in msg and "content" in msg
            ), f"Dict should have 'role' and 'content', got keys: {msg.keys()}"
            assert (
                "lc" not in msg
            ), "Should not have 'lc' field in application-facing dict"
        else:
            # Should be a proper message object
            assert isinstance(msg, HumanMessage)
            assert msg.content == "nihao1111"

    finally:
        redis_container.stop()


if __name__ == "__main__":
    # Run tests
    test_message_coercion_failure_second_message()
    test_message_serialization_roundtrip()
    test_message_dict_format_causes_error()
    print("All tests passed!")
