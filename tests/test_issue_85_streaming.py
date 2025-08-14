"""Test to reproduce issue #85 with streaming and subgraphs.

This test more closely mimics the exact scenario from the issue report where:
1. User uses stream mode with ["messages", "updates"]
2. User enables subgraphs=True
3. First message works fine
4. Second message with same thread_id causes MESSAGE_COERCION_FAILURE
"""

from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver


def test_streaming_with_subgraphs_second_message():
    """Test streaming with subgraphs doesn't cause MESSAGE_COERCION_FAILURE on second message."""

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

        # Create a simple chat model node
        def chat_model(state: GraphState):
            """Simulate a chat model response."""
            last_message = state["messages"][-1]
            response = f"You said: {last_message.content}"
            return {"messages": [AIMessage(content=response)]}

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
        first_messages = []
        first_updates = []

        # Stream with the exact parameters from the issue
        for node, mode, event_data in graph.stream(
            input=first_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            if mode == "messages":
                first_messages.append(event_data)
            elif mode == "updates":
                first_updates.append(event_data)

        assert (
            len(first_messages) > 0 or len(first_updates) > 0
        ), "First message should produce events"

        # Second message with same thread_id - this triggers the MESSAGE_COERCION_FAILURE
        second_input = {"messages": [HumanMessage(content="nihao1111")]}
        second_messages = []
        second_updates = []
        error_occurred = False
        error_message = None

        try:
            # This should NOT raise an error about message dict format
            for node, mode, event_data in graph.stream(
                input=second_input,
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                if mode == "messages":
                    second_messages.append(event_data)
                elif mode == "updates":
                    second_updates.append(event_data)
        except Exception as e:
            error_occurred = True
            error_message = str(e)

            # Check if it's the specific error from the issue
            if "Message dict must contain 'role' and 'content' keys" in error_message:
                pytest.fail(f"MESSAGE_COERCION_FAILURE occurred: {error_message}")
            else:
                # Re-raise other unexpected errors
                raise

        assert not error_occurred, f"No error should occur, but got: {error_message}"
        assert (
            len(second_messages) > 0 or len(second_updates) > 0
        ), "Second message should produce events"

        # Verify the state has all messages properly stored
        state = graph.get_state(config)
        assert len(state.values["messages"]) == 4  # 2 human, 2 AI messages

        # Verify all messages are proper message objects
        for i, msg in enumerate(state.values["messages"]):
            assert isinstance(
                msg, BaseMessage
            ), f"Message {i} should be BaseMessage, got {type(msg)}: {msg}"
            assert hasattr(msg, "content"), f"Message {i} should have content attribute"

            # Check that none have the problematic lc format
            if isinstance(msg, dict):
                assert "lc" not in msg, f"Message {i} shouldn't have 'lc' field: {msg}"
                assert (
                    "role" in msg and "content" in msg
                ), f"Message {i} should have role and content: {msg}"

    finally:
        redis_container.stop()


def test_complex_graph_with_subgraphs():
    """Test a more complex graph with actual subgraphs to ensure messages are handled correctly."""

    # Start Redis container
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Create checkpointer
        checkpointer = RedisSaver(redis_url)
        checkpointer.setup()

        # Define subgraph state
        class SubGraphState(TypedDict):
            messages: Annotated[list, add_messages]
            extra: str

        # Create subgraph
        def sub_node(state: SubGraphState):
            return {
                "extra": "processed",
                "messages": [AIMessage(content="From subgraph")],
            }

        sub_builder = StateGraph(SubGraphState)
        sub_builder.add_node("sub_node", sub_node)
        sub_builder.add_edge(START, "sub_node")
        subgraph = sub_builder.compile()

        # Define main graph state
        class MainGraphState(TypedDict):
            messages: Annotated[list, add_messages]

        # Create main graph nodes
        def main_node(state: MainGraphState):
            return {"messages": [AIMessage(content="From main node")]}

        # Build main graph with subgraph
        main_builder = StateGraph(MainGraphState)
        main_builder.add_node("main_node", main_node)
        main_builder.add_node("subgraph", subgraph)
        main_builder.add_edge(START, "main_node")
        main_builder.add_edge("main_node", "subgraph")

        # Compile with checkpointer
        graph = main_builder.compile(checkpointer=checkpointer)

        # Use same thread_id for multiple messages
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # First message
        first_input = {"messages": [HumanMessage(content="First message")]}

        for node, mode, event_data in graph.stream(
            input=first_input,
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            pass  # Just process

        # Second message - should not cause error
        second_input = {"messages": [HumanMessage(content="Second message")]}

        error_occurred = False
        try:
            for node, mode, event_data in graph.stream(
                input=second_input,
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                pass  # Just process
        except Exception as e:
            if "Message dict must contain 'role' and 'content' keys" in str(e):
                pytest.fail(f"MESSAGE_COERCION_FAILURE occurred: {e}")
            raise

        # Verify state
        state = graph.get_state(config)
        messages = state.values["messages"]

        # All messages should be proper BaseMessage objects
        for msg in messages:
            assert isinstance(
                msg, BaseMessage
            ), f"Message should be BaseMessage, got {type(msg)}"

    finally:
        redis_container.stop()


if __name__ == "__main__":
    # Run tests
    test_streaming_with_subgraphs_second_message()
    test_complex_graph_with_subgraphs()
    print("All streaming tests passed!")
