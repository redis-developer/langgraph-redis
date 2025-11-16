"""
Regression test for Issue #113: Interrupt objects not properly deserialized

When using interrupt() with RedisSaver, Interrupt objects are serialized to
dictionaries but not reconstructed back to Interrupt objects on deserialization.

This causes AttributeError: 'dict' object has no attribute 'id' when trying
to resume execution with Command(resume=...).

The error occurs in LangGraph's _pending_interrupts() method when it tries to
access value[0].id, but value[0] is a dict instead of an Interrupt object.
"""

import operator
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from langgraph.checkpoint.redis import RedisSaver


class AgentState(TypedDict):
    """State for the test agent."""

    messages: Annotated[list[AnyMessage], operator.add]


def review_node(state: AgentState):
    """Node that interrupts for review."""
    random_str = str(uuid4())
    print(f"Generated string: {random_str}")
    print("-------- entry interrupt --------")

    # This creates an Interrupt object that needs to be serialized
    user_input = interrupt({"test": "data"})

    print(f"Received input: {user_input.get('test')}")
    print("-------- exit interrupt --------")
    return {"messages": [random_str]}


def test_interrupt_serialization_roundtrip(redis_url: str) -> None:
    """
    Test that Interrupt objects are properly serialized and deserialized.

    This is a unit test that directly tests the serializer behavior.
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    serializer = JsonPlusRedisSerializer()

    # Create an Interrupt object
    original_interrupt = Interrupt(value={"test": "data"}, resumable=True)

    # Serialize it
    type_str, serialized = serializer.dumps_typed(original_interrupt)

    # Deserialize it
    deserialized = serializer.loads_typed((type_str, serialized))

    # This should be an Interrupt object, not a dict
    assert isinstance(deserialized, Interrupt), (
        f"Expected Interrupt object, got {type(deserialized)}. "
        f"This causes AttributeError when LangGraph tries to access attributes"
    )
    assert deserialized.value == {"test": "data"}
    assert deserialized.resumable is True


def test_interrupt_in_pending_sends(redis_url: str) -> None:
    """
    Test that Interrupt objects in pending_sends are properly deserialized.

    This tests the actual scenario from issue #113 where interrupts stored
    in checkpoint writes need to be reconstructed.
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    serializer = JsonPlusRedisSerializer()

    # Simulate what gets stored in pending_sends
    # In the real scenario, pending_sends contains tuples of (channel, value)
    # where value might be an Interrupt object
    pending_sends = [
        ("__interrupt__", [Interrupt(value={"test": "data"}, resumable=False)]),
        ("messages", ["some message"]),
    ]

    # Serialize the pending_sends
    type_str, serialized = serializer.dumps_typed(pending_sends)

    # Deserialize
    deserialized = serializer.loads_typed((type_str, serialized))

    # Check the structure
    assert isinstance(deserialized, list)
    assert len(deserialized) == 2

    # The first item should have reconstructed Interrupt object
    channel, value = deserialized[0]
    assert channel == "__interrupt__"
    assert isinstance(value, list)
    assert len(value) == 1

    # THIS IS THE CRITICAL CHECK - value[0] must be an Interrupt, not a dict
    assert isinstance(value[0], Interrupt), (
        f"Expected Interrupt object in pending_sends, got {type(value[0])}. "
        f"This is the root cause of 'dict' object has no attribute error"
    )
    assert value[0].value == {"test": "data"}
    assert value[0].resumable is False


def test_interrupt_resume_workflow(redis_url: str) -> None:
    """
    Integration test reproducing the exact scenario from issue #113.

    This test should fail with AttributeError until the fix is implemented.
    """
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        builder = StateGraph(AgentState)
        builder.add_node("review", review_node)
        builder.add_edge(START, "review")
        builder.add_edge("review", END)

        graph = builder.compile(checkpointer=checkpointer)

        # Use unique thread ID
        config = {"configurable": {"thread_id": f"test-interrupt-{uuid4()}"}}

        # First invocation - should hit the interrupt
        initial = graph.invoke({}, config=config)
        print(f"Initial result: {initial}")

        # Resume with Command - this is where the error occurs
        # The error happens because pending_sends contains dicts instead of Interrupt objects
        # When LangGraph tries to access Interrupt attributes
        # It fails because value[0] is {'value': ..., 'resumable': ..., 'ns': ..., 'when': ...} not Interrupt(...)
        final_state = graph.invoke(Command(resume={"test": "response"}), config=config)

        # If we get here, the test passed
        assert "messages" in final_state
        print(f"Final messages: {final_state['messages']}")
