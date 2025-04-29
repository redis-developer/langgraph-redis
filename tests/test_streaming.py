"""Tests for streaming with Redis checkpointing."""

import asyncio
from typing import Any, Dict, List, Literal, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.redis import RedisSaver


class State(TypedDict):
    counter: int
    values: List[str]


def count_node(state: State) -> Dict[str, Any]:
    """Simple counting node."""
    return {"counter": state["counter"] + 1}


def values_node(state: State) -> Dict[str, Any]:
    """Add a value to the list."""
    return {"values": state["values"] + [f"value_{state['counter']}"]}


def conditional_router(state: State) -> Literal["count_node", "END"]:
    """Route based on counter value."""
    if state["counter"] < 5:
        return "count_node"
    return "END"


@pytest.fixture
def graph_with_redis_checkpointer(redis_url: str):
    """Create a graph with Redis checkpointer."""
    builder = StateGraph(State)
    builder.add_node("count_node", count_node)
    builder.add_node("values_node", values_node)
    builder.add_edge(START, "count_node")
    builder.add_edge("count_node", "values_node")
    builder.add_conditional_edges(
        "values_node", conditional_router, {"count_node": "count_node", "END": END}
    )

    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        yield graph


def test_streaming_values_with_redis_checkpointer(graph_with_redis_checkpointer):
    """Test streaming with 'values' mode."""
    # Create a thread config with a unique ID
    thread_config = {"configurable": {"thread_id": "test_stream_values"}}

    # Stream with values mode
    results = []
    for chunk in graph_with_redis_checkpointer.stream(
        {"counter": 0, "values": []}, thread_config, stream_mode="values"
    ):
        results.append(chunk)

    # Verify results
    assert len(results) == 11  # 5 iterations x 2 nodes + initial state

    # Check state history from the checkpointer
    states = list(graph_with_redis_checkpointer.get_state_history(thread_config))
    assert len(states) > 0
    final_state = states[-1]
    assert final_state.values["counter"] == 5
    assert len(final_state.values["values"]) == 5


def test_streaming_updates_with_redis_checkpointer(graph_with_redis_checkpointer):
    """Test streaming with 'updates' mode."""
    # Create a thread config with a unique ID
    thread_config = {"configurable": {"thread_id": "test_stream_updates"}}

    # Stream with updates mode
    results = []
    for chunk in graph_with_redis_checkpointer.stream(
        {"counter": 0, "values": []}, thread_config, stream_mode="updates"
    ):
        results.append(chunk)

    # Verify results - we should get an update from each node
    assert len(results) == 10  # 5 iterations x 2 nodes

    # Check that each update contains the expected keys
    for i, update in enumerate(results):
        if i % 2 == 0:  # count_node
            assert "count_node" in update
            assert "counter" in update["count_node"]
        else:  # values_node
            assert "values_node" in update
            assert "values" in update["values_node"]

    # Check state history from the checkpointer
    states = list(graph_with_redis_checkpointer.get_state_history(thread_config))
    assert len(states) > 0
    final_state = states[-1]
    assert final_state.values["counter"] == 5
    assert len(final_state.values["values"]) == 5


@pytest.mark.asyncio
async def test_streaming_with_cancellation(graph_with_redis_checkpointer):
    """Test streaming with cancellation."""
    # Create a thread config with a unique ID
    thread_config = {"configurable": {"thread_id": "test_stream_cancel"}}

    # Create a task that streams with interruption
    async def stream_with_cancel():
        results = []
        try:
            for chunk in graph_with_redis_checkpointer.stream(
                {"counter": 0, "values": []}, thread_config, stream_mode="values"
            ):
                results.append(chunk)
                if len(results) >= 3:
                    # Simulate cancellation after 3 chunks
                    raise asyncio.CancelledError()
        except asyncio.CancelledError:
            # Expected - just pass
            pass
        return results

    # Run the task
    task = asyncio.create_task(stream_with_cancel())
    await asyncio.sleep(0.1)  # Let it run a bit
    results = await task

    # Verify results - we should have 3 chunks
    assert len(results) == 3

    # Check state history from the checkpointer
    states = list(graph_with_redis_checkpointer.get_state_history(thread_config))

    # We expect some state to be saved even after cancellation
    assert len(states) > 0

    # Should be able to continue from the last saved state
    last_state = graph_with_redis_checkpointer.get_state(thread_config)
    continuation_results = []

    for chunk in graph_with_redis_checkpointer.stream(
        None, thread_config, stream_mode="values"  # No input, continue from last state
    ):
        continuation_results.append(chunk)

    # Verify we can continue after cancellation
    assert len(continuation_results) > 0
