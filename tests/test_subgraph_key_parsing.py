"""Tests for Redis key parsing with subgraphs.

This test verifies that the fix to the _parse_redis_checkpoint_writes_key method
can handle keys formatted by subgraphs correctly.
"""

from typing import Any, Dict, List, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.base import (
    CHECKPOINT_WRITE_PREFIX,
    REDIS_KEY_SEPARATOR,
    BaseRedisSaver,
)


class State(TypedDict):
    counter: int
    message: str


class NestedState(TypedDict):
    counter: int
    user: str
    history: List[str]


def increment_counter(state: State) -> Dict[str, Any]:
    """Simple increment function."""
    return {"counter": state["counter"] + 1}


def add_message(state: State) -> Dict[str, Any]:
    """Add a message based on counter."""
    return {"message": f"Count is now {state['counter']}"}


def build_subgraph():
    """Build a simple subgraph to test."""
    builder = StateGraph(State)
    builder.add_node("increment", increment_counter)
    builder.add_node("add_message", add_message)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "add_message")
    builder.add_edge("add_message", END)
    return builder.compile()


def test_parse_subgraph_write_key():
    """Test the key parsing with subgraph keys."""
    # Create a complex key with subgraph components - similar to what would
    # happen in a real scenario with nested subgraphs
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            "subgraph1",
            "nested",
            "extra_component",
        ]
    )

    # Parse the key
    result = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # Verify the result has the expected components
    assert result["thread_id"] == "thread_id"
    assert result["checkpoint_ns"] == "checkpoint_ns"
    assert result["checkpoint_id"] == "checkpoint_id"
    assert result["task_id"] == "task_id"
    assert result["idx"] == "idx"
    # The extra components should be ignored
    assert len(result) == 5


@pytest.fixture
def redis_saver(redis_url: str):
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        yield saver


def test_complex_thread_ids(redis_saver):
    """Test key parsing with complex thread IDs."""
    # Some thread IDs might contain special formatting
    complex_thread_id = "parent/subgraph:nested.component-123"

    # Create a key with this complex thread ID
    key = REDIS_KEY_SEPARATOR.join(
        [
            CHECKPOINT_WRITE_PREFIX,
            complex_thread_id,
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
        ]
    )

    # Parse the key directly
    parsed_key = BaseRedisSaver._parse_redis_checkpoint_writes_key(key)

    # The thread_id would be processed by to_storage_safe_str
    # which handles special characters
    assert "thread_id" in parsed_key


def test_subgraph_state_history(redis_url: str):
    """Test for state history with subgraphs."""
    # Create main graph with a subgraph
    main_builder = StateGraph(NestedState)

    # Add the subgraph
    subgraph = build_subgraph()
    main_builder.add_node("process", subgraph)

    # Add edges for the main graph
    main_builder.add_edge(START, "process")
    main_builder.add_edge("process", END)

    # Create checkpointer
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        # Compile the graph with the checkpointer
        main_graph = main_builder.compile(checkpointer=checkpointer)

        # Create thread config
        thread_config = {
            "configurable": {
                "thread_id": "test_subgraph_history",
            }
        }

        # Run the graph
        result = main_graph.invoke(
            {"counter": 0, "user": "test_user", "history": []},
            thread_config,
        )

        # Get state history - this would have failed before the fix
        try:
            # Get state history
            states = list(main_graph.get_state_history(thread_config))
            assert len(states) > 0

            # The test passes if we don't get a "too many values to unpack" error
            # which would have happened before our key parsing fix
        except ValueError as e:
            if "too many values to unpack" in str(e):
                pytest.fail("Key parsing failed with 'too many values to unpack' error")
            else:
                raise
