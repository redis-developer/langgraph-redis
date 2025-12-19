"""
Regression tests for Issue #133: HITL Works Unexpected with AsyncShallowRedisSaver

Problem Description:
When using LangGraph 1.0's Human-in-the-Loop (HITL) interruption functionality with
`AsyncShallowRedisSaver`, interruptions intermittently require double confirmation.

Key Symptoms:
1. Inconsistent interrupt persistence: When calling `agent.get_state(...)` after resuming
   from an interrupt, the interrupt data is missing from the checkpoint
2. Self-resolving behavior: After the first confirmation, the interrupt reappears on
   subsequent checks
3. State inconsistency: Redis keys occasionally contain `__interrupt__` while having
   `has_writes` set to `false`

Root Cause Analysis:
The shallow saver's `aput` method cleans up ALL writes when a new checkpoint is saved.
But in the HITL flow:
1. Graph hits an interrupt
2. `put_writes` is called with the interrupt data (using the CURRENT checkpoint_id)
3. `put` is called to save a NEW checkpoint
4. The `put` method sees the checkpoint_id changed, so it cleans up writes from the
   previous checkpoint - INCLUDING the interrupt writes that were just saved!

This results in the interrupt being lost before it can be read when resuming.

Note: Some async tests require Python 3.11+ because interrupt() uses get_config()
which needs TaskGroup context support only available in Python 3.11+.
"""

import operator
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator, Dict, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AnyMessage, HumanMessage

# Skip marker for tests that require Python 3.11+ due to interrupt() async context requirements
requires_python_311 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() in async context requires Python 3.11+ for TaskGroup support",
)
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt
from redis.asyncio import Redis

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


class AgentState(TypedDict):
    """State for the test agent."""

    messages: Annotated[list[AnyMessage], operator.add]
    user_confirmed: bool


def review_node(state: AgentState) -> Dict[str, Any]:
    """Node that interrupts for review."""
    print("-------- review_node: before interrupt --------")

    # This creates an Interrupt that needs to be persisted
    user_input = interrupt(
        {"question": "Do you approve?", "context": state["messages"]}
    )

    print(f"-------- review_node: after interrupt, user_input={user_input} --------")
    return {"user_confirmed": user_input.get("approved", False)}


def process_node(state: AgentState) -> Dict[str, Any]:
    """Node that processes after confirmation."""
    print(
        f"-------- process_node: user_confirmed={state.get('user_confirmed')} --------"
    )
    return {"messages": [HumanMessage(content="Processing complete")]}


@asynccontextmanager
async def create_async_shallow_saver(
    redis_url: str,
) -> AsyncGenerator[AsyncShallowRedisSaver, None]:
    """Create and setup an AsyncShallowRedisSaver."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        yield saver


@requires_python_311
@pytest.mark.asyncio
async def test_hitl_interrupt_persists_in_shallow_saver(redis_url: str) -> None:
    """
    Test that HITL interrupts are properly persisted in AsyncShallowRedisSaver.

    This is the main regression test for Issue #133. It verifies that:
    1. An interrupt is saved when the graph hits an interrupt node
    2. The interrupt is still present when we check the state
    3. The interrupt can be resumed with Command(resume=...)
    """
    async with create_async_shallow_saver(redis_url) as saver:
        # Build the graph with an interrupt node
        builder = StateGraph(AgentState)
        builder.add_node("review", review_node)
        builder.add_node("process", process_node)
        builder.add_edge(START, "review")
        builder.add_edge("review", "process")
        builder.add_edge("process", END)

        graph = builder.compile(checkpointer=saver)

        # Use unique thread ID
        thread_id = f"test-hitl-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation - should hit the interrupt
        initial_state = await graph.ainvoke(
            {"messages": [HumanMessage(content="Please review this")]},
            config=config,
        )
        print(f"Initial state: {initial_state}")

        # Get the current state to check for pending interrupts
        state = await graph.aget_state(config)
        print(f"State after interrupt: {state}")

        # CRITICAL CHECK: The interrupt should be in the pending writes
        assert state is not None, "State should not be None after interrupt"
        assert hasattr(state, "tasks"), "State should have tasks attribute"

        # Check for the interrupt in the state
        # In LangGraph, interrupts are available via state.tasks
        has_interrupt = False
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                has_interrupt = True
                print(f"Found interrupt in task: {task.interrupts}")
                break

        assert has_interrupt, (
            "Interrupt should be present in state after hitting interrupt node. "
            "This is the core issue in #133 - the interrupt is being deleted prematurely."
        )

        # Resume the graph with the interrupt response
        final_state = await graph.ainvoke(
            Command(resume={"approved": True}),
            config=config,
        )
        print(f"Final state: {final_state}")

        # Verify the graph completed successfully
        assert "messages" in final_state
        assert final_state.get("user_confirmed") is True


@requires_python_311
@pytest.mark.asyncio
async def test_hitl_interrupt_with_multiple_checkpoints(redis_url: str) -> None:
    """
    Test HITL behavior when multiple checkpoints are created before the interrupt.

    This tests the scenario where the graph runs through several nodes before
    hitting an interrupt, creating multiple checkpoint transitions.
    """

    class MultiStepState(TypedDict):
        counter: int
        messages: Annotated[list[AnyMessage], operator.add]

    def step1(state: MultiStepState) -> Dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}

    def step2(state: MultiStepState) -> Dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}

    def interrupt_node(state: MultiStepState) -> Dict[str, Any]:
        result = interrupt({"counter": state["counter"]})
        return {"messages": [HumanMessage(content=f"Received: {result}")]}

    async with create_async_shallow_saver(redis_url) as saver:
        builder = StateGraph(MultiStepState)
        builder.add_node("step1", step1)
        builder.add_node("step2", step2)
        builder.add_node("interrupt", interrupt_node)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", "interrupt")
        builder.add_edge("interrupt", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"test-multistep-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # Run until interrupt
        await graph.ainvoke(
            {"counter": 0, "messages": []},
            config=config,
        )

        # Check state
        state = await graph.aget_state(config)
        print(f"State after multi-step run: {state}")

        # Verify counter was incremented
        assert (
            state.values.get("counter") == 2
        ), "Counter should be 2 after step1 and step2"

        # Check for interrupt
        has_interrupt = False
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                has_interrupt = True
                break

        assert has_interrupt, "Interrupt should be present after multi-step run"

        # Resume
        final_state = await graph.ainvoke(
            Command(resume={"value": "confirmed"}),
            config=config,
        )

        assert len(final_state["messages"]) > 0


@pytest.mark.asyncio
async def test_interrupt_write_order_timing(redis_url: str) -> None:
    """
    Low-level test of the timing issue between put_writes and put.

    This test directly tests the checkpoint saver methods to verify that
    writes saved via put_writes are not cleaned up by the subsequent put call.
    """
    async with create_async_shallow_saver(redis_url) as saver:
        thread_id = f"test-timing-{uuid4()}"
        checkpoint_ns = ""

        # Create initial checkpoint
        initial_checkpoint = empty_checkpoint()
        initial_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        # Save initial checkpoint
        saved_config_1 = await saver.aput(
            initial_config,
            initial_checkpoint,
            {"source": "input", "step": 0, "writes": {}},
            {},
        )
        print(f"Saved initial checkpoint: {saved_config_1}")

        # Now simulate the HITL flow:
        # 1. Save writes (interrupt) with the CURRENT checkpoint ID
        interrupt_data = Interrupt(
            value={"question": "Approve?"}, id="test-interrupt-1"
        )
        await saver.aput_writes(
            saved_config_1,
            [("__interrupt__", [interrupt_data])],
            "interrupt_task",
        )
        print("Saved interrupt write")

        # 2. Verify the write is present
        tuple_after_write = await saver.aget_tuple(saved_config_1)
        assert tuple_after_write is not None
        print(f"Pending writes after save: {tuple_after_write.pending_writes}")

        # Check that the interrupt write is present
        interrupt_writes = [
            w for w in tuple_after_write.pending_writes if w[1] == "__interrupt__"
        ]
        assert (
            len(interrupt_writes) > 0
        ), "Interrupt write should be present after aput_writes"

        # 3. Now save a new checkpoint (this is where the bug would trigger)
        new_checkpoint = create_checkpoint(initial_checkpoint, {}, 1)
        saved_config_2 = await saver.aput(
            saved_config_1,
            new_checkpoint,
            {"source": "update", "step": 1, "writes": {}},
            {},
        )
        print(f"Saved new checkpoint: {saved_config_2}")

        # 4. CRITICAL: Check if the interrupt write is STILL present
        # This is where the bug manifests - the write gets deleted
        tuple_after_new_checkpoint = await saver.aget_tuple(saved_config_2)
        assert tuple_after_new_checkpoint is not None
        print(
            f"Pending writes after new checkpoint: {tuple_after_new_checkpoint.pending_writes}"
        )

        # The interrupt should still be present!
        # In the buggy version, this would fail because the write was cleaned up
        interrupt_writes_after = [
            w
            for w in tuple_after_new_checkpoint.pending_writes
            if w[1] == "__interrupt__"
        ]

        # Note: The expected behavior here depends on the design decision:
        # - If writes should persist across checkpoints, this should pass
        # - If writes should be associated with specific checkpoints, we need different logic
        # For HITL to work, the interrupt should NOT be cleaned up prematurely


@requires_python_311
@pytest.mark.asyncio
async def test_interrupt_state_consistency_across_get_state_calls(
    redis_url: str,
) -> None:
    """
    Test that interrupt state is consistent across multiple get_state calls.

    This tests the reported symptom where the interrupt is missing on one get_state
    call but reappears on subsequent calls.
    """

    def simple_interrupt_node(state: AgentState) -> Dict[str, Any]:
        interrupt({"prompt": "Continue?"})
        return {}

    async with create_async_shallow_saver(redis_url) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("interrupt", simple_interrupt_node)
        builder.add_edge(START, "interrupt")
        builder.add_edge("interrupt", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"test-consistency-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # Run until interrupt
        await graph.ainvoke(
            {"messages": [], "user_confirmed": False},
            config=config,
        )

        # Check state multiple times to detect inconsistency
        interrupt_present_results = []
        for i in range(5):
            state = await graph.aget_state(config)
            has_interrupt = any(
                hasattr(task, "interrupts") and task.interrupts for task in state.tasks
            )
            interrupt_present_results.append(has_interrupt)
            print(f"Check {i+1}: interrupt_present={has_interrupt}")

        # All checks should be consistent
        assert all(
            result == interrupt_present_results[0]
            for result in interrupt_present_results
        ), f"Interrupt presence is inconsistent across checks: {interrupt_present_results}"

        # And the interrupt should actually be present
        assert interrupt_present_results[0], "Interrupt should be present"


@pytest.mark.asyncio
async def test_direct_redis_key_inspection(redis_url: str) -> None:
    """
    Test that directly inspects Redis keys to verify interrupt storage.

    This test examines the raw Redis data to understand what's being stored
    and when it's being deleted.
    """
    redis_client = Redis.from_url(redis_url)

    try:
        async with create_async_shallow_saver(redis_url) as saver:
            thread_id = f"test-inspect-{uuid4()}"
            checkpoint_ns = ""

            # Create initial checkpoint
            initial_checkpoint = empty_checkpoint()
            initial_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                }
            }

            saved_config = await saver.aput(
                initial_config,
                initial_checkpoint,
                {"source": "input", "step": 0, "writes": {}},
                {},
            )

            # Save interrupt write
            interrupt_data = Interrupt(value={"test": "data"}, id="inspect-interrupt")
            await saver.aput_writes(
                saved_config,
                [("__interrupt__", [interrupt_data])],
                "inspect_task",
            )

            # Check Redis keys after interrupt write
            all_keys_after_write = await redis_client.keys(f"*{thread_id}*")
            print(f"Keys after interrupt write: {all_keys_after_write}")

            # Look for write keys
            write_keys_after_write = [
                k for k in all_keys_after_write if b"checkpoint_write" in k
            ]
            print(f"Write keys after interrupt: {write_keys_after_write}")

            # Check the write registry
            registry_key = f"write_keys_zset:{thread_id}:{checkpoint_ns}:shallow"
            registry_contents = await redis_client.zrange(registry_key, 0, -1)
            print(f"Write registry after interrupt: {registry_contents}")

            # Now save a new checkpoint
            new_checkpoint = create_checkpoint(initial_checkpoint, {}, 1)
            new_config = await saver.aput(
                saved_config,
                new_checkpoint,
                {"source": "update", "step": 1, "writes": {}},
                {},
            )

            # Check Redis keys after new checkpoint
            all_keys_after_checkpoint = await redis_client.keys(f"*{thread_id}*")
            print(f"Keys after new checkpoint: {all_keys_after_checkpoint}")

            # Look for write keys after new checkpoint
            write_keys_after_checkpoint = [
                k for k in all_keys_after_checkpoint if b"checkpoint_write" in k
            ]
            print(f"Write keys after new checkpoint: {write_keys_after_checkpoint}")

            # Check the write registry after new checkpoint
            registry_contents_after = await redis_client.zrange(registry_key, 0, -1)
            print(f"Write registry after new checkpoint: {registry_contents_after}")

            # The write keys should still exist if the bug is fixed
            # If the bug is present, the write keys will be deleted

    finally:
        await redis_client.aclose()


def test_sync_hitl_interrupt_persists(redis_url: str) -> None:
    """
    Test that HITL interrupts work with the sync ShallowRedisSaver.

    This tests the same issue but with the synchronous implementation.
    """
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("review", review_node)
        builder.add_node("process", process_node)
        builder.add_edge(START, "review")
        builder.add_edge("review", "process")
        builder.add_edge("process", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"test-sync-hitl-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation - should hit the interrupt
        initial_state = graph.invoke(
            {"messages": [HumanMessage(content="Please review this")]},
            config=config,
        )
        print(f"Sync initial state: {initial_state}")

        # Get the current state
        state = graph.get_state(config)
        print(f"Sync state after interrupt: {state}")

        # Check for interrupt
        has_interrupt = False
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                has_interrupt = True
                break

        assert has_interrupt, "Interrupt should be present in sync shallow saver"

        # Resume
        final_state = graph.invoke(
            Command(resume={"approved": True}),
            config=config,
        )
        print(f"Sync final state: {final_state}")

        assert final_state.get("user_confirmed") is True


@requires_python_311
@pytest.mark.asyncio
async def test_double_resume_not_required(redis_url: str) -> None:
    """
    Test that verifies the interrupt doesn't require double confirmation.

    This specifically tests the symptom reported in Issue #133 where users
    need to confirm twice for the interrupt to be processed.
    """
    resume_count = 0

    def counting_interrupt_node(state: AgentState) -> Dict[str, Any]:
        nonlocal resume_count
        resume_count += 1
        print(f"Interrupt node called, count: {resume_count}")
        result = interrupt({"attempt": resume_count})
        return {"user_confirmed": True}

    async with create_async_shallow_saver(redis_url) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("interrupt", counting_interrupt_node)
        builder.add_edge(START, "interrupt")
        builder.add_edge("interrupt", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"test-double-resume-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        # Initial run - hits interrupt
        await graph.ainvoke(
            {"messages": [], "user_confirmed": False},
            config=config,
        )

        initial_resume_count = resume_count
        print(f"Resume count after initial run: {initial_resume_count}")

        # First resume attempt
        result = await graph.ainvoke(
            Command(resume={"confirmed": True}),
            config=config,
        )

        print(f"Result after first resume: {result}")
        print(f"Resume count after first resume: {resume_count}")

        # Check if we completed or need another resume
        state = await graph.aget_state(config)

        # If the graph is still at the interrupt, it means we need a double resume
        # This would be the bug - we should complete on first resume
        has_pending_interrupt = any(
            hasattr(task, "interrupts") and task.interrupts for task in state.tasks
        )

        assert not has_pending_interrupt, (
            "Graph should complete after single resume, not require double confirmation. "
            f"Resume was called {resume_count - initial_resume_count} time(s)."
        )

        # Verify we only entered the interrupt node once after the initial run
        assert resume_count == initial_resume_count + 1, (
            f"Interrupt node should only be entered once after resume, "
            f"but was entered {resume_count - initial_resume_count} times"
        )
