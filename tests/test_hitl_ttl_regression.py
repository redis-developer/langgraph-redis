"""
Regression tests for HitL + TTL pipeline interaction (BOA customer report).

Root Cause (PROVEN):
In non-cluster mode, aput_writes previously built a SINGLE non-transactional
pipeline mixing critical write commands (JSON.SET, JSON.MERGE, ZADD) with
best-effort TTL commands (EXPIRE). On Redis Enterprise proxy, the pipeline
could fail at EXPIRE because the proxy decomposes the pipeline into per-shard
sub-pipelines, and JSON module commands may route differently from native
commands.

When the pipeline failed at EXPIRE, the entire pipeline.execute() raised, and
the writes that were JSON.SET before the failure were orphaned without the
has_writes flag and without registry entries.

FIX (Option B):
1. Remove ALL EXPIRE calls from the critical pipeline
2. Use raise_on_error=False on pipeline execution
3. Inspect per-command results and only raise on critical failures
4. Apply TTL separately after the pipeline succeeds (best-effort)

These tests verify:
1. HitL+TTL flow works on standard Redis (regression guards)
2. The pipeline no longer contains EXPIRE commands (fix verification)
3. Writes survive even when TTL application fails
4. TTL is still applied to keys after the pipeline succeeds
"""

import logging
import operator
import sys
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Any, AsyncGenerator, Dict, Generator, List, TypedDict
from uuid import uuid4

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

requires_python_311 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() in async context requires Python 3.11+ for TaskGroup support",
)

# Customer TTL config (matches BOA setup: 6 days in minutes)
TTL_CONFIG: Dict[str, Any] = {"default_ttl": 1440 * 6, "refresh_on_read": True}


# ── State / Nodes ────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_confirmed: bool


def review_node(state: AgentState) -> Dict[str, Any]:
    """Node that interrupts for human review."""
    from langgraph.types import interrupt

    user_input = interrupt({"question": "Do you approve?", "context": "review needed"})
    return {"user_confirmed": user_input.get("approved", False)}


def process_node(state: AgentState) -> Dict[str, Any]:
    """Node that runs after human confirmation."""
    return {"messages": [HumanMessage(content="Processing complete")]}


# ── Helpers ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _async_saver(
    redis_url: str, ttl: Any = None
) -> AsyncGenerator[AsyncRedisSaver, None]:
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:
        yield saver


@contextmanager
def _sync_saver(redis_url: str, ttl: Any = None) -> Generator[RedisSaver, None, None]:
    with RedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:
        saver.setup()
        yield saver


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Verify HitL+TTL works on standard Redis (regression guards)
# ══════════════════════════════════════════════════════════════════════════════


@requires_python_311
@pytest.mark.asyncio
async def test_async_hitl_with_ttl(redis_url: str) -> None:
    """AsyncRedisSaver + TTL + HitL interrupt/resume on standard Redis."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("review", review_node)
        builder.add_node("process", process_node)
        builder.add_edge(START, "review")
        builder.add_edge("review", "process")
        builder.add_edge("process", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"hitl-ttl-async-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Please review this")],
                "user_confirmed": False,
            },
            config=config,
        )

        state = await graph.aget_state(config)
        assert state is not None
        has_interrupt = any(
            hasattr(task, "interrupts") and task.interrupts for task in state.tasks
        )
        assert has_interrupt, "Interrupt should be present in state with TTL enabled"

        final_state = await graph.ainvoke(
            Command(resume={"approved": True}),
            config=config,
        )

        assert final_state.get("user_confirmed") is True
        assert any(
            "Processing complete" in msg.content
            for msg in final_state.get("messages", [])
            if hasattr(msg, "content")
        )


@requires_python_311
@pytest.mark.asyncio
async def test_async_hitl_ttl_multi_step(redis_url: str) -> None:
    """Multi-step graph with TTL + HitL stresses the pipeline more heavily."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    class MultiStepState(TypedDict):
        counter: int
        messages: Annotated[list[AnyMessage], operator.add]

    def step1(state: MultiStepState) -> Dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}

    def step2(state: MultiStepState) -> Dict[str, Any]:
        return {"counter": state.get("counter", 0) + 1}

    def interrupt_node(state: MultiStepState) -> Dict[str, Any]:
        result = interrupt({"counter": state["counter"]})
        return {"messages": [HumanMessage(content=f"Approved: {result}")]}

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        builder = StateGraph(MultiStepState)
        builder.add_node("step1", step1)
        builder.add_node("step2", step2)
        builder.add_node("interrupt", interrupt_node)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", "interrupt")
        builder.add_edge("interrupt", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"hitl-ttl-multi-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        await graph.ainvoke({"counter": 0, "messages": []}, config=config)

        state = await graph.aget_state(config)
        assert state.values.get("counter") == 2
        has_interrupt = any(
            hasattr(task, "interrupts") and task.interrupts for task in state.tasks
        )
        assert has_interrupt

        final_state = await graph.ainvoke(
            Command(resume={"value": "confirmed"}),
            config=config,
        )
        assert len(final_state["messages"]) > 0


@requires_python_311
@pytest.mark.asyncio
async def test_async_hitl_ttl_keys_have_expiry(redis_url: str) -> None:
    """Verify TTL is actually applied to checkpoint and write keys during HitL."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    redis_client = Redis.from_url(redis_url)

    try:
        async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
            builder = StateGraph(AgentState)
            builder.add_node("review", review_node)
            builder.add_node("process", process_node)
            builder.add_edge(START, "review")
            builder.add_edge("review", "process")
            builder.add_edge("process", END)

            graph = builder.compile(checkpointer=saver)

            thread_id = f"hitl-ttl-expiry-{uuid4()}"
            config = {"configurable": {"thread_id": thread_id}}

            await graph.ainvoke(
                {"messages": [HumanMessage(content="Review")], "user_confirmed": False},
                config=config,
            )

            # Verify keys exist AND have TTL
            all_keys = await redis_client.keys(f"*{thread_id}*")
            assert len(all_keys) > 0

            keys_with_ttl = 0
            keys_without_ttl = []
            for key in all_keys:
                ttl = await redis_client.ttl(key)
                if ttl > 0:
                    keys_with_ttl += 1
                else:
                    keys_without_ttl.append(key.decode())

            assert (
                keys_with_ttl > 0
            ), "At least some keys should have TTL when ttl_config is provided"

            # Resume and verify completion
            final_state = await graph.ainvoke(
                Command(resume={"approved": True}),
                config=config,
            )
            assert final_state.get("user_confirmed") is True
    finally:
        await redis_client.aclose()


@requires_python_311
@pytest.mark.asyncio
async def test_async_hitl_ttl_single_resume(redis_url: str) -> None:
    """TTL must not cause double-resume bug."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    resume_count = 0

    def counting_interrupt(state: AgentState) -> Dict[str, Any]:
        nonlocal resume_count
        resume_count += 1
        result = interrupt({"attempt": resume_count})
        return {"user_confirmed": True}

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("interrupt", counting_interrupt)
        builder.add_edge(START, "interrupt")
        builder.add_edge("interrupt", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"hitl-ttl-single-resume-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        await graph.ainvoke(
            {"messages": [], "user_confirmed": False},
            config=config,
        )

        count_before_resume = resume_count

        await graph.ainvoke(
            Command(resume={"confirmed": True}),
            config=config,
        )

        state = await graph.aget_state(config)
        has_pending = any(
            hasattr(task, "interrupts") and task.interrupts for task in state.tasks
        )
        assert (
            not has_pending
        ), "Graph should complete after single resume with TTL enabled"
        assert resume_count == count_before_resume + 1


def test_sync_hitl_with_ttl(redis_url: str) -> None:
    """Sync RedisSaver + TTL + HitL interrupt/resume flow."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    with _sync_saver(redis_url, ttl=TTL_CONFIG) as saver:
        builder = StateGraph(AgentState)
        builder.add_node("review", review_node)
        builder.add_node("process", process_node)
        builder.add_edge(START, "review")
        builder.add_edge("review", "process")
        builder.add_edge("process", END)

        graph = builder.compile(checkpointer=saver)

        thread_id = f"hitl-ttl-sync-{uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        graph.invoke(
            {
                "messages": [HumanMessage(content="Please review this")],
                "user_confirmed": False,
            },
            config=config,
        )

        state = graph.get_state(config)
        has_interrupt = any(
            hasattr(task, "interrupts") and task.interrupts for task in state.tasks
        )
        assert has_interrupt, "Interrupt should be present in sync saver with TTL"

        final_state = graph.invoke(
            Command(resume={"approved": True}),
            config=config,
        )

        assert final_state.get("user_confirmed") is True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Verify the fix — EXPIRE is no longer in the critical pipeline
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_pipeline_no_longer_contains_expire(redis_url: str) -> None:
    """
    Verify that the critical pipeline in aput_writes no longer contains EXPIRE.

    After the fix, EXPIRE is applied separately (best-effort) after the
    critical pipeline succeeds. This test intercepts the pipeline to capture
    the exact command sequence and confirm only critical commands are present.
    """
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.types import Interrupt

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        thread_id = f"pipeline-no-expire-{uuid4()}"
        checkpoint_ns = ""

        initial_checkpoint = empty_checkpoint()
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }

        saved_config = await saver.aput(
            config,
            initial_checkpoint,
            {"source": "input", "step": 0, "writes": {}},
            {},
        )

        # Intercept the pipeline to capture commands
        captured_commands: List[List[Any]] = []
        original_pipeline = saver._redis.pipeline

        class CapturingPipeline:
            """Wraps a real pipeline to capture commands before execution."""

            def __init__(self, real_pipeline: Any) -> None:
                self._real = real_pipeline
                self._commands: List[str] = []

            def __getattr__(self, name: str) -> Any:
                attr = getattr(self._real, name)
                if name == "expire":

                    def tracked_expire(*args: Any, **kwargs: Any) -> Any:
                        self._commands.append(f"EXPIRE {args[0]}")
                        return attr(*args, **kwargs)

                    return tracked_expire
                if name == "json":
                    pipeline_ref = self

                    def capturing_json_factory() -> Any:
                        real_json = attr()  # call pipeline.json()

                        class CapturingJson:
                            def __getattr__(self_json, json_name: str) -> Any:
                                json_attr = getattr(real_json, json_name)
                                if json_name in ("set", "merge"):

                                    def tracked_json_op(
                                        *args: Any, **kwargs: Any
                                    ) -> Any:
                                        cmd = f"JSON.{json_name.upper()} {args[0]}"
                                        pipeline_ref._commands.append(cmd)
                                        return json_attr(*args, **kwargs)

                                    return tracked_json_op
                                return json_attr

                        return CapturingJson()

                    return capturing_json_factory
                if name == "zadd":

                    def tracked_zadd(*args: Any, **kwargs: Any) -> Any:
                        self._commands.append(f"ZADD {args[0]}")
                        return attr(*args, **kwargs)

                    return tracked_zadd
                if name == "execute":

                    async def tracked_execute(*args: Any, **kwargs: Any) -> Any:
                        captured_commands.append(list(self._commands))
                        return await attr(*args, **kwargs)

                    return tracked_execute
                return attr

        def capturing_pipeline(**kwargs: Any) -> CapturingPipeline:
            return CapturingPipeline(original_pipeline(**kwargs))

        saver._redis.pipeline = capturing_pipeline  # type: ignore[assignment]

        try:
            interrupt_data = Interrupt(
                value={"question": "Approve?"}, id="test-interrupt-1"
            )
            await saver.aput_writes(
                saved_config,
                [("__interrupt__", [interrupt_data])],
                "interrupt_task",
            )
        finally:
            saver._redis.pipeline = original_pipeline  # type: ignore[assignment]

        # Verify we captured the pipeline commands
        assert len(captured_commands) > 0, "Should have captured pipeline execution"

        commands = captured_commands[0]
        print(f"\nPipeline commands ({len(commands)} total):")
        for i, cmd in enumerate(commands):
            print(f"  Command {i}: {cmd}")

        # Verify the pipeline contains critical commands
        json_set_indices = [i for i, c in enumerate(commands) if "JSON.SET" in c]
        assert len(json_set_indices) > 0, "Pipeline should contain JSON.SET commands"

        # VERIFY THE FIX: pipeline must NOT contain EXPIRE
        expire_indices = [i for i, c in enumerate(commands) if "EXPIRE" in c]
        assert len(expire_indices) == 0, (
            f"Pipeline should NOT contain EXPIRE commands after the fix, "
            f"but found {len(expire_indices)}: "
            f"{[commands[i] for i in expire_indices]}"
        )


@pytest.mark.asyncio
async def test_writes_survive_ttl_failure(redis_url: str) -> None:
    """
    Verify that interrupt writes survive even when TTL application fails.

    This simulates the Redis Enterprise scenario: the critical pipeline
    (JSON.SET, JSON.MERGE, ZADD) succeeds, but the subsequent EXPIRE calls
    fail. With the fix, writes should still be intact and findable.
    """
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.types import Interrupt

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        thread_id = f"ttl-fail-survive-{uuid4()}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        saved_config = await saver.aput(
            config,
            empty_checkpoint(),
            {"source": "input", "step": 0, "writes": {}},
            {},
        )

        # Patch _apply_ttl_to_keys and expire to simulate TTL failure
        original_apply_ttl = saver._apply_ttl_to_keys
        original_expire = saver._redis.expire

        async def failing_apply_ttl(*args: Any, **kwargs: Any) -> None:
            raise ResponseError("MOVED 12345 host:port")

        async def failing_expire(*args: Any, **kwargs: Any) -> Any:
            raise ResponseError("MOVED 12345 host:port")

        saver._apply_ttl_to_keys = failing_apply_ttl  # type: ignore[assignment]
        saver._redis.expire = failing_expire  # type: ignore[assignment]

        try:
            # This should NOT raise — TTL failures are best-effort
            await saver.aput_writes(
                saved_config,
                [("__interrupt__", [Interrupt(value={"q": "ok?"}, id="int-1")])],
                "task1",
            )
        finally:
            saver._apply_ttl_to_keys = original_apply_ttl  # type: ignore[assignment]
            saver._redis.expire = original_expire  # type: ignore[assignment]

        # VERIFY: writes survived despite TTL failure
        checkpoint_tuple = await saver.aget_tuple(saved_config)
        assert checkpoint_tuple is not None

        interrupt_writes = [
            w for w in checkpoint_tuple.pending_writes if w[1] == "__interrupt__"
        ]
        assert len(interrupt_writes) > 0, (
            "Interrupt writes should survive TTL failure — "
            "this is the core fix for the BOA customer issue"
        )
        print(f"\nInterrupt writes found after TTL failure: {len(interrupt_writes)}")
        print(f"Total pending writes: {len(checkpoint_tuple.pending_writes)}")


@pytest.mark.asyncio
async def test_ttl_failure_logs_warning(redis_url: str, caplog: Any) -> None:
    """
    Verify that TTL failure produces a warning log but does not raise.
    """
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.types import Interrupt

    async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
        thread_id = f"ttl-warning-{uuid4()}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        saved_config = await saver.aput(
            config,
            empty_checkpoint(),
            {"source": "input", "step": 0, "writes": {}},
            {},
        )

        # Patch to simulate TTL failure
        original_apply_ttl = saver._apply_ttl_to_keys
        original_expire = saver._redis.expire

        async def failing_apply_ttl(*args: Any, **kwargs: Any) -> None:
            raise ResponseError("MOVED 12345 host:port")

        async def failing_expire(*args: Any, **kwargs: Any) -> Any:
            raise ResponseError("MOVED 12345 host:port")

        saver._apply_ttl_to_keys = failing_apply_ttl  # type: ignore[assignment]
        saver._redis.expire = failing_expire  # type: ignore[assignment]

        try:
            with caplog.at_level(
                logging.WARNING, logger="langgraph.checkpoint.redis.aio"
            ):
                await saver.aput_writes(
                    saved_config,
                    [("__interrupt__", [Interrupt(value={"q": "ok?"}, id="int-1")])],
                    "task1",
                )

            # Verify warning was logged about TTL failure
            ttl_warnings = [
                r
                for r in caplog.records
                if "TTL" in r.message or "ttl" in r.message.lower()
            ]
            assert len(ttl_warnings) > 0, (
                f"Should have logged a TTL warning, but got: "
                f"{[r.message for r in caplog.records]}"
            )
        finally:
            # Restore originals BEFORE aget_tuple (which also calls
            # _apply_ttl_to_keys via refresh_on_read)
            saver._apply_ttl_to_keys = original_apply_ttl  # type: ignore[assignment]
            saver._redis.expire = original_expire  # type: ignore[assignment]

        # Verify writes survived (with originals restored so aget_tuple works)
        checkpoint_tuple = await saver.aget_tuple(saved_config)
        assert checkpoint_tuple is not None
        interrupt_writes = [
            w for w in checkpoint_tuple.pending_writes if w[1] == "__interrupt__"
        ]
        assert len(interrupt_writes) > 0


@pytest.mark.asyncio
async def test_ttl_applied_separately_after_pipeline(redis_url: str) -> None:
    """
    Verify that TTL is still applied to keys even though EXPIRE is not in
    the critical pipeline. TTL should be applied separately after the
    pipeline succeeds.
    """
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.types import Interrupt

    redis_client = Redis.from_url(redis_url)

    try:
        async with _async_saver(redis_url, ttl=TTL_CONFIG) as saver:
            thread_id = f"ttl-separate-{uuid4()}"
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }

            saved_config = await saver.aput(
                config,
                empty_checkpoint(),
                {"source": "input", "step": 0, "writes": {}},
                {},
            )

            await saver.aput_writes(
                saved_config,
                [("__interrupt__", [Interrupt(value={"q": "ok?"}, id="int-1")])],
                "task1",
            )

            # Check that write keys have TTL
            write_keys = await redis_client.keys(f"checkpoint_write:{thread_id}*")
            assert len(write_keys) > 0, "Should have checkpoint_write keys"

            keys_with_ttl = 0
            for key in write_keys:
                ttl = await redis_client.ttl(key)
                if ttl > 0:
                    keys_with_ttl += 1

            assert keys_with_ttl > 0, (
                "checkpoint_write keys should have TTL applied "
                "(TTL is set separately after the pipeline)"
            )

            # Check registry key has TTL too
            registry_keys = await redis_client.keys(f"write_keys_zset:{thread_id}*")
            for key in registry_keys:
                ttl = await redis_client.ttl(key)
                assert ttl > 0, f"Registry key {key.decode()} should have TTL"

    finally:
        await redis_client.aclose()
