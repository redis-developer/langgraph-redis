"""RedisSaver.get_tuple() must return writes saved before their checkpoint.

LangGraph (default durability="async") hands put() for a checkpoint and
put_writes() for that checkpoint's tasks to its background executor without
ordering them. When the writes land first, put() stores the checkpoint document
again with has_writes=False. The writes and their registry entries are in Redis,
but get_tuple() trusted the flag and returned no pending writes, so get_state()
lost a fresh interrupt and a finished parallel task ran again on resume.
AsyncRedisSaver.aget_tuple() reads the registry and was not affected.
"""

import uuid
from operator import add
from typing import Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

METADATA: CheckpointMetadata = {"source": "loop", "step": 1}
GRAPH_RUNS = 20


def _thread() -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": f"writes-first-{uuid.uuid4()}",
            "checkpoint_ns": "",
        }
    }


def _at(config: RunnableConfig, checkpoint_id: str) -> RunnableConfig:
    return {"configurable": {**config["configurable"], "checkpoint_id": checkpoint_id}}


def test_get_tuple_returns_writes_saved_before_the_checkpoint(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread()
        checkpoint = empty_checkpoint()

        # The order LangGraph can produce: writes first, then the checkpoint.
        saver.put_writes(_at(config, checkpoint["id"]), [("answer", "yes")], "task-1")
        saver.put(config, checkpoint, METADATA, {})

        # Precondition: the write is stored and registered for this checkpoint.
        thread_id = config["configurable"]["thread_id"]
        assert saver._key_registry.get_write_count(thread_id, "", checkpoint["id"]) == 1

        latest = saver.get_tuple(config)
        by_id = saver.get_tuple(_at(config, checkpoint["id"]))

    assert latest is not None and by_id is not None
    assert latest.pending_writes == [("task-1", "answer", "yes")]
    assert by_id.pending_writes == [("task-1", "answer", "yes")]


def test_get_tuple_returns_writes_saved_after_the_checkpoint(redis_url: str) -> None:
    """Control: the usual order worked before and still does."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread()
        checkpoint = empty_checkpoint()
        saver.put(config, checkpoint, METADATA, {})
        saver.put_writes(_at(config, checkpoint["id"]), [("answer", "yes")], "task-1")

        latest = saver.get_tuple(config)

    assert latest is not None
    assert latest.pending_writes == [("task-1", "answer", "yes")]


def test_get_tuple_without_writes_returns_none(redis_url: str) -> None:
    """Control: a checkpoint with no writes still has no pending writes."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread()
        checkpoint = empty_checkpoint()
        saver.put(config, checkpoint, METADATA, {})
        latest = saver.get_tuple(config)

    assert latest is not None
    assert latest.pending_writes == []


@pytest.mark.asyncio
async def test_aget_tuple_returns_writes_saved_before_the_checkpoint(
    redis_url: str,
) -> None:
    """Control: the async twin already read the registry, so it passes on both."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread()
        checkpoint = empty_checkpoint()
        await saver.aput_writes(
            _at(config, checkpoint["id"]), [("answer", "yes")], "task-1"
        )
        await saver.aput(config, checkpoint, METADATA, {})

        thread_id = config["configurable"]["thread_id"]
        count = await saver._key_registry.get_write_count(
            thread_id, "", checkpoint["id"]
        )
        assert count == 1
        latest = await saver.aget_tuple(config)

    assert latest is not None
    assert latest.pending_writes == [("task-1", "answer", "yes")]


class State(TypedDict, total=False):
    log: Annotated[list[str], add]


def _ask(state: State) -> dict:
    return {"log": [f"ask:{interrupt('approve?')}"]}


def test_get_state_shows_a_fresh_interrupt(redis_url: str) -> None:
    builder = StateGraph(State)
    builder.add_node("ask", _ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = builder.compile(checkpointer=saver)
        missing = 0
        for _ in range(GRAPH_RUNS):
            config = _thread()
            graph.invoke({}, config)
            missing += not graph.get_state(config).interrupts

    assert missing == 0, f"interrupt missing from get_state in {missing}/{GRAPH_RUNS}"


def test_finished_parallel_task_does_not_run_again_on_resume(redis_url: str) -> None:
    calls: list[str] = []

    def work(state: State) -> dict:
        calls.append("work")
        return {"log": ["work"]}

    builder = StateGraph(State)
    builder.add_node("ask", _ask)
    builder.add_node("work", work)
    builder.add_edge(START, "ask")
    builder.add_edge(START, "work")
    builder.add_edge("ask", END)
    builder.add_edge("work", END)

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = builder.compile(checkpointer=saver)
        reran = 0
        for _ in range(GRAPH_RUNS):
            config = _thread()
            calls.clear()
            graph.invoke({}, config)
            result = graph.invoke(Command(resume="yes"), config)
            assert sorted(result["log"]) == ["ask:yes", "work"]
            reran += len(calls) > 1

    assert reran == 0, f"work ran again on resume in {reran}/{GRAPH_RUNS}"
