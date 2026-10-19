"""The shallow savers must return only the pending writes of the stored checkpoint.

The shallow savers keep one write registry per thread and namespace, and since
Issue #133 that registry is no longer cleared when a new checkpoint is saved. The
loaders returned every write in it, so get_tuple / aget_tuple handed back writes
made against earlier checkpoints as pending writes of the current one. LangGraph
applies a checkpoint's pending writes from the null task (for example the update
in Command(update=..., resume=...)) at the start of every run, so such an update
was applied again on each later resume.

RedisSaver and LangGraph's InMemorySaver keep pending writes per checkpoint id.
"""

import operator
import sys
import uuid
from typing import Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Durability, interrupt

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# interrupt() inside an async node needs Python 3.11+ (see test_issue_133_hitl_shallow.py)
requires_python_311 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() in async context requires Python 3.11+ for TaskGroup support",
)

METADATA: CheckpointMetadata = {"source": "loop", "step": 1, "writes": {}}


def _thread_config(prefix: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": f"{prefix}-{uuid.uuid4()}",
            "checkpoint_ns": "",
        }
    }


class State(TypedDict):
    log: Annotated[list, operator.add]


def _ask(name: str):
    def node(state: State) -> dict:
        return {"log": [f"{name}:{interrupt(name)}"]}

    return node


def _three_question_graph():
    builder = StateGraph(State)
    for name in ("ask1", "ask2", "ask3"):
        builder.add_node(name, _ask(name))
    builder.add_edge(START, "ask1")
    builder.add_edge("ask1", "ask2")
    builder.add_edge("ask2", "ask3")
    builder.add_edge("ask3", END)
    return builder


RESUMES = [
    Command(resume="A", update={"log": ["update"]}),
    Command(resume="B"),
    Command(resume="C"),
]
EXPECTED_LOG = ["start", "update", "ask1:A", "ask2:B", "ask3:C"]


def test_in_memory_reference_log() -> None:
    """Reference: what LangGraph's own InMemorySaver produces for the same run."""
    graph = _three_question_graph().compile(checkpointer=InMemorySaver())
    config = _thread_config("memory")
    graph.invoke({"log": ["start"]}, config)
    for command in RESUMES:
        result = graph.invoke(command, config)
    assert result["log"] == EXPECTED_LOG


def test_shallow_get_tuple_excludes_writes_of_earlier_checkpoint(
    redis_url: str,
) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread_config("shallow-stale")
        first = saver.put(config, empty_checkpoint(), METADATA, {})
        saver.put_writes(first, [("log", "old")], "task-1")
        second = saver.put(config, empty_checkpoint(), METADATA, {})

        result = saver.get_tuple(second)
        assert result is not None
        assert result.pending_writes == []

        saver.put_writes(second, [("log", "new")], "task-2")
        result = saver.get_tuple(second)
        assert result is not None
        assert result.pending_writes == [("task-2", "log", "new")]


async def test_ashallow_aget_tuple_excludes_writes_of_earlier_checkpoint(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        config = _thread_config("ashallow-stale")
        first = await saver.aput(config, empty_checkpoint(), METADATA, {})
        await saver.aput_writes(first, [("log", "old")], "task-1")
        second = await saver.aput(config, empty_checkpoint(), METADATA, {})

        result = await saver.aget_tuple(second)
        assert result is not None
        assert result.pending_writes == []

        await saver.aput_writes(second, [("log", "new")], "task-2")
        result = await saver.aget_tuple(second)
        assert result is not None
        assert result.pending_writes == [("task-2", "log", "new")]


def test_shallow_get_tuple_returns_writes_of_current_checkpoint(
    redis_url: str,
) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config = _thread_config("shallow-current")
        saved = saver.put(config, empty_checkpoint(), METADATA, {})
        saver.put_writes(saved, [("log", "a"), ("log", "b")], "task-1")

        result = saver.get_tuple(saved)
        assert result is not None
        assert result.pending_writes == [("task-1", "log", "a"), ("task-1", "log", "b")]


async def test_ashallow_aget_tuple_returns_writes_of_current_checkpoint(
    redis_url: str,
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        config = _thread_config("ashallow-current")
        saved = await saver.aput(config, empty_checkpoint(), METADATA, {})
        await saver.aput_writes(saved, [("log", "a"), ("log", "b")], "task-1")

        result = await saver.aget_tuple(saved)
        assert result is not None
        assert result.pending_writes == [("task-1", "log", "a"), ("task-1", "log", "b")]


# With durability="exit" the writes are saved before the checkpoint they belong to.
DURABILITIES = ["sync", "async", "exit"]


@pytest.mark.parametrize("durability", DURABILITIES)
def test_shallow_resume_update_is_applied_once(
    redis_url: str, durability: Durability
) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _three_question_graph().compile(checkpointer=saver)
        config = _thread_config("shallow-resume")
        graph.invoke({"log": ["start"]}, config, durability=durability)
        for command in RESUMES:
            result = graph.invoke(command, config, durability=durability)
        assert result["log"] == EXPECTED_LOG


@requires_python_311
@pytest.mark.parametrize("durability", DURABILITIES)
async def test_ashallow_resume_update_is_applied_once(
    redis_url: str, durability: Durability
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        graph = _three_question_graph().compile(checkpointer=saver)
        config = _thread_config("ashallow-resume")
        await graph.ainvoke({"log": ["start"]}, config, durability=durability)
        for command in RESUMES:
            result = await graph.ainvoke(command, config, durability=durability)
        assert result["log"] == EXPECTED_LOG
