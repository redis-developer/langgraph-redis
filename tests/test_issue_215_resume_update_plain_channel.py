"""Regression test for Issue #215.

With the shallow savers, resuming a thread a second time with
Command(resume=..., update=...) raised:

    InvalidUpdateError: At key 'limit': Can receive only one value per step

The update from the first resume was loaded again as a pending write of the
later checkpoint, next to the new update, and a plain (LastValue) channel can
take only one value per step. Fixed in #213; this runs the graph from the
issue end to end and compares it with LangGraph's InMemorySaver.
"""

import operator
import sys
import uuid
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# interrupt() inside an async node needs Python 3.11+ (see test_issue_133_hitl_shallow.py)
requires_python_311 = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() in async context requires Python 3.11+ for TaskGroup support",
)


class State(TypedDict):
    limit: int  # plain channel: one value per step
    log: Annotated[list[str], operator.add]


def ask(state: State) -> dict[str, list[str]]:
    answer = interrupt("approve?")
    return {"log": [f"answer={answer}"]}


def _compile(checkpointer: Any) -> Any:
    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=checkpointer)


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


TURNS = (1, 2)
EXPECTED_FINAL = {
    "limit": 2,
    "log": ["turn 1", "answer=yes", "turn 2", "answer=yes"],
}


def _run_turns(graph: Any, config: RunnableConfig) -> list[dict]:
    results = []
    for turn in TURNS:
        graph.invoke({"limit": 5, "log": [f"turn {turn}"]}, config)
        # Resume and set the plain channel in the same call.
        results.append(
            graph.invoke(Command(resume="yes", update={"limit": turn}), config)
        )
    return results


async def _arun_turns(graph: Any, config: RunnableConfig) -> list[dict]:
    results = []
    for turn in TURNS:
        await graph.ainvoke({"limit": 5, "log": [f"turn {turn}"]}, config)
        results.append(
            await graph.ainvoke(Command(resume="yes", update={"limit": turn}), config)
        )
    return results


def test_shallow_second_resume_with_update_matches_in_memory(redis_url: str) -> None:
    reference = _run_turns(_compile(InMemorySaver()), _config())
    assert reference[-1] == EXPECTED_FINAL

    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _compile(saver)
        config = _config()

        assert _run_turns(graph, config) == reference
        assert graph.get_state(config).values == EXPECTED_FINAL


@requires_python_311
async def test_ashallow_second_resume_with_update_matches_in_memory(
    redis_url: str,
) -> None:
    reference = await _arun_turns(_compile(InMemorySaver()), _config())
    assert reference[-1] == EXPECTED_FINAL

    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        graph = _compile(saver)
        config = _config()

        assert await _arun_turns(graph, config) == reference
        assert (await graph.aget_state(config)).values == EXPECTED_FINAL
