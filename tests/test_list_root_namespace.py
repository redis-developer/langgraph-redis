"""list() must treat ``checkpoint_ns=""`` as the root namespace, not as "any".

LangGraph stores a subgraph's checkpoints on the parent's thread under a child
namespace and reads the parent's history with ``checkpoint_ns=""``. The savers
only filtered on a truthy namespace, so the root listing returned every
namespace and ``get_state_history`` mixed the subgraph's checkpoints into the
parent's history. A config with no ``checkpoint_ns`` key still lists them all.
"""

import uuid
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

CHILD_NS = "child:1"
METADATA: CheckpointMetadata = {"source": "loop", "step": 1}


class State(TypedDict):
    count: int


def _increment(state: State) -> dict:
    return {"count": state["count"] + 1}


def _graph(saver):
    child = StateGraph(State)
    child.add_node("first", _increment)
    child.add_node("second", _increment)
    child.add_edge(START, "first")
    child.add_edge("first", "second")
    child.add_edge("second", END)

    parent = StateGraph(State)
    parent.add_node("child", child.compile())
    parent.add_node("after", _increment)
    parent.add_edge(START, "child")
    parent.add_edge("child", "after")
    parent.add_edge("after", END)
    return parent.compile(checkpointer=saver)


def _config(thread_id: str, checkpoint_ns: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}


def _namespaces(tuples: list) -> list[str]:
    return sorted(t.config["configurable"]["checkpoint_ns"] for t in tuples)


def _history_steps(history: list) -> list:
    return [(s.metadata["step"], s.values) for s in history]


def test_get_state_history_excludes_subgraph_checkpoints(redis_url: str) -> None:
    thread = {"configurable": {"thread_id": f"root-ns-{uuid.uuid4()}"}}
    reference = _graph(InMemorySaver())
    reference.invoke({"count": 0}, thread)
    expected = _history_steps(list(reference.get_state_history(thread)))

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        app = _graph(saver)
        app.invoke({"count": 0}, thread)
        history = list(app.get_state_history(thread))

    assert {s.config["configurable"]["checkpoint_ns"] for s in history} == {""}
    assert _history_steps(history) == expected


@pytest.mark.asyncio
async def test_aget_state_history_excludes_subgraph_checkpoints(
    redis_url: str,
) -> None:
    thread = {"configurable": {"thread_id": f"root-ns-{uuid.uuid4()}"}}
    reference = _graph(InMemorySaver())
    await reference.ainvoke({"count": 0}, thread)
    expected = _history_steps([s async for s in reference.aget_state_history(thread)])

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        app = _graph(saver)
        await app.ainvoke({"count": 0}, thread)
        history = [s async for s in app.aget_state_history(thread)]

    assert {s.config["configurable"]["checkpoint_ns"] for s in history} == {""}
    assert _history_steps(history) == expected


def test_list_filters_by_root_namespace(redis_url: str) -> None:
    thread_id = f"root-ns-{uuid.uuid4()}"

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        for checkpoint_ns in ("", CHILD_NS):
            config = _config(thread_id, checkpoint_ns)
            saver.put(config, empty_checkpoint(), METADATA, {})

        root = list(saver.list(_config(thread_id, "")))
        child = list(saver.list(_config(thread_id, CHILD_NS)))
        every = list(saver.list({"configurable": {"thread_id": thread_id}}))

    assert _namespaces(root) == [""]
    assert _namespaces(child) == [CHILD_NS]
    assert _namespaces(every) == ["", CHILD_NS]


@pytest.mark.asyncio
async def test_alist_filters_by_root_namespace(redis_url: str) -> None:
    thread_id = f"root-ns-{uuid.uuid4()}"

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        for checkpoint_ns in ("", CHILD_NS):
            config = _config(thread_id, checkpoint_ns)
            await saver.aput(config, empty_checkpoint(), METADATA, {})

        root = [t async for t in saver.alist(_config(thread_id, ""))]
        child = [t async for t in saver.alist(_config(thread_id, CHILD_NS))]
        every = [
            t async for t in saver.alist({"configurable": {"thread_id": thread_id}})
        ]

    assert _namespaces(root) == [""]
    assert _namespaces(child) == [CHILD_NS]
    assert _namespaces(every) == ["", CHILD_NS]
