"""AsyncShallowRedisSaver.alist must return what ShallowRedisSaver.list returns.

alist asked the index only for a few top-level fields and built every tuple from
them: an empty ``channel_values``, ``v: 1``, metadata made of ``source``/``step``
with ``writes: {}`` and a ``score``, and no pending writes. ``list`` returns the
stored checkpoint, metadata and pending writes, as ``aget_tuple`` does. LangGraph
builds ``aget_state_history`` from ``alist``, so the async history showed
``values == {}`` and ``next == ()`` at a pause where the sync history shows the
real state.

alist also skipped a filter key it does not support, where ``list`` raises
ValueError.
"""

import uuid
from typing import Any, Dict, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

WRITES = [("channel1", "value1"), ("channel2", {"nested": [1, 2]})]
CHANNEL_VALUES = {"count": 3, "messages": [HumanMessage("hi", id="m-1")]}


class State(TypedDict):
    count: int
    log: list


def _a(state: State) -> Dict[str, Any]:
    return {"count": state["count"] + 1, "log": state["log"] + ["a"]}


def _b(state: State) -> Dict[str, Any]:
    return {"count": state["count"] + 10, "log": state["log"] + ["b"]}


def _graph(saver):
    builder = StateGraph(State)
    builder.add_node("a", _a)
    builder.add_node("b", _b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=saver, interrupt_before=["b"])


def _thread_config(prefix: str, checkpoint_ns: str = "") -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": f"{prefix}-{uuid.uuid4()}",
            "checkpoint_ns": checkpoint_ns,
        }
    }


def _checkpoint() -> Checkpoint:
    checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
    checkpoint["channel_values"] = dict(CHANNEL_VALUES)
    checkpoint["channel_versions"] = {"count": "1", "messages": "1"}
    return checkpoint


METADATA: CheckpointMetadata = {
    "source": "loop",
    "step": 4,
    "parents": {},
    "custom": "kept",
}


def _same_tuple(listed, fetched) -> None:
    assert listed.config == fetched.config
    assert listed.checkpoint == fetched.checkpoint
    assert listed.metadata == fetched.metadata
    assert listed.parent_config == fetched.parent_config
    assert listed.pending_writes == fetched.pending_writes


async def test_alist_returns_the_stored_checkpoint(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saved = await saver.aput(_thread_config("alist"), _checkpoint(), METADATA, {})

        listed = [t async for t in saver.alist(saved)]
        fetched = await saver.aget_tuple(saved)

        assert len(listed) == 1
        assert fetched is not None
        assert listed[0].checkpoint["channel_values"] == CHANNEL_VALUES
        assert listed[0].metadata["custom"] == "kept"
        _same_tuple(listed[0], fetched)


async def test_alist_returns_pending_writes(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saved = await saver.aput(_thread_config("alist"), _checkpoint(), METADATA, {})
        await saver.aput_writes(saved, WRITES, "task-1")

        listed = [t async for t in saver.alist(saved)]
        fetched = await saver.aget_tuple(saved)

        assert len(listed) == 1
        assert fetched is not None
        assert [(w[1], w[2]) for w in listed[0].pending_writes] == WRITES
        _same_tuple(listed[0], fetched)


async def test_alist_each_namespace_matches_aget_tuple(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        root = _thread_config("alist")
        thread_id = root["configurable"]["thread_id"]
        inner: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": "inner"}
        }
        await saver.aput(root, _checkpoint(), METADATA, {})
        saved_inner = await saver.aput(inner, _checkpoint(), METADATA, {})
        await saver.aput_writes(saved_inner, WRITES, "task-1")

        # No checkpoint_ns in the config: every namespace on the thread
        thread: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        listed = [t async for t in saver.alist(thread)]

        assert sorted(t.config["configurable"]["checkpoint_ns"] for t in listed) == [
            "",
            "inner",
        ]
        for tup in listed:
            fetched = await saver.aget_tuple(tup.config)
            assert fetched is not None
            _same_tuple(tup, fetched)


async def test_aget_state_history_at_a_pause(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        graph = _graph(saver)
        config = _thread_config("history")
        await graph.ainvoke({"count": 0, "log": []}, config)

        state = await graph.aget_state(config)
        history = [s async for s in graph.aget_state_history(config)]

        assert state.values == {"count": 1, "log": ["a"]}
        assert state.next == ("b",)
        assert len(history) == 1
        assert history[0].values == state.values
        assert history[0].next == state.next


async def test_alist_unsupported_filter_key_raises(redis_url: str) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saved = await saver.aput(_thread_config("alist"), _checkpoint(), METADATA, {})

        with pytest.raises(ValueError, match="Unsupported filter key: custom"):
            [t async for t in saver.alist(saved, filter={"custom": "kept"})]


# Controls: unchanged behaviour.


def test_list_returns_the_stored_checkpoint(redis_url: str) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saved = saver.put(_thread_config("list"), _checkpoint(), METADATA, {})
        saver.put_writes(saved, WRITES, "task-1")

        listed = list(saver.list(saved))

        assert len(listed) == 1
        assert listed[0].checkpoint["channel_values"] == CHANNEL_VALUES
        assert listed[0].metadata["custom"] == "kept"
        assert [(w[1], w[2]) for w in listed[0].pending_writes] == WRITES


def test_get_state_history_at_a_pause(redis_url: str) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _graph(saver)
        config = _thread_config("history")
        graph.invoke({"count": 0, "log": []}, config)

        history = list(graph.get_state_history(config))

        assert len(history) == 1
        assert history[0].values == {"count": 1, "log": ["a"]}
        assert history[0].next == ("b",)


def test_list_unsupported_filter_key_raises(redis_url: str) -> None:
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saved = saver.put(_thread_config("list"), _checkpoint(), METADATA, {})

        with pytest.raises(ValueError, match="Unsupported filter key: custom"):
            list(saver.list(saved, filter={"custom": "kept"}))


@pytest.mark.parametrize(
    "filter_, expected",
    [
        ({"source": "loop"}, 1),
        ({"source": "input"}, 0),
        ({"step": 4}, 1),
        ({"step": 5}, 0),
        ({"run_id": "run-1"}, 1),
        ({"run_id": "run-2"}, 0),
    ],
)
async def test_alist_supported_filter_keys(
    redis_url: str, filter_: Dict[str, Any], expected: int
) -> None:
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config = _thread_config("alist")
        config["configurable"]["run_id"] = "run-1"
        await saver.aput(config, _checkpoint(), METADATA, {})
        await saver.aput(
            _thread_config("other"), _checkpoint(), {"source": "input", "step": 9}, {}
        )

        thread = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
        listed = [t async for t in saver.alist(thread, filter=filter_)]
        by_thread = [
            t
            async for t in saver.alist(
                None, filter={"thread_id": config["configurable"]["thread_id"]}
            )
        ]

        assert len(listed) == expected
        assert len(by_thread) == 1
