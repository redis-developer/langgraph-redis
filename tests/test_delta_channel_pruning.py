import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import pytest
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.graph import START, StateGraph

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def _append_writes(state: list[int], writes: list[list[int]]) -> list[int]:
    return state + [value for write in writes for value in write]


def _compile_graph(checkpointer: Any, snapshot_frequency: int) -> Any:
    class State(TypedDict, total=False):
        value: int
        delta: Annotated[
            list[int],
            DeltaChannel(_append_writes, snapshot_frequency=snapshot_frequency),
        ]
        accumulated: Annotated[list[int], operator.add]

    builder = StateGraph(State)
    builder.add_node(
        "append",
        lambda state: {
            "delta": [state["value"]],
            "accumulated": [state["value"]],
        },
    )
    builder.add_edge(START, "append")
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _assert_state(state: dict[str, Any], values: list[int], value: int) -> None:
    assert state == {
        "value": value,
        "delta": values,
        "accumulated": list(range(1, value + 1)),
    }


@pytest.mark.parametrize(
    ("snapshot_frequency", "expected_retention"),
    [
        pytest.param(100, "all", id="before-first-snapshot"),
        pytest.param(2, "prefix-pruned", id="ancestor-snapshot"),
        pytest.param(3, "latest-only", id="latest-is-snapshot"),
    ],
)
def test_prune_preserves_delta_channel_history(
    redis_url: str, snapshot_frequency: int, expected_retention: str
) -> None:
    thread_id = f"delta-prune-{uuid4()}"
    config = _config(thread_id)

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _compile_graph(saver, snapshot_frequency)
        for value in (1, 2, 3):
            graph.invoke({"value": value}, config, durability="sync")

        _assert_state(graph.get_state(config).values, [1, 2, 3], 3)
        before = list(saver.list(config))
        latest = saver.get_tuple(config)
        assert latest is not None
        if expected_retention == "latest-only":
            assert isinstance(
                latest.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
            )

        saver.prune([thread_id], strategy="keep_latest")

        _assert_state(graph.get_state(config).values, [1, 2, 3], 3)
        after = list(saver.list(config))
        if expected_retention == "all":
            assert len(after) == len(before)
        elif expected_retention == "prefix-pruned":
            assert 1 < len(after) < len(before)
        else:
            assert len(after) == 1

        graph.invoke({"value": 4}, config, durability="sync")
        _assert_state(graph.get_state(config).values, [1, 2, 3, 4], 4)


@pytest.mark.parametrize(
    ("snapshot_frequency", "expected_retention"),
    [
        pytest.param(100, "all", id="before-first-snapshot"),
        pytest.param(2, "prefix-pruned", id="ancestor-snapshot"),
        pytest.param(3, "latest-only", id="latest-is-snapshot"),
    ],
)
async def test_aprune_preserves_delta_channel_history(
    redis_url: str, snapshot_frequency: int, expected_retention: str
) -> None:
    thread_id = f"delta-aprune-{uuid4()}"
    config = _config(thread_id)

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        graph = _compile_graph(saver, snapshot_frequency)
        for value in (1, 2, 3):
            await graph.ainvoke({"value": value}, config, durability="sync")

        _assert_state((await graph.aget_state(config)).values, [1, 2, 3], 3)
        before = [checkpoint async for checkpoint in saver.alist(config)]
        latest = await saver.aget_tuple(config)
        assert latest is not None
        if expected_retention == "latest-only":
            assert isinstance(
                latest.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
            )

        await saver.aprune([thread_id], strategy="keep_latest")

        _assert_state((await graph.aget_state(config)).values, [1, 2, 3], 3)
        after = [checkpoint async for checkpoint in saver.alist(config)]
        if expected_retention == "all":
            assert len(after) == len(before)
        elif expected_retention == "prefix-pruned":
            assert 1 < len(after) < len(before)
        else:
            assert len(after) == 1

        await graph.ainvoke({"value": 4}, config, durability="sync")
        _assert_state((await graph.aget_state(config)).values, [1, 2, 3, 4], 4)
