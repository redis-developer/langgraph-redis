import operator
from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import pytest
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.graph import START, StateGraph

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# Rows: (snapshot frequency, keep_last, oldest checkpoint step to retain).
# Three invocations produce steps -1 through 7: -1 is the initial checkpoint;
# snapshots after the second and third invocations are at steps 4 and 7.
# Retaining from -1 keeps the whole chain when no snapshot can seed a retained state.
RETENTION_CASES = (
    pytest.param(100, 1, -1, id="before-first-snapshot"),
    pytest.param(2, 1, 4, id="ancestor-snapshot"),
    pytest.param(3, 1, 7, id="latest-is-snapshot"),
    pytest.param(2, 2, 4, id="overlapping-chains"),
    pytest.param(3, 2, -1, id="older-retained-checkpoint-needs-ancestors"),
)


@pytest.fixture
def saver(redis_url: str) -> Iterator[RedisSaver]:
    with RedisSaver.from_conn_string(redis_url) as checkpoint_saver:
        checkpoint_saver.setup()
        yield checkpoint_saver


@pytest.fixture
async def async_saver(redis_url: str) -> AsyncIterator[AsyncRedisSaver]:
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpoint_saver:
        await checkpoint_saver.asetup()
        yield checkpoint_saver


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


def _compile_multi_channel_graph(checkpointer: Any) -> Any:
    class State(TypedDict):
        value: int
        fast: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=2)]
        slow: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=3)]

    builder = StateGraph(State)
    builder.add_node(
        "append", lambda state: {"fast": [state["value"]], "slow": [state["value"]]}
    )
    builder.add_edge(START, "append")
    return builder.compile(checkpointer=checkpointer)


def _config(thread_id: str, checkpoint_ns: str = "") -> dict[str, dict[str, str]]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }


def _compile_subgraph(checkpointer: Any, snapshot_frequency: int) -> Any:
    class ParentState(TypedDict):
        value: int

    class SubgraphState(TypedDict, total=False):
        value: int
        delta: Annotated[
            list[int],
            DeltaChannel(_append_writes, snapshot_frequency=snapshot_frequency),
        ]
        accumulated: Annotated[list[int], operator.add]

    subgraph_builder = StateGraph(SubgraphState)
    for value in (1, 2, 3):
        node_name = f"append_{value}"
        subgraph_builder.add_node(
            node_name,
            lambda state, value=value: {
                "value": value,
                "delta": [value],
                "accumulated": [value],
            },
        )
        if value == 1:
            subgraph_builder.add_edge(START, node_name)
        else:
            subgraph_builder.add_edge(f"append_{value - 1}", node_name)

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("subgraph", subgraph_builder.compile())
    parent_builder.add_edge(START, "subgraph")
    return parent_builder.compile(checkpointer=checkpointer)


def _subgraph_namespace(checkpoints: list[Any]) -> str:
    namespaces = {
        checkpoint.config["configurable"]["checkpoint_ns"]
        for checkpoint in checkpoints
        if checkpoint.config["configurable"]["checkpoint_ns"]
    }
    assert len(namespaces) == 1
    return namespaces.pop()


def _assert_state(state: dict[str, Any], values: list[int], value: int) -> None:
    assert state == {
        "value": value,
        "delta": values,
        "accumulated": list(range(1, value + 1)),
    }


def _assert_retention(
    before: list[Any],
    after: list[Any],
    oldest_step: int,
) -> None:
    expected = [cp for cp in before if cp.metadata["step"] >= oldest_step]
    assert [cp.config for cp in after] == [cp.config for cp in expected]


@pytest.mark.parametrize(
    ("snapshot_frequency", "keep_last", "oldest_step"),
    RETENTION_CASES,
)
def test_prune_preserves_delta_channel_history(
    saver: RedisSaver,
    snapshot_frequency: int,
    keep_last: int,
    oldest_step: int,
) -> None:
    thread_id = f"delta-prune-{uuid4()}"
    config = _config(thread_id)

    graph = _compile_graph(saver, snapshot_frequency)
    for value in (1, 2, 3):
        graph.invoke({"value": value}, config, durability="sync")

    _assert_state(graph.get_state(config).values, [1, 2, 3], 3)
    before = list(saver.list(config))
    retained = before[:keep_last]
    expected_states = [graph.get_state(cp.config).values for cp in retained]
    latest = saver.get_tuple(config)
    assert latest is not None
    if snapshot_frequency == 3:
        assert isinstance(
            latest.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
        )
        if keep_last == 2:
            assert "delta" not in retained[1].checkpoint["channel_values"]

    saver.prune([thread_id], keep_last=keep_last)

    _assert_state(graph.get_state(config).values, [1, 2, 3], 3)
    _assert_retention(before, list(saver.list(config)), oldest_step)
    for checkpoint, expected_state in zip(retained, expected_states):
        assert saver.get_tuple(checkpoint.config) is not None
        assert graph.get_state(checkpoint.config).values == expected_state

    graph.invoke({"value": 4}, config, durability="sync")
    _assert_state(graph.get_state(config).values, [1, 2, 3, 4], 4)


@pytest.mark.parametrize(
    ("snapshot_frequency", "keep_last", "oldest_step"),
    RETENTION_CASES,
)
async def test_aprune_preserves_delta_channel_history(
    async_saver: AsyncRedisSaver,
    snapshot_frequency: int,
    keep_last: int,
    oldest_step: int,
) -> None:
    thread_id = f"delta-aprune-{uuid4()}"
    config = _config(thread_id)

    graph = _compile_graph(async_saver, snapshot_frequency)
    for value in (1, 2, 3):
        await graph.ainvoke({"value": value}, config, durability="sync")

    _assert_state((await graph.aget_state(config)).values, [1, 2, 3], 3)
    before = [checkpoint async for checkpoint in async_saver.alist(config)]
    retained = before[:keep_last]
    expected_states = [(await graph.aget_state(cp.config)).values for cp in retained]
    latest = await async_saver.aget_tuple(config)
    assert latest is not None
    if snapshot_frequency == 3:
        assert isinstance(
            latest.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
        )
        if keep_last == 2:
            assert "delta" not in retained[1].checkpoint["channel_values"]

    await async_saver.aprune([thread_id], keep_last=keep_last)

    _assert_state((await graph.aget_state(config)).values, [1, 2, 3], 3)
    after = [checkpoint async for checkpoint in async_saver.alist(config)]
    _assert_retention(before, after, oldest_step)
    for checkpoint, expected_state in zip(retained, expected_states):
        assert await async_saver.aget_tuple(checkpoint.config) is not None
        assert (await graph.aget_state(checkpoint.config)).values == expected_state

    await graph.ainvoke({"value": 4}, config, durability="sync")
    _assert_state((await graph.aget_state(config)).values, [1, 2, 3, 4], 4)


def test_prune_preserves_delta_channel_history_in_subgraph_namespace(
    saver: RedisSaver,
) -> None:
    thread_id = f"delta-subgraph-prune-{uuid4()}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    graph = _compile_subgraph(saver, snapshot_frequency=100)
    graph.invoke({"value": 0}, thread_config, durability="sync")

    all_checkpoints = list(saver.list(thread_config))
    subgraph_config = _config(thread_id, _subgraph_namespace(all_checkpoints))
    before = list(saver.list(subgraph_config))
    assert len(before) > 1
    _assert_state(graph.get_state(subgraph_config).values, [1, 2, 3], 3)

    saver.prune([thread_id], strategy="keep_latest")

    _assert_state(graph.get_state(subgraph_config).values, [1, 2, 3], 3)
    _assert_retention(before, list(saver.list(subgraph_config)), oldest_step=-1)


async def test_aprune_preserves_delta_channel_history_in_subgraph_namespace(
    async_saver: AsyncRedisSaver,
) -> None:
    thread_id = f"delta-subgraph-aprune-{uuid4()}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    graph = _compile_subgraph(async_saver, snapshot_frequency=100)
    await graph.ainvoke({"value": 0}, thread_config, durability="sync")

    all_checkpoints = [
        checkpoint async for checkpoint in async_saver.alist(thread_config)
    ]
    subgraph_config = _config(thread_id, _subgraph_namespace(all_checkpoints))
    before = [checkpoint async for checkpoint in async_saver.alist(subgraph_config)]
    assert len(before) > 1
    _assert_state(
        (await graph.aget_state(subgraph_config)).values,
        [1, 2, 3],
        3,
    )

    await async_saver.aprune([thread_id], strategy="keep_latest")

    _assert_state(
        (await graph.aget_state(subgraph_config)).values,
        [1, 2, 3],
        3,
    )
    after = [checkpoint async for checkpoint in async_saver.alist(subgraph_config)]
    _assert_retention(before, after, oldest_step=-1)


def test_prune_preserves_independent_delta_snapshots(saver: RedisSaver) -> None:
    thread_id = f"delta-multi-prune-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_multi_channel_graph(saver)
    for value in range(1, 6):
        graph.invoke({"value": value}, config, durability="sync")

    expected = {"value": 5, "fast": [1, 2, 3, 4, 5], "slow": [1, 2, 3, 4, 5]}
    assert graph.get_state(config).values == expected
    before = list(saver.list(config))
    # The nearest fast seed is at step 10; the slow seed needs history through 7.
    for channel, index in (("fast", 3), ("slow", 6)):
        seed = saver.get_tuple(before[index].config)
        assert seed is not None
        assert isinstance(seed.checkpoint["channel_values"][channel], _DeltaSnapshot)

    saver.prune([thread_id], keep_last=1)

    _assert_retention(before, list(saver.list(config)), oldest_step=7)
    assert graph.get_state(config).values == expected
    graph.invoke({"value": 6}, config, durability="sync")
    assert graph.get_state(config).values == {
        "value": 6,
        "fast": [1, 2, 3, 4, 5, 6],
        "slow": [1, 2, 3, 4, 5, 6],
    }


async def test_aprune_preserves_independent_delta_snapshots(
    async_saver: AsyncRedisSaver,
) -> None:
    thread_id = f"delta-multi-aprune-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_multi_channel_graph(async_saver)
    for value in range(1, 6):
        await graph.ainvoke({"value": value}, config, durability="sync")

    expected = {"value": 5, "fast": [1, 2, 3, 4, 5], "slow": [1, 2, 3, 4, 5]}
    assert (await graph.aget_state(config)).values == expected
    before = [cp async for cp in async_saver.alist(config)]
    for channel, index in (("fast", 3), ("slow", 6)):
        seed = await async_saver.aget_tuple(before[index].config)
        assert seed is not None
        assert isinstance(seed.checkpoint["channel_values"][channel], _DeltaSnapshot)

    await async_saver.aprune([thread_id], keep_last=1)

    after = [cp async for cp in async_saver.alist(config)]
    _assert_retention(before, after, oldest_step=7)
    assert (await graph.aget_state(config)).values == expected
    await graph.ainvoke({"value": 6}, config, durability="sync")
    assert (await graph.aget_state(config)).values == {
        "value": 6,
        "fast": [1, 2, 3, 4, 5, 6],
        "slow": [1, 2, 3, 4, 5, 6],
    }
