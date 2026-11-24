import operator
from collections.abc import AsyncIterator
from functools import partial
from typing import Annotated, Any, TypedDict, Union
from unittest.mock import DEFAULT, Mock, call, patch
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import set_config_context
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from redis.exceptions import ResponseError

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver, _PruneCheckpoint

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
# A declared DeltaChannel that is never written must not change what is retained.
UNUSED_CHANNEL_CASES = (
    pytest.param(False, id="written-channels-only"),
    pytest.param(True, id="with-unwritten-channel"),
)


class _Checkpointer:
    """Drive RedisSaver and AsyncRedisSaver through one async test body."""

    def __init__(self, saver: Union[RedisSaver, AsyncRedisSaver]) -> None:
        self.saver = saver
        self.is_async = isinstance(saver, AsyncRedisSaver)
        self.get_tuple_name = "aget_tuple" if self.is_async else "get_tuple"
        self.read_name = (
            "_aread_prune_checkpoint" if self.is_async else "_read_prune_checkpoint"
        )

    async def invoke(
        self, graph: Any, value: Any, config: Any, *, durability: str = "sync"
    ) -> Any:
        graph_input = value if isinstance(value, Command) else {"value": value}
        if self.is_async:
            return await graph.ainvoke(graph_input, config, durability=durability)
        return graph.invoke(graph_input, config, durability=durability)

    async def update_state(self, graph: Any, config: Any, values: Any) -> Any:
        if self.is_async:
            return await graph.aupdate_state(config, values)
        return graph.update_state(config, values)

    async def state(self, graph: Any, config: Any) -> Any:
        if self.is_async:
            return await graph.aget_state(config)
        return graph.get_state(config)

    async def checkpoints(self, config: Any) -> list[Any]:
        if self.is_async:
            return [checkpoint async for checkpoint in self.saver.alist(config)]
        return list(self.saver.list(config))

    async def get_tuple(self, config: Any) -> Any:
        if self.is_async:
            return await self.saver.aget_tuple(config)
        return self.saver.get_tuple(config)

    async def prune(self, thread_ids: list[str], **kwargs: Any) -> None:
        if self.is_async:
            await self.saver.aprune(thread_ids, **kwargs)
        else:
            self.saver.prune(thread_ids, **kwargs)


@pytest.fixture(params=["sync", "async"])
async def checkpointer(
    request: pytest.FixtureRequest, redis_url: str
) -> AsyncIterator[_Checkpointer]:
    mode, _, decoded = request.param.partition("-")
    connection_args = {"decode_responses": True} if decoded else None
    if mode == "sync":
        with RedisSaver.from_conn_string(
            redis_url, connection_args=connection_args
        ) as saver:
            saver.setup()
            yield _Checkpointer(saver)
    else:
        async with AsyncRedisSaver.from_conn_string(
            redis_url, connection_args=connection_args
        ) as async_saver:
            await async_saver.asetup()
            yield _Checkpointer(async_saver)


def _append_writes(state: list[int], writes: list[list[int]]) -> list[int]:
    return state + [value for write in writes for value in write]


def _compile_graph(
    checkpointer: Any, snapshot_frequency: int, *, unused_channel: bool = False
) -> Any:
    class State(TypedDict, total=False):
        value: int
        delta: Annotated[
            list[int],
            DeltaChannel(_append_writes, snapshot_frequency=snapshot_frequency),
        ]
        accumulated: Annotated[list[int], operator.add]

    class StateWithUnusedChannel(State, total=False):
        # Declared but never written: LangGraph still counts it in
        # counters_since_delta_snapshot, but never versions, snapshots or replays it.
        unused: Annotated[
            list[int],
            DeltaChannel(_append_writes, snapshot_frequency=snapshot_frequency),
        ]

    builder = StateGraph(StateWithUnusedChannel if unused_channel else State)
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


def _compile_late_channel_graph(checkpointer: Any) -> Any:
    class State(TypedDict, total=False):
        value: int
        delta: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=2)]
        late: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=2)]

    def append(state: dict[str, Any]) -> dict[str, list[int]]:
        update = {"delta": [state["value"]]}
        # `late` gets its first version on the third invocation.
        if state["value"] >= 3:
            update["late"] = [state["value"]]
        return update

    builder = StateGraph(State)
    builder.add_node("append", append)
    builder.add_edge(START, "append")
    return builder.compile(checkpointer=checkpointer)


def _compile_interrupt_graph(checkpointer: Any) -> Any:
    class State(TypedDict, total=False):
        value: int
        delta: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=2)]

    def ask(_state: State, config: RunnableConfig) -> dict[str, list[int]]:
        # LangGraph does not set the runnable context on async Python 3.10 calls.
        with set_config_context(config) as context:
            return {"delta": [context.run(interrupt, "next value")]}

    builder = StateGraph(State)
    builder.add_node("append", lambda state: {"delta": [state["value"]]})
    builder.add_node("ask", ask)
    builder.add_edge(START, "append")
    builder.add_edge("append", "ask")
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


def _assert_state(
    state: dict[str, Any],
    values: list[int],
    value: int,
    *,
    unused_channel: bool = False,
) -> None:
    expected = {
        "value": value,
        "delta": values,
        "accumulated": list(range(1, value + 1)),
    }
    if unused_channel:
        expected["unused"] = []
    assert state == expected


def _assert_retention(
    before: list[Any],
    after: list[Any],
    oldest_step: int,
) -> None:
    expected = [cp for cp in before if cp.metadata["step"] >= oldest_step]
    assert [cp.config for cp in after] == [cp.config for cp in expected]


def _nearest_seed_step(checkpoints: list[Any], *channels: str) -> int:
    """Oldest step a linear thread needs: each channel's newest stored value.

    `checkpoints` is newest-first. A channel with no stored value anywhere is
    rebuilt from the start of the thread, so it needs every checkpoint (-1).
    """
    steps = []
    for channel in channels:
        stored = [
            cp.metadata["step"]
            for cp in checkpoints
            if channel in cp.checkpoint["channel_values"]
        ]
        steps.append(stored[0] if stored else -1)
    return min(steps)


def _expected_fork_retention(
    checkpoints: list[Any],
    fork: Any,
    branches: tuple[Any, Any],
    *,
    separate_snapshots: bool,
) -> list[Any]:
    # Each invocation adds three checkpoints. A second invocation on branch A
    # creates its own snapshot; branch B still needs the snapshot at the fork.
    # History is newest-first: B occupies indices 0..2; A occupies 3..5 for one
    # invocation or 3..8 for two. Their input checkpoints at 2 and 5/8 point to fork.
    branch_a_input = 8 if separate_snapshots else 5
    assert (
        checkpoints[2].parent_config
        == checkpoints[branch_a_input].parent_config
        == fork
    )
    assert "delta" not in checkpoints[0].checkpoint["channel_values"]
    # keep_last=4 selects B's three checkpoints and A's head. The walk must
    # additionally protect any ancestors needed to reconstruct those states.
    retained = [cp.config for cp in checkpoints[:4]]
    for branch in branches:
        assert branch in retained
    # 4 = B's three checkpoints + A's snapshot head; 6 = three checkpoints per
    # branch when neither head is a snapshot. Append the fork seed separately.
    branch_checkpoint_count = 4 if separate_snapshots else 6
    return [cp.config for cp in checkpoints[:branch_checkpoint_count]] + [fork]


def _assert_branch_state(state: dict[str, Any], values: list[int]) -> None:
    assert state == {"value": values[-1], "delta": values, "accumulated": values}


def _cycle_parent_update(
    saver: Any, checkpoints: list[Any], cycle_length: int
) -> tuple[str, str]:
    head = checkpoints[0].config["configurable"]
    target = checkpoints[cycle_length - 1].config["configurable"]
    return saver._make_redis_checkpoint_key(**target), head["checkpoint_id"]


@pytest.mark.parametrize("unused_channel", UNUSED_CHANNEL_CASES)
@pytest.mark.parametrize(
    ("snapshot_frequency", "keep_last", "oldest_step"),
    RETENTION_CASES,
)
async def test_prune_preserves_delta_channel_history(
    checkpointer: _Checkpointer,
    snapshot_frequency: int,
    keep_last: int,
    oldest_step: int,
    unused_channel: bool,
) -> None:
    saver = checkpointer.saver
    thread_id = f"delta-prune-{uuid4()}"
    config = _config(thread_id)
    assert_state = partial(_assert_state, unused_channel=unused_channel)

    graph = _compile_graph(saver, snapshot_frequency, unused_channel=unused_channel)
    for value in (1, 2, 3):
        await checkpointer.invoke(graph, value, config)

    assert_state((await checkpointer.state(graph, config)).values, [1, 2, 3], 3)
    before = await checkpointer.checkpoints(config)
    retained = before[:keep_last]
    expected_states = [
        (await checkpointer.state(graph, cp.config)).values for cp in retained
    ]
    latest = await checkpointer.get_tuple(config)
    assert latest is not None
    if unused_channel:
        # Counted like every declared DeltaChannel, but never given a version.
        assert "unused" in latest.metadata["counters_since_delta_snapshot"]
        assert "unused" not in latest.checkpoint["channel_versions"]
    if snapshot_frequency == 3:
        assert isinstance(
            latest.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
        )
        if keep_last == 2:
            assert "delta" not in retained[1].checkpoint["channel_values"]

    with (
        patch.object(
            saver,
            checkpointer.get_tuple_name,
            side_effect=AssertionError("full read"),
        ),
        patch.object(
            saver,
            "_queue_prune_checkpoint_read",
            wraps=saver._queue_prune_checkpoint_read,
        ) as reads,
    ):
        await checkpointer.prune([thread_id], keep_last=keep_last)
        read_ids = [
            read.args[1]["configurable"]["checkpoint_id"]
            for read in reads.call_args_list
        ]
        assert len(read_ids) == len(set(read_ids))
        if keep_last > 1:
            # A second call must build a fresh cache, reading the surviving chain again.
            reads.reset_mock()
            await checkpointer.prune([thread_id], keep_last=keep_last)
            assert [
                read.args[1]["configurable"]["checkpoint_id"]
                for read in reads.call_args_list
            ] == read_ids
    assert [
        read.args[1] for read in reads.call_args_list if read.kwargs["include_metadata"]
    ] == [checkpoint.config for checkpoint in retained]

    assert_state((await checkpointer.state(graph, config)).values, [1, 2, 3], 3)
    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step)
    for checkpoint, expected_state in zip(retained, expected_states):
        assert await checkpointer.get_tuple(checkpoint.config) is not None
        assert (
            await checkpointer.state(graph, checkpoint.config)
        ).values == expected_state

    await checkpointer.invoke(graph, 4, config)
    assert_state((await checkpointer.state(graph, config)).values, [1, 2, 3, 4], 4)


async def test_prune_preserves_delta_channel_history_in_subgraph_namespace(
    checkpointer: _Checkpointer,
) -> None:
    thread_id = f"delta-subgraph-prune-{uuid4()}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    graph = _compile_subgraph(checkpointer.saver, snapshot_frequency=100)
    await checkpointer.invoke(graph, 0, thread_config)

    all_checkpoints = await checkpointer.checkpoints(thread_config)
    subgraph_config = _config(thread_id, _subgraph_namespace(all_checkpoints))
    before = await checkpointer.checkpoints(subgraph_config)
    assert len(before) > 1
    state = await checkpointer.state(graph, subgraph_config)
    _assert_state(state.values, [1, 2, 3], 3)

    await checkpointer.prune([thread_id], strategy="keep_latest")

    state = await checkpointer.state(graph, subgraph_config)
    _assert_state(state.values, [1, 2, 3], 3)
    after = await checkpointer.checkpoints(subgraph_config)
    _assert_retention(before, after, oldest_step=-1)


@pytest.mark.parametrize("keep_last", [1, 2])
async def test_prune_preserves_independent_delta_snapshots(
    checkpointer: _Checkpointer, keep_last: int
) -> None:
    thread_id = f"delta-multi-prune-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_multi_channel_graph(checkpointer.saver)
    for value in range(1, 6):
        await checkpointer.invoke(graph, value, config)

    expected = {"value": 5, "fast": [1, 2, 3, 4, 5], "slow": [1, 2, 3, 4, 5]}
    assert (await checkpointer.state(graph, config)).values == expected
    before = await checkpointer.checkpoints(config)
    retained_states = [
        (await checkpointer.state(graph, cp.config)).values for cp in before[:keep_last]
    ]
    # The nearest fast seed is at step 10; the slow seed needs history through 7.
    for channel, index in (("fast", 3), ("slow", 6)):
        seed = await checkpointer.get_tuple(before[index].config)
        assert seed is not None
        assert isinstance(seed.checkpoint["channel_values"][channel], _DeltaSnapshot)

    await checkpointer.prune([thread_id], keep_last=keep_last)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step=7)
    assert (await checkpointer.state(graph, config)).values == expected
    assert [
        (await checkpointer.state(graph, cp.config)).values for cp in before[:keep_last]
    ] == retained_states
    await checkpointer.invoke(graph, 6, config)
    assert (await checkpointer.state(graph, config)).values == {
        "value": 6,
        "fast": list(range(1, 7)),
        "slow": list(range(1, 7)),
    }


@pytest.mark.parametrize(
    "separate_snapshots", [False, True], ids=["shared-snapshot", "different-snapshots"]
)
async def test_prune_preserves_forked_delta_history(
    checkpointer: _Checkpointer, separate_snapshots: bool
) -> None:
    thread_id = f"delta-fork-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(checkpointer.saver, snapshot_frequency=2)
    for value in (1, 2):
        await checkpointer.invoke(graph, value, config)
    fork = (await checkpointer.state(graph, config)).config

    await checkpointer.invoke(graph, 3, fork)
    if separate_snapshots:
        await checkpointer.invoke(graph, 4, config)
    branch_a = (await checkpointer.state(graph, config)).config
    if separate_snapshots:
        seed = await checkpointer.get_tuple(branch_a)
        assert seed is not None
        assert isinstance(
            seed.checkpoint["channel_values"].get("delta"), _DeltaSnapshot
        )
    # 30 and 31 are arbitrary branch-B values, distinct from branch A's data.
    await checkpointer.invoke(graph, 30, fork)
    branch_b = (await checkpointer.state(graph, config)).config
    before = await checkpointer.checkpoints(config)
    expected = _expected_fork_retention(
        before, fork, (branch_a, branch_b), separate_snapshots=separate_snapshots
    )

    await checkpointer.prune([thread_id], keep_last=4)

    assert [cp.config for cp in await checkpointer.checkpoints(config)] == expected
    a_values = [1, 2, 3, 4] if separate_snapshots else [1, 2, 3]
    for branch, values, next_value in (
        (branch_a, a_values, a_values[-1] + 1),
        (branch_b, [1, 2, 30], 31),
    ):
        _assert_branch_state((await checkpointer.state(graph, branch)).values, values)
        result = await checkpointer.invoke(graph, next_value, branch)
        _assert_branch_state(result, values + [next_value])


@pytest.mark.parametrize(
    "checkpointer", ["sync-decoded", "async-decoded"], indirect=True
)
@pytest.mark.parametrize("unused_channel", UNUSED_CHANNEL_CASES)
async def test_prune_preserves_delta_channel_history_with_decoded_responses(
    checkpointer: _Checkpointer, unused_channel: bool
) -> None:
    # decode_responses=True returns str channel names instead of bytes.
    thread_id = f"delta-decoded-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(
        checkpointer.saver, snapshot_frequency=2, unused_channel=unused_channel
    )
    for value in (1, 2, 3):
        await checkpointer.invoke(graph, value, config)
    before = await checkpointer.checkpoints(config)

    await checkpointer.prune([thread_id], keep_last=1)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step=4)
    state = await checkpointer.state(graph, config)
    _assert_state(state.values, [1, 2, 3], 3, unused_channel=unused_channel)


@pytest.mark.parametrize("durability", ["async", "exit"])
async def test_prune_preserves_delta_channel_history_with_durability(
    checkpointer: _Checkpointer, durability: str
) -> None:
    thread_id = f"delta-durability-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(checkpointer.saver, snapshot_frequency=2)
    for value in range(1, 6):
        await checkpointer.invoke(graph, value, config, durability=durability)
    before = await checkpointer.checkpoints(config)
    oldest_step = _nearest_seed_step(before, "delta")
    assert oldest_step > before[-1].metadata["step"]

    await checkpointer.prune([thread_id], keep_last=1)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step)
    state = await checkpointer.state(graph, config)
    _assert_state(state.values, [1, 2, 3, 4, 5], 5)
    await checkpointer.invoke(graph, 6, config, durability=durability)
    state = await checkpointer.state(graph, config)
    _assert_state(state.values, list(range(1, 7)), 6)


@pytest.mark.parametrize(
    "invocations", [3, 5], ids=["late-not-snapshotted", "late-snapshotted"]
)
async def test_prune_preserves_channel_first_written_later(
    checkpointer: _Checkpointer, invocations: int
) -> None:
    thread_id = f"delta-late-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_late_channel_graph(checkpointer.saver)
    for value in range(1, invocations + 1):
        await checkpointer.invoke(graph, value, config)
    before = await checkpointer.checkpoints(config)
    oldest_step = _nearest_seed_step(before, "delta", "late")
    # Until `late` has a snapshot, rebuilding it needs the whole thread.
    assert (oldest_step == -1) == (invocations == 3)

    await checkpointer.prune([thread_id], keep_last=1)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step)
    expected = {
        "value": invocations,
        "delta": list(range(1, invocations + 1)),
        "late": list(range(3, invocations + 1)),
    }
    assert (await checkpointer.state(graph, config)).values == expected
    await checkpointer.invoke(graph, invocations + 1, config)
    state = await checkpointer.state(graph, config)
    assert state.values["late"] == list(range(3, invocations + 2))


async def test_prune_preserves_delta_channel_history_after_update_state(
    checkpointer: _Checkpointer,
) -> None:
    thread_id = f"delta-update-state-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(checkpointer.saver, snapshot_frequency=2)
    for value in (1, 2):
        await checkpointer.invoke(graph, value, config)
    await checkpointer.update_state(graph, config, {"delta": [99]})
    before = await checkpointer.checkpoints(config)
    # The update_state head stores no value, so it needs the snapshot at value 2.
    assert before[0].metadata["source"] == "update"
    assert "delta" not in before[0].checkpoint["channel_values"]
    oldest_step = _nearest_seed_step(before, "delta")
    assert oldest_step > before[-1].metadata["step"]

    await checkpointer.prune([thread_id], keep_last=1)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step)
    state = await checkpointer.state(graph, config)
    assert state.values["delta"] == [1, 2, 99]
    await checkpointer.invoke(graph, 3, config)
    state = await checkpointer.state(graph, config)
    assert state.values["delta"] == [1, 2, 99, 3]


async def test_prune_preserves_interrupted_delta_run(
    checkpointer: _Checkpointer,
) -> None:
    thread_id = f"delta-interrupt-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_interrupt_graph(checkpointer.saver)
    for value in (1, 2):
        await checkpointer.invoke(graph, value, config)
        await checkpointer.invoke(graph, Command(resume=value * 10), config)
    await checkpointer.invoke(graph, 3, config)  # Pauses at the interrupt.
    paused = await checkpointer.state(graph, config)
    assert paused.next == ("ask",)
    before = await checkpointer.checkpoints(config)
    oldest_step = _nearest_seed_step(before, "delta")
    assert oldest_step > before[-1].metadata["step"]

    await checkpointer.prune([thread_id], keep_last=1)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step)
    state = await checkpointer.state(graph, config)
    assert (state.values, state.next) == (paused.values, paused.next)
    result = await checkpointer.invoke(graph, Command(resume=30), config)
    assert result["delta"] == [1, 10, 2, 20, 3, 30]


@pytest.mark.parametrize(
    "cycle_length", [1, 2], ids=["self-parent", "two-checkpoint-cycle"]
)
async def test_prune_rejects_cyclic_ancestry(
    checkpointer: _Checkpointer, client: Any, cycle_length: int
) -> None:
    saver = checkpointer.saver
    thread_id = f"delta-cycle-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(saver, snapshot_frequency=100)
    for value in (1, 2, 3):
        await checkpointer.invoke(graph, value, config)
    checkpoints = await checkpointer.checkpoints(config)
    key, parent_id = _cycle_parent_update(saver, checkpoints, cycle_length)
    client.json().set(key, "$.parent_checkpoint_id", parent_id)
    before_keys = set(client.scan_iter(match=f"*{thread_id}*"))

    # Bound advancement so a broken guard fails instead of hanging the test.
    with (
        patch.object(
            saver,
            "_advance_delta_prune_dependencies",
            wraps=saver._advance_delta_prune_dependencies,
            side_effect=[DEFAULT] * cycle_length,
        ),
        pytest.raises(ValueError, match="Cyclic checkpoint ancestry"),
    ):
        await checkpointer.prune([thread_id], keep_last=1)

    assert set(client.scan_iter(match=f"*{thread_id}*")) == before_keys


async def test_prune_handles_checkpoint_expiring_during_walk(
    checkpointer: _Checkpointer, client: Any
) -> None:
    saver = checkpointer.saver
    thread_id = f"delta-prune-expiry-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(saver, snapshot_frequency=2)
    for value in (1, 2, 3):
        await checkpointer.invoke(graph, value, config)
    latest, ancestor, *_ = await checkpointer.checkpoints(config)
    ancestor_key = saver._make_redis_checkpoint_key(
        thread_id, "", ancestor.config["configurable"]["checkpoint_id"]
    )

    def expire_ancestor(read_config: Any, *, include_metadata: bool) -> Any:
        if read_config == ancestor.config:
            # Remove the ancestor after the search found it, just before its read.
            assert client.delete(ancestor_key) == 1
        return DEFAULT

    with patch.object(
        saver,
        checkpointer.read_name,
        wraps=getattr(saver, checkpointer.read_name),
        side_effect=expire_ancestor,
    ) as reads:
        await checkpointer.prune([thread_id], keep_last=1)

    assert [read.args[0] for read in reads.call_args_list] == [
        latest.config,
        ancestor.config,
    ]
    assert await checkpointer.get_tuple(latest.config) is not None


@pytest.mark.parametrize("checkpointer", ["sync"], indirect=True)
@pytest.mark.parametrize("include_metadata", [False, True])
def test_prune_reads_only_dependency_fields(
    checkpointer: _Checkpointer, include_metadata: bool
) -> None:
    config = _config("field-selection")
    config["configurable"]["checkpoint_id"] = "checkpoint-id"
    key = "checkpoint:field-selection:__empty__:checkpoint-id"
    pipeline = Mock()

    checkpointer.saver._queue_prune_checkpoint_read(
        pipeline, config, include_metadata=include_metadata
    )

    expected = [call.json(), call.json().objkeys(key, "$.checkpoint.channel_values")]
    if include_metadata:
        expected.extend(
            [call.json(), call.json().objkeys(key, "$.checkpoint.channel_versions")]
        )
    expected.extend([call.json(), call.json().get(key, "$.parent_checkpoint_id")])
    if include_metadata:
        expected.extend([call.json(), call.json().get(key, "$.metadata")])
    assert pipeline.mock_calls == expected


async def test_prune_does_not_refresh_ttls(
    checkpointer: _Checkpointer, client: Any
) -> None:
    checkpointer.saver.ttl_config = {"default_ttl": 10, "refresh_on_read": True}
    thread_id = f"delta-prune-ttl-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_graph(checkpointer.saver, snapshot_frequency=100)
    for value in (1, 2, 3):
        await checkpointer.invoke(graph, value, config)

    # Include checkpoints, writes, registries, and the latest pointer. Shorten
    # their TTLs so an accidental read refresh to the 10-minute default is visible.
    keys = list(client.scan_iter(match=f"*{thread_id}*"))
    assert keys
    for key in keys:
        client.expire(key, 60)
    before = {key: client.pttl(key) for key in keys}

    await checkpointer.prune([thread_id], keep_last=1)

    for key, ttl in before.items():
        assert 0 < client.pttl(key) <= ttl
    _assert_state((await checkpointer.state(graph, config)).values, [1, 2, 3], 3)


_COUNTERS = '{"counters_since_delta_snapshot": {"delta": [1, 3], "unused": [0, 3]}}'


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([None, None], None),  # Expired checkpoint.
        ([ResponseError("missing key"), None], None),  # OBJKEYS on an expired key.
        ([[], [""]], None),  # Missing channel-values path.
        ([[None], [""]], None),  # Non-object channel values.
        ([[[]], None], None),  # Expired between pipelined commands.
        # Starting checkpoints: channel values, channel versions, parent, metadata.
        ([ResponseError("missing"), ResponseError("missing"), None, None], None),
        ([[[]], ResponseError("missing"), None, None], None),  # Expired mid-read.
        ([[[]], [["delta"]], [""], None], None),
        (
            [[[]], [["delta"]], [""], [_COUNTERS]],
            _PruneCheckpoint(
                channel_names=set(), parent_checkpoint_id="", delta_channels={"delta"}
            ),
        ),
        (  # Counted but never written: no version, so nothing to replay.
            [[[]], [[]], [""], [_COUNTERS]],
            _PruneCheckpoint(
                channel_names=set(), parent_checkpoint_id="", delta_channels=set()
            ),
        ),
        ([ResponseError("OBJKEYS failed"), [""]], ResponseError),
        ([[[]], ResponseError("parent GET failed")], ResponseError),
        ([[[]], ResponseError("OBJKEYS failed"), [""], [_COUNTERS]], ResponseError),
        (
            [[[]], [["delta"]], [""], ResponseError("metadata GET failed")],
            ResponseError,
        ),
    ],
)
def test_prune_reader_responses(results: list[Any], expected: Any) -> None:
    if expected is ResponseError:
        with pytest.raises(ResponseError) as raised:
            BaseRedisSaver._parse_prune_checkpoint(results)
        assert any(raised.value is reply for reply in results)
    else:
        assert BaseRedisSaver._parse_prune_checkpoint(results) == expected
