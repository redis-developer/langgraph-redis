import operator
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
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
# Step 0 holds the channel's first write. Retaining from 0 keeps every checkpoint
# the channel's history needs when no snapshot can seed a retained state; step -1
# predates the channel's first version and holds nothing it needs.
RETENTION_CASES = (
    pytest.param(100, 1, 0, id="before-first-snapshot"),
    pytest.param(2, 1, 4, id="ancestor-snapshot"),
    pytest.param(3, 1, 7, id="latest-is-snapshot"),
    pytest.param(2, 2, 4, id="overlapping-chains"),
    pytest.param(3, 2, 0, id="older-retained-checkpoint-needs-ancestors"),
)
# A declared DeltaChannel that is never written must not change what is retained.
UNUSED_CHANNEL_CASES = (
    pytest.param(False, id="written-channels-only"),
    pytest.param(True, id="with-unwritten-channel"),
)

# (Invocation count, keep_last, oldest retained step, use the exact search limit.)
# With staggered snapshots, support checkpoints still need each other's history.
# The sixth invocation stores both channel snapshots together at step 16. Both
# channels are first written on step 0, so the whole chain from there may remain.
MULTI_CHANNEL_RETENTION_CASES = (
    pytest.param(5, 1, 0, True, id="staggered-snapshots-exact-limit"),
    pytest.param(6, 1, 16, False, id="common-snapshot"),
    pytest.param(6, 2, 0, False, id="checkpoint-before-common-snapshot"),
    pytest.param(7, 1, 16, False, id="ancestor-common-snapshot"),
)

# Forty invocations keep 120 checkpoints alive when the channel never snapshots.
# Retaining twenty of them makes repeated ancestor walks easy to detect.
LONG_CHAIN_INVOCATIONS = 40
KEEP_LAST = 20

# A head self-cycle, a head two-checkpoint cycle, and a cycle only reached when
# preserving support ancestors. Five invocations place the latest step at 13.
ANCESTRY_CYCLE_CASES = (
    pytest.param(1, 13, id="self-parent"),
    pytest.param(2, 13, id="two-checkpoint-cycle"),
    pytest.param(2, 6, id="support-ancestor-cycle"),
)

FORK_RETENTION_CASES = (
    pytest.param(False, True, id="shared-snapshot"),
    pytest.param(True, True, id="different-snapshots"),
    pytest.param(True, False, id="truncated-fork-history"),
)


class _Checkpointer:
    """Drive RedisSaver and AsyncRedisSaver through one async test body."""

    def __init__(self, saver: Union[RedisSaver, AsyncRedisSaver]) -> None:
        self.saver = saver
        self.is_async = isinstance(saver, AsyncRedisSaver)
        self.get_tuple_name = "aget_tuple" if self.is_async else "get_tuple"
        self.read_name = (
            "_aread_prune_checkpoints" if self.is_async else "_read_prune_checkpoints"
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


def _compile_single_late_channel_graph(checkpointer: Any, first_value: int) -> Any:
    """Write a never-snapshotted DeltaChannel from ``first_value`` on."""

    class State(TypedDict, total=False):
        value: int
        late: Annotated[list[int], DeltaChannel(_append_writes, snapshot_frequency=100)]

    def append(state: dict[str, Any]) -> dict[str, Any]:
        if state["value"] < first_value:
            return {}
        return {"late": [state["value"]]}

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

    `checkpoints` is newest-first. A channel with no stored value is rebuilt from
    its first write, which is recorded on the step before its first version. A
    channel that was never written needs no history.
    """
    steps = [checkpoints[0].metadata["step"]]
    for channel in channels:
        stored = [
            cp.metadata["step"]
            for cp in checkpoints
            if channel in cp.checkpoint["channel_values"]
        ]
        versioned = [
            cp.metadata["step"]
            for cp in checkpoints
            if channel in cp.checkpoint["channel_versions"]
        ]
        if stored:
            steps.append(stored[0])
        elif versioned:
            steps.append(versioned[-1] - 1)
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


def _expected_multi_channel_update(state: dict[str, Any], value: int) -> dict[str, Any]:
    return {
        "value": value,
        **{channel: state.get(channel, []) + [value] for channel in ("fast", "slow")},
    }


def _checkpoint_at_step(checkpoints: list[Any], step: int) -> Any:
    return next(cp for cp in checkpoints if cp.metadata["step"] == step)


def _long_chain_checkpoints_to_check(checkpoints: list[Any]) -> list[Any]:
    """The requested window plus the oldest checkpoint the chain still needs."""
    return checkpoints[:KEEP_LAST] + [_checkpoint_at_step(checkpoints, 0)]


@contextmanager
def _assert_linear_walk_cost(saver: Any, chain: int) -> Iterator[None]:
    """Fail if pruning inside the block repeats walks or reads ancestors one by one."""
    with (
        patch.object(
            saver,
            "_advance_delta_prune_dependencies",
            wraps=saver._advance_delta_prune_dependencies,
        ) as steps,
        patch.object(saver._redis, "pipeline", wraps=saver._redis.pipeline) as trips,
    ):
        yield
    # Walks from overlapping retained checkpoints share their ancestors.
    assert steps.call_count <= chain + KEEP_LAST
    # Reads are pipelined in batches rather than one round trip per ancestor.
    assert trips.call_count <= chain // 10


def _cycle_parent_update(
    saver: Any, checkpoints: list[Any], cycle_length: int, cycle_step: int
) -> tuple[str, str]:
    checkpoints = [cp for cp in checkpoints if cp.metadata["step"] <= cycle_step]
    head = checkpoints[0].config["configurable"]
    target = checkpoints[cycle_length - 1].config["configurable"]
    return saver._make_redis_checkpoint_key(**target), head["checkpoint_id"]


def _bounded_dependency_walk(saver: Any, max_steps: int) -> Any:
    """Fail instead of hanging if pruning stops detecting ancestry cycles."""
    return patch.object(
        saver,
        "_advance_delta_prune_dependencies",
        wraps=saver._advance_delta_prune_dependencies,
        side_effect=[DEFAULT] * max_steps,
    )


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
            # A second call must build a fresh cache, reading the surviving chain
            # again. Ancestors prefetched past the protected history were deleted.
            surviving = {
                cp.checkpoint["id"] for cp in await checkpointer.checkpoints(config)
            }
            reads.reset_mock()
            await checkpointer.prune([thread_id], keep_last=keep_last)
            assert [
                read.args[1]["configurable"]["checkpoint_id"]
                for read in reads.call_args_list
            ] == [read_id for read_id in read_ids if read_id in surviving]

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
    # append_1's write is recorded on the subgraph's step 0, its oldest dependency.
    _assert_retention(before, after, oldest_step=0)


@pytest.mark.parametrize(
    ("invocations", "keep_last", "oldest_step", "exact_limit"),
    MULTI_CHANNEL_RETENTION_CASES,
)
async def test_prune_preserves_independent_delta_snapshots(
    checkpointer: _Checkpointer,
    invocations: int,
    keep_last: int,
    oldest_step: int,
    exact_limit: bool,
) -> None:
    thread_id = f"delta-multi-prune-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_multi_channel_graph(checkpointer.saver)
    for value in range(1, invocations + 1):
        await checkpointer.invoke(graph, value, config)

    values = list(range(1, invocations + 1))
    expected = {"value": invocations, "fast": values, "slow": values}
    assert (await checkpointer.state(graph, config)).values == expected
    before = await checkpointer.checkpoints(config)
    expected_states = {
        cp.checkpoint["id"]: (await checkpointer.state(graph, cp.config)).values
        for cp in before
    }
    max_results = len(before) if exact_limit else 10000
    for _ in range(2):
        await checkpointer.prune(
            [thread_id], keep_last=keep_last, max_results=max_results
        )
        after = await checkpointer.checkpoints(config)
        # Check support ancestors as well as the requested keep_last window.
        for checkpoint in after:
            assert (
                await checkpointer.state(graph, checkpoint.config)
            ).values == expected_states[checkpoint.checkpoint["id"]]
        _assert_retention(before, after, oldest_step)

    # Resume an older support checkpoint, not just the latest requested state.
    support = after[min(4, len(after)) - 1]
    previous = expected_states[support.checkpoint["id"]]
    assert await checkpointer.invoke(
        graph, 99, support.config
    ) == _expected_multi_channel_update(previous, 99)


async def test_prune_releases_history_before_delta_channel_was_written(
    checkpointer: _Checkpointer,
) -> None:
    thread_id = f"delta-unversioned-prune-{uuid4()}"
    config = _config(thread_id)
    # Invocation v stores its input at step 3v - 4, records the node's writes on
    # step 3v - 3, and versions the channel at step 3v - 2. The channel never
    # snapshots, so only the checkpoint holding its first write (step 6, from
    # invocation 3) bounds retention.
    graph = _compile_single_late_channel_graph(checkpointer.saver, first_value=3)
    for value in range(1, 6):
        await checkpointer.invoke(graph, value, config)
    before = await checkpointer.checkpoints(config)
    expected_states = {
        cp.checkpoint["id"]: (await checkpointer.state(graph, cp.config)).values
        for cp in before
    }

    await checkpointer.prune([thread_id], keep_last=1)

    after = await checkpointer.checkpoints(config)
    # The first versioned checkpoint still replays the write kept on its parent.
    _assert_retention(before, after, oldest_step=6)
    for checkpoint in after:
        assert (
            await checkpointer.state(graph, checkpoint.config)
        ).values == expected_states[checkpoint.checkpoint["id"]]


async def test_prune_walks_long_retained_chains_in_linear_time(
    checkpointer: _Checkpointer,
) -> None:
    thread_id = f"delta-long-chain-prune-{uuid4()}"
    config = _config(thread_id)
    # The channel never snapshots, so every checkpoint from step 0 must remain.
    graph = _compile_graph(checkpointer.saver, snapshot_frequency=10_000)
    for value in range(1, LONG_CHAIN_INVOCATIONS + 1):
        await checkpointer.invoke(graph, value, config)
    before = await checkpointer.checkpoints(config)
    checked = _long_chain_checkpoints_to_check(before)
    expected_states = [
        (await checkpointer.state(graph, cp.config)).values for cp in checked
    ]

    with _assert_linear_walk_cost(checkpointer.saver, chain=len(before)):
        await checkpointer.prune([thread_id], keep_last=KEEP_LAST)

    _assert_retention(before, await checkpointer.checkpoints(config), oldest_step=0)
    assert [
        (await checkpointer.state(graph, cp.config)).values for cp in checked
    ] == expected_states


@pytest.mark.parametrize(
    ("separate_snapshots", "complete_search"), FORK_RETENTION_CASES
)
async def test_prune_preserves_forked_delta_history(
    checkpointer: _Checkpointer, separate_snapshots: bool, complete_search: bool
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
    if not complete_search:
        # Three branch-B checkpoints are omitted; the separate-snapshot case
        # reproduces max_results=12 deleting the seed for [1, 2, 30].
        with pytest.raises(ValueError, match="max_results"):
            await checkpointer.prune([thread_id], max_results=len(before) - 3)
        after = await checkpointer.checkpoints(config)
        assert [cp.config for cp in after] == [cp.config for cp in before]
        state = await checkpointer.state(graph, config)
        _assert_branch_state(state.values, [1, 2, 30])
        await checkpointer.prune([thread_id], max_results=len(before))
        state = await checkpointer.state(graph, config)
        _assert_branch_state(state.values, [1, 2, 30])
        return
    expected = _expected_fork_retention(
        before, fork, (branch_a, branch_b), separate_snapshots=separate_snapshots
    )

    await checkpointer.prune([thread_id], keep_last=4, max_results=len(before))

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
    # Until `late` has a snapshot, its first write bounds what it needs, so
    # history before that and before `delta`'s snapshot is still released.
    assert oldest_step > before[-1].metadata["step"]

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


@pytest.mark.parametrize(("cycle_length", "cycle_step"), ANCESTRY_CYCLE_CASES)
async def test_prune_rejects_cyclic_ancestry(
    checkpointer: _Checkpointer, client: Any, cycle_length: int, cycle_step: int
) -> None:
    saver = checkpointer.saver
    thread_id = f"delta-cycle-{uuid4()}"
    config = _config(thread_id)
    graph = _compile_multi_channel_graph(saver)
    for value in range(1, 6):
        await checkpointer.invoke(graph, value, config)
    checkpoints = await checkpointer.checkpoints(config)
    key, parent_id = _cycle_parent_update(saver, checkpoints, cycle_length, cycle_step)
    client.json().set(key, "$.parent_checkpoint_id", parent_id)
    before_keys = set(client.scan_iter(match=f"*{thread_id}*"))

    # Bound advancement so a broken guard fails instead of hanging the test.
    with (
        _bounded_dependency_walk(saver, len(checkpoints) + cycle_length),
        pytest.raises(ValueError, match="Cyclic checkpoint ancestry"),
    ):
        await checkpointer.prune([thread_id], keep_last=1)

    assert set(client.scan_iter(match=f"*{thread_id}*")) == before_keys
    # Complete deletion remains available even for a corrupt dependency chain.
    await checkpointer.prune([thread_id], keep_last=0)
    assert not list(client.scan_iter(match=f"*{thread_id}*"))


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

    def expire_ancestor(batch: list[Any]) -> Any:
        if ancestor.config in batch:
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

    # Each batch starts at the checkpoint the walk needs next; it stops after the
    # expired ancestor instead of failing or reading further for it.
    assert [read.args[0][0] for read in reads.call_args_list] == [
        latest.config,
        ancestor.config,
    ]
    assert await checkpointer.get_tuple(latest.config) is not None


@pytest.mark.parametrize("checkpointer", ["sync"], indirect=True)
def test_prune_reads_only_dependency_fields(checkpointer: _Checkpointer) -> None:
    config = _config("field-selection")
    config["configurable"]["checkpoint_id"] = "checkpoint-id"
    key = "checkpoint:field-selection:__empty__:checkpoint-id"
    pipeline = Mock()

    checkpointer.saver._queue_prune_checkpoint_read(pipeline, config)

    assert pipeline.mock_calls == [
        call.json(),
        call.json().objkeys(key, "$.checkpoint.channel_values"),
        call.json(),
        call.json().get(key, "$.parent_checkpoint_id"),
        call.json(),
        call.json().get(key, "$.metadata"),
        call.json(),
        call.json().get(key, "$.checkpoint.channel_versions"),
    ]


def test_prune_walk_keeps_channels_when_versions_are_unreadable() -> None:
    config = _config("walk-versions")
    config["configurable"]["checkpoint_id"] = "child"
    checkpoint = _PruneCheckpoint(
        channel_names=set(),
        parent_checkpoint_id="parent",
        delta_channels={"delta"},
        versioned_channels=None,
    )

    remaining, parent = BaseRedisSaver._advance_delta_prune_dependencies(
        config, checkpoint, None
    )

    assert remaining == {"delta"}
    assert parent is not None
    assert parent["configurable"]["checkpoint_id"] == "parent"


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
        remaining = client.pttl(key)
        # -2 means pruning deleted the key. Every surviving key keeps its TTL.
        if remaining != -2:
            assert 0 < remaining <= ttl
    _assert_state((await checkpointer.state(graph, config)).values, [1, 2, 3], 3)


_COUNTERS = '{"counters_since_delta_snapshot": {"delta": [1, 3], "unused": [0, 3]}}'


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        # Replies: channel values, parent, metadata, channel versions.
        ([None, None, None, None], None),  # Expired checkpoint.
        # OBJKEYS on an expired key.
        ([ResponseError("missing key"), None, None, None], None),
        ([[], [""], ["{}"], [{}]], None),  # Missing channel-values path.
        ([[None], [""], ["{}"], [{}]], None),  # Non-object channel values.
        ([[[]], None, None, None], None),  # Expired between pipelined commands.
        ([[[]], [""], None, None], None),  # Expired before the metadata read.
        (
            [[["value"]], ["parent"], [], [{"value": "1"}]],  # Missing metadata path.
            _PruneCheckpoint(
                channel_names={"value"},
                parent_checkpoint_id="parent",
                delta_channels=set(),
                versioned_channels={"value"},
            ),
        ),
        (  # Counted but never written: the walk drops it for having no version.
            [[[]], [""], [_COUNTERS], [{"delta": "2"}]],
            _PruneCheckpoint(
                channel_names=set(),
                parent_checkpoint_id="",
                delta_channels={"delta", "unused"},
                versioned_channels={"delta"},
            ),
        ),
        (
            # An unreadable version map keeps every delta channel unresolved.
            [[[]], [""], [_COUNTERS], []],
            _PruneCheckpoint(
                channel_names=set(),
                parent_checkpoint_id="",
                delta_channels={"delta", "unused"},
                versioned_channels=None,
            ),
        ),
        ([ResponseError("OBJKEYS failed"), [""], ["{}"], [{}]], ResponseError),
        ([[[]], ResponseError("parent GET failed"), ["{}"], [{}]], ResponseError),
        ([[[]], [""], ResponseError("metadata GET failed"), [{}]], ResponseError),
    ],
)
def test_prune_reader_responses(results: list[Any], expected: Any) -> None:
    if expected is ResponseError:
        with pytest.raises(ResponseError) as raised:
            BaseRedisSaver._parse_prune_checkpoint(results)
        assert any(raised.value is reply for reply in results)
    else:
        assert BaseRedisSaver._parse_prune_checkpoint(results) == expected
