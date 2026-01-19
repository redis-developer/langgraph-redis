import uuid
from this import d

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def _make_checkpoint(checkpoint_id: str) -> Checkpoint:
    return Checkpoint(
        v=1,
        id=checkpoint_id,
        ts="2024-01-01T00:00:00Z",
        channel_values={"messages": [f"checkpoint-{checkpoint_id}"]},
        channel_versions={"messages": "1"},
        versions_seen={"agent": {"messages": "1"}},
        pending_sends=[],
        tasks=[],
    )


def test_list_filters_run_id_and_thread_id(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        thread_id = "thread-filter-sync"
        run_id_1 = "run-1"
        run_id_2 = "run-2"

        config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "1",
                "run_id": run_id_1,
            }
        }
        config_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "2",
            }
        }

        checkpointer.put(
            config=config_1,
            checkpoint=_make_checkpoint("1"),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        checkpointer.put(
            config=config_2,
            checkpoint=_make_checkpoint("2"),
            metadata=CheckpointMetadata(
                source="input", step=1, writes={}, run_id=run_id_2
            ),
            new_versions={"messages": "1"},
        )

        run_id_results = list(checkpointer.list(None, filter={"run_id": run_id_1}))
        assert len(run_id_results) == 1
        assert run_id_results[0].checkpoint["id"] == "1"

        thread_id_results = list(
            checkpointer.list(None, filter={"thread_id": thread_id})
        )
        assert len(thread_id_results) == 2


@pytest.mark.asyncio
async def test_alist_filters_run_id_and_thread_id(redis_url: str) -> None:
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        thread_id = "thread-filter-async"
        run_id_1 = "run-async-1"
        run_id_2 = "run-async-2"

        config_1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "1",
                "run_id": run_id_1,
            }
        }
        config_2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "2",
            }
        }

        await checkpointer.aput(
            config=config_1,
            checkpoint=_make_checkpoint("1"),
            metadata=CheckpointMetadata(source="input", step=0, writes={}),
            new_versions={"messages": "1"},
        )

        await checkpointer.aput(
            config=config_2,
            checkpoint=_make_checkpoint("2"),
            metadata=CheckpointMetadata(
                source="input", step=1, writes={}, run_id=run_id_2
            ),
            new_versions={"messages": "1"},
        )

        run_id_results = [
            item async for item in checkpointer.alist(None, filter={"run_id": run_id_1})
        ]
        assert len(run_id_results) == 1
        assert run_id_results[0].checkpoint["id"] == "1"

        thread_id_results = [
            item
            async for item in checkpointer.alist(None, filter={"thread_id": thread_id})
        ]
        assert len(thread_id_results) == 2


# Now test using the higher-level configuration
async def test_aget_state_history_run_id_and_thread_id_pregel(redis_url: str) -> None:
    class State(TypedDict):
        foo: str

    def node_1(state: State):
        return {"foo": "hi node1! " + state["foo"]}

    def node_2(state: State):
        return {"foo": "hi node2! " + state["foo"]}

    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        await checkpointer.asetup()
        thread_id = uuid.uuid4()
        run_id_1 = uuid.uuid4()
        run_id_2 = uuid.uuid4()
        thread_id_2 = uuid.uuid4()
        run_id_3 = uuid.uuid4()
        graph = builder.compile(checkpointer=checkpointer)
        expected = {"foo": "hi node2! hi node1! input1"}
        result = await graph.ainvoke(
            {"foo": "input1"},
            config={"configurable": {"thread_id": thread_id, "run_id": run_id_1}},
        )
        assert result == expected
        expected = {"foo": "hi node2! hi node1! input2"}
        result = await graph.ainvoke(
            {"foo": "input2"},
            config={"configurable": {"thread_id": thread_id, "run_id": run_id_2}},
        )
        assert result == expected
        expected = {"foo": "hi node2! hi node1! input3"}
        result = await graph.ainvoke(
            {"foo": "input3"},
            config={"configurable": {"thread_id": thread_id_2, "run_id": run_id_3}},
        )
        assert result == expected
        full_history = []
        async for item in graph.aget_state_history(
            config={"configurable": {"thread_id": thread_id}}
        ):
            full_history.append(item)
        history_run_1 = []
        async for item in graph.aget_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_1},
        ):
            history_run_1.append(item)
        history_run_2 = []
        async for item in graph.aget_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_2},
        ):
            history_run_2.append(item)
        assert len(full_history) == 8
        assert full_history == history_run_1 + history_run_2
        history_run_3 = []
        async for item in graph.aget_state_history(
            config={"configurable": {"thread_id": thread_id_2}},
            filter={"run_id": run_id_3},
        ):
            history_run_3.append(item)
        assert len(history_run_3) == 4
        # run_id_3 was NOT on thread_id, so we expect no items
        async for item in graph.aget_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_3},
        ):
            raise Exception("Should not have any items")


def test_get_state_history_run_id_and_thread_id_pregel(redis_url: str) -> None:
    class State(TypedDict):
        foo: str

    def node_1(state: State):
        return {"foo": "hi node1! " + state["foo"]}

    def node_2(state: State):
        return {"foo": "hi node2! " + state["foo"]}

    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()
        thread_id = uuid.uuid4()
        run_id_1 = uuid.uuid4()
        run_id_2 = uuid.uuid4()
        thread_id_2 = uuid.uuid4()
        run_id_3 = uuid.uuid4()
        graph = builder.compile(checkpointer=checkpointer)
        expected = {"foo": "hi node2! hi node1! input1"}
        result = graph.invoke(
            {"foo": "input1"},
            config={"configurable": {"thread_id": thread_id, "run_id": run_id_1}},
        )
        assert result == expected
        expected = {"foo": "hi node2! hi node1! input2"}
        result = graph.invoke(
            {"foo": "input2"},
            config={"configurable": {"thread_id": thread_id, "run_id": run_id_2}},
        )
        assert result == expected
        expected = {"foo": "hi node2! hi node1! input3"}
        result = graph.invoke(
            {"foo": "input3"},
            config={"configurable": {"thread_id": thread_id_2, "run_id": run_id_3}},
        )
        assert result == expected
        full_history = []
        for item in graph.get_state_history(
            config={"configurable": {"thread_id": thread_id}}
        ):
            full_history.append(item)
        history_run_1 = []
        for item in graph.get_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_1},
        ):
            history_run_1.append(item)
        history_run_2 = []
        for item in graph.get_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_2},
        ):
            history_run_2.append(item)
        assert len(full_history) == 8
        assert full_history == history_run_1 + history_run_2
        history_run_3 = []
        for item in graph.get_state_history(
            config={"configurable": {"thread_id": thread_id_2}},
            filter={"run_id": run_id_3},
        ):
            history_run_3.append(item)
        assert len(history_run_3) == 4
        # run_id_3 was NOT on thread_id, so we expect no items
        for item in graph.get_state_history(
            config={"configurable": {"thread_id": thread_id}},
            filter={"run_id": run_id_3},
        ):
            raise Exception("Should not have any items")
