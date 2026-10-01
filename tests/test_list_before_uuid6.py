"""list(before=...) must filter for the checkpoint ids LangGraph actually generates.

LangGraph creates checkpoint ids with uuid6() (36 characters). The savers built the
``before`` filter by parsing the id as a ULID, and when that failed they dropped the
filter, so ``get_state_history(config, before=...)`` returned the whole history.
"""

import operator
import uuid
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


class State(TypedDict):
    messages: Annotated[list, operator.add]


def _reply(state: State) -> dict:
    return {"messages": [f"reply to {state['messages'][-1]}"]}


def _graph(saver):
    builder = StateGraph(State)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


def _config() -> dict:
    return {"configurable": {"thread_id": f"before-{uuid.uuid4()}"}}


def test_get_state_history_before_uuid6_checkpoint(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        app = _graph(saver)
        config = _config()
        for turn in range(3):
            app.invoke({"messages": [f"turn {turn}"]}, config)

        history = list(app.get_state_history(config))
        ids = [s.config["configurable"]["checkpoint_id"] for s in history]
        assert len(ids[0]) == 36  # LangGraph's own uuid6 ids, not ULIDs
        middle = history[len(history) // 2]

        before = list(app.get_state_history(config, before=middle.config))
        before_ids = [s.config["configurable"]["checkpoint_id"] for s in before]

        expected = [
            i for i in ids if i < middle.config["configurable"]["checkpoint_id"]
        ]
        assert before_ids == expected
        assert middle.config["configurable"]["checkpoint_id"] not in before_ids


@pytest.mark.asyncio
async def test_aget_state_history_before_uuid6_checkpoint(redis_url: str) -> None:
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        app = _graph(saver)
        config = _config()
        for turn in range(3):
            await app.ainvoke({"messages": [f"turn {turn}"]}, config)

        history = [s async for s in app.aget_state_history(config)]
        ids = [s.config["configurable"]["checkpoint_id"] for s in history]
        middle = history[len(history) // 2]

        before = [s async for s in app.aget_state_history(config, before=middle.config)]
        before_ids = [s.config["configurable"]["checkpoint_id"] for s in before]

        expected = [
            i for i in ids if i < middle.config["configurable"]["checkpoint_id"]
        ]
        assert before_ids == expected


def test_checkpoint_id_timestamp_formats() -> None:
    import time

    from langgraph.checkpoint.base import empty_checkpoint
    from ulid import ULID

    from langgraph.checkpoint.redis.util import checkpoint_id_timestamp

    # LangGraph's uuid6 ids decode to milliseconds since the epoch, in order.
    first, second = empty_checkpoint()["id"], empty_checkpoint()["id"]
    first_ts, second_ts = checkpoint_id_timestamp(first), checkpoint_id_timestamp(
        second
    )
    assert first_ts is not None and second_ts is not None
    assert abs(first_ts - time.time() * 1000) < 60_000
    assert first_ts < second_ts

    # ULIDs keep the value the savers have always stored for them.
    ulid = ULID()
    assert checkpoint_id_timestamp(str(ulid)) == ulid.timestamp

    # Ids that encode no time give None, and the caller falls back.
    assert checkpoint_id_timestamp(str(uuid.uuid4())) is None
    assert checkpoint_id_timestamp("not-an-id") is None
