"""list(before=...) must filter for the checkpoint ids LangGraph actually generates.

LangGraph creates checkpoint ids with uuid6() (36 characters). The savers built the
``before`` filter by parsing the id as a ULID, and when that failed they dropped the
filter, so ``get_state_history(config, before=...)`` returned the whole history.

The shallow savers stamped ``checkpoint_ts`` the same way, falling back to
``checkpoint["ts"]`` for every UUIDv6 id. ULID ids keep working as before.
"""

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, empty_checkpoint
from langgraph.graph import END, START, StateGraph
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# A LangGraph checkpoint id (the one from Issue #136). Its UUIDv6 time field is
# 2025-11-10 13:03:23.206485 UTC.
UUID6_ID = "1f0be35a-360e-6154-8002-cb3ee66bf299"
UUID6_ID_MS = (
    datetime(2025, 11, 10, 13, 3, 23, 206485, tzinfo=timezone.utc).timestamp() * 1000
)
# A checkpoint["ts"] almost two years earlier, so the fallback cannot pass for the id.
OTHER_TS = "2024-01-01T00:00:00+00:00"
OTHER_TS_MS = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


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
    first = empty_checkpoint()["id"]
    time.sleep(0.001)  # ids made within one microsecond decode to the same value
    second = empty_checkpoint()["id"]
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


def _uuid6_checkpoint() -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = UUID6_ID
    checkpoint["ts"] = OTHER_TS
    return checkpoint


def _thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def test_shallow_put_stores_uuid6_id_timestamp(redis_url: str) -> None:
    """checkpoint_ts comes from the UUIDv6 id, not the checkpoint["ts"] fallback."""
    thread_id = f"shallow-{uuid.uuid4()}"
    metadata: CheckpointMetadata = {"source": "input", "step": 1}

    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saver.put(_thread_config(thread_id), _uuid6_checkpoint(), metadata, {})
        stored = saver._redis.json().get(
            saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        )

    assert stored["checkpoint_id"] == UUID6_ID
    assert stored["checkpoint_ts"] == pytest.approx(UUID6_ID_MS, abs=1)
    assert stored["checkpoint_ts"] != pytest.approx(OTHER_TS_MS, abs=1)


@pytest.mark.asyncio
async def test_ashallow_aput_stores_uuid6_id_timestamp(redis_url: str) -> None:
    """checkpoint_ts comes from the UUIDv6 id, not the checkpoint["ts"] fallback."""
    thread_id = f"ashallow-{uuid.uuid4()}"
    metadata: CheckpointMetadata = {"source": "input", "step": 1}

    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        await saver.aput(_thread_config(thread_id), _uuid6_checkpoint(), metadata, {})
        stored = await saver._redis.json().get(
            saver._make_shallow_redis_checkpoint_key_cached(thread_id, "")
        )

    assert stored["checkpoint_id"] == UUID6_ID
    assert stored["checkpoint_ts"] == pytest.approx(UUID6_ID_MS, abs=1)
    assert stored["checkpoint_ts"] != pytest.approx(OTHER_TS_MS, abs=1)


def _ulid_ids(count: int) -> list[str]:
    """Explicit ULID checkpoint ids one second apart, oldest first."""
    start = datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp()
    return [str(ULID.from_timestamp(start + i)) for i in range(count)]


def _ulid_checkpoint(checkpoint_id: str) -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = checkpoint_id
    return checkpoint


def _list_ids(tuples: list) -> list[str]:
    return [t.config["configurable"]["checkpoint_id"] for t in tuples]


def test_list_before_ulid_checkpoint(redis_url: str) -> None:
    """Explicit ULID ids still filter: list(before=...) returns the older ones."""
    thread_id = f"ulid-{uuid.uuid4()}"
    ids = _ulid_ids(4)

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config: RunnableConfig = _thread_config(thread_id)
        for step, checkpoint_id in enumerate(ids):
            metadata: CheckpointMetadata = {"source": "loop", "step": step}
            config = saver.put(config, _ulid_checkpoint(checkpoint_id), metadata, {})

        thread = _thread_config(thread_id)
        assert _list_ids(list(saver.list(thread))) == ids[::-1]

        before = {"configurable": {**thread["configurable"], "checkpoint_id": ids[2]}}
        assert _list_ids(list(saver.list(thread, before=before))) == [ids[1], ids[0]]


@pytest.mark.asyncio
async def test_alist_before_ulid_checkpoint(redis_url: str) -> None:
    """Explicit ULID ids still filter: alist(before=...) returns the older ones."""
    thread_id = f"aulid-{uuid.uuid4()}"
    ids = _ulid_ids(4)

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config: RunnableConfig = _thread_config(thread_id)
        for step, checkpoint_id in enumerate(ids):
            metadata: CheckpointMetadata = {"source": "loop", "step": step}
            config = await saver.aput(
                config, _ulid_checkpoint(checkpoint_id), metadata, {}
            )

        thread = _thread_config(thread_id)
        assert _list_ids([t async for t in saver.alist(thread)]) == ids[::-1]

        before = {"configurable": {**thread["configurable"], "checkpoint_id": ids[2]}}
        assert _list_ids([t async for t in saver.alist(thread, before=before)]) == [
            ids[1],
            ids[0],
        ]


# Five checkpoint ids LangGraph made on a Linux CI runner (run 38835913127), oldest
# first; the last is the one get_state_history(before=...) returned there. Linux's
# clock carries the 100 ns digit, so their timestamps need 17 significant digits.
CI_IDS = [
    "1f1c902d-58c6-6e18-bfff-3b1e1314fdf4",
    "1f1c902d-58c8-616c-8000-2f9c6a1dbb6d",
    "1f1c902d-58cd-69e9-8001-94923f4c07c0",
    "1f1c902d-58d9-6471-8002-89e408eacaeb",
    "1f1c902d-58da-60f2-8003-4602ef4666e2",
]
# `before` checkpoints printed by other failing CI runs, 12 to 15 Oct 2026.
CI_BEFORE_IDS = [
    "1f1c9088-894a-6ba6-8003-2b9b78b77676",
    "1f1c9088-ed9c-66b9-8003-7fa526d3ec3e",
    "1f1c908c-233d-6dd2-8003-4d8bc0be861f",
    "1f1c902e-0e2b-60e1-8003-296b05e03f6e",
    "1f1c7761-bce6-6221-8003-679fd903ee16",
    "1f1c724c-e87a-6ae1-8003-54e0758e5421",
    "1f1c724b-d02b-6bfd-8003-cea4a0430ec4",
    "1f1c6aee-6a9b-6619-8003-c6e8706dd7ea",
    "1f1c6af1-1f74-6952-8003-12a71ca320c0",
    "1f1c5e63-efcb-659d-8003-f03b3060c16a",
]


def _put_ids(saver: RedisSaver, thread_id: str, ids: list[str]) -> None:
    config: RunnableConfig = _thread_config(thread_id)
    for step, checkpoint_id in enumerate(ids):
        metadata: CheckpointMetadata = {"source": "loop", "step": step}
        config = saver.put(config, _ulid_checkpoint(checkpoint_id), metadata, {})


def _before(thread_id: str, checkpoint_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }


@pytest.mark.parametrize("checkpoint_id", CI_IDS[-1:] + CI_BEFORE_IDS)
def test_stored_checkpoint_ts_is_the_before_bound(
    redis_url: str, checkpoint_id: str
) -> None:
    """The checkpoint_ts Redis stores is the value list(before=...) compares with."""
    from langgraph.checkpoint.redis.util import checkpoint_id_timestamp

    thread_id = f"ts-{uuid.uuid4()}"
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        _put_ids(saver, thread_id, [checkpoint_id])
        stored = saver._redis.json().get(
            saver._make_redis_checkpoint_key_cached(thread_id, "", checkpoint_id)
        )

    assert stored["checkpoint_ts"] == checkpoint_id_timestamp(checkpoint_id)


# CI_IDS[4] is the id Redis stores one unit in the last place low; CI_IDS[3] is a
# control whose 17-digit timestamp it already stores exactly.
@pytest.mark.parametrize("boundary", [4, 3])
def test_list_before_excludes_the_before_checkpoint(
    redis_url: str, boundary: int
) -> None:
    thread_id = f"bound-{uuid.uuid4()}"
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        _put_ids(saver, thread_id, CI_IDS)
        thread = _thread_config(thread_id)
        assert _list_ids(list(saver.list(thread))) == CI_IDS[::-1]

        before = _before(thread_id, CI_IDS[boundary])
        listed = _list_ids(list(saver.list(thread, before=before)))
        assert listed == CI_IDS[boundary - 1 :: -1]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", [4, 3])
async def test_alist_before_excludes_the_before_checkpoint(
    redis_url: str, boundary: int
) -> None:
    thread_id = f"abound-{uuid.uuid4()}"
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        config: RunnableConfig = _thread_config(thread_id)
        for step, checkpoint_id in enumerate(CI_IDS):
            metadata: CheckpointMetadata = {"source": "loop", "step": step}
            config = await saver.aput(
                config, _ulid_checkpoint(checkpoint_id), metadata, {}
            )
        thread = _thread_config(thread_id)

        before = _before(thread_id, CI_IDS[boundary])
        listed = _list_ids([t async for t in saver.alist(thread, before=before)])
        assert listed == CI_IDS[boundary - 1 :: -1]


def test_shallow_list_before_excludes_the_before_checkpoint(redis_url: str) -> None:
    thread_id = f"sbound-{uuid.uuid4()}"
    metadata: CheckpointMetadata = {"source": "loop", "step": 1}
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saver.put(_thread_config(thread_id), _ulid_checkpoint(CI_IDS[-1]), metadata, {})
        thread = _thread_config(thread_id)
        assert _list_ids(list(saver.list(thread))) == [CI_IDS[-1]]

        before = _before(thread_id, CI_IDS[-1])
        assert _list_ids(list(saver.list(thread, before=before))) == []


@pytest.mark.asyncio
async def test_ashallow_alist_before_excludes_the_before_checkpoint(
    redis_url: str,
) -> None:
    thread_id = f"asbound-{uuid.uuid4()}"
    metadata: CheckpointMetadata = {"source": "loop", "step": 1}
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        await saver.aput(
            _thread_config(thread_id), _ulid_checkpoint(CI_IDS[-1]), metadata, {}
        )
        thread = _thread_config(thread_id)
        assert _list_ids([t async for t in saver.alist(thread)]) == [CI_IDS[-1]]

        before = _before(thread_id, CI_IDS[-1])
        assert _list_ids([t async for t in saver.alist(thread, before=before)]) == []
