"""Regression tests for issue #201: _DeltaSnapshot lost through Redis JSON.

JsonPlusRedisSerializer used orjson and treated _DeltaSnapshot (a NamedTuple)
as a plain tuple, so the snapshot seed became [[messages]]. DeltaChannel then
failed with NotImplementedError: Message as a sequence must be (role string,
template).
"""

import base64
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict
from unittest.mock import Mock
from uuid import uuid4

import orjson
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


def _messages_delta_reducer(
    state: list[BaseMessage], writes: Sequence[Any]
) -> list[BaseMessage]:
    result = state
    for write in writes:
        result = add_messages(result, write)
    return result


class _DeltaGraphState(TypedDict):
    messages: Annotated[
        list, DeltaChannel(_messages_delta_reducer, snapshot_frequency=2)
    ]


TURN_COUNT = 3


def _chat_model(state: _DeltaGraphState) -> dict[str, list[AIMessage]]:
    n = len(state["messages"])
    return {"messages": [AIMessage(content=f"reply-{n}", id=f"ai-{n}")]}


def _compile_delta_graph(checkpointer: Any) -> Any:
    builder = StateGraph(_DeltaGraphState)
    builder.add_node("chat", _chat_model)
    builder.add_edge(START, "chat")
    return builder.compile(checkpointer=checkpointer)


def _turn_input(i: int) -> dict[str, list[HumanMessage]]:
    return {"messages": [HumanMessage(content=f"turn-{i}", id=f"human-{i}")]}


def _assert_stored_delta_snapshot(history: dict[str, Any]) -> None:
    seed = history["messages"].get("seed")
    assert isinstance(
        seed, _DeltaSnapshot
    ), f"DeltaChannel never wrote a _DeltaSnapshot seed, got {type(seed)}"


def _expected_messages(turn_count: int) -> list[tuple[str, str, str]]:
    expected = []
    for i in range(turn_count):
        expected.append(("human", f"human-{i}", f"turn-{i}"))

        ai_index = len(expected)
        expected.append(("ai", f"ai-{ai_index}", f"reply-{ai_index}"))
    return expected


def _assert_expected_messages(messages: list[BaseMessage], *, turn_count: int) -> None:
    assert [(msg.type, msg.id, msg.content) for msg in messages] == (
        _expected_messages(turn_count)
    )


def _dump_and_restore_messages_channel(value: Any) -> tuple[dict[str, Any], Any]:
    saver = RedisSaver(redis_client=Mock())
    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(),
        channels={"messages": []},
        step=1,
    )
    checkpoint["channel_values"]["messages"] = value

    dumped = saver._dump_checkpoint(checkpoint)
    restored = saver._deserialize_channel_values(dumped["channel_values"])["messages"]
    return dumped, restored


def test_delta_snapshot_roundtrip() -> None:
    """Tests that nested _DeltaSnapshot values dump as a marker dict and load back.

    Intended behavior: JsonPlusRedisSerializer should write
    {"__delta_snapshot__": True, "value": ...} and revive that as _DeltaSnapshot
    with HumanMessage contents, not as a nested list.
    """
    serializer = JsonPlusRedisSerializer()
    original = {
        "channel_values": {
            "messages": _DeltaSnapshot([HumanMessage(content="hello", id="human-1")])
        }
    }

    type_str, data_bytes = serializer.dumps_typed(original)
    raw = orjson.loads(data_bytes)
    result = serializer.loads_typed((type_str, data_bytes))

    marker = raw["channel_values"]["messages"]
    assert marker["__delta_snapshot__"] is True
    assert set(marker) == {"__delta_snapshot__", "value"}

    seed = result["channel_values"]["messages"]
    assert isinstance(seed, _DeltaSnapshot)
    assert isinstance(seed.value[0], HumanMessage)


def test_plain_dict_not_snapshot() -> None:
    """Tests that a plain {value: ...} dict is not treated as a _DeltaSnapshot.

    Intended behavior: loads_typed should return a non-_DeltaSnapshot value when
    the payload has a value field but no __delta_snapshot__ marker.
    """
    serializer = JsonPlusRedisSerializer()
    payload = {"value": [HumanMessage(content="plain", id="human-plain")]}

    type_str, data_bytes = serializer.dumps_typed(payload)
    result = serializer.loads_typed((type_str, data_bytes))

    assert not isinstance(result, _DeltaSnapshot)


@pytest.mark.parametrize(
    ("snapshot_value", "expected_value"),
    [
        pytest.param([b"payload"], [b"payload"], id="bytes"),
        pytest.param([bytearray(b"payload")], [b"payload"], id="bytearray"),
        pytest.param({1: "one"}, {"1": "one"}, id="non-string-key"),
    ],
)
def test_msgpack_snapshot_roundtrip(snapshot_value: Any, expected_value: Any) -> None:
    """Tests that msgpack fallback preserves _DeltaSnapshot.

    Intended behavior: _DeltaSnapshot should restore as _DeltaSnapshot when
    msgpack fallback is needed for Redis JSON-incompatible values.
    """
    dumped, restored = _dump_and_restore_messages_channel(
        _DeltaSnapshot(snapshot_value)
    )

    assert dumped["type"] == "msgpack"
    assert isinstance(restored, _DeltaSnapshot)
    assert restored.value == expected_value


def test_msgpack_tuple_not_snapshot() -> None:
    """Tests that the msgpack normalization order does not over-match tuples.

    Intended behavior: only real _DeltaSnapshot objects should get the
    __delta_snapshot__ marker; plain tuples remain plain sequence values.
    """
    payload = b"payload"
    dumped, restored = _dump_and_restore_messages_channel((payload,))
    encoded = base64.b64encode(payload).decode()

    assert dumped["type"] == "msgpack"
    assert dumped["channel_values"]["messages"] == [{"__bytes__": encoded}]
    assert not isinstance(restored, _DeltaSnapshot)
    assert restored == [payload]


def test_get_state_after_snapshot(redis_url: str) -> None:
    """Tests that get_state works after DeltaChannel writes a snapshot.

    Intended behavior: RedisSaver should reconstruct DeltaChannel state so
    get_state returns BaseMessage instances instead of raising NotImplementedError.
    """

    thread_id = f"delta-graph-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _compile_delta_graph(saver)

        for i in range(TURN_COUNT):
            graph.invoke(_turn_input(i), config)

        state = graph.get_state(config)
        _assert_expected_messages(state.values["messages"], turn_count=TURN_COUNT)
        _assert_stored_delta_snapshot(
            saver.get_delta_channel_history(config=config, channels=["messages"])
        )


async def test_aget_state_after_snapshot(redis_url: str) -> None:
    """Tests that aget_state works after DeltaChannel writes a snapshot.

    Intended behavior: AsyncRedisSaver should reconstruct DeltaChannel state so
    aget_state returns BaseMessage instances instead of raising NotImplementedError.
    """

    thread_id = f"delta-graph-async-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        graph = _compile_delta_graph(saver)

        for i in range(TURN_COUNT):
            await graph.ainvoke(_turn_input(i), config)

        state = await graph.aget_state(config)
        _assert_expected_messages(state.values["messages"], turn_count=TURN_COUNT)
        _assert_stored_delta_snapshot(
            await saver.aget_delta_channel_history(config=config, channels=["messages"])
        )
