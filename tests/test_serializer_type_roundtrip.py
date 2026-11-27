"""Regression coverage for lossless Redis serializer type round-trips."""

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Any
from uuid import UUID

import orjson
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.types import Interrupt, Send

from langgraph.checkpoint.redis import (
    AsyncRedisSaver,
    AsyncShallowRedisSaver,
    RedisSaver,
    ShallowRedisSaver,
)
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.store.redis import RedisStore
from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.store.redis.base import BaseRedisStore


class StringValue(str, Enum):
    VALUE = "value"


class IntegerValue(IntEnum):
    VALUE = 1


ENUM_CASES: dict[str, Enum] = {
    "string-enum": StringValue.VALUE,
    "integer-enum": IntegerValue.VALUE,
}

ROUNDTRIP_CASES: dict[str, Any] = {
    "decimal": Decimal("1.10"),
    "timedelta": timedelta(days=1, seconds=2),
    "frozenset": frozenset({1, 2}),
    "deque": deque([1, 2]),
    "datetime": datetime(2024, 1, 2, 3, 4, 5),
    "date": date(2024, 1, 2),
    "time": time(3, 4, 5),
    "uuid": UUID("12345678-1234-5678-1234-567812345678"),
    **ENUM_CASES,
    "positive-infinity": float("inf"),
    "negative-infinity": float("-inf"),
    "nan": float("nan"),
    "mixed-dict-keys": {1: "integer", "1": "string"},
}


def _assert_same_value(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


@pytest.mark.parametrize("value", ROUNDTRIP_CASES.values(), ids=ROUNDTRIP_CASES.keys())
def test_serializer_preserves_python_types(value: Any) -> None:
    serializer = JsonPlusRedisSerializer()

    type_, data = serializer.dumps_typed({"value": value})
    restored = serializer.loads_typed((type_, data))["value"]

    _assert_same_value(restored, value)


@pytest.mark.parametrize("value", ENUM_CASES.values(), ids=ENUM_CASES.keys())
def test_store_filters_enum_values_by_plain_value(redis_url: str, value: Enum) -> None:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(("enum",), "item", {"status": value, "tenant": "acme"})
        items = store.search(("enum",), filter={"tenant": "acme"})
        assert [item.key for item in items] == ["item"]
        _assert_same_value(items[0].value["status"], value)
        assert store.search(("enum",), filter={"tenant": None}) == []
        assert [
            item.key for item in store.search(("enum",), filter={"status": value.value})
        ] == ["item"]


@pytest.mark.parametrize("value", ENUM_CASES.values(), ids=ENUM_CASES.keys())
async def test_async_store_filters_enum_values_by_plain_value(
    redis_url: str, value: Enum
) -> None:
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()
        await store.aput(("enum",), "item", {"status": value, "tenant": "acme"})
        items = await store.asearch(("enum",), filter={"tenant": "acme"})
        assert [item.key for item in items] == ["item"]
        _assert_same_value(items[0].value["status"], value)
        assert await store.asearch(("enum",), filter={"tenant": None}) == []
        matches = await store.asearch(("enum",), filter={"status": value.value})
        assert [item.key for item in matches] == ["item"]


class _DummySaver(BaseRedisSaver):
    def create_indexes(self) -> None:
        pass

    def configure_client(self, **kwargs: Any) -> None:
        self._redis = None


def test_checkpoint_normalization_preserves_lossy_json_values() -> None:
    saver = _DummySaver(redis_client=object())
    values = ROUNDTRIP_CASES
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"payload": values}

    dumped = saver._dump_checkpoint(checkpoint)

    # The persisted checkpoint remains valid RedisJSON.
    orjson.dumps(dumped)
    restored = saver._deserialize_channel_values(
        dumped["channel_values"], type_=dumped["type"]
    )["payload"]
    for name, expected in values.items():
        _assert_same_value(restored[name], expected)


def test_store_wraps_values_that_json_would_change() -> None:
    store = object.__new__(BaseRedisStore)
    store._serde = JsonPlusRedisSerializer()
    value = {
        "datetime": datetime(2024, 1, 2, 3, 4, 5),
        "mapping": {1: "integer", "1": "string"},
    }

    serialized = store._serialize_value(value)
    restored = store._deserialize_value(serialized)

    assert set(serialized) == {"__serde_type__", "__serde_data__"}
    assert restored == value
    assert type(restored["datetime"]) is datetime
    assert set(restored["mapping"]) == {1, "1"}


def test_store_keeps_native_json_values_unwrapped() -> None:
    store = object.__new__(BaseRedisStore)
    store._serde = JsonPlusRedisSerializer()
    value = {"nested": [1, 2.5, True, None, "value"]}

    assert store._serialize_value(value) is value


def _roundtrip_put_args(saver_cls: type) -> tuple[Any, ...]:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"payload": ROUNDTRIP_CASES}
    checkpoint["channel_versions"] = {"payload": "1"}
    config = {
        "configurable": {
            "thread_id": f"serializer-type-roundtrip-{saver_cls.__name__}",
            "checkpoint_ns": "",
        }
    }
    metadata = {"source": "input", "step": 0, "writes": {}}
    return config, checkpoint, metadata, {"payload": "1"}


def _assert_restored_roundtrip_values(restored_checkpoint: Any) -> None:
    assert restored_checkpoint is not None
    restored = restored_checkpoint["channel_values"]["payload"]
    for name, expected in ROUNDTRIP_CASES.items():
        _assert_same_value(restored[name], expected)


@pytest.mark.parametrize(
    "saver_cls", [RedisSaver, ShallowRedisSaver], ids=lambda cls: cls.__name__
)
def test_redis_saver_preserves_lossy_json_values(
    redis_url: str, saver_cls: type
) -> None:
    with saver_cls.from_conn_string(redis_url) as saver:
        saver.setup()
        saved_config = saver.put(*_roundtrip_put_args(saver_cls))
        _assert_restored_roundtrip_values(saver.get(saved_config))


@pytest.mark.parametrize(
    "saver_cls",
    [AsyncRedisSaver, AsyncShallowRedisSaver],
    ids=lambda cls: cls.__name__,
)
async def test_async_redis_saver_preserves_lossy_json_values(
    redis_url: str, saver_cls: type
) -> None:
    async with saver_cls.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saved_config = await saver.aput(*_roundtrip_put_args(saver_cls))
        _assert_restored_roundtrip_values(await saver.aget(saved_config))


def test_redis_saver_returns_plain_value_for_blocked_enum(redis_url: str) -> None:
    """With msgpack revival locked down, an unlisted Enum degrades to its value."""
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"payload": ENUM_CASES}
    checkpoint["channel_versions"] = {"payload": "1"}
    config = {
        "configurable": {"thread_id": "serializer-blocked-enum", "checkpoint_ns": ""}
    }

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        # Equivalent to LANGGRAPH_STRICT_MSGPACK=true with no allowlist.
        saver.serde = JsonPlusRedisSerializer(allowed_msgpack_modules=None)
        saved_config = saver.put(
            config, checkpoint, {"source": "input", "step": 0}, {"payload": "1"}
        )
        restored_checkpoint = saver.get(saved_config)

    assert restored_checkpoint is not None
    restored = restored_checkpoint["channel_values"]["payload"]
    for name, member in ENUM_CASES.items():
        _assert_same_value(restored[name], member.value)


def test_redis_store_preserves_lossy_json_values(redis_url: str) -> None:
    values = ROUNDTRIP_CASES

    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(("serializer",), "types", values)
        item = store.get(("serializer",), "types")

    assert item is not None
    for name, expected in values.items():
        _assert_same_value(item.value[name], expected)


LITERAL_ENVELOPES = [
    pytest.param(
        {"__redis_jsonplus_typed__": True, "type": "bytes", "data": "YQ=="},
        id="valid-bytes",
    ),
    pytest.param(
        {"__redis_jsonplus_typed__": True, "type": "unknown", "data": ""},
        id="unknown-codec",
    ),
    pytest.param(
        {"__redis_jsonplus_typed__": True, "type": "msgpack", "data": "wQ=="},
        id="malformed-msgpack",
    ),
    pytest.param(
        {"__redis_jsonplus_typed__": True, "type": "bytes", "data": "!"},
        id="malformed-base64",
    ),
    pytest.param(
        {
            "__redis_jsonplus_typed__": True,
            "type": {"__send__": True, "node": "literal", "arg": 1},
            "data": {"__bytes__": "YQ=="},
        },
        id="nested-literal-markers",
    ),
]


@dataclass
class EnvelopeContainer:
    value: dict[str, Any]


@pytest.mark.parametrize("literal", LITERAL_ENVELOPES)
@pytest.mark.parametrize("binary", [False, True], ids=["json", "msgpack-fallback"])
def test_serializer_preserves_literal_envelopes(
    literal: dict[str, Any], binary: bool
) -> None:
    value = {"literal": literal, "amount": Decimal("1.10")}
    payload = {
        "plain": value,
        "snapshot": _DeltaSnapshot([value]),
        "send": Send("node", value),
        "interrupt": Interrupt(value=value),
        "message": HumanMessage(content="hello", additional_kwargs=value),
        "dataclass": EnvelopeContainer(value),
    }
    if binary:
        payload["binary"] = b"payload"
    serde = JsonPlusRedisSerializer()
    assert serde.loads_typed(serde.dumps_typed(payload)) == payload


@pytest.mark.parametrize(
    "value",
    [
        {"__redis_jsonplus_typed__": False, "type": "bytes", "data": "YQ=="},
        {"__redis_jsonplus_typed__": 1, "type": "bytes", "data": "YQ=="},
        {"__redis_jsonplus_typed__": True, "type": "bytes", "data": "YQ==", "extra": 1},
        {"__redis_jsonplus_typed__": True, "type": "bytes"},
        {"__redis_jsonplus_typed__": True, "type": 123, "data": []},
    ],
)
def test_serializer_preserves_incomplete_or_invalid_literal_envelopes(
    value: dict[str, Any],
) -> None:
    serde = JsonPlusRedisSerializer()
    assert serde.loads_typed(serde.dumps_typed(value)) == value


@pytest.mark.parametrize("literal", LITERAL_ENVELOPES[:-1])
def test_legacy_json_does_not_decode_new_envelopes(literal: dict[str, Any]) -> None:
    # Nested legacy Send/bytes markers retain their old meaning, so exclude those.
    value = {"literal": literal}
    assert JsonPlusRedisSerializer().loads_typed(("json", orjson.dumps(value))) == value
