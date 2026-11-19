"""Regression coverage for lossless Redis serializer type round-trips."""

import math
from collections import deque
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import orjson
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.store.redis import RedisStore
from langgraph.store.redis.base import BaseRedisStore


class StringValue(str, Enum):
    VALUE = "value"


ROUNDTRIP_CASES: list[tuple[str, Any]] = [
    ("decimal", Decimal("1.10")),
    ("timedelta", timedelta(days=1, seconds=2)),
    ("frozenset", frozenset({1, 2})),
    ("deque", deque([1, 2])),
    ("datetime", datetime(2024, 1, 2, 3, 4, 5)),
    ("date", date(2024, 1, 2)),
    ("time", time(3, 4, 5)),
    ("uuid", UUID("12345678-1234-5678-1234-567812345678")),
    ("enum", StringValue.VALUE),
    ("positive-infinity", float("inf")),
    ("negative-infinity", float("-inf")),
    ("nan", float("nan")),
    ("mixed-dict-keys", {1: "integer", "1": "string"}),
]

ROUNDTRIP_VALUES = [pytest.param(value, id=name) for name, value in ROUNDTRIP_CASES]


def _assert_same_value(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected


def _roundtrip_values() -> dict[str, Any]:
    return dict(ROUNDTRIP_CASES)


@pytest.mark.parametrize("value", ROUNDTRIP_VALUES)
def test_serializer_preserves_python_types(value: Any) -> None:
    serializer = JsonPlusRedisSerializer()

    type_, data = serializer.dumps_typed({"value": value})
    restored = serializer.loads_typed((type_, data))["value"]

    _assert_same_value(restored, value)


class _DummySaver(BaseRedisSaver):
    def create_indexes(self) -> None:
        pass

    def configure_client(self, **kwargs: Any) -> None:
        self._redis = None


def test_checkpoint_normalization_preserves_lossy_json_values() -> None:
    saver = _DummySaver(redis_client=object())
    values = _roundtrip_values()
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"payload": values}

    dumped = saver._dump_checkpoint(checkpoint)

    # The persisted checkpoint remains valid RedisJSON.
    orjson.dumps(dumped)
    restored = saver._deserialize_channel_values(dumped["channel_values"])["payload"]
    for name, expected in values.items():
        _assert_same_value(restored[name], expected)


def test_store_wraps_values_that_json_would_change() -> None:
    store = object.__new__(BaseRedisStore)
    store._serde = JsonPlusRedisSerializer()
    value = {
        "datetime": datetime(2024, 1, 2, 3, 4, 5),
        "enum": StringValue.VALUE,
        "mapping": {1: "integer", "1": "string"},
    }

    serialized = store._serialize_value(value)
    restored = store._deserialize_value(serialized)

    assert set(serialized) == {"__serde_type__", "__serde_data__"}
    assert restored == value
    assert type(restored["datetime"]) is datetime
    assert type(restored["enum"]) is StringValue
    assert set(restored["mapping"]) == {1, "1"}


def test_store_keeps_native_json_values_unwrapped() -> None:
    store = object.__new__(BaseRedisStore)
    store._serde = JsonPlusRedisSerializer()
    value = {"nested": [1, 2.5, True, None, "value"]}

    assert store._serialize_value(value) is value


def test_redis_saver_preserves_lossy_json_values(redis_url: str) -> None:
    values = _roundtrip_values()
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"payload": values}
    checkpoint["channel_versions"] = {"payload": "1"}
    config = {
        "configurable": {
            "thread_id": "serializer-type-roundtrip",
            "checkpoint_ns": "",
        }
    }

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saved_config = saver.put(
            config,
            checkpoint,
            {"source": "input", "step": 0, "writes": {}},
            {"payload": "1"},
        )
        restored_checkpoint = saver.get(saved_config)

    assert restored_checkpoint is not None
    restored = restored_checkpoint["channel_values"]["payload"]
    for name, expected in values.items():
        _assert_same_value(restored[name], expected)


def test_redis_store_preserves_lossy_json_values(redis_url: str) -> None:
    values = _roundtrip_values()

    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        store.put(("serializer",), "types", values)
        item = store.get(("serializer",), "types")

    assert item is not None
    for name, expected in values.items():
        _assert_same_value(item.value[name], expected)
