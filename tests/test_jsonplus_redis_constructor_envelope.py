"""Regression tests for issue #189 JsonPlusRedisSerializer compatibility."""

import dataclasses
import json
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


@dataclasses.dataclass
class ConstructorEnvelopePayload:
    value: str


@dataclasses.dataclass
class NestedConstructorEnvelopePayload:
    payload: ConstructorEnvelopePayload


def test_dataclass_roundtrip_when_constructor_helper_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that dataclass payloads roundtrip without the checkpoint helper.

    Intended behavior: JsonPlusRedisSerializer should preserve dataclass type
    information using its JSON constructor envelope even when the inherited
    checkpoint serializer does not provide a constructor helper.
    """
    monkeypatch.delattr(
        JsonPlusSerializer,
        "_encode_constructor_args",
        raising=False,
    )

    serializer = JsonPlusRedisSerializer()
    type_str, data_bytes = serializer.dumps_typed(
        {"payload": ConstructorEnvelopePayload(value="ok")}
    )
    result = serializer.loads_typed((type_str, data_bytes))

    assert type_str == "json"
    assert result["payload"] == ConstructorEnvelopePayload(value="ok")


def test_nested_dataclass_roundtrip_preserves_inner_type() -> None:
    """Dataclass serialization should not flatten nested dataclasses to dicts.

    A dataclass field whose value is another dataclass should roundtrip as that
    original inner dataclass type, not as a plain dictionary.
    """
    serializer = JsonPlusRedisSerializer()
    original = NestedConstructorEnvelopePayload(
        payload=ConstructorEnvelopePayload(value="nested")
    )

    type_str, data_bytes = serializer.dumps_typed({"payload": original})
    result = serializer.loads_typed((type_str, data_bytes))

    assert result["payload"] == original
    assert isinstance(result["payload"].payload, ConstructorEnvelopePayload)


def test_set_constructor_roundtrip_uses_default_constructor_args() -> None:
    """New set envelopes should be compatible with set(iterable).

    The serializer should write sets using constructor args rather than custom
    kwargs, and the decoder should restore the value as a Python set.
    """
    serializer = JsonPlusRedisSerializer()

    type_str, data_bytes = serializer.dumps_typed({"ids": {"msg1", "msg2"}})
    raw = json.loads(data_bytes)

    assert raw["ids"]["id"] == ["builtins", "set"]
    assert "kwargs" not in raw["ids"]
    assert sorted(raw["ids"]["args"][0]) == ["msg1", "msg2"]

    result = serializer.loads_typed((type_str, data_bytes))

    assert result["ids"] == {"msg1", "msg2"}


def test_legacy_set_constructor_kwargs_still_roundtrips() -> None:
    """Older Redis checkpoints used a custom __set_items__ kwarg.

    The decoder should continue to read that legacy shape so existing
    checkpoints written by earlier serializer versions still load correctly.
    """
    serializer = JsonPlusRedisSerializer()
    legacy_payload = {
        "ids": {
            "lc": 2,
            "type": "constructor",
            "id": ["builtins", "set"],
            "kwargs": {"__set_items__": ["msg1", "msg2"]},
        }
    }

    result = serializer.loads_typed(("json", json.dumps(legacy_payload).encode()))

    assert result["ids"] == {"msg1", "msg2"}


def test_checkpoint_dataclass_state_roundtrip(
    redis_url: str,
) -> None:
    """Tests that dataclass checkpoint state roundtrips through Redis.

    Intended behavior: RedisSaver should persist and restore dataclass channel
    values as their original type, not as plain dictionaries.
    """
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "context": ConstructorEnvelopePayload(value="persisted")
    }
    checkpoint["channel_versions"] = {"context": "1"}

    thread_id = f"constructor-envelope-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    metadata: CheckpointMetadata = {"source": "test", "step": 1, "writes": {}}

    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saved_config = saver.put(
            config,
            checkpoint,
            metadata,
            {"context": "1"},
        )
        restored = saver.get(saved_config)

    assert restored is not None
    payload = restored["channel_values"]["context"]
    assert payload == ConstructorEnvelopePayload(value="persisted")
