"""Regression tests for issue #189 JsonPlusRedisSerializer compatibility."""

import dataclasses
from uuid import uuid4

import pytest
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


@dataclasses.dataclass
class ConstructorEnvelopePayload:
    value: str


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
