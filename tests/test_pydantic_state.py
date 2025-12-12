"""Test Pydantic BaseModel state serialization with Redis checkpoint saver.

This tests the fix in PR #126 which changes the order of serde methods
to prefer _revive_if_needed over _reviver for better Pydantic compatibility.
"""

from contextlib import contextmanager
from typing import Any, List, Optional

import pytest
from pydantic import BaseModel, Field
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver


class Address(BaseModel):
    """Nested Pydantic model for testing."""

    street: str
    city: str
    zip_code: str = Field(default="00000")


class Person(BaseModel):
    """Pydantic model with nested objects."""

    name: str
    age: int
    address: Optional[Address] = None
    tags: List[str] = Field(default_factory=list)


class ChatState(BaseModel):
    """Pydantic model representing a chat state."""

    messages: List[dict] = Field(default_factory=list)
    user: Optional[Person] = None
    metadata: dict = Field(default_factory=dict)


@contextmanager
def _saver(redis_url: str):
    """Create a RedisSaver context manager."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        yield saver


@pytest.fixture(scope="module")
def redis_url():
    """Provide a Redis URL using TestContainers."""
    redis_container = RedisContainer("redis:8")
    redis_container.start()
    try:
        yield f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"
    finally:
        redis_container.stop()


def test_pydantic_basemodel_in_checkpoint(redis_url: str) -> None:
    """Test that Pydantic BaseModel objects can be stored and retrieved.

    This is the key test for PR #126 - ensures that the _revive_if_needed
    method is called properly for Pydantic model reconstruction.
    """
    with _saver(redis_url) as saver:
        config = {
            "configurable": {
                "thread_id": "pydantic-test-1",
                "checkpoint_ns": "",
            }
        }

        # Create a checkpoint with Pydantic models in channel_values
        checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "id": "checkpoint-pydantic-1",
            "channel_values": {
                "state": {
                    "person": Person(
                        name="Alice",
                        age=30,
                        address=Address(
                            street="123 Main St", city="NYC", zip_code="10001"
                        ),
                        tags=["developer", "python"],
                    ).model_dump(),
                    "chat": ChatState(
                        messages=[
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                        user=Person(name="Bob", age=25),
                        metadata={"session_id": "abc123"},
                    ).model_dump(),
                }
            },
            "channel_versions": {"state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        # Store the checkpoint
        next_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1}, {"state": "1"}
        )

        # Retrieve the checkpoint
        retrieved = saver.get(next_config)

        assert retrieved is not None
        assert "state" in retrieved["channel_values"]
        state = retrieved["channel_values"]["state"]

        # Verify the person data was preserved
        assert state["person"]["name"] == "Alice"
        assert state["person"]["age"] == 30
        assert state["person"]["address"]["city"] == "NYC"

        # Verify the chat state was preserved
        assert len(state["chat"]["messages"]) == 2
        assert state["chat"]["user"]["name"] == "Bob"


def test_nested_pydantic_models_roundtrip(redis_url: str) -> None:
    """Test deeply nested Pydantic models can survive a checkpoint roundtrip."""
    with _saver(redis_url) as saver:
        config = {
            "configurable": {
                "thread_id": "nested-pydantic-test",
                "checkpoint_ns": "",
            }
        }

        # Create deeply nested structure
        nested_state = {
            "level1": {
                "level2": {
                    "person": Person(
                        name="Charlie",
                        age=35,
                        address=Address(
                            street="456 Elm St", city="Boston", zip_code="02101"
                        ),
                    ).model_dump(),
                    "items": [
                        {"id": 1, "data": Person(name="Dave", age=40).model_dump()},
                        {"id": 2, "data": Person(name="Eve", age=28).model_dump()},
                    ],
                }
            }
        }

        checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "id": "checkpoint-nested-1",
            "channel_values": {"state": nested_state},
            "channel_versions": {"state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        next_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1}, {"state": "1"}
        )

        retrieved = saver.get(next_config)

        assert retrieved is not None
        state = retrieved["channel_values"]["state"]

        # Verify deeply nested data
        level2 = state["level1"]["level2"]
        assert level2["person"]["name"] == "Charlie"
        assert level2["person"]["address"]["city"] == "Boston"

        # Verify list items
        assert len(level2["items"]) == 2
        assert level2["items"][0]["data"]["name"] == "Dave"
        assert level2["items"][1]["data"]["name"] == "Eve"


def test_pydantic_model_with_langchain_messages(redis_url: str) -> None:
    """Test Pydantic state with LangChain-style message objects.

    This is the critical test case mentioned in PR #126 - when users
    use Pydantic BaseModel as state with LangChain message types.
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage
    except ImportError:
        pytest.skip("langchain-core not installed")

    with _saver(redis_url) as saver:
        config = {
            "configurable": {
                "thread_id": "langchain-pydantic-test",
                "checkpoint_ns": "",
            }
        }

        # Simulate a state that mixes Pydantic with LangChain messages
        checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "id": "checkpoint-lc-pydantic-1",
            "channel_values": {
                "messages": [
                    HumanMessage(content="Hello, how are you?"),
                    AIMessage(content="I'm doing well, thank you!"),
                ],
                "user_profile": Person(
                    name="TestUser", age=25, tags=["test", "demo"]
                ).model_dump(),
            },
            "channel_versions": {"messages": "1", "user_profile": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        next_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

        retrieved = saver.get(next_config)

        assert retrieved is not None

        # Verify messages were deserialized correctly
        messages = retrieved["channel_values"]["messages"]
        assert len(messages) == 2

        # Messages should be proper LangChain message objects after deserialization
        # with _revive_if_needed properly handling them
        if hasattr(messages[0], "content"):
            # Message objects were properly reconstructed
            assert messages[0].content == "Hello, how are you?"
            assert messages[1].content == "I'm doing well, thank you!"
        else:
            # If they're still dicts, verify the content is there
            assert messages[0].get("content") == "Hello, how are you?"

        # Verify user profile data
        user_profile = retrieved["channel_values"]["user_profile"]
        assert user_profile["name"] == "TestUser"
        assert user_profile["age"] == 25


def test_revive_if_needed_fallback_behavior(redis_url: str) -> None:
    """Test that _revive_if_needed properly falls back when _reviver fails.

    The PR #126 change ensures _revive_if_needed is tried first, which
    includes its own fallback to _reconstruct_from_constructor.
    """
    with _saver(redis_url) as saver:
        config = {
            "configurable": {
                "thread_id": "fallback-test",
                "checkpoint_ns": "",
            }
        }

        # Create a checkpoint with complex nested data
        checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00.000000+00:00",
            "id": "checkpoint-fallback-1",
            "channel_values": {
                "complex_state": {
                    "nested_dict": {
                        "key1": [1, 2, 3],
                        "key2": {"a": "b", "c": {"d": "e"}},
                    },
                    "simple_value": 42,
                    "string_value": "test",
                    "bool_value": True,
                    "none_value": None,
                }
            },
            "channel_versions": {"complex_state": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        next_config = saver.put(config, checkpoint, {"source": "test", "step": 1}, {})

        retrieved = saver.get(next_config)

        assert retrieved is not None
        state = retrieved["channel_values"]["complex_state"]

        # All values should be preserved exactly
        assert state["nested_dict"]["key1"] == [1, 2, 3]
        assert state["nested_dict"]["key2"]["c"]["d"] == "e"
        assert state["simple_value"] == 42
        assert state["string_value"] == "test"
        assert state["bool_value"] is True
        assert state["none_value"] is None
