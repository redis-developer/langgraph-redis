"""Tests for issue #128: HumanMessage serialization in RedisStore.

This test reproduces the issue where storing LangChain message objects in
RedisStore fails with:
    redisvl.exceptions.RedisVLerror: failed to load data: Object of type
    HumanMessage is not JSON serializable

The root cause is that RedisStore passes values directly to redisvl's
SearchIndex.load() which uses standard JSON serialization, unable to
handle LangChain message objects.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.store.redis import RedisStore


@pytest.fixture(scope="function")
def store(redis_url: str) -> Iterator[RedisStore]:
    """Fixture to create a Redis store."""
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        yield store


class TestIssue128MessageSerialization:
    """Test suite for issue #128: HumanMessage serialization in RedisStore."""

    def test_store_human_message_in_value(self, store: RedisStore) -> None:
        """Test storing a HumanMessage object directly in the store value.

        This is the core issue: storing LangChain message objects fails because
        redisvl cannot serialize them with standard JSON.
        """
        namespace = ("test", "messages")
        key = "message1"

        # This is the value that causes the issue - contains HumanMessage
        value = {
            "messages": [
                HumanMessage(content="Hello, how are you?"),
            ]
        }

        # This should work but currently fails with:
        # redisvl.exceptions.RedisVLerror: failed to load data:
        # Object of type HumanMessage is not JSON serializable
        store.put(namespace, key, value)

        # Verify we can retrieve it back
        item = store.get(namespace, key)
        assert item is not None
        assert "messages" in item.value
        assert len(item.value["messages"]) == 1

        # Verify the message is properly deserialized
        retrieved_message = item.value["messages"][0]
        assert isinstance(retrieved_message, HumanMessage)
        assert retrieved_message.content == "Hello, how are you?"

    def test_store_multiple_message_types(self, store: RedisStore) -> None:
        """Test storing multiple message types in the store value."""
        namespace = ("test", "conversation")
        key = "conv1"

        value = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is Python?"),
                AIMessage(content="Python is a programming language."),
            ]
        }

        store.put(namespace, key, value)

        item = store.get(namespace, key)
        assert item is not None
        assert len(item.value["messages"]) == 3

        # Verify message types are preserved
        messages = item.value["messages"]
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)

    def test_store_message_with_additional_kwargs(self, store: RedisStore) -> None:
        """Test storing messages with additional kwargs like name, id, etc."""
        namespace = ("test", "messages")
        key = "msg_with_kwargs"

        value = {
            "messages": [
                HumanMessage(
                    content="Hello!",
                    name="User",
                    id="msg-123",
                    additional_kwargs={"custom": "data"},
                ),
            ]
        }

        store.put(namespace, key, value)

        item = store.get(namespace, key)
        assert item is not None

        retrieved_message = item.value["messages"][0]
        assert isinstance(retrieved_message, HumanMessage)
        assert retrieved_message.content == "Hello!"
        assert retrieved_message.name == "User"
        assert retrieved_message.id == "msg-123"
        assert retrieved_message.additional_kwargs == {"custom": "data"}

    def test_store_nested_message_structure(self, store: RedisStore) -> None:
        """Test storing messages in nested data structures."""
        namespace = ("test", "nested")
        key = "nested1"

        value = {
            "conversation": {
                "id": "conv-1",
                "messages": [
                    HumanMessage(content="First message"),
                ],
                "metadata": {
                    "last_message": HumanMessage(content="Last message"),
                },
            }
        }

        store.put(namespace, key, value)

        item = store.get(namespace, key)
        assert item is not None

        # Verify nested messages are preserved
        conv = item.value["conversation"]
        assert isinstance(conv["messages"][0], HumanMessage)
        assert isinstance(conv["metadata"]["last_message"], HumanMessage)

    def test_store_ai_message_with_tool_calls(self, store: RedisStore) -> None:
        """Test storing AIMessage with tool calls."""
        namespace = ("test", "tools")
        key = "tool_call"

        value = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "name": "get_weather",
                            "args": {"location": "NYC"},
                        }
                    ],
                ),
            ]
        }

        store.put(namespace, key, value)

        item = store.get(namespace, key)
        assert item is not None

        retrieved_message = item.value["messages"][0]
        assert isinstance(retrieved_message, AIMessage)
        assert len(retrieved_message.tool_calls) == 1
        assert retrieved_message.tool_calls[0]["name"] == "get_weather"

    def test_search_with_message_values(self, store: RedisStore) -> None:
        """Test searching for items that contain message values.

        Note: When values contain non-JSON-serializable objects like HumanMessage,
        the entire value is serialized using the serde wrapper. This means filters
        on nested fields won't work for such values. This test verifies that
        search works and messages are properly deserialized.
        """
        namespace = ("test", "searchable")

        # Store items with messages
        for i in range(3):
            store.put(
                namespace,
                f"msg{i}",
                {
                    "topic": f"topic_{i}",
                    "messages": [HumanMessage(content=f"Message {i}")],
                },
            )

        # Search without filter (filters don't work on serialized values)
        results = store.search(namespace)
        assert len(results) == 3

        # Verify messages are properly deserialized in all results
        for result in results:
            assert "messages" in result.value
            assert isinstance(result.value["messages"][0], HumanMessage)

        # Verify we can get specific items by key
        item = store.get(namespace, "msg1")
        assert item is not None
        assert item.value["topic"] == "topic_1"
        assert isinstance(item.value["messages"][0], HumanMessage)
        assert item.value["messages"][0].content == "Message 1"
