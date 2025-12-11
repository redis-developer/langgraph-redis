"""Async tests for issue #128: HumanMessage serialization in AsyncRedisStore.

This test suite mirrors the sync tests to ensure the async store implementation
properly handles LangChain message serialization.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.store.redis.aio import AsyncRedisStore


@pytest.fixture
async def async_store(redis_url: str) -> AsyncIterator[AsyncRedisStore]:
    """Fixture to create an async Redis store."""
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()
        yield store


class TestIssue128AsyncMessageSerialization:
    """Async test suite for issue #128: HumanMessage serialization."""

    @pytest.mark.asyncio
    async def test_store_human_message_in_value(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test storing a HumanMessage object in async store."""
        namespace = ("test", "async_messages")
        key = "message1"

        value = {
            "messages": [
                HumanMessage(content="Hello from async!"),
            ]
        }

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert "messages" in item.value
        assert len(item.value["messages"]) == 1

        retrieved_message = item.value["messages"][0]
        assert isinstance(retrieved_message, HumanMessage)
        assert retrieved_message.content == "Hello from async!"

    @pytest.mark.asyncio
    async def test_store_multiple_message_types(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test storing multiple message types in async store."""
        namespace = ("test", "async_conversation")
        key = "conv1"

        value = {
            "messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="What is Python?"),
                AIMessage(content="Python is a programming language."),
            ]
        }

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert len(item.value["messages"]) == 3

        messages = item.value["messages"]
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)

    @pytest.mark.asyncio
    async def test_store_ai_message_with_tool_calls(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test storing AIMessage with tool calls in async store."""
        namespace = ("test", "async_tools")
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

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None

        retrieved_message = item.value["messages"][0]
        assert isinstance(retrieved_message, AIMessage)
        assert len(retrieved_message.tool_calls) == 1
        assert retrieved_message.tool_calls[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_search_with_message_values(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test async search for items containing message values."""
        namespace = ("test", "async_searchable")

        for i in range(3):
            await async_store.aput(
                namespace,
                f"msg{i}",
                {
                    "topic": f"topic_{i}",
                    "messages": [HumanMessage(content=f"Async Message {i}")],
                },
            )

        results = await async_store.asearch(namespace)
        assert len(results) == 3

        for result in results:
            assert "messages" in result.value
            assert isinstance(result.value["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_backward_compatibility_plain_json(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test that plain JSON values work correctly in async store."""
        namespace = ("test", "async_plain_json")
        key = "simple_value"

        value = {
            "name": "async_test",
            "count": 42,
            "tags": ["a", "b", "c"],
        }

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert item.value == value

        # Verify filters work on plain JSON values
        results = await async_store.asearch(namespace, filter={"name": "async_test"})
        assert len(results) == 1
        assert results[0].key == key

    @pytest.mark.asyncio
    async def test_serde_key_collision_prevention(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test that user data with serde-like keys is handled correctly."""
        namespace = ("test", "async_collision")
        key = "user_data_with_serde_keys"

        value = {
            "__serde_type__": "user_defined_type",
            "__serde_data__": "user_defined_data",
            "extra_key": "this makes it not a serde wrapper",
        }

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert item.value == value
        assert item.value["extra_key"] == "this makes it not a serde wrapper"

    @pytest.mark.asyncio
    async def test_bytes_value_serialization(
        self, async_store: AsyncRedisStore
    ) -> None:
        """Test that bytes values are properly serialized in async store."""
        namespace = ("test", "async_bytes")
        key = "binary_data"

        value = {
            "data": b"async binary content",
            "name": "test",
        }

        await async_store.aput(namespace, key, value)

        item = await async_store.aget(namespace, key)
        assert item is not None
        assert item.value["data"] == b"async binary content"
        assert item.value["name"] == "test"
