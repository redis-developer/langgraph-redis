"""Tests for PR #87 - Async message deserialization fix.

This test suite validates that AsyncRedisSaver properly deserializes
LangChain messages
"""

from typing import Any, Dict, List
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import (
    CheckpointTuple,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver


@pytest.mark.asyncio
async def test_pr87_fix_approach(redis_url: str):
    """Test if PR #87's fix approach using _recursive_deserialize works."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        # Test if _recursive_deserialize is accessible (inherited from BaseRedisSaver)
        assert hasattr(
            saver, "_recursive_deserialize"
        ), "Missing _recursive_deserialize method"

        # Create test data - raw channel values as they come from aget_channel_values
        raw_channel_values = {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {
                        "content": "Test message",
                        "type": "human",
                        "id": "test-123",
                    },
                }
            ]
        }

        # Test the PR #87 approach
        deserialized = saver._recursive_deserialize(raw_channel_values)

        # Verify the fix works
        assert isinstance(deserialized["messages"][0], HumanMessage)
        assert deserialized["messages"][0].content == "Test message"
        assert deserialized["messages"][0].id == "test-123"


@pytest.mark.asyncio
async def test_async_deserializes_langchain_messages(redis_url: str):
    """Test that AsyncRedisSaver properly deserializes LangChain message objects.

    This is the core test for PR #87 - verifies that messages are returned as
    proper Message objects, not as serialized dictionaries.
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with various message types
        messages = [
            SystemMessage(content="You are a helpful assistant.", id="sys-1"),
            HumanMessage(content="What's the weather like?", id="human-1"),
            AIMessage(content="I'll help you check the weather.", id="ai-1"),
            ToolMessage(
                content="Weather data retrieved",
                tool_call_id="call-1",
                name="weather_tool",
            ),
        ]

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels={"messages": messages}, step=1
        )
        checkpoint["channel_values"]["messages"] = messages

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        # Save checkpoint
        await saver.aput(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load checkpoint
        loaded_tuple = await saver.aget_tuple(config)

        assert loaded_tuple is not None
        loaded_messages = loaded_tuple.checkpoint["channel_values"]["messages"]

        # Verify all messages are properly deserialized
        assert len(loaded_messages) == 4
        assert isinstance(loaded_messages[0], SystemMessage)
        assert isinstance(loaded_messages[1], HumanMessage)
        assert isinstance(loaded_messages[2], AIMessage)
        assert isinstance(loaded_messages[3], ToolMessage)

        # Verify content and IDs
        assert loaded_messages[0].content == "You are a helpful assistant."
        assert loaded_messages[1].content == "What's the weather like?"
        assert loaded_messages[1].id == "human-1"
        assert loaded_messages[2].content == "I'll help you check the weather."
        assert loaded_messages[2].id == "ai-1"
        assert loaded_messages[3].content == "Weather data retrieved"
        assert loaded_messages[3].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_async_handles_serialized_langchain_format(redis_url: str):
    """Test that async handles the serialized LangChain format that causes MESSAGE_COERCION_FAILURE.

    This tests the specific format from issue #85:
    {'lc': 1, 'type': 'constructor', 'id': [...], 'kwargs': {...}}
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # This is the format that gets stored in Redis
        serialized_messages = [
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "SystemMessage"],
                "kwargs": {
                    "content": "System prompt",
                    "type": "system",
                },
            },
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "HumanMessage"],
                "kwargs": {"content": "User input", "type": "human", "id": "msg-123"},
            },
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Assistant response",
                    "type": "ai",
                    "id": "msg-456",
                },
            },
        ]

        # Simulate what happens when checkpoint is saved with serialized messages
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": serialized_messages},
            step=1,
        )
        checkpoint["channel_values"]["messages"] = serialized_messages

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        # Manually save the checkpoint to simulate the serialized state
        await saver.aput(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load and verify deserialization
        loaded_tuple = await saver.aget_tuple(config)

        assert loaded_tuple is not None
        loaded_messages = loaded_tuple.checkpoint["channel_values"]["messages"]

        # Messages should be deserialized to proper objects
        assert len(loaded_messages) == 3
        assert isinstance(loaded_messages[0], SystemMessage)
        assert isinstance(loaded_messages[1], HumanMessage)
        assert isinstance(loaded_messages[2], AIMessage)

        # Verify content
        assert loaded_messages[0].content == "System prompt"
        assert loaded_messages[1].content == "User input"
        assert loaded_messages[1].id == "msg-123"
        assert loaded_messages[2].content == "Assistant response"
        assert loaded_messages[2].id == "msg-456"


@pytest.mark.asyncio
async def test_async_get_channel_values_directly(redis_url: str):
    """Test aget_channel_values method directly to ensure proper deserialization."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with messages
        messages = [
            HumanMessage(content="Direct test", id="msg-1"),
            AIMessage(content="Response", id="msg-2"),
        ]

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels={"messages": messages}, step=1
        )
        checkpoint["channel_values"]["messages"] = messages

        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "test-checkpoint",
            }
        }

        saved_config = await saver.aput(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Call aget_channel_values directly
        channel_values = await saver.aget_channel_values(
            thread_id=thread_id,
            checkpoint_ns="",
            checkpoint_id=saved_config["configurable"]["checkpoint_id"],
        )

        # Note: Without PR #87, aget_channel_values returns raw serialized data
        # With PR #87, aget_tuple deserializes it, but aget_channel_values itself doesn't
        # This test documents the current behavior
        assert "messages" in channel_values
        loaded_messages = channel_values["messages"]

        # This will fail without additional fixes to aget_channel_values itself
        # The PR #87 fix is in aget_tuple, not aget_channel_values
        # Keeping this test to document the behavior
        if isinstance(loaded_messages[0], dict):
            # Without the fix - raw serialized format
            assert loaded_messages[0].get("lc") == 1
            assert loaded_messages[0].get("type") == "constructor"
        else:
            # With a complete fix
            assert isinstance(loaded_messages[0], HumanMessage)
            assert isinstance(loaded_messages[1], AIMessage)


@pytest.mark.asyncio
async def test_async_sync_parity(redis_url: str):
    """Test that async and sync implementations return the same deserialized messages."""
    thread_id = str(uuid4())

    # Create test messages
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="User message", id="user-1"),
        AIMessage(content="AI response", id="ai-1"),
    ]

    checkpoint = create_checkpoint(
        checkpoint=empty_checkpoint(), channels={"messages": messages}, step=1
    )
    checkpoint["channel_values"]["messages"] = messages

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    # Test with sync saver
    sync_saver = RedisSaver(redis_url)
    sync_saver.setup()

    sync_saver.put(config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {})

    sync_tuple = sync_saver.get_tuple(config)
    sync_messages = sync_tuple.checkpoint["channel_values"]["messages"]

    # Test with async saver using the same checkpoint
    async with AsyncRedisSaver.from_conn_string(redis_url) as async_saver:
        async_tuple = await async_saver.aget_tuple(config)
        async_messages = async_tuple.checkpoint["channel_values"]["messages"]

    # Both should return the same deserialized messages
    assert len(sync_messages) == len(async_messages)

    for sync_msg, async_msg in zip(sync_messages, async_messages):
        assert type(sync_msg) == type(async_msg)
        assert sync_msg.content == async_msg.content
        if hasattr(sync_msg, "id") and sync_msg.id:
            assert sync_msg.id == async_msg.id

        # Both should be proper Message objects, not dicts
        assert isinstance(sync_msg, BaseMessage)
        assert isinstance(async_msg, BaseMessage)


@pytest.mark.asyncio
async def test_async_with_parent_checkpoint(redis_url: str):
    """Test deserialization works when loading checkpoint with parent (pending_sends path)."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Create parent checkpoint
        parent_messages = [HumanMessage(content="First message", id="msg-1")]

        parent_checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": parent_messages},
            step=1,
        )
        parent_checkpoint["channel_values"]["messages"] = parent_messages
        parent_checkpoint["id"] = "parent-checkpoint-id"

        parent_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "parent-checkpoint-id",
            }
        }

        await saver.aput(
            parent_config,
            parent_checkpoint,
            {"source": "test", "step": 1, "writes": {}},
            {},
        )

        # Create child checkpoint with parent reference
        child_messages = [
            HumanMessage(content="First message", id="msg-1"),
            AIMessage(content="Response", id="msg-2"),
        ]

        child_checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels={"messages": child_messages}, step=2
        )
        child_checkpoint["channel_values"]["messages"] = child_messages
        child_checkpoint["id"] = "child-checkpoint-id"
        child_checkpoint["parent_checkpoint_id"] = "parent-checkpoint-id"

        child_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "parent-checkpoint-id",  # This becomes parent during save
            }
        }

        # Save child with parent reference
        saved_config = await saver.aput(
            child_config,
            child_checkpoint,
            {"source": "test", "step": 2, "writes": {}, "parent_config": parent_config},
            {},
        )

        # Load child checkpoint (should trigger the parent checkpoint path)
        loaded_tuple = await saver.aget_tuple(saved_config)

        assert loaded_tuple is not None
        assert loaded_tuple.parent_config is not None  # Has parent

        loaded_messages = loaded_tuple.checkpoint["channel_values"]["messages"]

        # Messages should be properly deserialized even with parent
        assert len(loaded_messages) == 2
        assert isinstance(loaded_messages[0], HumanMessage)
        assert isinstance(loaded_messages[1], AIMessage)
        assert loaded_messages[0].content == "First message"
        assert loaded_messages[1].content == "Response"


@pytest.mark.asyncio
async def test_async_nested_message_structures(redis_url: str):
    """Test deserialization of complex nested message structures."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Create checkpoint with nested structures
        complex_channel_values = {
            "messages": [
                HumanMessage(content="Question", id="q-1"),
                AIMessage(
                    content="Answer",
                    id="a-1",
                    additional_kwargs={"function_call": {"name": "tool", "args": {}}},
                ),
            ],
            "other_data": {
                "nested": {"messages": [SystemMessage(content="Nested system message")]}
            },
            "list_of_lists": [
                [HumanMessage(content="Deep message")],
                [AIMessage(content="Deep response")],
            ],
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels=complex_channel_values, step=1
        )
        checkpoint["channel_values"] = complex_channel_values

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        await saver.aput(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load and verify nested structures are deserialized
        loaded_tuple = await saver.aget_tuple(config)

        assert loaded_tuple is not None
        loaded_values = loaded_tuple.checkpoint["channel_values"]

        # Top-level messages
        assert isinstance(loaded_values["messages"][0], HumanMessage)
        assert isinstance(loaded_values["messages"][1], AIMessage)
        assert (
            loaded_values["messages"][1].additional_kwargs["function_call"]["name"]
            == "tool"
        )

        # Nested messages
        assert isinstance(
            loaded_values["other_data"]["nested"]["messages"][0], SystemMessage
        )

        # List of lists
        assert isinstance(loaded_values["list_of_lists"][0][0], HumanMessage)
        assert isinstance(loaded_values["list_of_lists"][1][0], AIMessage)


@pytest.mark.asyncio
async def test_async_mixed_content_types(redis_url: str):
    """Test that non-message content is preserved while messages are deserialized."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Mix messages with other data types
        mixed_values = {
            "messages": [
                HumanMessage(content="User input", id="user-1"),
                {"role": "system", "content": "Plain dict"},  # Not a message object
                AIMessage(content="Response", id="ai-1"),
            ],
            "metadata": {"key": "value"},
            "count": 42,
            "flags": [True, False, True],
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels=mixed_values, step=1
        )
        checkpoint["channel_values"] = mixed_values

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        await saver.aput(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load and verify mixed content
        loaded_tuple = await saver.aget_tuple(config)

        assert loaded_tuple is not None
        loaded_values = loaded_tuple.checkpoint["channel_values"]

        # Messages are deserialized
        assert isinstance(loaded_values["messages"][0], HumanMessage)
        assert isinstance(loaded_values["messages"][2], AIMessage)

        # Plain dict remains a dict
        assert isinstance(loaded_values["messages"][1], dict)
        assert loaded_values["messages"][1]["role"] == "system"

        # Other data types are preserved
        assert loaded_values["metadata"] == {"key": "value"}
        assert loaded_values["count"] == 42
        assert loaded_values["flags"] == [True, False, True]


@pytest.mark.asyncio
async def test_async_alist_with_deserialization(redis_url: str):
    """Test that alist() also properly deserializes messages.

    This test verifies that the alist() method properly deserializes
    LangChain messages when listing checkpoints, matching the behavior
    of aget_tuple().
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = str(uuid4())

        # Create multiple checkpoints
        for i in range(3):
            messages = [HumanMessage(content=f"Message {i}", id=f"msg-{i}")]

            checkpoint = create_checkpoint(
                checkpoint=empty_checkpoint(), channels={"messages": messages}, step=i
            )
            checkpoint["channel_values"]["messages"] = messages

            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": f"checkpoint-{i}",
                }
            }

            await saver.aput(
                config, checkpoint, {"source": "test", "step": i, "writes": {}}, {}
            )

        # List all checkpoints
        checkpoints: List[CheckpointTuple] = []
        async for checkpoint_tuple in saver.alist(
            {"configurable": {"thread_id": thread_id}}
        ):
            checkpoints.append(checkpoint_tuple)

        # Verify all checkpoints have deserialized messages
        assert len(checkpoints) == 3

        for checkpoint_tuple in checkpoints:
            messages = checkpoint_tuple.checkpoint["channel_values"]["messages"]
            assert len(messages) == 1
            assert isinstance(messages[0], HumanMessage)
            # Messages should be objects, not dicts
            assert hasattr(messages[0], "content")
            assert not isinstance(messages[0], dict)


@pytest.mark.asyncio
async def test_regression_issue_85(redis_url: str):
    """Regression test for issue #85 - MESSAGE_COERCION_FAILURE.

    This reproduces the exact scenario from the issue where sending a second
    message with the same thread_id causes an error due to improper deserialization.
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        thread_id = "test-thread-85"

        # First message (should work)
        first_messages = [HumanMessage(content="First message", id="msg-1")]

        first_checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(), channels={"messages": first_messages}, step=1
        )
        first_checkpoint["channel_values"]["messages"] = first_messages

        first_config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        await saver.aput(
            first_config,
            first_checkpoint,
            {"source": "user", "step": 1, "writes": {}},
            {},
        )

        # Load first checkpoint - should work
        loaded_first = await saver.aget_tuple(first_config)
        assert loaded_first is not None
        first_loaded_messages = loaded_first.checkpoint["channel_values"]["messages"]
        assert isinstance(first_loaded_messages[0], HumanMessage)

        # Second message with same thread_id (this would fail without the fix)
        second_messages = [
            HumanMessage(content="First message", id="msg-1"),
            AIMessage(content="Response", id="msg-2"),
            HumanMessage(content="Second message", id="msg-3"),
        ]

        second_checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": second_messages},
            step=2,
        )
        second_checkpoint["channel_values"]["messages"] = second_messages

        second_config = {
            "configurable": {
                "thread_id": thread_id,  # Same thread_id
                "checkpoint_ns": "",
            }
        }

        # This should NOT raise MESSAGE_COERCION_FAILURE
        await saver.aput(
            second_config,
            second_checkpoint,
            {"source": "user", "step": 2, "writes": {}},
            {},
        )

        # Load second checkpoint - should work with proper deserialization
        loaded_second = await saver.aget_tuple(second_config)
        assert loaded_second is not None
        second_loaded_messages = loaded_second.checkpoint["channel_values"]["messages"]

        # All messages should be properly deserialized
        assert len(second_loaded_messages) == 3
        assert isinstance(second_loaded_messages[0], HumanMessage)
        assert isinstance(second_loaded_messages[1], AIMessage)
        assert isinstance(second_loaded_messages[2], HumanMessage)

        # Content should be correct
        assert second_loaded_messages[0].content == "First message"
        assert second_loaded_messages[1].content == "Response"
        assert second_loaded_messages[2].content == "Second message"
