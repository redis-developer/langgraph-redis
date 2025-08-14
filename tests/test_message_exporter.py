"""Test message exporter functionality."""

from typing import Any, Dict, List
from uuid import uuid4

import orjson
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver


class TestLangChainRecipe:
    """Test the default LangChain message recipe."""

    def test_extract_human_message_object(self):
        """Test extracting from a HumanMessage object."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        msg = HumanMessage(content="Hello, AI!", id="msg-123", name="John")

        result = recipe.extract(msg)

        assert result is not None
        assert result["role"] == "human"
        assert result["content"] == "Hello, AI!"
        assert result["type"] == "HumanMessage"
        assert result["id"] == "msg-123"
        assert result["metadata"]["name"] == "John"

    def test_extract_ai_message_object(self):
        """Test extracting from an AIMessage object."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        msg = AIMessage(content="Hello, human!", id="ai-456")

        result = recipe.extract(msg)

        assert result["role"] == "ai"
        assert result["content"] == "Hello, human!"
        assert result["type"] == "AIMessage"
        assert result["id"] == "ai-456"

    def test_extract_tool_message_with_tool_calls(self):
        """Test extracting from a ToolMessage with tool call info."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        msg = ToolMessage(
            content="Weather is sunny", tool_call_id="call-789", name="weather_tool"
        )

        result = recipe.extract(msg)

        assert result["role"] == "tool"
        assert result["content"] == "Weather is sunny"
        assert result["type"] == "ToolMessage"
        assert result["metadata"]["name"] == "weather_tool"

    def test_extract_serialized_langchain_format(self):
        """Test extracting from serialized LangChain format."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        serialized_msg = {
            "lc": 1,
            "type": "constructor",
            "id": ["langchain", "schema", "messages", "HumanMessage"],
            "kwargs": {
                "content": "What's the weather?",
                "type": "human",
                "id": "msg-abc",
            },
        }

        result = recipe.extract(serialized_msg)

        assert result["role"] == "human"
        assert result["content"] == "What's the weather?"
        assert result["type"] == "HumanMessage"
        assert result["id"] == "msg-abc"

    def test_extract_simple_dict_format(self):
        """Test extracting from simple dict with role and content."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        msg = {"role": "user", "content": "Hello"}

        result = recipe.extract(msg)

        assert result == msg  # Should return as-is

    def test_extract_plain_string(self):
        """Test extracting from plain string."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()
        msg = "Just a plain message"

        result = recipe.extract(msg)

        assert result["role"] == "unknown"
        assert result["content"] == "Just a plain message"
        assert result["type"] == "string"

    def test_extract_returns_none_for_unsupported(self):
        """Test that extract returns None for unsupported types."""
        from langgraph.checkpoint.redis.message_exporter import LangChainRecipe

        recipe = LangChainRecipe()

        assert recipe.extract(123) is None
        assert recipe.extract([]) is None
        assert recipe.extract(None) is None


class TestMessageExporter:
    """Test the main MessageExporter class."""

    def test_export_from_latest_checkpoint(self):
        """Test exporting messages from the latest checkpoint."""
        redis_container = RedisContainer("redis:8")
        redis_container.start()

        try:
            redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

            # Setup
            saver = RedisSaver(redis_url)
            saver.setup()

            thread_id = str(uuid4())
            messages = [
                HumanMessage(content="Hello", id="msg-1"),
                AIMessage(content="Hi there", id="msg-2"),
            ]

            # Create and save checkpoint
            checkpoint = create_checkpoint(
                checkpoint=empty_checkpoint(), channels={"messages": messages}, step=1
            )
            checkpoint["channel_values"]["messages"] = messages

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            saver.put(
                config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
            )

            # Test export
            from langgraph.checkpoint.redis.message_exporter import MessageExporter

            exporter = MessageExporter(saver)
            result = exporter.export(thread_id=thread_id)

            assert len(result) == 2
            assert result[0]["role"] == "human"
            assert result[0]["content"] == "Hello"
            assert result[0]["id"] == "msg-1"
            assert result[1]["role"] == "ai"
            assert result[1]["content"] == "Hi there"
            assert result[1]["id"] == "msg-2"

        finally:
            redis_container.stop()

    def test_export_from_specific_checkpoint(self):
        """Test exporting from a specific checkpoint ID."""
        redis_container = RedisContainer("redis:8")
        redis_container.start()

        try:
            redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

            saver = RedisSaver(redis_url)
            saver.setup()

            thread_id = str(uuid4())

            # Create first checkpoint
            messages1 = [HumanMessage(content="First", id="msg-1")]
            checkpoint1 = create_checkpoint(
                checkpoint=empty_checkpoint(), channels={"messages": messages1}, step=1
            )
            checkpoint1["channel_values"]["messages"] = messages1

            config1 = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "checkpoint-1",
                }
            }

            saved_config1 = saver.put(
                config1, checkpoint1, {"source": "test", "step": 1, "writes": {}}, {}
            )

            # Create second checkpoint
            messages2 = [
                HumanMessage(content="First", id="msg-1"),
                AIMessage(content="Response", id="msg-2"),
                HumanMessage(content="Second", id="msg-3"),
            ]
            checkpoint2 = create_checkpoint(
                checkpoint=empty_checkpoint(), channels={"messages": messages2}, step=2
            )
            checkpoint2["channel_values"]["messages"] = messages2

            config2 = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": "checkpoint-2",
                }
            }

            saver.put(
                config2, checkpoint2, {"source": "test", "step": 2, "writes": {}}, {}
            )

            # Test export from specific checkpoint
            from langgraph.checkpoint.redis.message_exporter import MessageExporter

            exporter = MessageExporter(saver)

            # Export from first checkpoint
            result1 = exporter.export(
                thread_id=thread_id,
                checkpoint_id=saved_config1["configurable"]["checkpoint_id"],
            )
            assert len(result1) == 1
            assert result1[0]["content"] == "First"

            # Export from latest (should be checkpoint-2)
            result2 = exporter.export(thread_id=thread_id)
            assert len(result2) == 3
            assert result2[2]["content"] == "Second"

        finally:
            redis_container.stop()

    def test_export_empty_checkpoint(self):
        """Test exporting from checkpoint with no messages."""
        redis_container = RedisContainer("redis:8")
        redis_container.start()

        try:
            redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

            saver = RedisSaver(redis_url)
            saver.setup()

            thread_id = str(uuid4())

            # Create checkpoint with no messages
            checkpoint = create_checkpoint(
                checkpoint=empty_checkpoint(), channels={}, step=1
            )

            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

            saver.put(
                config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
            )

            from langgraph.checkpoint.redis.message_exporter import MessageExporter

            exporter = MessageExporter(saver)
            result = exporter.export(thread_id=thread_id)

            assert result == []

        finally:
            redis_container.stop()

    def test_export_nonexistent_thread(self):
        """Test exporting from a thread that doesn't exist."""
        redis_container = RedisContainer("redis:8")
        redis_container.start()

        try:
            redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

            saver = RedisSaver(redis_url)
            saver.setup()

            from langgraph.checkpoint.redis.message_exporter import MessageExporter

            exporter = MessageExporter(saver)
            result = exporter.export(thread_id="nonexistent-thread")

            assert result == []

        finally:
            redis_container.stop()

    def test_export_thread_all_checkpoints(self):
        """Test exporting all messages from all checkpoints in a thread."""
        redis_container = RedisContainer("redis:8")
        redis_container.start()

        try:
            redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

            saver = RedisSaver(redis_url)
            saver.setup()

            thread_id = str(uuid4())

            # Create multiple checkpoints
            for i in range(3):
                messages = [HumanMessage(content=f"Message {i}", id=f"msg-{i}")]
                checkpoint = create_checkpoint(
                    checkpoint=empty_checkpoint(),
                    channels={"messages": messages},
                    step=i,
                )
                checkpoint["channel_values"]["messages"] = messages

                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": f"checkpoint-{i}",
                    }
                }

                saver.put(
                    config, checkpoint, {"source": "test", "step": i, "writes": {}}, {}
                )

            from langgraph.checkpoint.redis.message_exporter import MessageExporter

            exporter = MessageExporter(saver)
            result = exporter.export_thread(thread_id=thread_id)

            assert "thread_id" in result
            assert result["thread_id"] == thread_id
            assert "messages" in result
            assert len(result["messages"]) == 3
            assert "export_timestamp" in result

            # Check messages have checkpoint info
            for msg in result["messages"]:
                assert "checkpoint_id" in msg
                assert "checkpoint_ts" in msg

        finally:
            redis_container.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
