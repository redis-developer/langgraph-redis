"""Integration test demonstrating MessageExporter usage."""

from uuid import uuid4

import orjson
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import MessageExporter, MessageRecipe, RedisSaver


def test_export_conversation_with_multiple_checkpoints():
    """Test exporting messages from a conversation with multiple checkpoints."""
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # Initialize saver
        saver = RedisSaver(redis_url)
        saver.setup()

        thread_id = str(uuid4())

        # Simulate a conversation with multiple checkpoints

        # Checkpoint 1: Initial user message
        checkpoint1 = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="What's the weather like?", id="msg-1"),
                ]
            },
            step=1,
        )
        checkpoint1["channel_values"]["messages"] = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like?", id="msg-1"),
        ]

        config1 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }
        saved_config1 = saver.put(
            config1, checkpoint1, {"source": "user", "step": 1, "writes": {}}, {}
        )

        # Checkpoint 2: AI response
        checkpoint2 = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="What's the weather like?", id="msg-1"),
                    AIMessage(
                        content="I'd be happy to help with weather information. However, I don't have access to real-time weather data. Could you tell me which city you're interested in?",
                        id="msg-2",
                    ),
                ]
            },
            step=2,
        )
        checkpoint2["channel_values"]["messages"] = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like?", id="msg-1"),
            AIMessage(
                content="I'd be happy to help with weather information. However, I don't have access to real-time weather data. Could you tell me which city you're interested in?",
                id="msg-2",
            ),
        ]

        config2 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-2",
            }
        }
        saver.put(
            config2, checkpoint2, {"source": "assistant", "step": 2, "writes": {}}, {}
        )

        # Checkpoint 3: Follow-up
        checkpoint3 = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={
                "messages": [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content="What's the weather like?", id="msg-1"),
                    AIMessage(
                        content="I'd be happy to help with weather information. However, I don't have access to real-time weather data. Could you tell me which city you're interested in?",
                        id="msg-2",
                    ),
                    HumanMessage(content="San Francisco", id="msg-3"),
                ]
            },
            step=3,
        )
        checkpoint3["channel_values"]["messages"] = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like?", id="msg-1"),
            AIMessage(
                content="I'd be happy to help with weather information. However, I don't have access to real-time weather data. Could you tell me which city you're interested in?",
                id="msg-2",
            ),
            HumanMessage(content="San Francisco", id="msg-3"),
        ]

        config3 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-3",
            }
        }
        saver.put(config3, checkpoint3, {"source": "user", "step": 3, "writes": {}}, {})

        # Now use MessageExporter to extract messages
        exporter = MessageExporter(saver)

        # Export from latest checkpoint
        latest_messages = exporter.export(thread_id=thread_id)

        print("\n=== Latest Checkpoint Messages ===")
        print(orjson.dumps(latest_messages, option=orjson.OPT_INDENT_2).decode())

        assert len(latest_messages) == 4
        assert latest_messages[0]["role"] == "system"
        assert latest_messages[1]["role"] == "human"
        assert latest_messages[2]["role"] == "ai"
        assert latest_messages[3]["content"] == "San Francisco"

        # Export entire thread (all checkpoints)
        thread_export = exporter.export_thread(thread_id=thread_id)

        print("\n=== Full Thread Export ===")
        print(f"Thread ID: {thread_export['thread_id']}")
        print(f"Total unique messages: {len(thread_export['messages'])}")
        print(f"Export timestamp: {thread_export['export_timestamp']}")

        # Should have 6 messages total (system message has no ID so appears 3 times)
        assert len(thread_export["messages"]) == 6

        # Messages should have checkpoint metadata
        for msg in thread_export["messages"]:
            assert "checkpoint_id" in msg
            assert "checkpoint_ts" in msg

        # Export from specific checkpoint (using actual checkpoint ID from saved config)
        early_messages = exporter.export(
            thread_id=thread_id,
            checkpoint_id=saved_config1["configurable"]["checkpoint_id"],
        )
        assert len(early_messages) == 2  # Only system and first human message

        print("\n=== Successfully exported messages for context feeding ===")

    finally:
        redis_container.stop()


def test_export_with_custom_recipe_implementation():
    """Test exporting messages using a custom recipe implementation."""
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        saver = RedisSaver(redis_url)
        saver.setup()

        thread_id = str(uuid4())

        # Create checkpoint with custom message format
        custom_messages = [
            {
                "sender": "user",
                "text": "Hello bot",
                "msg_id": "custom-1",
                "timestamp": "2024-01-01T10:00:00Z",
            },
            {
                "sender": "assistant",
                "text": "Hello! How can I help?",
                "msg_id": "custom-2",
                "timestamp": "2024-01-01T10:00:05Z",
            },
        ]

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": custom_messages},
            step=1,
        )
        checkpoint["channel_values"]["messages"] = custom_messages

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        saver.put(config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {})

        # Create custom recipe implementation
        class CustomFormatRecipe:
            def extract(self, message):
                if isinstance(message, dict) and "sender" in message:
                    return {
                        "role": message.get("sender"),
                        "content": message.get("text", ""),
                        "id": message.get("msg_id"),
                        "metadata": {"timestamp": message.get("timestamp")},
                    }
                return None

        recipe = CustomFormatRecipe()

        exporter = MessageExporter(saver, recipe=recipe)
        messages = exporter.export(thread_id=thread_id)

        print("\n=== Custom Format Messages ===")
        print(orjson.dumps(messages, option=orjson.OPT_INDENT_2).decode())

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello bot"
        assert messages[0]["id"] == "custom-1"
        assert messages[0]["metadata"]["timestamp"] == "2024-01-01T10:00:00Z"

        print("\n=== Successfully exported custom format messages ===")

    finally:
        redis_container.stop()


if __name__ == "__main__":
    test_export_conversation_with_multiple_checkpoints()
    test_export_with_custom_recipe_implementation()
