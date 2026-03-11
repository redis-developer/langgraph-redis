# Message Exporter

The `MessageExporter` utility extracts conversation messages from Redis
checkpoints into structured dictionaries. This is useful for analytics,
debugging, auditing, and exporting conversation data to external systems.

## Basic Usage

Create a `MessageExporter` with a checkpoint saver, then export messages from
a thread:

```python
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.message_exporter import MessageExporter

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()

    # Run your graph to produce checkpoints...
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "my-thread"}}
    graph.invoke({"messages": [("human", "What is Redis?")]}, config)

    # Export messages from the latest checkpoint
    exporter = MessageExporter(saver)
    messages = exporter.export("my-thread")

    for msg in messages:
        print(f"{msg['role']}: {msg['content']}")
```

Each message is a dictionary with at least `role` and `content` keys:

```python
{
    "role": "human",
    "content": "What is Redis?",
    "type": "HumanMessage",
    "id": "msg-123",
    "metadata": {
        "name": None,
        "tool_calls": None,
        "additional_kwargs": {},
    },
}
```

## Exporting a Specific Checkpoint

Pass a `checkpoint_id` to export messages from a specific point in time rather
than the latest:

```python
messages = exporter.export("my-thread", checkpoint_id="1ef4f797-8335-6428-8001-8a1503f9b875")
```

## Exporting an Entire Thread

`export_thread` iterates over all checkpoints in a thread, deduplicates
messages by ID, and annotates each message with checkpoint metadata:

```python
result = exporter.export_thread("my-thread")

print(f"Thread: {result['thread_id']}")
print(f"Exported at: {result['export_timestamp']}")

for msg in result["messages"]:
    print(f"[{msg['checkpoint_ts']}] {msg['role']}: {msg['content']}")
```

The returned dictionary has this structure:

```python
{
    "thread_id": "my-thread",
    "messages": [
        {
            "role": "human",
            "content": "What is Redis?",
            "type": "HumanMessage",
            "id": "msg-123",
            "checkpoint_id": "1ef4f797-...",
            "checkpoint_ts": "2024-07-31T20:14:19.804150+00:00",
            "metadata": {...},
        },
        # ...
    ],
    "export_timestamp": "2024-08-01T10:30:00.000000+00:00",
}
```

Messages are deduplicated by their `id` field, so the same message appearing
in multiple checkpoints is only included once.

## Recipes

The exporter uses a **recipe** to convert raw message objects into structured
dictionaries. The default `LangChainRecipe` handles:

- **LangChain message objects** (`HumanMessage`, `AIMessage`, `ToolMessage`, etc.)
- **Serialized LangChain format** (dicts with `lc` and `type` keys)
- **Simple dict messages** (dicts with `role` and `content` keys)
- **Plain strings** (mapped to `role: "unknown"`)

### Custom Recipes

Implement the `MessageRecipe` protocol to support custom message formats:

```python
from typing import Any, Dict, Optional
from langgraph.checkpoint.redis.message_exporter import MessageExporter, MessageRecipe


class MyCustomRecipe:
    """Extract messages from a custom format."""

    def extract(self, message: Any) -> Optional[Dict[str, Any]]:
        if isinstance(message, dict) and "sender" in message:
            return {
                "role": message["sender"],
                "content": message.get("text", ""),
                "timestamp": message.get("ts"),
            }
        return None


exporter = MessageExporter(saver, recipe=MyCustomRecipe())
messages = exporter.export("my-thread")
```

The `extract` method should return a dictionary with at least `role` and
`content` keys, or `None` if the message format is not recognized.

## Example: Export to JSON

```python
import json
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.message_exporter import MessageExporter

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()

    exporter = MessageExporter(saver)
    result = exporter.export_thread("my-thread")

    with open("conversation.json", "w") as f:
        json.dump(result, f, indent=2)
```

## API Summary

| Method | Description |
|--------|-------------|
| `MessageExporter(saver, recipe=None)` | Create an exporter with a checkpoint saver and optional recipe |
| `export(thread_id, checkpoint_id=None)` | Export messages from a single checkpoint (latest if no ID given) |
| `export_thread(thread_id)` | Export all messages across all checkpoints in a thread, deduplicated |

| Class | Description |
|-------|-------------|
| `LangChainRecipe` | Default recipe that handles LangChain message objects and serialized formats |
| `MessageRecipe` | Protocol for custom recipes — implement `extract(message) -> dict or None` |
