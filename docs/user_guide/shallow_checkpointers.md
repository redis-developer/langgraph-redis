---
title: Shallow Checkpointers
---

# Shallow Checkpointers

Shallow checkpointers store only the **most recent checkpoint** per thread and
namespace, rather than maintaining a full history. This trades off history
traversal for significantly reduced storage consumption.

## When to Use Shallow Checkpointers

Use a shallow checkpointer when:

- You only need the current state of a conversation, not its full history.
- Storage efficiency is a priority (e.g., high-volume, short-lived threads).
- Your application does not require time-travel debugging or checkpoint replay.
- You want to minimize the number of Redis keys per thread.

Use the full `RedisSaver` when:

- You need to traverse or replay checkpoint history.
- Your workflow uses `before` checkpoints for interrupt/resume patterns that
  require referencing prior states.
- You need audit trails of state transitions.

## ShallowRedisSaver

The synchronous shallow checkpointer:

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

with ShallowRedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "shallow-thread"}}
    result = graph.invoke(inputs, config)
```

### Import Path

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
```

### Cache Configuration

`ShallowRedisSaver` includes internal LRU caches for key lookups and channel
data. You can configure their sizes:

```python
saver = ShallowRedisSaver(
    redis_url="redis://localhost:6379",
    key_cache_max_size=2000,      # Default: 1000
    channel_cache_max_size=200,   # Default: 100
)
saver.setup()
```

Larger cache sizes reduce Redis round trips at the cost of more memory in
your application process.

## AsyncShallowRedisSaver

The async variant for use with `asyncio`:

```python
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver

async with AsyncShallowRedisSaver.from_conn_string("redis://localhost:6379") as saver:
    await saver.asetup()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "async-shallow-thread"}}
    result = await graph.ainvoke(inputs, config)
```

### Import Path

```python
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
```

## Same API Surface

Shallow checkpointers implement the same `BaseCheckpointSaver` interface as
the full checkpointers. You can use them as drop-in replacements:

```python
# These all work the same way as with RedisSaver
checkpoint_tuple = saver.get_tuple(config)

for ct in saver.list(config):
    print(ct.metadata)

saver.put_writes(config, writes, task_id)
```

The key difference is that `list()` returns at most one checkpoint per
thread-namespace combination.

## Trade-offs

| Feature | RedisSaver | ShallowRedisSaver |
|---------|-----------|-------------------|
| Checkpoint history | Full history | Latest only |
| Storage per thread | Grows with steps | Constant |
| Time travel | Supported | Not supported |
| `list()` results | All checkpoints | Latest checkpoint |
| Redis key pattern | `checkpoint:{thread}:{ns}:{id}` | `checkpoint:{thread}:{ns}` |
| Interrupt/resume | Full support | Supported for current state |

## Using with TTL

Shallow checkpointers support TTL configuration, just like the full
checkpointers:

```python
saver = ShallowRedisSaver(
    redis_url="redis://localhost:6379",
    ttl={"default_ttl": 60, "refresh_on_read": True},
)
saver.setup()
```

See {doc}`ttl` for more details on TTL configuration.

## Example: High-Volume Chat Application

For a chat application with many concurrent users where only the latest
conversation state matters:

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

with ShallowRedisSaver.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 120, "refresh_on_read": True},  # 2-hour TTL
) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    # Each user session gets its own thread
    for user_id, message in incoming_messages:
        config = {"configurable": {"thread_id": f"chat:{user_id}"}}
        result = graph.invoke({"messages": [message]}, config)
        send_response(user_id, result)
```

## Next Steps

- {doc}`checkpointers` -- full checkpointer reference
- {doc}`ttl` -- configure automatic expiration
- {doc}`stores` -- add cross-thread persistence with `RedisStore`
