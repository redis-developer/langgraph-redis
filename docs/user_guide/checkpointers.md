---
title: Checkpointers
---

# Checkpointers

Checkpointers persist the state of a LangGraph workflow at each step, enabling
time travel, fault recovery, and multi-turn conversations. The
`langgraph-checkpoint-redis` library provides two primary checkpointer classes
that store full checkpoint history.

## RedisSaver

`RedisSaver` is the synchronous checkpointer for standard Python applications.

### Creating with a Connection String

The `from_conn_string` class method is a context manager that handles connection
lifecycle:

```python
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)
    result = graph.invoke(inputs, config)
```

The connection URL supports multiple schemes:

- `redis://host:port` -- standard unencrypted connection
- `rediss://host:port` -- SSL/TLS encrypted connection
- `redis+sentinel://host:26379/service_name/db` -- Redis Sentinel

### Creating with a Redis Client

Pass an existing Redis client for full control over connection settings:

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

client = Redis(host="localhost", port=6379, decode_responses=False)
saver = RedisSaver(redis_client=client)
saver.setup()
```

When you provide your own client, the saver does not close it automatically.
You are responsible for managing the client lifecycle.

### Calling setup()

You **must** call `setup()` before using the checkpointer. This method creates
the RediSearch indices that power checkpoint queries:

```python
saver.setup()  # Creates indices if they do not already exist
```

The call is idempotent -- it will not overwrite existing indices.

### Core Operations

**put** -- Save a checkpoint:

```python
next_config = saver.put(config, checkpoint, metadata, new_versions)
```

This is called automatically by LangGraph during graph execution. You rarely
need to call it directly.

**get_tuple** -- Retrieve the latest checkpoint for a thread:

```python
config = {"configurable": {"thread_id": "thread-1"}}
checkpoint_tuple = saver.get_tuple(config)
# Returns: CheckpointTuple(config, checkpoint, metadata, parent_config, pending_writes)
```

To retrieve a specific checkpoint, include the checkpoint ID:

```python
config = {
    "configurable": {
        "thread_id": "thread-1",
        "checkpoint_id": "01HZXYZ...",  # Specific ULID
    }
}
checkpoint_tuple = saver.get_tuple(config)
```

**list** -- Iterate over checkpoint history:

```python
config = {"configurable": {"thread_id": "thread-1"}}
for checkpoint_tuple in saver.list(config, limit=10):
    print(checkpoint_tuple.metadata)
```

The `list` method supports filtering:

```python
# Filter by metadata fields
for ct in saver.list(config, filter={"source": "loop", "step": 3}):
    print(ct.checkpoint)
```

### Thread Management

**delete_thread** -- Remove all checkpoints and writes for a thread:

```python
saver.delete_thread("thread-1")
```

**prune** -- Keep only recent checkpoints to manage storage:

```python
# Keep only the latest checkpoint per namespace
saver.prune(["thread-1", "thread-2"])

# Keep the last 5 checkpoints per namespace
saver.prune(["thread-1"], keep_last=5)

# Delete all checkpoints for the given threads
saver.prune(["thread-1"], strategy="delete")
```

## AsyncRedisSaver

`AsyncRedisSaver` provides the same API for async applications, using async
Redis client connections.

### Creating an Async Checkpointer

```python
from langgraph.checkpoint.redis import AsyncRedisSaver

async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as saver:
    await saver.asetup()
    graph = builder.compile(checkpointer=saver)
    result = await graph.ainvoke(inputs, config)
```

Or with a direct async client:

```python
from redis.asyncio import Redis as AsyncRedis
from langgraph.checkpoint.redis import AsyncRedisSaver

client = AsyncRedis(host="localhost", port=6379)
saver = AsyncRedisSaver(redis_client=client)
await saver.asetup()
```

### Async Operations

The async methods mirror the sync API:

```python
# Get a checkpoint
checkpoint_tuple = await saver.aget_tuple(config)

# List checkpoints
async for ct in saver.alist(config, limit=10):
    print(ct.metadata)

# Save writes
await saver.aput_writes(config, writes, task_id)
```

## Using with LangGraph compile()

The standard pattern for integrating a checkpointer with LangGraph:

```python
from langgraph.graph import StateGraph

builder = StateGraph(MyState)
# ... add nodes and edges ...

# Compile with checkpointer
with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    # Each invocation with the same thread_id continues from the last checkpoint
    config = {"configurable": {"thread_id": "user-session-123"}}
    result = graph.invoke({"messages": [user_message]}, config)
```

## Configuration Options

The `RedisSaver` and `AsyncRedisSaver` constructors accept the following
keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | `str` | `None` | Redis connection URL |
| `redis_client` | `Redis` | `None` | Existing Redis client |
| `connection_args` | `dict` | `None` | Extra connection keyword arguments |
| `ttl` | `dict` | `None` | TTL configuration (see {doc}`ttl`) |
| `checkpoint_prefix` | `str` | `"checkpoint"` | Key prefix for checkpoint documents |
| `checkpoint_write_prefix` | `str` | `"checkpoint_write"` | Key prefix for write documents |

## Cluster Support

Both `RedisSaver` and `AsyncRedisSaver` support Redis Cluster. Cluster mode is
detected automatically based on the client type:

```python
from redis.cluster import RedisCluster

client = RedisCluster(host="cluster-node", port=7000)
saver = RedisSaver(redis_client=client)
saver.setup()
```

In cluster mode, the saver uses individual key operations instead of
pipelines to avoid cross-slot errors.

## Next Steps

- {doc}`shallow_checkpointers` -- use shallow savers for reduced storage
- {doc}`ttl` -- configure automatic expiration
- {doc}`stores` -- add cross-thread key-value persistence
