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

**prune** -- Keep recent checkpoints and their required history to manage storage:

```python
# Request the latest checkpoint per namespace, plus any required ancestors
saver.prune(["thread-1", "thread-2"])

# Request the last 5 checkpoints per namespace, plus any required ancestors
saver.prune(["thread-1"], keep_last=5)

# Delete all checkpoints for the given threads
saver.prune(["thread-1"], strategy="delete")
```

#### DeltaChannel retention

`DeltaChannel` values can depend on snapshots and writes stored in ancestor
checkpoints. Pruning preserves those dependencies for **every checkpoint that
remains**, including ancestors kept to reconstruct other checkpoints. These
support checkpoints remain visible in history and can still be inspected or
resumed with their complete state.

As a result, `keep_last` specifies the recent checkpoint window to retain; it is
**not a strict limit on stored checkpoints or Redis memory**. Channels that
snapshot at different steps can require the entire chain to remain. For example,
channels snapshotting every 2 and 3 updates may need the history before either
channel's most recent snapshot to reconstruct an intermediate checkpoint. Older
history can be removed once the retained checkpoints and their support ancestors
can all be reconstructed from later stored values.

`strategy="delete"` and `keep_last=0` bypass dependency preservation and remove
the selected history. Only `"keep_latest"` and `"delete"` are valid strategies;
an explicit `keep_last` overrides either valid strategy.

#### Search limits and expiration

Prune only inactive threads: checkpoint creation, updates, and other pruning calls
for those threads must be paused until pruning finishes. The search inventory and
deletions are not atomic with concurrent writers.

Pruning's searches disable the Search query timeout for themselves (`TIMEOUT 0`),
so they always return their complete result set. Under Search's default
`search-on-timeout` policy (`RETURN`), a search that times out returns a partial
set whose reported total matches its returned count, which pruning could not
tell apart from a complete result. The per-query override avoids this without
reading or changing any server configuration, so pruning works on Redis Cloud and
Redis Software and does not need `FT.CONFIG` permission. A search over a very
large thread runs until it finishes instead of stopping at the server's timeout.

`max_results` defaults to 10,000 and limits each search: the initial checkpoint
search across all namespaces in a thread, and each search for an evicted
checkpoint's writes. Pruning compares the total matches with the returned
records. If any of these searches is truncated, it raises `ValueError` before
deleting any keys for that thread. This also applies to explicit deletion and
calls with a `keep_last` override.

Set the limit high enough to cover both the thread's full checkpoint history and
the writes for each checkpoint being deleted, then retry. A result count equal
to the limit is accepted when the complete result set was returned. A too-small
limit no longer causes partial deletion or silently removes a surviving branch's
required history.

This parameter is not a deletion batch size or a cursor for incremental pruning,
and it does not limit dependency-walk reads. Completeness checks apply per thread:
if a later thread in the same call fails, previously processed threads are not
rolled back.

Write cleanup uses the full checkpoint identity: thread ID, namespace, and
checkpoint ID. Reusing a checkpoint ID in another namespace does not remove that
namespace's pending writes.

Pruning does not refresh TTLs, even with `refresh_on_read=True`. Dependency
preservation assumes the required checkpoints and writes are still present; it
cannot restore history that has already expired or been deleted. Cycles in
required ancestry raise `ValueError` before pruning deletes keys for that thread.

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

For standalone Redis workloads that may schedule more concurrent operations
than the connection pool can serve at once, use a bounded
`BlockingConnectionPool`. When all connections are busy, additional operations
wait for a connection instead of failing immediately with
`MaxConnectionsError`:

```python
from redis.asyncio import BlockingConnectionPool, Redis as AsyncRedis
from langgraph.checkpoint.redis import AsyncRedisSaver

pool = BlockingConnectionPool.from_url(
    "redis://localhost:6379",
    max_connections=20,
    timeout=5,
)
client = AsyncRedis(connection_pool=pool)

try:
    async with AsyncRedisSaver.from_conn_string(redis_client=client) as saver:
        graph = builder.compile(checkpointer=saver)
        result = await graph.ainvoke(inputs, config)
finally:
    await client.aclose()
    await pool.disconnect()
```

Choose `max_connections` based on the concurrency your Redis deployment can
support. The `timeout` applies to each attempt to acquire a connection from the
pool. If no connection becomes available within that interval, direct redis-py
operations raise `redis.exceptions.ConnectionError`. Operations performed
through RedisVL may instead raise `RedisVLError`, with the original connection
error preserved as its cause.

This provides connection-level backpressure, but it does not limit how many
asyncio tasks the application creates. Applications that can schedule an
unbounded number of tasks should also use an application-level concurrency
limit, such as an `asyncio.Semaphore` or a bounded worker queue.

`BlockingConnectionPool` is for standalone Redis clients. Redis Cluster and
Redis Sentinel use topology-specific connection management; for those
deployments, inject the appropriate async client and configure the limits it
supports. Because the client in this example is supplied by the application,
the application is also responsible for closing it and disconnecting its pool.

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

# Prune with the same dependency-retention rules as the synchronous saver
await saver.aprune(["thread-1"], keep_last=5)
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
