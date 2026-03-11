# Architecture

`langgraph-checkpoint-redis` provides Redis-backed persistence for LangGraph across three packages: checkpoint savers, stores, and middleware. Each package follows a shared set of design principles that keep the codebase maintainable while supporting both sync and async workloads.

## Package Structure

The library is organized into three top-level packages:

- **`langgraph.checkpoint.redis`** -- Checkpoint savers that persist graph state at every superstep.
- **`langgraph.store.redis`** -- Key-value stores with optional vector search for long-lived, cross-thread memory.
- **`langgraph.middleware.redis`** -- Agent middleware for caching, memory injection, and semantic routing.

## Dual Implementation Strategy

Every major component exists in both sync and async variants. The bulk of the business logic lives in an abstract base class, and thin concrete subclasses handle Redis client management and I/O:

| Base Class | Sync | Async |
|---|---|---|
| `BaseRedisSaver` | `RedisSaver` | `AsyncRedisSaver` |
| `BaseRedisSaver` | `ShallowRedisSaver` | `AsyncShallowRedisSaver` |
| `BaseRedisStore` | `RedisStore` | `AsyncRedisStore` |
| `AsyncRedisMiddleware` | -- | `SemanticCacheMiddleware`, etc. |

This pattern avoids duplicating logic across sync and async code paths. The base classes define methods like `_load_checkpoint`, `_apply_ttl_to_keys`, and schema definitions, while the concrete implementations call into them using the appropriate sync or async Redis client.

## Type System

Generic type parameters maintain type safety across the sync/async boundary:

```python
class BaseRedisSaver(BaseCheckpointSaver[str], Generic[RedisClientType, IndexType]):
    _redis: RedisClientType
    checkpoints_index: IndexType
    checkpoint_writes_index: IndexType
```

`RedisClientType` is bound to `Union[Redis, AsyncRedis, RedisCluster, AsyncRedisCluster]`, and `IndexType` is bound to `Union[SearchIndex, AsyncSearchIndex]`. This lets the type checker verify that a sync saver uses a sync client and a sync index without requiring separate class hierarchies.

## Redis Module Dependencies

The library requires two Redis modules:

- **RedisJSON** -- Stores checkpoints, writes, and store items as structured JSON documents rather than flat key-value pairs. This enables partial reads and atomic updates on nested fields.
- **RediSearch** -- Provides secondary indexing over JSON documents. Checkpoint queries (e.g., "find the latest checkpoint for thread X in namespace Y") run as index queries rather than key scans.

Redis 8.0+ ships with both modules built in. For earlier versions, use [Redis Stack](https://redis.io/docs/stack/).

## Schema-Driven Indexing

Both checkpoints and stores define their RediSearch index schemas as structured dictionaries. For example, the checkpoint index includes fields for `thread_id`, `checkpoint_ns`, `checkpoint_id`, `source`, and `step`:

```python
{
    "index": {
        "name": "checkpoint",
        "prefix": "checkpoint:",
        "storage_type": "json",
    },
    "fields": [
        {"name": "thread_id", "type": "tag"},
        {"name": "checkpoint_ns", "type": "tag"},
        {"name": "checkpoint_id", "type": "tag"},
        {"name": "source", "type": "tag"},
        {"name": "step", "type": "numeric"},
        # ...
    ],
}
```

Indexes are created lazily via `setup()` with `overwrite=False`, so they are safe to call on every application startup. The `redisvl` library handles index creation and query building from these schemas.

## Redis Key Patterns

Every Redis key follows a deterministic pattern built from the data it represents:

| Data | Key Pattern |
|---|---|
| Checkpoint | `checkpoint:{thread_id}:{namespace}:{checkpoint_id}` |
| Checkpoint blob | `checkpoint_blob:{thread_id}:{namespace}:{channel}:{version}` |
| Checkpoint write | `checkpoint_write:{thread_id}:{namespace}:{checkpoint_id}:{task_id}` |
| Store item | `store:{uuid}` |
| Store vector | `store_vectors:{uuid}` |

Key prefixes are configurable via constructor parameters (`checkpoint_prefix`, `checkpoint_write_prefix`, `store_prefix`, `vector_prefix`), which allows multiple independent saver or store instances to coexist on the same Redis deployment.

## Connection Management

All components support two connection modes:

1. **URL-based** -- Pass a `redis_url` string and the component creates and owns the client, closing it on cleanup.
2. **Client injection** -- Pass an existing `redis_client` and the component uses it without taking ownership.

Client injection is essential for sharing a single Redis connection across checkpointers, stores, and middleware in the same application, reducing connection overhead and simplifying resource management.

```python
from redis import Redis

redis_client = Redis.from_url("redis://localhost:6379")

saver = RedisSaver(redis_client=redis_client)
store = RedisStore(conn=redis_client)
```
