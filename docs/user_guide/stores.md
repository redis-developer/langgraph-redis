---
title: Stores
---

# Stores

`RedisStore` and `AsyncRedisStore` provide cross-thread key-value persistence
with optional vector similarity search. While checkpointers store per-thread
graph state, stores hold shared data that any thread can read and write --
user profiles, knowledge bases, configuration, and more.

## Basic Usage

### Creating a Store

Use `from_conn_string` for managed connection lifecycle:

```python
from langgraph.store.redis import RedisStore

with RedisStore.from_conn_string("redis://localhost:6379") as store:
    store.setup()
    # Use the store...
```

Or pass a Redis client directly:

```python
from redis import Redis
from langgraph.store.redis import RedisStore

client = Redis.from_url("redis://localhost:6379")
store = RedisStore(client)
store.setup()
```

### AsyncRedisStore

The async variant:

```python
from langgraph.store.redis import AsyncRedisStore

async with AsyncRedisStore.from_conn_string("redis://localhost:6379") as store:
    await store.asetup()
    # Use the store...
```

## Store Operations

### put -- Store an Item

Items are organized by **namespace** (a tuple of strings) and **key** (a string):

```python
store.put(
    namespace=("users", "profiles"),
    key="user-123",
    value={"name": "Alice", "role": "admin"},
)
```

### get -- Retrieve an Item

```python
item = store.get(namespace=("users", "profiles"), key="user-123")
if item:
    print(item.value)  # {"name": "Alice", "role": "admin"}
    print(item.key)    # "user-123"
```

### delete -- Remove an Item

```python
store.delete(namespace=("users", "profiles"), key="user-123")
```

### search -- Find Items by Namespace Prefix

Search returns all items under a namespace prefix:

```python
results = store.search(namespace_prefix=("users",))
for item in results:
    print(f"{item.namespace}/{item.key}: {item.value}")
```

Filter results by value fields:

```python
results = store.search(
    namespace_prefix=("users", "profiles"),
    filter={"role": "admin"},
)
```

Control pagination with `limit` and `offset`:

```python
results = store.search(
    namespace_prefix=("users",),
    limit=10,
    offset=20,
)
```

## Namespaces

Namespaces are tuples of strings that form a hierarchical key space. They work
like directory paths:

```python
# Store items in nested namespaces
store.put(("app", "settings"), "theme", {"color": "dark"})
store.put(("app", "settings"), "language", {"locale": "en"})
store.put(("app", "users", "alice"), "preferences", {"notifications": True})

# Search across a namespace prefix
results = store.search(namespace_prefix=("app",))  # Returns all items under "app"

# List unique namespaces
namespaces = store.list_namespaces(prefix=("app",))
# Returns: [("app", "settings"), ("app", "users", "alice")]
```

## Vector Search

`RedisStore` supports semantic similarity search using vector embeddings. This
requires configuring an `IndexConfig` with embedding dimensions and a
compatible embedding model.

### Configuring Vector Search

```python
from langgraph.store.redis import RedisStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_config = {
    "dims": 1536,
    "distance_type": "cosine",
    "fields": ["text"],
    "embed": embeddings,
}

with RedisStore.from_conn_string(
    "redis://localhost:6379",
    index=index_config,
) as store:
    store.setup()

    # Store items with text fields for embedding
    store.put(
        ("docs",), "intro",
        {"text": "LangGraph enables stateful AI workflows."},
    )
    store.put(
        ("docs",), "redis",
        {"text": "Redis provides fast in-memory data storage."},
    )

    # Semantic search
    results = store.search(
        namespace_prefix=("docs",),
        query="How do I build stateful agents?",
        limit=5,
    )
    for item in results:
        print(f"{item.key}: score={item.score:.3f}")
```

### Index Configuration Options

| Field | Type | Description |
|-------|------|-------------|
| `dims` | `int` | Embedding vector dimensions (e.g., 1536 for OpenAI) |
| `distance_type` | `str` | Distance metric: `"cosine"`, `"l2"`, or `"inner_product"` |
| `fields` | `list[str]` | Value fields to embed for search |
| `embed` | `Embeddings` | LangChain-compatible embeddings instance |

## Using with LangGraph compile()

Pass a store alongside a checkpointer when compiling a graph:

```python
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()

    with RedisStore.from_conn_string("redis://localhost:6379") as store:
        store.setup()

        graph = builder.compile(checkpointer=saver, store=store)
        result = graph.invoke(inputs, config)
```

Inside graph nodes, access the store via the `store` parameter:

```python
from langgraph.store.base import BaseStore


def my_node(state: MyState, store: BaseStore) -> dict:
    # Read from the store
    user = store.get(("users",), state["user_id"])

    # Write to the store
    store.put(("interactions",), state["interaction_id"], {"result": "success"})

    return {"output": user.value["name"] if user else "unknown"}
```

## TTL for Store Items

Store items support per-item TTL as well as global TTL configuration:

```python
# Global TTL configuration
with RedisStore.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 60},  # 60 minutes
) as store:
    store.setup()

    # Per-item TTL (overrides global)
    store.put(
        ("cache",), "temp-result",
        {"data": "expires soon"},
        ttl=5,  # 5 minutes
    )
```

See {doc}`ttl` for more details.

## Next Steps

- {doc}`ttl` -- configure expiration policies
- {doc}`checkpointers` -- per-thread state with `RedisSaver`
- {doc}`middleware` -- add caching and routing middleware
