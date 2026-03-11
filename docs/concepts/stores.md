# Stores

LangGraph stores provide persistent key-value storage that is independent of graph execution history. While checkpoints capture per-thread, per-step state, stores hold long-lived data that can be shared across threads and accessed by any graph node.

`langgraph-checkpoint-redis` implements the LangGraph `BaseStore` interface using Redis with optional vector search.

## Stores vs Checkpoints

The distinction between stores and checkpoints is fundamental:

| | Checkpoints | Stores |
|---|---|---|
| **Scope** | Per-thread execution history | Cross-thread shared memory |
| **Lifetime** | Tied to a conversation thread | Independent of any thread |
| **Access pattern** | Sequential (latest or by ID) | Random (by namespace and key) |
| **Use case** | Conversation state, resumption | User profiles, learned preferences, shared knowledge |

A store is the right choice when data needs to outlive a single conversation or be accessible from multiple concurrent threads.

## Namespaces and Keys

Store items are organized into a two-level hierarchy:

- **Namespace** -- A tuple of strings forming a hierarchical path, such as `("users", "user-123", "preferences")`.
- **Key** -- A string identifying a specific item within the namespace.

This structure supports flexible data organization:

```python
from langgraph.store.redis import RedisStore

store = RedisStore(conn=redis_client)
store.setup()

# Store a user preference
store.put(("users", "user-123", "preferences"), "theme", {"value": "dark"})

# Retrieve it
item = store.get(("users", "user-123", "preferences"), "theme")

# List all items under a namespace prefix
items = store.search(("users", "user-123"))
```

## How Redis Stores Items

Each store item is a JSON document stored via RedisJSON:

```json
{
    "prefix": "users.user-123.preferences",
    "key": "theme",
    "value": "{\"value\": \"dark\"}",
    "created_at": 1700000000,
    "updated_at": 1700000000
}
```

The document key in Redis is `store:{uuid}`, where the UUID is deterministically derived from the namespace and key. This means `put()` with the same namespace and key performs an upsert.

A RediSearch index over the `prefix` and `key` fields enables efficient lookups and prefix-based listing without key scanning.

## Vector Search Integration

Stores optionally support vector search for semantic retrieval. When configured with an `IndexConfig`, the store computes embeddings for designated text fields and stores them alongside the item:

```python
from langchain_openai import OpenAIEmbeddings
from langgraph.store.redis import RedisStore

store = RedisStore(
    conn=redis_client,
    index={
        "dims": 1536,
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "fields": ["text"],
    },
)
store.setup()

# Store an item with searchable text
store.put(("notes",), "idea-1", {"text": "Redis is great for caching"})

# Semantic search
results = store.search(("notes",), query="database performance")
```

Vector embeddings are stored in a separate Redis document (`store_vectors:{uuid}`) with their own RediSearch index. This separation keeps the primary store index lightweight while enabling vector similarity queries via RediSearch's vector search capabilities.

### Vector Configuration

The `IndexConfig` accepts:

- **`dims`** -- Dimensionality of the embedding vectors.
- **`embed`** -- An embeddings provider (any LangChain `Embeddings` implementation).
- **`fields`** -- List of field names within the stored value to embed. The text from these fields is extracted and embedded automatically on `put()`.
- **`distance_type`** -- The similarity metric (`"cosine"`, `"l2"`, or `"ip"`). Defaults to `"cosine"`.

## Cross-Thread Memory

Because store items exist outside the checkpoint system, they are accessible from any thread. This enables patterns like:

- **User profiles** -- Store user preferences under `("users", user_id)` and read them from any conversation thread.
- **Learned facts** -- An agent can store facts it learns during one conversation and retrieve them in future conversations.
- **Shared knowledge** -- Multiple agents or graphs can read and write to shared namespaces.

LangGraph's `Store` is injected into graph nodes via the `store` parameter:

```python
from langgraph.store.base import BaseStore

def my_node(state, config, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    prefs = store.get(("users", user_id, "preferences"), "settings")
    # Use preferences in the node logic
    ...
```

## TTL Support

Stores support the same TTL mechanism as checkpoints. Items can have a per-item TTL set at write time, or a default TTL applied to all items. See the [TTL Management](ttl.md) page for details.

## Namespace Operations

The store supports listing and filtering namespaces:

```python
# List all namespaces matching a pattern
namespaces = store.list_namespaces(prefix=("users",))

# List with depth limiting
namespaces = store.list_namespaces(prefix=("users",), max_depth=2)
```

This is useful for discovering what data exists in the store without knowing specific keys.
