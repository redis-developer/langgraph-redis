---
title: Middleware
---

# Middleware

The `langgraph-checkpoint-redis` library includes a middleware system for
LangGraph agents. Middleware intercepts model calls and tool calls to add
caching, routing, and memory capabilities -- all backed by Redis.

Middleware implements LangChain's `AgentMiddleware` protocol and works with
`create_agent` and LangGraph's prebuilt agents.

## Available Middleware

### SemanticCacheMiddleware

Caches LLM responses by semantic similarity. When a new request is
semantically close to a previously cached one, the cached response is
returned without calling the LLM.

```python
from langgraph.middleware.redis import SemanticCacheMiddleware, SemanticCacheConfig

config = SemanticCacheConfig(
    redis_url="redis://localhost:6379",
    name="llmcache",
    distance_threshold=0.1,   # Lower = stricter matching
    ttl_seconds=3600,         # Cache entries expire after 1 hour
    cache_final_only=True,    # Only cache responses without tool calls
)

cache_middleware = SemanticCacheMiddleware(config)
```

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"llmcache"` | Redis index name for the cache |
| `distance_threshold` | `float` | `0.1` | Max vector distance for a cache hit |
| `ttl_seconds` | `int` | `None` | Cache entry TTL in seconds |
| `cache_final_only` | `bool` | `True` | Only cache responses without tool calls |
| `deterministic_tools` | `list[str]` | `None` | Tools whose results are safe to cache through |

### ToolResultCacheMiddleware

Caches tool call results using exact-match lookup. Same tool name plus same
arguments produces the same cache key. Uses direct Redis `GET`/`SET` rather
than vector similarity.

```python
from langgraph.middleware.redis import ToolResultCacheMiddleware, ToolCacheConfig

config = ToolCacheConfig(
    redis_url="redis://localhost:6379",
    name="toolcache",
    ttl_seconds=7200,
    cacheable_tools=["web_search", "calculate"],
    excluded_tools=["send_email"],
)

tool_middleware = ToolResultCacheMiddleware(config)
```

#### Tool Metadata for Cacheability

Tools can declare metadata to control caching behavior:

```python
from langchain_core.tools import tool

@tool(metadata={
    "cacheable": True,       # Explicitly mark as cacheable
    "read_only": True,       # Does not produce side effects
    "idempotent": True,      # Same input always gives same output
})
def search_database(query: str) -> str:
    """Search the database for information."""
    ...

@tool(metadata={
    "destructive": True,     # Produces side effects
    "volatile": True,        # Results change over time
})
def delete_record(record_id: str) -> str:
    """Delete a record from the database."""
    ...
```

The middleware respects these metadata fields:

| Metadata Field | Effect |
|---------------|--------|
| `cacheable=True` | Explicitly allow caching |
| `cacheable=False` | Explicitly prevent caching |
| `destructive=True` | Never cache (side effects) |
| `volatile=True` | Never cache (time-dependent results) |
| `read_only=True` | Safe to cache |
| `idempotent=True` | Safe to cache |

#### ToolCacheConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cacheable_tools` | `list[str]` | `None` | Allowlist of tools to cache (None = all non-excluded) |
| `excluded_tools` | `list[str]` | `[]` | Tools to never cache |
| `volatile_arg_names` | `set[str]` | `None` | Arg names that prevent caching (e.g., `"timestamp"`) |
| `ignored_arg_names` | `set[str]` | `None` | Arg names stripped from cache key |
| `side_effect_prefixes` | `tuple[str]` | `None` | Tool name prefixes that prevent caching (e.g., `"send_"`) |

### ConversationMemoryMiddleware

Injects semantically relevant past messages into the current context using
Redis-backed session management.

```python
from langgraph.middleware.redis import ConversationMemoryMiddleware, ConversationMemoryConfig

config = ConversationMemoryConfig(
    redis_url="redis://localhost:6379",
    name="conversation_memory",
    session_tag="user-123",
    top_k=5,
    distance_threshold=0.7,
    ttl_seconds=86400,
)

memory_middleware = ConversationMemoryMiddleware(config)
```

### SemanticRouterMiddleware

Routes requests based on semantic matching against predefined route
categories. Useful for directing different types of user queries to
specialized handlers.

```python
from langgraph.middleware.redis import SemanticRouterMiddleware, SemanticRouterConfig

config = SemanticRouterConfig(
    redis_url="redis://localhost:6379",
    name="semantic_router",
    routes=[
        {
            "name": "technical",
            "references": [
                "How do I fix this error?",
                "What API should I use?",
                "Debug this code",
            ],
        },
        {
            "name": "general",
            "references": [
                "Tell me a joke",
                "What's the weather like?",
                "How are you?",
            ],
        },
    ],
    max_k=3,
    aggregation_method="avg",
)

router_middleware = SemanticRouterMiddleware(config)
```

## MiddlewareStack

Compose multiple middleware into a single stack. Middleware are applied in
order: the first wraps the second, which wraps the third, and so on.

```python
from langgraph.middleware.redis import MiddlewareStack

stack = MiddlewareStack([
    cache_middleware,
    tool_middleware,
    memory_middleware,
])
```

Use the stack with `create_agent`:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search, calculate],
    middleware=[stack],
)
```

## from_configs Factory

Create a middleware stack from configuration objects, optionally sharing a
single Redis connection:

```python
from langgraph.middleware.redis import from_configs, SemanticCacheConfig, ToolCacheConfig

stack = from_configs(
    redis_url="redis://localhost:6379",
    configs=[
        SemanticCacheConfig(ttl_seconds=3600),
        ToolCacheConfig(cacheable_tools=["search"]),
    ],
)
```

## create_caching_stack Convenience

For the common pattern of combining semantic and tool caching:

```python
from langgraph.middleware.redis import create_caching_stack

stack = create_caching_stack(
    redis_url="redis://localhost:6379",
    semantic_cache_ttl=3600,
    tool_cache_ttl=7200,
    cacheable_tools=["search", "calculate"],
    excluded_tools=["send_email"],
    distance_threshold=0.1,
)
```

## Connection Sharing with IntegratedRedisMiddleware

When you already have a `RedisSaver` or `RedisStore`, use
`IntegratedRedisMiddleware` to create middleware that connects to the same
Redis server:

```python
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.middleware.redis import IntegratedRedisMiddleware, SemanticCacheConfig

async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as saver:
    await saver.asetup()

    stack = IntegratedRedisMiddleware.from_saver(
        saver,
        [
            SemanticCacheConfig(ttl_seconds=3600),
            ToolCacheConfig(cacheable_tools=["search"]),
        ],
    )

    graph = builder.compile(checkpointer=saver)
    # Both the checkpointer and middleware use the same Redis server
```

You can also create middleware from a store:

```python
from langgraph.store.redis import AsyncRedisStore
from langgraph.middleware.redis import IntegratedRedisMiddleware, SemanticCacheConfig

async with AsyncRedisStore.from_conn_string("redis://localhost:6379") as store:
    await store.asetup()

    stack = IntegratedRedisMiddleware.from_store(
        store,
        [SemanticCacheConfig(ttl_seconds=3600)],
    )
```

## Base Configuration

All middleware configs inherit from `MiddlewareConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `redis_url` | `str` | `None` | Redis connection URL |
| `redis_client` | `Redis` | `None` | Existing Redis client instance |
| `connection_args` | `dict` | `None` | Extra connection arguments |
| `graceful_degradation` | `bool` | `True` | Pass through on Redis errors instead of failing |

When `graceful_degradation` is `True` (the default), middleware failures are
logged as warnings and the request passes through to the underlying handler
without caching.

## Next Steps

- {doc}`checkpointers` -- persist graph state with Redis checkpointers
- {doc}`stores` -- cross-thread key-value storage
- {doc}`azure_enterprise` -- deploy middleware with enterprise Redis
