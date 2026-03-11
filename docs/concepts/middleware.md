# Middleware

Agent middleware intercepts model calls and tool calls during graph execution, enabling caching, memory injection, and routing without modifying agent or graph code. The middleware system in `langgraph-checkpoint-redis` is built on LangChain's `AgentMiddleware` protocol and uses Redis as its backing store.

## AsyncRedisMiddleware Base Class

All middleware extend `AsyncRedisMiddleware`, which provides:

- **Redis client lifecycle** -- Creates or accepts a Redis connection, with automatic cleanup for owned clients.
- **Lazy initialization** -- Resources (caches, indexes, routers) are created on first use with double-checked locking for thread safety.
- **Graceful degradation** -- When `graceful_degradation=True`, Redis errors are caught and the middleware falls through to the original handler rather than failing the request.
- **AgentMiddleware protocol** -- Implements `awrap_model_call` and `awrap_tool_call` as defined by LangChain.

Subclasses implement `_setup_async()` to initialize their specific resources and override one or both of the wrapping methods.

## The Four Middleware Types

### SemanticCacheMiddleware

Caches LLM responses based on **semantic similarity** of the input prompt. When a new prompt is semantically close to a previously seen prompt, the cached response is returned without calling the LLM.

This uses vector embeddings via `redisvl.extensions.cache.llm.SemanticCache`. Semantic similarity is the correct approach here because natural language prompts with similar meaning should produce equivalent responses. "What is Python?" and "Tell me about the Python language" are genuinely similar queries.

```python
from langgraph.middleware.redis import SemanticCacheMiddleware, SemanticCacheConfig

config = SemanticCacheConfig(
    redis_url="redis://localhost:6379",
    distance_threshold=0.2,
)
cache = SemanticCacheMiddleware(config)
```

### ToolResultCacheMiddleware

Caches tool call results using **exact-match** key-value lookup. The cache key is derived from the tool name and its arguments (JSON-serialized, sorted by key). Same tool + same arguments = cache hit.

Exact match is correct for tool caching because tool calls are deterministic: `search(query="redis", page=1)` always returns the same result. Semantic similarity would produce false cache hits: `search(query="redis", page=1)` and `search(query="redis", page=2)` have nearly identical embeddings but return different data.

```python
from langgraph.middleware.redis import ToolResultCacheMiddleware, ToolCacheConfig

config = ToolCacheConfig(
    redis_url="redis://localhost:6379",
    cacheable_tools=["search", "calculate"],
    ttl_seconds=3600,
)
tool_cache = ToolResultCacheMiddleware(config)
```

### ConversationMemoryMiddleware

Retrieves semantically relevant past messages and injects them into the current conversation context. Uses `redisvl.extensions.message_history.SemanticMessageHistory` to store and search conversation history.

This enables long-term memory: an agent can recall relevant information from prior conversations even if they occurred in different threads.

```python
from langgraph.middleware.redis import ConversationMemoryMiddleware, ConversationMemoryConfig

config = ConversationMemoryConfig(
    redis_url="redis://localhost:6379",
    session_tag="user-123",
    top_k=5,
    distance_threshold=0.7,
)
memory = ConversationMemoryMiddleware(config)
```

### SemanticRouterMiddleware

Routes requests to different handlers based on semantic matching against predefined intent categories. Uses `redisvl.extensions.router.SemanticRouter` to classify user intents.

```python
from langgraph.middleware.redis import SemanticRouterMiddleware, SemanticRouterConfig

config = SemanticRouterConfig(
    redis_url="redis://localhost:6379",
    routes=[
        {"name": "greeting", "references": ["hello", "hi", "hey"]},
        {"name": "support", "references": ["help", "issue", "problem"]},
    ],
)
router = SemanticRouterMiddleware(config)

@router.register_route_handler("greeting")
async def handle_greeting(request, route_match):
    return {"content": "Hello! How can I help?"}
```

## Cacheability Decision Chain

The `ToolResultCacheMiddleware` uses a 7-level priority chain to decide whether a tool call should be cached, inspired by SQL function volatility categories and the MCP `ToolAnnotations` specification:

1. **`metadata["cacheable"]`** -- Explicit override. If the tool's metadata includes a `cacheable` flag, it takes highest priority.
2. **`metadata["destructive"]`** -- Tools marked as destructive are never cached.
3. **`metadata["volatile"]`** -- Tools marked as volatile are never cached.
4. **`metadata["read_only"]` and `metadata["idempotent"]`** -- Tools that are both read-only and idempotent are cached.
5. **Side-effect prefix** -- Tool names starting with `send_`, `delete_`, `create_`, `update_`, `remove_`, `write_`, `post_`, `put_`, or `patch_` are never cached.
6. **Volatile argument names** -- If the tool's arguments contain keys like `timestamp`, `current_time`, or `now` (at any nesting depth), the call is not cached.
7. **Config whitelist/blacklist** -- Falls back to `cacheable_tools` (whitelist) or `excluded_tools` (blacklist) from the middleware config.

This chain ensures sensible defaults while allowing tool authors to override behavior explicitly.

## Connection Sharing

Middleware can share a Redis connection with checkpointers and stores by injecting an existing client:

```python
from redis.asyncio import Redis as AsyncRedis

redis_client = AsyncRedis.from_url("redis://localhost:6379")

cache_config = SemanticCacheConfig(redis_client=redis_client)
tool_config = ToolCacheConfig(redis_client=redis_client)
```

When a `redis_client` is provided, the middleware does not own the connection and will not close it on cleanup. This avoids opening multiple connections to the same Redis instance.

## MiddlewareStack

`MiddlewareStack` composes multiple middleware into a single `AgentMiddleware` instance. Middleware are applied in order: the first middleware's pre-processing runs first, and its post-processing runs last.

```python
from langgraph.middleware.redis import MiddlewareStack

stack = MiddlewareStack([
    SemanticCacheMiddleware(cache_config),
    ToolResultCacheMiddleware(tool_config),
    ConversationMemoryMiddleware(memory_config),
])

# Use with LangGraph's create_agent
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[stack],
)
```

The stack also sanitizes outgoing requests by stripping provider-specific IDs from cached AI messages, preventing duplicate ID errors when using APIs like OpenAI's Responses API.
