# Tool Result Caching: Design Document

## Problem Statement

The `ToolResultCacheMiddleware` previously used `redisvl.SemanticCache` with vector embeddings
to cache tool call results. This was architecturally wrong: tool caching is **deterministic** --
the same tool called with the same arguments always produces the same result. Semantic similarity
(vector distance) is the wrong abstraction for this use case.

### Why Semantic Similarity Fails for Tool Caching

Vector embeddings map text into a high-dimensional space where "similar meaning" maps to nearby
points. This works well for natural language (LLM prompts/responses), but creates false cache
hits for tool calls:

- `get_app_details(app_name="SOMEAPP")` and `get_app_details(app_name="SOMEAPP2")` produce
  nearly identical embeddings, but return completely different results.
- `search(query="redis", page=1)` and `search(query="redis", page=2)` are semantically
  similar but return different data.
- `lookup(entity="ProjectAlpha")` vs `lookup(entity="ProjectAlpha2")` -- the vector distance
  is below any reasonable threshold.

Tool caching needs **exact match**: same tool + same args = cache hit. Different args (even
slightly different) = cache miss.

### When Semantic Caching IS Appropriate

Semantic caching remains correct for **LLM response caching** (`SemanticCacheMiddleware`),
where the goal is to return a cached response for a semantically similar *user prompt*.
"What is Python?" and "Tell me about the Python language" are genuinely similar queries
that can share a cached response.

## Reference Implementations

### FastMCP CachingMiddleware

The [FastMCP CachingMiddleware](https://github.com/jlowin/fastmcp/blob/main/src/fastmcp/server/middleware/caching.py)
uses exact key-value lookup:

```python
cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
cached = self._cache.get(cache_key)
```

This is the correct approach: deterministic, no vectorizer needed, no false hits.

### MCP ToolAnnotations Spec

The [MCP specification](https://modelcontextprotocol.io/legacy/concepts/tools) defines
tool annotations that signal cacheability:

- `readOnlyHint` -- tool does not modify state
- `destructiveHint` -- tool performs destructive operations
- `idempotentHint` -- calling multiple times with same args has same effect
- `openWorldHint` -- tool interacts with external entities

Our cacheability decision chain maps these concepts to metadata fields.

### Known Pitfalls

- [promptfoo cache key bug](https://github.com/promptfoo/promptfoo/issues/7467):
  cache key must include full tool config, not just args
- [OpenAI MCP stale caching](https://community.openai.com/t/agent-builder-mcp-calls-will-cache-invalid-endpoints-even-though-they-may-be-valid-later-preventing-you-from-using-them/1368481):
  caching tool results incorrectly leads to stale data

## Cache Key Design

### SQL Cache Analogy

Think of tool caching like a database table:

| Concept | SQL Analogy | Tool Cache |
|---------|------------|------------|
| Tool name | Table name | `search` |
| Arg keys | Column names | `query`, `page` |
| Arg values | Row values | `"redis"`, `1` |
| Cache key | Primary key | `toolcache:search:{"page": 1, "query": "redis"}` |

### Key Format

```
{config.name}:{tool_name}:{sorted_json_args}
```

Where `sorted_json_args = json.dumps(effective_args, sort_keys=True)`.

### Key Normalization

1. **Sort keys**: `json.dumps(args, sort_keys=True)` ensures `{"b": 2, "a": 1}` and
   `{"a": 1, "b": 2}` produce the same key.
2. **Strip ignored args**: Remove `request_id`, `trace_id`, etc. before key generation.
3. **Volatile arg detection**: If args contain volatile names (`timestamp`, `now`, etc.),
   skip caching entirely rather than producing non-deterministic keys.

## Cacheability Decision Chain

Before looking up or storing a cache entry, the middleware evaluates whether the tool
call is cacheable. The priority chain (highest to lowest):

```
1. metadata["cacheable"]           -- explicit override (highest priority)
2. metadata["destructive"] = True  -- never cache
3. metadata["volatile"] = True     -- never cache
4. metadata["read_only"] + ["idempotent"] = True -- cache
5. Side-effect prefix match        -- never cache (send_, delete_, create_, ...)
6. Volatile arg name in args       -- never cache (timestamp, now, date, ...)
7. Config whitelist / blacklist    -- fallback
```

## Example Scenarios

### Correct Caching

```python
# First call: cache miss, executes tool, stores result
await cache.awrap_tool_call(
    ToolCallRequest(tool_call={"name": "search", "args": {"q": "redis"}, "id": "c1"}, ...),
    handler
)
# key = "toolcache:search:{\"q\": \"redis\"}"
# Result stored: "Found 42 results for redis"

# Second call: exact same args, cache hit
await cache.awrap_tool_call(
    ToolCallRequest(tool_call={"name": "search", "args": {"q": "redis"}, "id": "c2"}, ...),
    handler
)
# key = "toolcache:search:{\"q\": \"redis\"}"  -- exact match, returns cached result
```

### Correct Cache Miss

```python
# Different args produce different keys -- no false hit
await cache.awrap_tool_call(
    ToolCallRequest(tool_call={"name": "search", "args": {"q": "redis", "page": 1}, "id": "c1"}, ...),
    handler
)
# key = "toolcache:search:{\"page\": 1, \"q\": \"redis\"}"

await cache.awrap_tool_call(
    ToolCallRequest(tool_call={"name": "search", "args": {"q": "redis", "page": 2}, "id": "c2"}, ...),
    handler
)
# key = "toolcache:search:{\"page\": 2, \"q\": \"redis\"}"  -- different key, cache miss
```

### Ignored Args (Cache Hit)

```python
config = ToolCacheConfig(ignored_arg_names={"request_id"})

# These two calls share a cache entry because request_id is stripped
# key = "toolcache:search:{\"q\": \"redis\"}" for both
await cache.awrap_tool_call({"tool_name": "search", "args": {"q": "redis", "request_id": "r1"}}, handler)
await cache.awrap_tool_call({"tool_name": "search", "args": {"q": "redis", "request_id": "r2"}}, handler)
```

## Implementation

The tool cache uses direct Redis `GET`/`SET` operations instead of `SemanticCache`:

- **Store**: `SET key serialized_result [EX ttl]`
- **Lookup**: `GET key` -- exact match, no vector similarity
- **No vectorizer needed** -- the `vectorizer` and `distance_threshold` fields on
  `ToolCacheConfig` are kept for backward compatibility but ignored.

The semantic cache (`SemanticCacheMiddleware`) continues to use `redisvl.SemanticCache`
with vector embeddings, as semantic similarity is the correct abstraction for LLM
response caching.
