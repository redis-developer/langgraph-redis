"""Integration tests for middleware cache false-hit prevention.

Verifies that:
- Tool cache uses exact-match (Redis GET/SET), so different args never collide.
- Semantic cache uses vector similarity controlled by distance_threshold, so
  truly similar prompts share cache entries while distinct prompts do not
  (depending on threshold).

These tests use TestContainers to spin up a real Redis instance.
"""

import pytest
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    SemanticCacheConfig,
    SemanticCacheMiddleware,
    ToolCacheConfig,
    ToolResultCacheMiddleware,
)

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)


def _extract_args(request):
    """Extract args dict from either a dict or ToolCallRequest."""
    if isinstance(request, dict):
        return request.get("args", {})
    tool_call = getattr(request, "tool_call", {})
    return tool_call.get("args", {}) if isinstance(tool_call, dict) else {}


def _extract_tool_call_id(request, fallback="call_1"):
    """Extract tool_call_id from either a dict or ToolCallRequest."""
    if isinstance(request, dict):
        return request.get("id", fallback)
    tool_call = getattr(request, "tool_call", {})
    return tool_call.get("id", fallback) if isinstance(tool_call, dict) else fallback


@pytest.fixture(scope="module")
def redis_url():
    """Provide a Redis URL using TestContainers."""
    redis_container = RedisContainer("redis/redis-stack-server:latest")
    redis_container.start()
    try:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        redis_container.stop()


# ---------------------------------------------------------------------------
# Negative tests: calls that MUST NOT share cache entries
# ---------------------------------------------------------------------------


class TestToolCacheFalseHitPrevention:
    """Tests that tool cache does NOT return cached results for different args."""

    @pytest.mark.asyncio
    async def test_different_args_same_tool_not_cached(self, redis_url: str) -> None:
        """Two calls to same tool with different args must return different results."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_false_hit",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                app_name = _extract_args(request).get("app_name", "unknown")
                return LangChainToolMessage(
                    content=f"Details for {app_name}",
                    name="get_app_details",
                    tool_call_id=_extract_tool_call_id(request),
                )

            result1 = await middleware.awrap_tool_call(
                {
                    "tool_name": "get_app_details",
                    "args": {"app_name": "SOMEAPP"},
                    "id": "c1",
                },
                handler,
            )
            assert call_count[0] == 1
            assert "SOMEAPP" in result1.content

            result2 = await middleware.awrap_tool_call(
                {
                    "tool_name": "get_app_details",
                    "args": {"app_name": "SOMEAPP2"},
                    "id": "c2",
                },
                handler,
            )
            assert call_count[0] == 2, (
                f"Handler should be called twice — got call_count={call_count[0]}. "
                "Cache returned a false hit for different arguments."
            )
            assert "SOMEAPP2" in result2.content

    @pytest.mark.asyncio
    async def test_similar_entity_names_not_confused(self, redis_url: str) -> None:
        """Entities that differ by only a suffix must not collide in cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_entity_names",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            results = []

            async def handler(request):
                entity = _extract_args(request).get("entity", "unknown")
                return LangChainToolMessage(
                    content=f"Data for {entity}",
                    name="lookup",
                    tool_call_id=_extract_tool_call_id(request),
                )

            entities = ["ProjectAlpha", "ProjectAlpha2", "ProjectBeta", "ProjectBeta2"]
            for entity in entities:
                result = await middleware.awrap_tool_call(
                    {
                        "tool_name": "lookup",
                        "args": {"entity": entity},
                        "id": f"c_{entity}",
                    },
                    handler,
                )
                results.append(result.content)

            for i, entity in enumerate(entities):
                assert entity in results[i], (
                    f"Expected '{entity}' in result[{i}]='{results[i]}'. "
                    "Cache returned wrong entity's data."
                )

    @pytest.mark.asyncio
    async def test_numeric_arg_difference(self, redis_url: str) -> None:
        """Args that differ only by a numeric value must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_numeric_diff",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                page = _extract_args(request).get("page", 0)
                return LangChainToolMessage(
                    content=f"Page {page} results",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            r1 = await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "redis", "page": 1},
                    "id": "c1",
                },
                handler,
            )
            r2 = await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "redis", "page": 2},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 2
            assert "Page 1" in r1.content
            assert "Page 2" in r2.content

    @pytest.mark.asyncio
    async def test_extra_arg_present_vs_absent(self, redis_url: str) -> None:
        """Calls with an extra arg vs without must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_extra_arg",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                args = _extract_args(request)
                return LangChainToolMessage(
                    content=f"args={args}",
                    name="fetch",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {
                    "tool_name": "fetch",
                    "args": {"url": "https://example.com"},
                    "id": "c1",
                },
                handler,
            )
            await middleware.awrap_tool_call(
                {
                    "tool_name": "fetch",
                    "args": {"url": "https://example.com", "timeout": 30},
                    "id": "c2",
                },
                handler,
            )
            assert call_count[0] == 2, "Extra arg should cause a cache miss"

    @pytest.mark.asyncio
    async def test_different_tool_names_same_args(self, redis_url: str) -> None:
        """Different tool names with identical args must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_cross_tool",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                if isinstance(request, dict):
                    name = request.get("tool_name", "")
                else:
                    tc = getattr(request, "tool_call", {})
                    name = tc.get("name", "") if isinstance(tc, dict) else ""
                return LangChainToolMessage(
                    content=f"from {name}",
                    name=name,
                    tool_call_id=_extract_tool_call_id(request),
                )

            r1 = await middleware.awrap_tool_call(
                {"tool_name": "search_web", "args": {"q": "hello"}, "id": "c1"},
                handler,
            )
            r2 = await middleware.awrap_tool_call(
                {"tool_name": "search_db", "args": {"q": "hello"}, "id": "c2"},
                handler,
            )

            assert call_count[0] == 2
            assert "search_web" in r1.content
            assert "search_db" in r2.content

    @pytest.mark.asyncio
    async def test_nested_object_one_field_differs(self, redis_url: str) -> None:
        """Nested args differing in one field must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_nested_diff",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                args = _extract_args(request)
                return LangChainToolMessage(
                    content=f"result={args}",
                    name="query",
                    tool_call_id=_extract_tool_call_id(request),
                )

            args_a = {"filter": {"status": "active", "region": "us-east-1"}}
            args_b = {"filter": {"status": "active", "region": "us-west-2"}}

            await middleware.awrap_tool_call(
                {"tool_name": "query", "args": args_a, "id": "c1"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "query", "args": args_b, "id": "c2"}, handler
            )

            assert call_count[0] == 2, "Nested arg difference should cause a cache miss"

    @pytest.mark.asyncio
    async def test_empty_string_vs_missing_key(self, redis_url: str) -> None:
        """Args with empty string value vs absent key must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_empty_vs_missing",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                args = _extract_args(request)
                return LangChainToolMessage(
                    content=f"args={args}",
                    name="tool",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {"tool_name": "tool", "args": {"name": ""}, "id": "c1"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "tool", "args": {}, "id": "c2"}, handler
            )

            assert call_count[0] == 2, "Empty string and missing key are different args"

    @pytest.mark.asyncio
    async def test_interleaved_calls_return_correct_results(
        self, redis_url: str
    ) -> None:
        """Alternating A/B/A/B calls must return the right result each time."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_interleaved",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                city = _extract_args(request).get("city", "")
                return LangChainToolMessage(
                    content=f"Weather in {city}",
                    name="weather",
                    tool_call_id=_extract_tool_call_id(request),
                )

            results = []
            for city in ["Paris", "London", "Paris", "London"]:
                r = await middleware.awrap_tool_call(
                    {"tool_name": "weather", "args": {"city": city}, "id": f"c_{city}"},
                    handler,
                )
                results.append(r.content)

            # Handler called twice (Paris miss, London miss, Paris hit, London hit)
            assert call_count[0] == 2
            assert "Weather in Paris" in results[0] and "Weather in Paris" in results[2]
            assert (
                "Weather in London" in results[1] and "Weather in London" in results[3]
            )

    @pytest.mark.asyncio
    async def test_tool_call_request_objects_not_confused(self, redis_url: str) -> None:
        """ToolCallRequest objects with different args must not share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_tcr_objects",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                args = _extract_args(request)
                return LangChainToolMessage(
                    content=f"result for {args.get('id', '')}",
                    name="lookup",
                    tool_call_id=_extract_tool_call_id(request),
                )

            tcr1 = ToolCallRequest(
                tool_call={"name": "lookup", "args": {"id": "user_101"}, "id": "c1"},
                tool=None,
                state={},
                runtime={},
            )
            tcr2 = ToolCallRequest(
                tool_call={"name": "lookup", "args": {"id": "user_102"}, "id": "c2"},
                tool=None,
                state={},
                runtime={},
            )

            r1 = await middleware.awrap_tool_call(tcr1, handler)
            r2 = await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 2
            assert "user_101" in r1.content
            assert "user_102" in r2.content


# ---------------------------------------------------------------------------
# Positive tests: calls that MUST share cache entries
# ---------------------------------------------------------------------------


class TestToolCachePositiveHits:
    """Tests that identical tool calls DO return cached results."""

    @pytest.mark.asyncio
    async def test_same_args_same_tool_is_cached(self, redis_url: str) -> None:
        """Two calls with identical args should use cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_exact_hit",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="App details result",
                    name="get_app_details",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "get_app_details",
                "args": {"app_name": "SOMEAPP"},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)

            req2 = {
                "tool_name": "get_app_details",
                "args": {"app_name": "SOMEAPP"},
                "id": "c2",
            }
            await middleware.awrap_tool_call(req2, handler)

            assert call_count[0] == 1, "Identical args should hit cache"

    @pytest.mark.asyncio
    async def test_different_key_order_same_values_cached(self, redis_url: str) -> None:
        """Args with different key insertion order but same values should cache hit.

        _build_cache_key uses json.dumps(args, sort_keys=True), so
        {"b": 2, "a": 1} and {"a": 1, "b": 2} produce the same key.
        """
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_key_order",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {"tool_name": "search", "args": {"b": 2, "a": 1}, "id": "c1"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "search", "args": {"a": 1, "b": 2}, "id": "c2"}, handler
            )

            assert call_count[0] == 1, "sort_keys=True should normalise key order"

    @pytest.mark.asyncio
    async def test_cached_result_preserves_tool_message_fields(
        self, redis_url: str
    ) -> None:
        """Cached ToolMessage must have correct name and content."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_fields",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:

            async def handler(request):
                return LangChainToolMessage(
                    content="42",
                    name="calculate",
                    tool_call_id=_extract_tool_call_id(request),
                )

            # Store
            await middleware.awrap_tool_call(
                {"tool_name": "calculate", "args": {"expr": "6*7"}, "id": "c1"},
                handler,
            )
            # Retrieve from cache
            cached = await middleware.awrap_tool_call(
                {"tool_name": "calculate", "args": {"expr": "6*7"}, "id": "c2"},
                handler,
            )

            assert isinstance(cached, LangChainToolMessage)
            assert cached.name == "calculate"
            assert "42" in cached.content

    @pytest.mark.asyncio
    async def test_multiple_tools_cached_independently(self, redis_url: str) -> None:
        """Different tools with same args each get their own cache entry."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_independent",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                if isinstance(request, dict):
                    name = request.get("tool_name", "")
                else:
                    tc = getattr(request, "tool_call", {})
                    name = tc.get("name", "") if isinstance(tc, dict) else ""
                return LangChainToolMessage(
                    content=f"from {name}",
                    name=name,
                    tool_call_id=_extract_tool_call_id(request),
                )

            args = {"q": "test"}

            # Two different tools, same args — 2 handler calls
            await middleware.awrap_tool_call(
                {"tool_name": "tool_a", "args": args, "id": "c1"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "tool_b", "args": args, "id": "c2"}, handler
            )
            assert call_count[0] == 2

            # Re-call each — both should now be cached
            ra = await middleware.awrap_tool_call(
                {"tool_name": "tool_a", "args": args, "id": "c3"}, handler
            )
            rb = await middleware.awrap_tool_call(
                {"tool_name": "tool_b", "args": args, "id": "c4"}, handler
            )
            assert call_count[0] == 2, "Re-calls should hit cache"
            assert "tool_a" in ra.content
            assert "tool_b" in rb.content

    @pytest.mark.asyncio
    async def test_many_identical_calls_handler_once(self, redis_url: str) -> None:
        """Calling same tool 10 times with identical args invokes handler once."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_many_calls",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="pong",
                    name="ping",
                    tool_call_id=_extract_tool_call_id(request),
                )

            for i in range(10):
                r = await middleware.awrap_tool_call(
                    {"tool_name": "ping", "args": {"host": "redis.io"}, "id": f"c{i}"},
                    handler,
                )
                assert "pong" in r.content

            assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_tool_call_request_object_exact_cache_hit(
        self, redis_url: str
    ) -> None:
        """ToolCallRequest objects with identical args should cache hit."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tcr_exact_hit",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="user data",
                    name="get_user",
                    tool_call_id=_extract_tool_call_id(request),
                )

            tcr1 = ToolCallRequest(
                tool_call={"name": "get_user", "args": {"id": "u42"}, "id": "c1"},
                tool=None,
                state={},
                runtime={},
            )
            tcr2 = ToolCallRequest(
                tool_call={"name": "get_user", "args": {"id": "u42"}, "id": "c2"},
                tool=None,
                state={},
                runtime={},
            )

            await middleware.awrap_tool_call(tcr1, handler)
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 1, "Identical ToolCallRequest args should cache hit"


# ---------------------------------------------------------------------------
# Config-based caching behaviour
# ---------------------------------------------------------------------------


class TestToolCacheConfigBehaviour:
    """Tests for cacheable_tools / excluded_tools config options."""

    @pytest.mark.asyncio
    async def test_excluded_tool_always_calls_handler(self, redis_url: str) -> None:
        """Tool in excluded_tools must never be cached."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_excluded",
            excluded_tools=["random_number"],
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content=f"roll {call_count[0]}",
                    name="random_number",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {"tool_name": "random_number", "args": {"sides": 6}, "id": "c1"}
            r1 = await middleware.awrap_tool_call(req, handler)
            r2 = await middleware.awrap_tool_call(
                {"tool_name": "random_number", "args": {"sides": 6}, "id": "c2"},
                handler,
            )

            assert call_count[0] == 2, "Excluded tool should never cache"
            assert r1.content != r2.content

    @pytest.mark.asyncio
    async def test_cacheable_tools_whitelist(self, redis_url: str) -> None:
        """Only tools in cacheable_tools should be cached."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_whitelist",
            cacheable_tools=["search"],
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            search_count = [0]
            calc_count = [0]

            async def handler(request):
                if isinstance(request, dict):
                    name = request.get("tool_name", "")
                else:
                    tc = getattr(request, "tool_call", {})
                    name = tc.get("name", "") if isinstance(tc, dict) else ""
                if name == "search":
                    search_count[0] += 1
                else:
                    calc_count[0] += 1
                return LangChainToolMessage(
                    content=f"{name} result",
                    name=name,
                    tool_call_id=_extract_tool_call_id(request),
                )

            args = {"q": "hello"}

            # search: cached after first call
            await middleware.awrap_tool_call(
                {"tool_name": "search", "args": args, "id": "c1"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "search", "args": args, "id": "c2"}, handler
            )
            assert search_count[0] == 1, "search should be cached"

            # calculate: not in whitelist, always calls handler
            await middleware.awrap_tool_call(
                {"tool_name": "calculate", "args": args, "id": "c3"}, handler
            )
            await middleware.awrap_tool_call(
                {"tool_name": "calculate", "args": args, "id": "c4"}, handler
            )
            assert (
                calc_count[0] == 2
            ), "calculate not in cacheable_tools, should not cache"


# ---------------------------------------------------------------------------
# Semantic cache (model-level) tests
# ---------------------------------------------------------------------------


@requires_sentence_transformers
class TestSemanticCacheFalseHitPrevention:
    """Tests that semantic model cache behaviour is controlled by distance_threshold."""

    @pytest.mark.asyncio
    async def test_similar_prompts_different_entities_tight_threshold(
        self, redis_url: str
    ) -> None:
        """With a tight threshold, prompts differing by entity should not share cache."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_semantic_tight_threshold",
            distance_threshold=0.05,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                messages = (
                    request.get("messages", [])
                    if isinstance(request, dict)
                    else getattr(request, "messages", [])
                )
                prompt = messages[-1].get("content", "") if messages else ""
                return {"content": f"Response to: {prompt}"}

            await middleware.awrap_model_call(
                {
                    "messages": [
                        {"role": "user", "content": "What are app details for SOMEAPP"}
                    ]
                },
                handler,
            )
            await middleware.awrap_model_call(
                {
                    "messages": [
                        {"role": "user", "content": "What are app details for SOMEAPP2"}
                    ]
                },
                handler,
            )
            assert call_count[0] == 2, (
                f"Handler should be called twice — got call_count={call_count[0]}. "
                "Tight threshold should prevent false hit for different entity."
            )

    @pytest.mark.asyncio
    async def test_identical_prompts_cached(self, redis_url: str) -> None:
        """Identical prompts must return cached result."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_semantic_exact_hit",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return {"content": "The answer is 42"}

            msg = {
                "messages": [
                    {"role": "user", "content": "What is the meaning of life?"}
                ]
            }

            await middleware.awrap_model_call(msg, handler)
            await middleware.awrap_model_call(msg, handler)

            assert call_count[0] == 1, "Identical prompts should hit cache"

    @pytest.mark.asyncio
    async def test_numeric_suffix_prompts_tight_threshold(self, redis_url: str) -> None:
        """With tight threshold, 'order 1001' vs 'order 1002' should not share cache."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_semantic_numeric_tight",
            distance_threshold=0.05,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                messages = (
                    request.get("messages", [])
                    if isinstance(request, dict)
                    else getattr(request, "messages", [])
                )
                prompt = messages[-1].get("content", "") if messages else ""
                return {"content": f"Response: {prompt}"}

            await middleware.awrap_model_call(
                {"messages": [{"role": "user", "content": "Show me order 1001"}]},
                handler,
            )
            await middleware.awrap_model_call(
                {"messages": [{"role": "user", "content": "Show me order 1002"}]},
                handler,
            )

            assert call_count[0] == 2, "Different order numbers must not share cache"

    @pytest.mark.asyncio
    async def test_truly_different_prompts_miss_cache(self, redis_url: str) -> None:
        """Completely different prompts must always miss cache."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_semantic_different",
            distance_threshold=0.3,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return {"content": f"Response {call_count[0]}"}

            await middleware.awrap_model_call(
                {
                    "messages": [
                        {"role": "user", "content": "What is the weather in Paris?"}
                    ]
                },
                handler,
            )
            await middleware.awrap_model_call(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "How do I bake a chocolate cake?",
                        }
                    ]
                },
                handler,
            )

            assert call_count[0] == 2, "Completely different prompts must miss cache"


# ---------------------------------------------------------------------------
# Volatile arg detection tests
# ---------------------------------------------------------------------------


class TestVolatileArgDetection:
    """Tests that volatile arg names prevent caching."""

    @pytest.mark.asyncio
    async def test_timestamp_arg_skips_cache(self, redis_url: str) -> None:
        """Args containing a volatile name like 'timestamp' should skip cache."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_VOLATILE_ARG_NAMES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_volatile_ts",
            distance_threshold=0.2,
            ttl_seconds=60,
            volatile_arg_names=DEFAULT_VOLATILE_ARG_NAMES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "search",
                "args": {"query": "x", "timestamp": 123},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(req, handler)

            assert call_count[0] == 2, "Volatile arg 'timestamp' should prevent caching"

    @pytest.mark.asyncio
    async def test_nested_date_arg_skips_cache(self, redis_url: str) -> None:
        """Nested volatile arg name should also prevent caching."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_VOLATILE_ARG_NAMES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_volatile_nested",
            distance_threshold=0.2,
            ttl_seconds=60,
            volatile_arg_names=DEFAULT_VOLATILE_ARG_NAMES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="query",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "query",
                "args": {"filter": {"date": "2024-01-01"}},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(req, handler)

            assert call_count[0] == 2, "Nested 'date' arg should prevent caching"

    @pytest.mark.asyncio
    async def test_no_volatile_args_caches_normally(self, redis_url: str) -> None:
        """Without volatile args, caching should work normally."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_VOLATILE_ARG_NAMES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_volatile_normal",
            distance_threshold=0.2,
            ttl_seconds=60,
            volatile_arg_names=DEFAULT_VOLATILE_ARG_NAMES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {"tool_name": "search", "args": {"query": "hello"}, "id": "c1"}
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {"tool_name": "search", "args": {"query": "hello"}, "id": "c2"},
                handler,
            )

            assert call_count[0] == 1, "No volatile args — should cache normally"

    @pytest.mark.asyncio
    async def test_volatile_detection_disabled_empty_set(self, redis_url: str) -> None:
        """volatile_arg_names=set() should disable volatile detection."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_volatile_disabled",
            distance_threshold=0.2,
            ttl_seconds=60,
            volatile_arg_names=set(),
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "search",
                "args": {"query": "x", "timestamp": 123},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "x", "timestamp": 123},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 1, "Empty set disables volatile detection"

    @pytest.mark.asyncio
    async def test_custom_volatile_arg_names(self, redis_url: str) -> None:
        """Custom volatile names like 'as_of' should prevent caching."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_volatile_custom",
            distance_threshold=0.2,
            ttl_seconds=60,
            volatile_arg_names={"as_of", "effective_date"},
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="report",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "report",
                "args": {"metric": "revenue", "as_of": "2024-03-01"},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(req, handler)

            assert call_count[0] == 2, "Custom volatile 'as_of' should prevent caching"


# ---------------------------------------------------------------------------
# Ignored args tests
# ---------------------------------------------------------------------------


class TestIgnoredArgs:
    """Tests that ignored args are stripped from cache keys."""

    @pytest.mark.asyncio
    async def test_different_request_id_hits_cache(self, redis_url: str) -> None:
        """Calls differing only by request_id should share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_ignored_rid",
            distance_threshold=0.2,
            ttl_seconds=60,
            ignored_arg_names={"request_id", "trace_id"},
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "request_id": "r1"},
                    "id": "c1",
                },
                handler,
            )
            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "request_id": "r2"},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 1, "Different request_id should still hit cache"

    @pytest.mark.asyncio
    async def test_different_trace_id_hits_cache(self, redis_url: str) -> None:
        """Calls differing only by trace_id should share cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_ignored_tid",
            distance_threshold=0.2,
            ttl_seconds=60,
            ignored_arg_names={"request_id", "trace_id"},
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "trace_id": "t1"},
                    "id": "c1",
                },
                handler,
            )
            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "trace_id": "t2"},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 1, "Different trace_id should still hit cache"

    @pytest.mark.asyncio
    async def test_non_ignored_arg_difference_misses_cache(
        self, redis_url: str
    ) -> None:
        """Calls differing by a non-ignored arg should miss cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_ignored_miss",
            distance_threshold=0.2,
            ttl_seconds=60,
            ignored_arg_names={"request_id", "trace_id"},
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="search",
                    tool_call_id=_extract_tool_call_id(request),
                )

            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "user_id": "u1"},
                    "id": "c1",
                },
                handler,
            )
            await middleware.awrap_tool_call(
                {
                    "tool_name": "search",
                    "args": {"query": "hello", "user_id": "u2"},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 2, "Different user_id is not ignored — cache miss"


# ---------------------------------------------------------------------------
# MCP-style metadata tests
# ---------------------------------------------------------------------------


class TestMCPStyleMetadata:
    """Tests for MCP-style metadata fields: destructive, volatile, read_only, idempotent."""

    @pytest.mark.asyncio
    async def test_destructive_metadata_never_cached(self, redis_url: str) -> None:
        """Tool with metadata={'destructive': True} should never cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_mcp_destructive",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="deleted",
                    name="wipe_data",
                    tool_call_id=_extract_tool_call_id(request),
                )

            from unittest.mock import MagicMock

            mock_tool = MagicMock()
            mock_tool.metadata = {"destructive": True}
            tcr = ToolCallRequest(
                tool_call={"name": "wipe_data", "args": {"table": "users"}, "id": "c1"},
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr, handler)

            tcr2 = ToolCallRequest(
                tool_call={"name": "wipe_data", "args": {"table": "users"}, "id": "c2"},
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 2, "Destructive metadata should prevent caching"

    @pytest.mark.asyncio
    async def test_volatile_metadata_never_cached(self, redis_url: str) -> None:
        """Tool with metadata={'volatile': True} should never cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_mcp_volatile",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="time is now",
                    name="get_time",
                    tool_call_id=_extract_tool_call_id(request),
                )

            from unittest.mock import MagicMock

            mock_tool = MagicMock()
            mock_tool.metadata = {"volatile": True}
            tcr = ToolCallRequest(
                tool_call={"name": "get_time", "args": {}, "id": "c1"},
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr, handler)

            tcr2 = ToolCallRequest(
                tool_call={"name": "get_time", "args": {}, "id": "c2"},
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 2, "Volatile metadata should prevent caching"

    @pytest.mark.asyncio
    async def test_read_only_idempotent_cached(self, redis_url: str) -> None:
        """Tool with read_only=True + idempotent=True should cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_mcp_readonly",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="data",
                    name="get_user",
                    tool_call_id=_extract_tool_call_id(request),
                )

            from unittest.mock import MagicMock

            mock_tool = MagicMock()
            mock_tool.metadata = {"read_only": True, "idempotent": True}
            tcr = ToolCallRequest(
                tool_call={
                    "name": "get_user",
                    "args": {"id": "u42"},
                    "id": "c1",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr, handler)

            tcr2 = ToolCallRequest(
                tool_call={
                    "name": "get_user",
                    "args": {"id": "u42"},
                    "id": "c2",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 1, "read_only+idempotent should cache"

    @pytest.mark.asyncio
    async def test_cacheable_true_overrides_destructive(self, redis_url: str) -> None:
        """Explicit cacheable=True should override destructive=True."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_mcp_override",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="result",
                    name="special_tool",
                    tool_call_id=_extract_tool_call_id(request),
                )

            from unittest.mock import MagicMock

            mock_tool = MagicMock()
            mock_tool.metadata = {"cacheable": True, "destructive": True}
            tcr = ToolCallRequest(
                tool_call={
                    "name": "special_tool",
                    "args": {"x": 1},
                    "id": "c1",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr, handler)

            tcr2 = ToolCallRequest(
                tool_call={
                    "name": "special_tool",
                    "args": {"x": 1},
                    "id": "c2",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 1, "cacheable=True overrides destructive"


# ---------------------------------------------------------------------------
# Side-effect prefix tests
# ---------------------------------------------------------------------------


class TestSideEffectPrefixes:
    """Tests for side-effect prefix matching."""

    @pytest.mark.asyncio
    async def test_send_prefix_never_cached(self, redis_url: str) -> None:
        """Tool named 'send_email' should skip cache when prefixes enabled."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_SIDE_EFFECT_PREFIXES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_prefix_send",
            distance_threshold=0.2,
            ttl_seconds=60,
            side_effect_prefixes=DEFAULT_SIDE_EFFECT_PREFIXES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="sent",
                    name="send_email",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "send_email",
                "args": {"to": "user@test.com"},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {
                    "tool_name": "send_email",
                    "args": {"to": "user@test.com"},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 2, "send_ prefix should prevent caching"

    @pytest.mark.asyncio
    async def test_delete_prefix_never_cached(self, redis_url: str) -> None:
        """Tool named 'delete_record' should skip cache."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_SIDE_EFFECT_PREFIXES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_prefix_delete",
            distance_threshold=0.2,
            ttl_seconds=60,
            side_effect_prefixes=DEFAULT_SIDE_EFFECT_PREFIXES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="deleted",
                    name="delete_record",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "delete_record",
                "args": {"id": "r1"},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {"tool_name": "delete_record", "args": {"id": "r1"}, "id": "c2"},
                handler,
            )

            assert call_count[0] == 2, "delete_ prefix should prevent caching"

    @pytest.mark.asyncio
    async def test_get_prefix_cached_normally(self, redis_url: str) -> None:
        """Tool named 'get_data' should cache normally."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_SIDE_EFFECT_PREFIXES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_prefix_get",
            distance_threshold=0.2,
            ttl_seconds=60,
            side_effect_prefixes=DEFAULT_SIDE_EFFECT_PREFIXES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="data",
                    name="get_data",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {"tool_name": "get_data", "args": {"key": "a"}, "id": "c1"}
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {"tool_name": "get_data", "args": {"key": "a"}, "id": "c2"},
                handler,
            )

            assert call_count[0] == 1, "get_ prefix should cache normally"

    @pytest.mark.asyncio
    async def test_cacheable_metadata_overrides_prefix(self, redis_url: str) -> None:
        """cacheable=True in metadata should override side-effect prefix."""
        from langgraph.middleware.redis.tool_cache import DEFAULT_SIDE_EFFECT_PREFIXES

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_prefix_override",
            distance_threshold=0.2,
            ttl_seconds=60,
            side_effect_prefixes=DEFAULT_SIDE_EFFECT_PREFIXES,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="ok",
                    name="delete_user",
                    tool_call_id=_extract_tool_call_id(request),
                )

            from unittest.mock import MagicMock

            mock_tool = MagicMock()
            mock_tool.metadata = {"cacheable": True}

            tcr = ToolCallRequest(
                tool_call={
                    "name": "delete_user",
                    "args": {"id": "u1"},
                    "id": "c1",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr, handler)

            tcr2 = ToolCallRequest(
                tool_call={
                    "name": "delete_user",
                    "args": {"id": "u1"},
                    "id": "c2",
                },
                tool=mock_tool,
                state={},
                runtime={},
            )
            await middleware.awrap_tool_call(tcr2, handler)

            assert call_count[0] == 1, "cacheable=True overrides side-effect prefix"

    @pytest.mark.asyncio
    async def test_side_effect_disabled_empty_tuple(self, redis_url: str) -> None:
        """side_effect_prefixes=() should disable prefix checking."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_prefix_disabled",
            distance_threshold=0.2,
            ttl_seconds=60,
            side_effect_prefixes=(),
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def handler(request):
                call_count[0] += 1
                return LangChainToolMessage(
                    content="sent",
                    name="send_email",
                    tool_call_id=_extract_tool_call_id(request),
                )

            req = {
                "tool_name": "send_email",
                "args": {"to": "user@test.com"},
                "id": "c1",
            }
            await middleware.awrap_tool_call(req, handler)
            await middleware.awrap_tool_call(
                {
                    "tool_name": "send_email",
                    "args": {"to": "user@test.com"},
                    "id": "c2",
                },
                handler,
            )

            assert call_count[0] == 1, "Empty tuple disables prefix check"
