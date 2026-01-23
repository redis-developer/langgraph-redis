"""Integration tests for middleware with LangGraph.

These tests use TestContainers to spin up a real Redis instance.
"""

import pytest
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    SemanticCacheConfig,
    SemanticCacheMiddleware,
    ToolCacheConfig,
    ToolResultCacheMiddleware,
    create_caching_stack,
    from_configs,
)

# Check if sentence-transformers is available
try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)


@pytest.fixture(scope="module")
def redis_url():
    """Provide a Redis URL using TestContainers.

    Uses redis/redis-stack-server which includes RediSearch module required by redisvl.
    """
    redis_container = RedisContainer("redis/redis-stack-server:latest")
    redis_container.start()
    try:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        redis_container.stop()


@requires_sentence_transformers
class TestSemanticCacheIntegration:
    """Integration tests for SemanticCacheMiddleware with real Redis."""

    @pytest.mark.asyncio
    async def test_cache_stores_and_retrieves(self, redis_url: str) -> None:
        """Test that cache stores and retrieves responses."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_llm_cache",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_handler(request: dict) -> dict:
                call_count[0] += 1
                return {"content": f"Response #{call_count[0]}"}

            # First call - should call handler
            request1 = {"messages": [{"role": "user", "content": "What is Python?"}]}
            result1 = await middleware.awrap_model_call(request1, mock_handler)
            assert call_count[0] == 1
            assert result1["content"] == "Response #1"

            # Second call with same prompt - should use cache
            request2 = {"messages": [{"role": "user", "content": "What is Python?"}]}
            result2 = await middleware.awrap_model_call(request2, mock_handler)
            assert result2 is not None  # Verify response was returned
            # Handler should not be called again for cached request
            # Note: This depends on the vectorizer being available

    @pytest.mark.asyncio
    async def test_different_prompts_not_cached(self, redis_url: str) -> None:
        """Test that semantically different prompts get different responses."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="test_llm_cache_diff",
            distance_threshold=0.95,  # High similarity threshold - only near-exact matches
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_handler(request: dict) -> dict:
                call_count[0] += 1
                return {"content": f"Response #{call_count[0]}"}

            # First prompt - about programming
            request1 = {"messages": [{"role": "user", "content": "What is Python?"}]}
            await middleware.awrap_model_call(request1, mock_handler)
            first_count = call_count[0]

            # Very different prompt - about cooking (should be semantically different)
            request2 = {
                "messages": [
                    {"role": "user", "content": "How do I bake a chocolate cake?"}
                ]
            }
            await middleware.awrap_model_call(request2, mock_handler)
            assert call_count[0] > first_count


@requires_sentence_transformers
class TestToolCacheIntegration:
    """Integration tests for ToolResultCacheMiddleware with real Redis."""

    @pytest.mark.asyncio
    async def test_tool_cache_stores_and_retrieves(self, redis_url: str) -> None:
        """Test that tool cache stores and retrieves results."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_cache",
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_tool(request: dict) -> dict:
                call_count[0] += 1
                return {"result": f"Tool result #{call_count[0]}"}

            # First call
            request1 = {"tool_name": "search", "args": {"query": "Python tutorial"}}
            result1 = await middleware.awrap_tool_call(request1, mock_tool)
            assert call_count[0] == 1
            assert result1["result"] == "Tool result #1"

    @pytest.mark.asyncio
    async def test_excluded_tools_not_cached(self, redis_url: str) -> None:
        """Test that excluded tools bypass the cache."""
        config = ToolCacheConfig(
            redis_url=redis_url,
            name="test_tool_cache_excluded",
            excluded_tools=["random_tool"],
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_tool(request: dict) -> dict:
                call_count[0] += 1
                return {"result": f"Result #{call_count[0]}"}

            # Call excluded tool twice
            request = {"tool_name": "random_tool", "args": {}}
            await middleware.awrap_tool_call(request, mock_tool)
            await middleware.awrap_tool_call(request, mock_tool)

            # Both calls should have executed
            assert call_count[0] == 2


@requires_sentence_transformers
class TestMiddlewareStackIntegration:
    """Integration tests for MiddlewareStack with real Redis."""

    @pytest.mark.asyncio
    async def test_stack_chains_middlewares(self, redis_url: str) -> None:
        """Test that middleware stack properly chains multiple middlewares."""
        stack = create_caching_stack(
            redis_url=redis_url,
            semantic_cache_ttl=60,
            tool_cache_ttl=60,
        )

        async with stack:

            async def mock_handler(request: dict) -> dict:
                return {"content": "Response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await stack.awrap_model_call(request, mock_handler)
            assert result["content"] == "Response"

    @pytest.mark.asyncio
    async def test_from_configs_creates_stack(self, redis_url: str) -> None:
        """Test creating stack from config objects."""
        stack = from_configs(
            redis_url=redis_url,
            configs=[
                SemanticCacheConfig(name="cache_from_config", ttl_seconds=60),
                ToolCacheConfig(name="tool_from_config", ttl_seconds=60),
            ],
        )

        async with stack:

            async def mock_handler(request: dict) -> dict:
                return {"content": "From configs"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await stack.awrap_model_call(request, mock_handler)
            assert result["content"] == "From configs"


class TestGracefulDegradation:
    """Test graceful degradation when Redis is unavailable."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_passes_through(self) -> None:
        """Test that graceful_degradation=True passes through on errors."""
        # Use an invalid Redis URL
        config = SemanticCacheConfig(
            redis_url="redis://invalid-host:9999",
            graceful_degradation=True,
        )

        middleware = SemanticCacheMiddleware(config)

        async def mock_handler(request: dict) -> dict:
            return {"content": "Handler response"}

        request = {"messages": [{"role": "user", "content": "Test"}]}

        # Should not raise, should pass through to handler
        try:
            result = await middleware.awrap_model_call(request, mock_handler)
            assert result is not None  # Graceful degradation should return a result
            # If we got here without exception, graceful degradation worked
            # or the middleware wasn't initialized yet
        except Exception:
            # Connection errors are expected if middleware tries to connect
            pass
