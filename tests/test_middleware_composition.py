"""Unit tests for middleware composition."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.middleware.redis.types import (
    SemanticCacheConfig,
    ToolCacheConfig,
)


class TestMiddlewareStack:
    """Tests for MiddlewareStack class."""

    @pytest.mark.asyncio
    async def test_init_with_middlewares(self) -> None:
        """Test initialization with a list of middlewares."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware
        from langgraph.middleware.redis.composition import MiddlewareStack

        class MockMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        mock_client = AsyncMock()
        from langgraph.middleware.redis.types import MiddlewareConfig

        m1 = MockMiddleware(MiddlewareConfig(redis_client=mock_client))
        m2 = MockMiddleware(MiddlewareConfig(redis_client=mock_client))

        stack = MiddlewareStack([m1, m2])
        assert len(stack._middlewares) == 2

    @pytest.mark.asyncio
    async def test_awrap_model_call_chains_middlewares(self) -> None:
        """Test that model calls are chained through all middlewares."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware
        from langgraph.middleware.redis.composition import MiddlewareStack

        call_order = []

        class FirstMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

            async def awrap_model_call(self, request, handler):
                call_order.append("first_before")
                result = await handler(request)
                call_order.append("first_after")
                return result

        class SecondMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

            async def awrap_model_call(self, request, handler):
                call_order.append("second_before")
                result = await handler(request)
                call_order.append("second_after")
                return result

        mock_client = AsyncMock()
        from langgraph.middleware.redis.types import MiddlewareConfig

        m1 = FirstMiddleware(MiddlewareConfig(redis_client=mock_client))
        m2 = SecondMiddleware(MiddlewareConfig(redis_client=mock_client))

        stack = MiddlewareStack([m1, m2])

        async def final_handler(request):
            call_order.append("handler")
            return {"content": "response"}

        await stack.awrap_model_call({}, final_handler)

        # First middleware wraps second, which wraps handler
        assert call_order == [
            "first_before",
            "second_before",
            "handler",
            "second_after",
            "first_after",
        ]

    @pytest.mark.asyncio
    async def test_awrap_tool_call_chains_middlewares(self) -> None:
        """Test that tool calls are chained through all middlewares."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware
        from langgraph.middleware.redis.composition import MiddlewareStack

        call_order = []

        class TrackingMiddleware(AsyncRedisMiddleware):
            def __init__(self, config, tracking_name):
                super().__init__(config)
                self._tracking_name = tracking_name

            async def _setup_async(self) -> None:
                pass

            async def awrap_tool_call(self, request, handler):
                call_order.append(f"{self._tracking_name}_before")
                result = await handler(request)
                call_order.append(f"{self._tracking_name}_after")
                return result

        mock_client = AsyncMock()
        from langgraph.middleware.redis.types import MiddlewareConfig

        m1 = TrackingMiddleware(MiddlewareConfig(redis_client=mock_client), "first")
        m2 = TrackingMiddleware(MiddlewareConfig(redis_client=mock_client), "second")

        stack = MiddlewareStack([m1, m2])

        async def final_handler(request):
            call_order.append("handler")
            return {"result": "tool_result"}

        await stack.awrap_tool_call({}, final_handler)

        assert call_order == [
            "first_before",
            "second_before",
            "handler",
            "second_after",
            "first_after",
        ]

    @pytest.mark.asyncio
    async def test_empty_stack_passes_through(self) -> None:
        """Test that empty stack passes through to handler."""
        from langgraph.middleware.redis.composition import MiddlewareStack

        stack = MiddlewareStack([])

        async def handler(request):
            return {"content": "direct response"}

        result = await stack.awrap_model_call({}, handler)
        assert result == {"content": "direct response"}

    @pytest.mark.asyncio
    async def test_aclose_closes_all_middlewares(self) -> None:
        """Test that aclose() closes all middlewares."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware
        from langgraph.middleware.redis.composition import MiddlewareStack

        closed = []

        class CloseableMiddleware(AsyncRedisMiddleware):
            def __init__(self, config, tracking_name):
                super().__init__(config)
                self._tracking_name = tracking_name

            async def _setup_async(self) -> None:
                pass

            async def aclose(self) -> None:
                closed.append(self._tracking_name)
                await super().aclose()

        mock_client = AsyncMock()
        from langgraph.middleware.redis.types import MiddlewareConfig

        m1 = CloseableMiddleware(MiddlewareConfig(redis_client=mock_client), "first")
        m2 = CloseableMiddleware(MiddlewareConfig(redis_client=mock_client), "second")

        stack = MiddlewareStack([m1, m2])
        await stack.aclose()

        assert "first" in closed
        assert "second" in closed

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test that MiddlewareStack works as async context manager."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware
        from langgraph.middleware.redis.composition import MiddlewareStack

        closed = []

        class TrackingMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

            async def aclose(self) -> None:
                closed.append(True)

        mock_client = AsyncMock()
        from langgraph.middleware.redis.types import MiddlewareConfig

        m1 = TrackingMiddleware(MiddlewareConfig(redis_client=mock_client))

        async with MiddlewareStack([m1]) as stack:
            assert stack is not None

        assert len(closed) == 1


class TestFromConfigs:
    """Tests for from_configs factory function."""

    @pytest.mark.asyncio
    async def test_creates_stack_from_configs(self) -> None:
        """Test creating middleware stack from config objects."""
        from langgraph.middleware.redis.composition import from_configs

        mock_client = AsyncMock()

        with (
            patch("langgraph.middleware.redis.semantic_cache.SemanticCache"),
            patch("langgraph.middleware.redis.tool_cache.SemanticCache"),
        ):
            stack = from_configs(
                redis_client=mock_client,
                configs=[
                    SemanticCacheConfig(name="cache1"),
                    ToolCacheConfig(name="tool_cache1"),
                ],
            )

            assert len(stack._middlewares) == 2

    @pytest.mark.asyncio
    async def test_creates_from_redis_url(self) -> None:
        """Test creating middleware stack from redis_url."""
        from langgraph.middleware.redis.composition import from_configs

        with (
            patch("langgraph.middleware.redis.semantic_cache.SemanticCache"),
            patch("redis.asyncio.Redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = AsyncMock()

            stack = from_configs(
                redis_url="redis://localhost:6379",
                configs=[SemanticCacheConfig()],
            )

            assert len(stack._middlewares) == 1


class TestCreateCachingStack:
    """Tests for create_caching_stack convenience function."""

    @pytest.mark.asyncio
    async def test_creates_semantic_and_tool_cache(self) -> None:
        """Test that create_caching_stack creates both cache types."""
        from langgraph.middleware.redis.composition import create_caching_stack

        mock_client = AsyncMock()

        with (
            patch("langgraph.middleware.redis.semantic_cache.SemanticCache"),
            patch("langgraph.middleware.redis.tool_cache.SemanticCache"),
        ):
            stack = create_caching_stack(
                redis_client=mock_client,
                semantic_cache_ttl=3600,
                tool_cache_ttl=7200,
            )

            assert len(stack._middlewares) == 2

    @pytest.mark.asyncio
    async def test_respects_cacheable_tools(self) -> None:
        """Test that cacheable_tools is passed to tool cache."""
        from langgraph.middleware.redis.composition import create_caching_stack
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()

        with (
            patch("langgraph.middleware.redis.semantic_cache.SemanticCache"),
            patch("langgraph.middleware.redis.tool_cache.SemanticCache"),
        ):
            stack = create_caching_stack(
                redis_client=mock_client,
                cacheable_tools=["search", "calculate"],
            )

            # Find the tool cache middleware
            tool_cache = None
            for m in stack._middlewares:
                if isinstance(m, ToolResultCacheMiddleware):
                    tool_cache = m
                    break

            assert tool_cache is not None
            assert tool_cache._config.cacheable_tools == ["search", "calculate"]


class TestIntegratedRedisMiddleware:
    """Tests for IntegratedRedisMiddleware connection sharing."""

    @pytest.mark.asyncio
    async def test_from_saver_extracts_url(self) -> None:
        """Test creating middleware that extracts redis_url from saver.

        Note: redisvl requires sync Redis connections, so middleware creates
        its own connection using the extracted redis_url.
        """
        from langgraph.middleware.redis.composition import IntegratedRedisMiddleware

        # Mock a RedisSaver with connection pool info
        mock_saver = MagicMock()
        mock_saver._redis_url = "redis://localhost:6379"

        with patch("langgraph.middleware.redis.semantic_cache.SemanticCache"):
            stack = IntegratedRedisMiddleware.from_saver(
                saver=mock_saver,
                configs=[SemanticCacheConfig()],
            )

            # Middleware should have the redis_url configured
            assert stack._middlewares[0]._config.redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_from_store_extracts_url(self) -> None:
        """Test creating middleware that extracts redis_url from store.

        Note: redisvl requires sync Redis connections, so middleware creates
        its own connection using the extracted redis_url.
        """
        from langgraph.middleware.redis.composition import IntegratedRedisMiddleware

        # Mock a RedisStore with connection pool info
        mock_store = MagicMock()
        mock_store._redis_url = "redis://localhost:6379"

        with patch("langgraph.middleware.redis.semantic_cache.SemanticCache"):
            stack = IntegratedRedisMiddleware.from_store(
                store=mock_store,
                configs=[SemanticCacheConfig()],
            )

            # Middleware should have the redis_url configured
            assert stack._middlewares[0]._config.redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_extracts_url_from_connection_pool(self) -> None:
        """Test extracting redis_url from connection pool when not directly available."""
        from langgraph.middleware.redis.composition import IntegratedRedisMiddleware

        # Mock a saver without direct redis_url but with connection pool
        mock_saver = MagicMock()
        mock_saver._redis_url = None
        mock_saver.redis_url = None
        mock_pool = MagicMock()
        mock_pool.connection_kwargs = {"host": "myhost", "port": 6380}
        mock_saver._redis.connection_pool = mock_pool

        with patch("langgraph.middleware.redis.semantic_cache.SemanticCache"):
            stack = IntegratedRedisMiddleware.from_saver(
                saver=mock_saver,
                configs=[SemanticCacheConfig()],
            )

            # Middleware should have the reconstructed redis_url
            assert stack._middlewares[0]._config.redis_url == "redis://myhost:6380"
