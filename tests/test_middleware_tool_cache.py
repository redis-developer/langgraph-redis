"""Unit tests for ToolResultCacheMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.middleware.redis.types import ToolCacheConfig


class TestToolResultCacheMiddleware:
    """Tests for ToolResultCacheMiddleware class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self) -> None:
        """Test initialization with ToolCacheConfig."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            name="test_tool_cache",
            distance_threshold=0.15,
        )
        middleware = ToolResultCacheMiddleware(config)
        assert middleware._config.name == "test_tool_cache"
        assert middleware._config.distance_threshold == 0.15

    @pytest.mark.asyncio
    async def test_is_tool_cacheable_with_no_restrictions(self) -> None:
        """Test that all tools are cacheable when no restrictions set."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        assert middleware._is_tool_cacheable("search") is True
        assert middleware._is_tool_cacheable("calculate") is True
        assert middleware._is_tool_cacheable("any_tool") is True

    @pytest.mark.asyncio
    async def test_is_tool_cacheable_with_cacheable_list(self) -> None:
        """Test that only listed tools are cacheable."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            cacheable_tools=["search", "calculate"],
        )
        middleware = ToolResultCacheMiddleware(config)

        # Test with dict-style requests (no tool object, falls back to config)
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "calculate"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "random_tool"}) is False

    @pytest.mark.asyncio
    async def test_is_tool_cacheable_with_excluded_list(self) -> None:
        """Test that excluded tools are not cacheable."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            excluded_tools=["random_tool", "dangerous_tool"],
        )
        middleware = ToolResultCacheMiddleware(config)

        # Test with dict-style requests (no tool object, falls back to config)
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "random_tool"}) is False
        assert middleware._is_tool_cacheable({"tool_name": "dangerous_tool"}) is False

    @pytest.mark.asyncio
    async def test_cacheable_takes_precedence_over_excluded(self) -> None:
        """Test that cacheable_tools takes precedence over excluded_tools."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            cacheable_tools=["search"],
            excluded_tools=["search"],  # Excluded but in cacheable list
        )
        middleware = ToolResultCacheMiddleware(config)

        # When cacheable_tools is set, only those tools are cached
        # Test with dict-style request
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True

    @pytest.mark.asyncio
    async def test_tool_metadata_cacheable_overrides_config(self) -> None:
        """Test that tool.metadata['cacheable'] overrides config settings."""
        from unittest.mock import MagicMock

        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        # Config says only 'search' is cacheable
        config = ToolCacheConfig(
            redis_client=mock_client,
            cacheable_tools=["search"],
        )
        middleware = ToolResultCacheMiddleware(config)

        # Create a mock tool with metadata={'cacheable': True}
        mock_tool = MagicMock()
        mock_tool.metadata = {"cacheable": True}

        # Create request with tool object (simulating ToolCallRequest)
        class MockRequest:
            def __init__(self, name: str, tool: MagicMock):
                self.name = name
                self.tool = tool

        # Tool 'calculate' is NOT in cacheable_tools, but has metadata={'cacheable': True}
        request = MockRequest("calculate", mock_tool)
        assert middleware._is_tool_cacheable(request) is True

        # Tool 'search' IS in cacheable_tools, but has metadata={'cacheable': False}
        mock_tool.metadata = {"cacheable": False}
        request = MockRequest("search", mock_tool)
        assert middleware._is_tool_cacheable(request) is False

        # Tool with no metadata falls back to config
        mock_tool.metadata = None
        request = MockRequest("search", mock_tool)
        assert middleware._is_tool_cacheable(request) is True  # In cacheable_tools

        request = MockRequest("calculate", mock_tool)
        assert middleware._is_tool_cacheable(request) is False  # Not in cacheable_tools

    @pytest.mark.asyncio
    async def test_build_cache_key_from_request(self) -> None:
        """Test cache key generation from tool request."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        request = {
            "tool_name": "search",
            "args": {"query": "Python programming"},
        }
        cache_key = middleware._build_cache_key(request)
        assert "search" in cache_key
        assert "Python programming" in cache_key

    @pytest.mark.asyncio
    async def test_build_cache_key_with_complex_args(self) -> None:
        """Test cache key generation with complex arguments."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        request = {
            "tool_name": "api_call",
            "args": {"endpoint": "/users", "method": "GET", "params": {"id": 123}},
        }
        cache_key = middleware._build_cache_key(request)
        assert "api_call" in cache_key

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self) -> None:
        """Test that cache hit returns cached result without calling handler."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(
                return_value=[{"response": '{"result": "cached"}', "metadata": {}}]
            )
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            handler_called = []

            async def mock_handler(request: dict) -> dict:
                handler_called.append(True)
                return {"result": "new"}

            request = {"tool_name": "search", "args": {"query": "test"}}
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result == {"result": "cached"}
            assert len(handler_called) == 0

    @pytest.mark.asyncio
    async def test_cache_miss_calls_handler(self) -> None:
        """Test that cache miss calls handler and stores result."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])  # Cache miss
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "new result"}

            request = {"tool_name": "search", "args": {"query": "test"}}
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result == {"result": "new result"}
            mock_cache.astore.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_cacheable_tool_skips_cache(self) -> None:
        """Test that non-cacheable tools bypass the cache entirely."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            excluded_tools=["random_tool"],
        )

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock()
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "random"}

            request = {"tool_name": "random_tool", "args": {}}
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result == {"result": "random"}
            mock_cache.acheck.assert_not_called()
            mock_cache.astore.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_cache_error(self) -> None:
        """Test that middleware passes through on cache error."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client, graceful_degradation=True)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(side_effect=Exception("Redis error"))
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "handler response"}

            request = {"tool_name": "search", "args": {"query": "test"}}
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result == {"result": "handler response"}

    @pytest.mark.asyncio
    async def test_raises_on_cache_error_without_graceful_degradation(self) -> None:
        """Test that middleware raises on cache error when graceful_degradation=False."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client, graceful_degradation=False)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(side_effect=Exception("Redis error"))
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "handler response"}

            request = {"tool_name": "search", "args": {"query": "test"}}
            with pytest.raises(Exception, match="Redis error"):
                await middleware.awrap_tool_call(request, mock_handler)

    @pytest.mark.asyncio
    async def test_ttl_configuration(self) -> None:
        """Test that TTL is passed to cache."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client, ttl_seconds=7200)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            call_kwargs = mock_cache_class.call_args.kwargs
            assert call_kwargs.get("ttl") == 7200

    @pytest.mark.asyncio
    async def test_stores_with_tool_name_metadata(self) -> None:
        """Test that tool name is stored as metadata for filtering."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "data"}

            request = {"tool_name": "api_search", "args": {"q": "test"}}
            await middleware.awrap_tool_call(request, mock_handler)

            # Verify astore was called
            mock_cache.astore.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_missing_tool_name(self) -> None:
        """Test handling of requests without tool_name."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.tool_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache

            middleware = ToolResultCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"result": "handled"}

            # Request without tool_name
            request = {"args": {"query": "test"}}
            result = await middleware.awrap_tool_call(request, mock_handler)

            # Should pass through to handler
            assert result == {"result": "handled"}

    @pytest.mark.asyncio
    async def test_awrap_model_call_passes_through(self) -> None:
        """Test that awrap_model_call passes through (tool cache only for tools)."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        async def mock_handler(request: dict) -> dict:
            return {"content": "model response"}

        request = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await middleware.awrap_model_call(request, mock_handler)

        assert result == {"content": "model response"}
