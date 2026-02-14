"""Unit tests for ToolResultCacheMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from langgraph.middleware.redis.types import ToolCacheConfig


def _make_tool_call_request(
    tool_name: str,
    args: dict | None = None,
    tool_call_id: str = "call_123",
    tool: MagicMock | None = None,
) -> ToolCallRequest:
    """Create a ToolCallRequest for testing."""
    tool_call = {"name": tool_name, "args": args or {}, "id": tool_call_id}
    return ToolCallRequest(
        tool_call=tool_call,
        tool=tool,
        state={},
        runtime=MagicMock(),
    )


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
    async def test_is_tool_cacheable_with_no_restrictions_dict(self) -> None:
        """Test that all tools are cacheable when no restrictions set (dict path)."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "calculate"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "any_tool"}) is True

    @pytest.mark.asyncio
    async def test_is_tool_cacheable_with_no_restrictions_request(self) -> None:
        """Test that all tools are cacheable when no restrictions set (ToolCallRequest)."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        assert middleware._is_tool_cacheable(_make_tool_call_request("search")) is True
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("calculate")) is True
        )
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("any_tool")) is True
        )

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

        # Dict-style requests
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "calculate"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "random_tool"}) is False

        # ToolCallRequest objects
        assert middleware._is_tool_cacheable(_make_tool_call_request("search")) is True
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("calculate")) is True
        )
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("random_tool"))
            is False
        )

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

        # Dict-style requests
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable({"tool_name": "random_tool"}) is False
        assert middleware._is_tool_cacheable({"tool_name": "dangerous_tool"}) is False

        # ToolCallRequest objects
        assert middleware._is_tool_cacheable(_make_tool_call_request("search")) is True
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("random_tool"))
            is False
        )
        assert (
            middleware._is_tool_cacheable(_make_tool_call_request("dangerous_tool"))
            is False
        )

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
        assert middleware._is_tool_cacheable({"tool_name": "search"}) is True
        assert middleware._is_tool_cacheable(_make_tool_call_request("search")) is True

    @pytest.mark.asyncio
    async def test_tool_metadata_cacheable_overrides_config(self) -> None:
        """Test that tool.metadata['cacheable'] overrides config settings."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(
            redis_client=mock_client,
            cacheable_tools=["search"],
        )
        middleware = ToolResultCacheMiddleware(config)

        # Tool 'calculate' is NOT in cacheable_tools, but has metadata={'cacheable': True}
        mock_tool = MagicMock()
        mock_tool.metadata = {"cacheable": True}
        request = _make_tool_call_request("calculate", tool=mock_tool)
        assert middleware._is_tool_cacheable(request) is True

        # Tool 'search' IS in cacheable_tools, but has metadata={'cacheable': False}
        mock_tool.metadata = {"cacheable": False}
        request = _make_tool_call_request("search", tool=mock_tool)
        assert middleware._is_tool_cacheable(request) is False

        # Tool with no metadata falls back to config
        mock_tool.metadata = None
        request = _make_tool_call_request("search", tool=mock_tool)
        assert middleware._is_tool_cacheable(request) is True  # In cacheable_tools

        request = _make_tool_call_request("calculate", tool=mock_tool)
        assert middleware._is_tool_cacheable(request) is False  # Not in cacheable_tools

    @pytest.mark.asyncio
    async def test_build_cache_key_from_dict(self) -> None:
        """Test cache key generation from dict request."""
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
    async def test_build_cache_key_from_tool_call_request(self) -> None:
        """Test cache key generation from ToolCallRequest."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        request = _make_tool_call_request(
            "search", args={"query": "Python programming"}
        )
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

        request = _make_tool_call_request(
            "api_call",
            args={"endpoint": "/users", "method": "GET", "params": {"id": 123}},
        )
        cache_key = middleware._build_cache_key(request)
        assert "api_call" in cache_key

    @pytest.mark.asyncio
    async def test_cache_hit_returns_tool_message(self) -> None:
        """Test that cache hit returns a proper ToolMessage."""
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

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                handler_called.append(True)
                return LangChainToolMessage(
                    content="new", name="search", tool_call_id="call_123"
                )

            request = _make_tool_call_request(
                "search", args={"query": "test"}, tool_call_id="call_456"
            )
            result = await middleware.awrap_tool_call(request, mock_handler)

            # Should return a ToolMessage, not a raw dict
            assert isinstance(result, LangChainToolMessage)
            assert result.name == "search"
            assert result.tool_call_id == "call_456"
            assert len(handler_called) == 0

    @pytest.mark.asyncio
    async def test_cache_hit_with_dict_request(self) -> None:
        """Test cache hit with dict-style request for backward compat."""
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

            request = {"tool_name": "search", "args": {"query": "test"}, "id": "c1"}
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert isinstance(result, LangChainToolMessage)
            assert result.name == "search"
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

            expected = LangChainToolMessage(
                content="new result", name="search", tool_call_id="call_123"
            )

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                return expected

            request = _make_tool_call_request("search", args={"query": "test"})
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result is expected
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

            expected = LangChainToolMessage(
                content="random", name="random_tool", tool_call_id="call_123"
            )

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                return expected

            request = _make_tool_call_request("random_tool")
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result is expected
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

            expected = LangChainToolMessage(
                content="handler response", name="search", tool_call_id="call_123"
            )

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                return expected

            request = _make_tool_call_request("search", args={"query": "test"})
            result = await middleware.awrap_tool_call(request, mock_handler)

            assert result is expected

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

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                return LangChainToolMessage(
                    content="response", name="search", tool_call_id="call_123"
                )

            request = _make_tool_call_request("search", args={"query": "test"})
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
        """Test that tool results are stored in cache."""
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

            async def mock_handler(
                request: ToolCallRequest,
            ) -> LangChainToolMessage:
                return LangChainToolMessage(
                    content="data", name="api_search", tool_call_id="call_123"
                )

            request = _make_tool_call_request("api_search", args={"q": "test"})
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

    @pytest.mark.asyncio
    async def test_deserialize_tool_result_json_content(self) -> None:
        """Test deserialization of cached JSON content into ToolMessage."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        result = middleware._deserialize_tool_result(
            '{"content": "hello world"}', "search", "call_1"
        )
        assert isinstance(result, LangChainToolMessage)
        assert result.content == "hello world"
        assert result.name == "search"
        assert result.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_deserialize_tool_result_plain_string(self) -> None:
        """Test deserialization of plain string cached content."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        result = middleware._deserialize_tool_result(
            '"plain text result"', "calc", "call_2"
        )
        assert isinstance(result, LangChainToolMessage)
        assert result.content == "plain text result"
        assert result.name == "calc"

    @pytest.mark.asyncio
    async def test_deserialize_tool_result_invalid_json(self) -> None:
        """Test deserialization gracefully handles invalid JSON."""
        from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware

        mock_client = AsyncMock()
        config = ToolCacheConfig(redis_client=mock_client)
        middleware = ToolResultCacheMiddleware(config)

        result = middleware._deserialize_tool_result(
            "not valid json {{{", "tool", "call_3"
        )
        assert isinstance(result, LangChainToolMessage)
        assert result.content == "not valid json {{{"
