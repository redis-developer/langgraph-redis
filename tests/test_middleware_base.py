"""Unit tests for base middleware classes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.middleware.redis.types import MiddlewareConfig


class TestBaseRedisMiddleware:
    """Tests for the sync BaseRedisMiddleware class."""

    def test_init_with_redis_url(self) -> None:
        """Test initialization with redis_url."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("redis.Redis") as mock_redis:
            mock_redis.return_value = MagicMock()
            middleware = ConcreteMiddleware(config)
            assert middleware._config.redis_url == "redis://localhost:6379"

    def test_init_with_redis_client(self) -> None:
        """Test initialization with existing redis_client."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        assert middleware._owns_client is False

    def test_init_requires_url_or_client(self) -> None:
        """Test that initialization requires redis_url or redis_client."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        config = MiddlewareConfig()
        with pytest.raises(ValueError, match="Either redis_url or redis_client"):
            ConcreteMiddleware(config)

    def test_owns_client_when_created_from_url(self) -> None:
        """Test that middleware owns client when created from URL."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("redis.Redis") as mock_redis:
            mock_redis.from_url.return_value = MagicMock()
            middleware = ConcreteMiddleware(config)
            assert middleware._owns_client is True

    def test_graceful_degradation_default(self) -> None:
        """Test that graceful_degradation is True by default."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        assert middleware._graceful_degradation is True

    def test_graceful_degradation_disabled(self) -> None:
        """Test that graceful_degradation can be disabled."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client, graceful_degradation=False)
        middleware = ConcreteMiddleware(config)
        assert middleware._graceful_degradation is False

    def test_close_client_when_owned(self) -> None:
        """Test that close() closes client when middleware owns it."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("langgraph.middleware.redis.base.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client
            middleware = ConcreteMiddleware(config)
            middleware.close()
            mock_client.close.assert_called_once()

    def test_close_does_not_close_external_client(self) -> None:
        """Test that close() doesn't close externally provided client."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        middleware.close()
        mock_client.close.assert_not_called()

    def test_ensure_initialized_calls_setup(self) -> None:
        """Test that _ensure_initialized_sync calls _setup_sync."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        setup_called = []

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                setup_called.append(True)

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        middleware._ensure_initialized_sync()
        assert len(setup_called) == 1

    def test_ensure_initialized_only_once(self) -> None:
        """Test that _setup_sync is only called once."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        setup_count = [0]

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                setup_count[0] += 1

        mock_client = MagicMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        middleware._ensure_initialized_sync()
        middleware._ensure_initialized_sync()
        middleware._ensure_initialized_sync()
        assert setup_count[0] == 1

    def test_context_manager(self) -> None:
        """Test that middleware works as context manager."""
        from langgraph.middleware.redis.base import BaseRedisMiddleware

        class ConcreteMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("langgraph.middleware.redis.base.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_redis.from_url.return_value = mock_client
            with ConcreteMiddleware(config) as middleware:
                assert middleware is not None
            mock_client.close.assert_called_once()


class TestAsyncRedisMiddleware:
    """Tests for the async AsyncRedisMiddleware class."""

    @pytest.mark.asyncio
    async def test_init_with_redis_url(self) -> None:
        """Test async initialization with redis_url."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("redis.asyncio.Redis") as mock_redis:
            mock_redis.from_url.return_value = AsyncMock()
            middleware = ConcreteMiddleware(config)
            assert middleware._config.redis_url == "redis://localhost:6379"

    @pytest.mark.asyncio
    async def test_init_with_async_client(self) -> None:
        """Test initialization with existing async redis_client."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        assert middleware._owns_client is False

    @pytest.mark.asyncio
    async def test_aclose_client_when_owned(self) -> None:
        """Test that aclose() closes client when middleware owns it."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("langgraph.middleware.redis.aio.AsyncRedis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.from_url.return_value = mock_client
            middleware = ConcreteMiddleware(config)
            await middleware.aclose()
            mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_does_not_close_external_client(self) -> None:
        """Test that aclose() doesn't close externally provided client."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        await middleware.aclose()
        mock_client.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ensure_initialized_async_calls_setup(self) -> None:
        """Test that _ensure_initialized_async calls _setup_async."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        setup_called = []

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                setup_called.append(True)

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        await middleware._ensure_initialized_async()
        assert len(setup_called) == 1

    @pytest.mark.asyncio
    async def test_ensure_initialized_async_only_once(self) -> None:
        """Test that _setup_async is only called once (double-check locking)."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        setup_count = [0]

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                setup_count[0] += 1

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)
        await middleware._ensure_initialized_async()
        await middleware._ensure_initialized_async()
        await middleware._ensure_initialized_async()
        assert setup_count[0] == 1

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test that middleware works as async context manager."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        config = MiddlewareConfig(redis_url="redis://localhost:6379")

        with patch("langgraph.middleware.redis.aio.AsyncRedis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.from_url.return_value = mock_client
            async with ConcreteMiddleware(config) as middleware:
                assert middleware is not None
            mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_awrap_model_call_default(self) -> None:
        """Test default awrap_model_call passes through."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)

        # Create a mock handler
        async def mock_handler(request: dict) -> dict:
            return {"response": "test"}

        request = {"messages": []}
        result = await middleware.awrap_model_call(request, mock_handler)
        assert result == {"response": "test"}

    @pytest.mark.asyncio
    async def test_awrap_tool_call_default(self) -> None:
        """Test default awrap_tool_call passes through."""
        from langgraph.middleware.redis.aio import AsyncRedisMiddleware

        class ConcreteMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                pass

        mock_client = AsyncMock()
        config = MiddlewareConfig(redis_client=mock_client)
        middleware = ConcreteMiddleware(config)

        # Create a mock handler
        async def mock_handler(request: dict) -> dict:
            return {"tool_result": "success"}

        request = {"tool_name": "test_tool"}
        result = await middleware.awrap_tool_call(request, mock_handler)
        assert result == {"tool_result": "success"}
