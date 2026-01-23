"""Unit tests for SemanticRouterMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.middleware.redis.types import SemanticRouterConfig


class TestSemanticRouterMiddleware:
    """Tests for SemanticRouterMiddleware class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self) -> None:
        """Test initialization with SemanticRouterConfig."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        routes = [
            {"name": "greeting", "references": ["hello", "hi"]},
            {"name": "farewell", "references": ["bye", "goodbye"]},
        ]
        config = SemanticRouterConfig(
            redis_client=mock_client,
            name="test_router",
            routes=routes,
        )
        middleware = SemanticRouterMiddleware(config)
        assert middleware._config.name == "test_router"
        assert len(middleware._config.routes) == 2

    @pytest.mark.asyncio
    async def test_extract_query_from_messages(self) -> None:
        """Test query extraction from messages list."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(redis_client=mock_client)
        middleware = SemanticRouterMiddleware(config)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello there!"},
        ]
        query = middleware._extract_query(messages)
        assert query == "Hello there!"

    @pytest.mark.asyncio
    async def test_extract_query_handles_empty_messages(self) -> None:
        """Test query extraction with empty messages."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(redis_client=mock_client)
        middleware = SemanticRouterMiddleware(config)

        query = middleware._extract_query([])
        assert query == ""

    @pytest.mark.asyncio
    async def test_route_detection(self) -> None:
        """Test that routes are detected from user input."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        routes = [
            {"name": "greeting", "references": ["hello", "hi", "hey"]},
        ]
        config = SemanticRouterConfig(redis_client=mock_client, routes=routes)

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_route_match = MagicMock()
            mock_route_match.name = "greeting"
            mock_route_match.distance = 0.05
            mock_router.return_value = mock_route_match
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            route = middleware._get_route("Hello there!")
            assert route is not None

    @pytest.mark.asyncio
    async def test_adds_routing_info_to_request(self) -> None:
        """Test that routing info is added to request runtime."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        routes = [
            {"name": "greeting", "references": ["hello", "hi"]},
        ]
        config = SemanticRouterConfig(redis_client=mock_client, routes=routes)

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_route_match = MagicMock()
            mock_route_match.name = "greeting"
            mock_route_match.distance = 0.05
            mock_router.return_value = mock_route_match
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                # Check that routing info was added
                assert "runtime" in request
                assert request["runtime"].get("route") == "greeting"
                return {"content": "Hello!"}

            request = {"messages": [{"role": "user", "content": "Hi!"}]}
            await middleware.awrap_model_call(request, mock_handler)

    @pytest.mark.asyncio
    async def test_no_route_match_passes_through(self) -> None:
        """Test that requests without route matches pass through unchanged."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(redis_client=mock_client, routes=[])

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router.return_value = None  # No route match
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            request_seen = []

            async def mock_handler(request: dict) -> dict:
                request_seen.append(request)
                return {"content": "Response"}

            request = {"messages": [{"role": "user", "content": "Random text"}]}
            await middleware.awrap_model_call(request, mock_handler)

            # Request should still be passed to handler
            assert len(request_seen) == 1

    @pytest.mark.asyncio
    async def test_register_route_handler(self) -> None:
        """Test registering custom route handlers."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        routes = [
            {"name": "greeting", "references": ["hello"]},
        ]
        config = SemanticRouterConfig(redis_client=mock_client, routes=routes)

        middleware = SemanticRouterMiddleware(config)

        handler_called = []

        async def greeting_handler(request: dict, route_match: dict) -> dict:
            handler_called.append(route_match)
            return {"content": "Custom greeting response"}

        middleware.register_route_handler("greeting", greeting_handler)

        assert "greeting" in middleware._route_handlers

    @pytest.mark.asyncio
    async def test_custom_handler_invoked_on_route_match(self) -> None:
        """Test that custom handler is invoked when route matches."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        routes = [
            {"name": "greeting", "references": ["hello"]},
        ]
        config = SemanticRouterConfig(redis_client=mock_client, routes=routes)

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_route_match = MagicMock()
            mock_route_match.name = "greeting"
            mock_route_match.distance = 0.05
            mock_router.return_value = mock_route_match
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            custom_handler_called = []

            async def custom_greeting(request: dict, route_match: dict) -> dict:
                custom_handler_called.append(True)
                return {"content": "Custom greeting!"}

            middleware.register_route_handler("greeting", custom_greeting)

            async def mock_handler(request: dict) -> dict:
                return {"content": "Default response"}

            request = {"messages": [{"role": "user", "content": "Hello!"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert len(custom_handler_called) == 1
            assert result == {"content": "Custom greeting!"}

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_router_error(self) -> None:
        """Test that middleware passes through on router error."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(
            redis_client=mock_client, graceful_degradation=True
        )

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router.side_effect = Exception("Router error")
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "Handler response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert result == {"content": "Handler response"}

    @pytest.mark.asyncio
    async def test_raises_on_router_error_without_graceful_degradation(self) -> None:
        """Test that middleware raises on router error when graceful_degradation=False."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(
            redis_client=mock_client, graceful_degradation=False
        )

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router.side_effect = Exception("Router error")
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "Handler response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            with pytest.raises(Exception, match="Router error"):
                await middleware.awrap_model_call(request, mock_handler)

    @pytest.mark.asyncio
    async def test_max_k_parameter(self) -> None:
        """Test that max_k is passed to router."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(redis_client=mock_client, max_k=5)

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify router was created with max_k in routing_config
            call_kwargs = mock_router_class.call_args.kwargs
            routing_config = call_kwargs.get("routing_config")
            assert routing_config is not None
            assert routing_config.max_k == 5

    @pytest.mark.asyncio
    async def test_aggregation_method_parameter(self) -> None:
        """Test that aggregation_method is passed to router via routing_config."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(
            redis_client=mock_client, aggregation_method="sum"
        )

        with patch(
            "langgraph.middleware.redis.semantic_router.SemanticRouter"
        ) as mock_router_class:
            mock_router = MagicMock()
            mock_router_class.return_value = mock_router

            middleware = SemanticRouterMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify router was created with aggregation_method in routing_config
            call_kwargs = mock_router_class.call_args.kwargs
            routing_config = call_kwargs.get("routing_config")
            assert routing_config is not None
            assert routing_config.aggregation_method.value == "sum"

    @pytest.mark.asyncio
    async def test_tool_call_passes_through(self) -> None:
        """Test that awrap_tool_call passes through (router is for model calls)."""
        from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware

        mock_client = AsyncMock()
        config = SemanticRouterConfig(redis_client=mock_client)
        middleware = SemanticRouterMiddleware(config)

        async def mock_handler(request: dict) -> dict:
            return {"result": "tool result"}

        request = {"tool_name": "search", "args": {"query": "test"}}
        result = await middleware.awrap_tool_call(request, mock_handler)

        assert result == {"result": "tool result"}
