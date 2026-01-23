"""Semantic router middleware for intent-based routing.

This module provides a middleware that routes requests based on
semantic similarity to predefined intents using Redis and vector embeddings.
Compatible with LangChain's AgentMiddleware protocol for use with create_agent.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from redisvl.extensions.router import RoutingConfig, SemanticRouter
from redisvl.extensions.router.schema import Route

from .aio import AsyncRedisMiddleware
from .types import SemanticRouterConfig

logger = logging.getLogger(__name__)

# Type for route handler functions
RouteHandler = Callable[[ModelRequest, Dict[str, Any]], Awaitable[ModelCallResult]]


class SemanticRouterMiddleware(AsyncRedisMiddleware):
    """Middleware that routes requests based on semantic similarity.

    Uses redisvl.extensions.router.SemanticRouter to classify user intents
    and route requests to appropriate handlers. This is useful for:
    - Directing queries to specialized agents
    - Triggering specific workflows based on intent
    - Adding routing metadata for downstream processing

    Example:
        ```python
        from langgraph.middleware.redis import (
            SemanticRouterMiddleware,
            SemanticRouterConfig,
        )

        routes = [
            {"name": "greeting", "references": ["hello", "hi", "hey"]},
            {"name": "support", "references": ["help", "issue", "problem"]},
        ]

        config = SemanticRouterConfig(
            redis_url="redis://localhost:6379",
            routes=routes,
        )

        middleware = SemanticRouterMiddleware(config)

        # Register custom handler for greeting route
        @middleware.register_route_handler("greeting")
        async def handle_greeting(request, route_match):
            return {"content": "Hello! How can I help you today?"}
        ```
    """

    _router: SemanticRouter
    _config: SemanticRouterConfig
    _route_handlers: Dict[str, RouteHandler]

    def __init__(self, config: SemanticRouterConfig) -> None:
        """Initialize the semantic router middleware.

        Args:
            config: Configuration for the semantic router.
        """
        super().__init__(config)
        self._config = config
        self._route_handlers = {}

    async def _setup_async(self) -> None:
        """Set up the SemanticRouter instance.

        Note: SemanticRouter from redisvl uses synchronous Redis operations
        internally, so we must provide redis_url and let it manage its own
        sync connection rather than passing our async client.
        """
        # Convert route configs to Route objects
        routes = []
        for route_config in self._config.routes:
            route_kwargs: dict[str, Any] = {
                "name": route_config["name"],
                "references": route_config["references"],
            }
            # Only add distance_threshold if explicitly set
            if route_config.get("distance_threshold") is not None:
                route_kwargs["distance_threshold"] = route_config["distance_threshold"]
            route = Route(**route_kwargs)
            routes.append(route)

        # Create routing config
        routing_config = RoutingConfig(
            max_k=self._config.max_k,
            aggregation_method=self._config.aggregation_method,
        )

        router_kwargs: dict[str, Any] = {
            "name": self._config.name,
            "routes": routes,
            "routing_config": routing_config,
        }

        # SemanticRouter requires a sync Redis connection
        # Use redis_url to let it create its own connection
        if self._config.redis_url:
            router_kwargs["redis_url"] = self._config.redis_url
        elif self._config.connection_args:
            router_kwargs["connection_kwargs"] = self._config.connection_args

        if self._config.vectorizer is not None:
            router_kwargs["vectorizer"] = self._config.vectorizer

        self._router = SemanticRouter(**router_kwargs)

    def _extract_query(self, messages: List[Union[dict[str, Any], Any]]) -> str:
        """Extract the query to use for routing.

        Args:
            messages: List of messages from the request.

        Returns:
            The extracted query string.
        """
        if not messages:
            return ""

        # Find the last user message
        for message in reversed(messages):
            if isinstance(message, dict):
                role = message.get("role", "")
                if role == "user":
                    return message.get("content", "")
            else:
                msg_type = getattr(message, "type", None) or getattr(
                    message, "role", None
                )
                if msg_type in ("user", "human"):
                    return getattr(message, "content", "")

        return ""

    def _get_route(self, query: str) -> Optional[Any]:
        """Get the matching route for a query.

        Args:
            query: The user query to route.

        Returns:
            The route match object, or None if no match.
        """
        if not query:
            return None
        return self._router(query)

    def register_route_handler(
        self, route_name: str, handler: Optional[RouteHandler] = None
    ) -> Callable[[RouteHandler], RouteHandler]:
        """Register a handler for a specific route.

        Can be used as a decorator or called directly.

        Args:
            route_name: The name of the route to handle.
            handler: Optional handler function. If not provided,
                returns a decorator.

        Returns:
            The handler function, or a decorator if handler not provided.

        Example:
            ```python
            # As decorator
            @middleware.register_route_handler("greeting")
            async def handle_greeting(request, route_match):
                return {"content": "Hello!"}

            # Direct registration
            middleware.register_route_handler("greeting", handle_greeting)
            ```
        """

        def decorator(fn: RouteHandler) -> RouteHandler:
            self._route_handlers[route_name] = fn
            return fn

        if handler is not None:
            return decorator(handler)
        return decorator

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap a model call with semantic routing.

        This method is part of the LangChain AgentMiddleware protocol.
        Determines the route based on the user's message and either:
        - Calls a registered route handler if one exists
        - Adds routing info to the request and calls the default handler

        Args:
            request: The model request containing messages.
            handler: The async function to call the model.

        Returns:
            The model response.

        Raises:
            Exception: If graceful_degradation is False and routing fails.
        """
        await self._ensure_initialized_async()

        # Support both dict-style and LangChain ModelRequest types
        if isinstance(request, dict):
            messages = request.get("messages", [])
        else:
            messages = getattr(request, "messages", [])
        query = self._extract_query(messages)

        if not query:
            return await handler(request)

        # Try to get route match
        route_match = None
        try:
            route_match = self._get_route(query)
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Router failed, passing through: {e}")

        if route_match is not None:
            route_name = route_match.name

            # Check if there's a custom handler for this route
            if route_name in self._route_handlers:
                route_info = {
                    "name": route_name,
                    "distance": getattr(route_match, "distance", None),
                }
                return await self._route_handlers[route_name](request, route_info)

            # Add routing info to request (for dict-style requests only)
            # LangChain ModelRequest has runtime as a read-only attribute
            if isinstance(request, dict):
                if "runtime" not in request:
                    request["runtime"] = {}
                request["runtime"]["route"] = route_name
                request["runtime"]["route_distance"] = getattr(
                    route_match, "distance", None
                )

        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Pass through tool calls without routing.

        This method is part of the LangChain AgentMiddleware protocol.
        Semantic router only applies to model calls, not tool calls.

        Args:
            request: The tool call request.
            handler: The async function to execute the tool.

        Returns:
            The tool result from the handler.
        """
        return await handler(request)
