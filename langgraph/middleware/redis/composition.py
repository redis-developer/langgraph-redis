"""Middleware composition and factory functions.

This module provides utilities for composing multiple middleware together
and creating middleware stacks that share Redis connections with
checkpointers and stores. Compatible with LangChain's AgentMiddleware protocol
for use with create_agent.
"""

import logging
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from redis.asyncio import Redis as AsyncRedis

from .aio import AsyncRedisMiddleware
from .conversation_memory import ConversationMemoryMiddleware
from .semantic_cache import SemanticCacheMiddleware
from .semantic_router import SemanticRouterMiddleware
from .tool_cache import ToolResultCacheMiddleware
from .types import (
    ConversationMemoryConfig,
    MiddlewareConfigType,
    SemanticCacheConfig,
    SemanticRouterConfig,
    ToolCacheConfig,
)

logger = logging.getLogger(__name__)


class MiddlewareStack(AgentMiddleware):
    """A stack of middleware that chains calls through all middlewares.

    Inherits from LangChain's AgentMiddleware, so can be used directly with
    create_agent(middleware=[stack]) or as a single middleware entry.

    Middleware are applied in order: the first middleware wraps the second,
    which wraps the third, etc. This means the first middleware's
    before-processing runs first, and its after-processing runs last.

    Example:
        ```python
        from langchain.agents import create_agent
        from langgraph.middleware.redis import (
            MiddlewareStack,
            SemanticCacheMiddleware,
            ToolResultCacheMiddleware,
        )

        stack = MiddlewareStack([
            SemanticCacheMiddleware(cache_config),
            ToolResultCacheMiddleware(tool_config),
        ])

        # Use with create_agent
        agent = create_agent(
            model="gpt-4o",
            tools=[...],
            middleware=[stack],  # Pass stack as middleware
        )
        ```
    """

    _middlewares: List[AsyncRedisMiddleware]

    def __init__(self, middlewares: Sequence[AsyncRedisMiddleware]) -> None:
        """Initialize the middleware stack.

        Args:
            middlewares: List of middleware to chain together.
        """
        self._middlewares = list(middlewares)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap a model call through all middleware.

        This method is part of the LangChain AgentMiddleware protocol.

        Args:
            request: The model request.
            handler: The final handler to call the model.

        Returns:
            The model response.
        """
        if not self._middlewares:
            return await handler(request)

        # Build the chain from inside out
        # Start with the final handler
        current_handler = handler

        # Wrap from last to first middleware
        for middleware in reversed(self._middlewares):

            # Create a closure to capture the current middleware and handler
            def make_wrapper(
                mw: AsyncRedisMiddleware,
                h: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> Callable[[ModelRequest], Awaitable[ModelResponse]]:
                async def wrapper(req: ModelRequest) -> ModelCallResult:
                    return await mw.awrap_model_call(req, h)

                return wrapper

            current_handler = make_wrapper(middleware, current_handler)

        return await current_handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Wrap a tool call through all middleware.

        This method is part of the LangChain AgentMiddleware protocol.

        Args:
            request: The tool call request.
            handler: The final handler to execute the tool.

        Returns:
            The tool result.
        """
        if not self._middlewares:
            return await handler(request)

        # Build the chain from inside out
        current_handler = handler

        for middleware in reversed(self._middlewares):

            def make_wrapper(
                mw: AsyncRedisMiddleware,
                h: Callable[
                    [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
                ],
            ) -> Callable[
                [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
            ]:
                async def wrapper(
                    req: ToolCallRequest,
                ) -> Union[LangChainToolMessage, Command]:
                    return await mw.awrap_tool_call(req, h)

                return wrapper

            current_handler = make_wrapper(middleware, current_handler)

        return await current_handler(request)

    async def aclose(self) -> None:
        """Close all middleware in the stack."""
        for middleware in self._middlewares:
            try:
                await middleware.aclose()
            except Exception as e:
                logger.warning(f"Error closing middleware: {e}")

    async def __aenter__(self) -> "MiddlewareStack":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit async context manager and close all middleware."""
        await self.aclose()


def _create_middleware_from_config(
    config: MiddlewareConfigType,
    redis_client: Optional[AsyncRedis] = None,
    redis_url: Optional[str] = None,
) -> AsyncRedisMiddleware:
    """Create a middleware instance from a config object.

    Args:
        config: The middleware configuration.
        redis_client: Optional Redis client to use (overrides config).
        redis_url: Optional Redis URL to use (overrides config).

    Returns:
        The created middleware instance.
    """
    # Override Redis connection if provided
    if redis_client is not None:
        config.redis_client = redis_client
        config.redis_url = None
    elif redis_url is not None:
        config.redis_url = redis_url
        config.redis_client = None

    if isinstance(config, SemanticCacheConfig):
        return SemanticCacheMiddleware(config)
    elif isinstance(config, ToolCacheConfig):
        return ToolResultCacheMiddleware(config)
    elif isinstance(config, SemanticRouterConfig):
        return SemanticRouterMiddleware(config)
    elif isinstance(config, ConversationMemoryConfig):
        return ConversationMemoryMiddleware(config)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def from_configs(
    configs: Sequence[MiddlewareConfigType],
    redis_client: Optional[AsyncRedis] = None,
    redis_url: Optional[str] = None,
) -> MiddlewareStack:
    """Create a middleware stack from configuration objects.

    All middlewares will share the same Redis connection.

    Args:
        configs: List of middleware configuration objects.
        redis_client: Optional Redis client to use for all middlewares.
        redis_url: Optional Redis URL to use for all middlewares.

    Returns:
        A MiddlewareStack with the configured middlewares.

    Example:
        ```python
        from langgraph.middleware.redis import (
            from_configs,
            SemanticCacheConfig,
            ToolCacheConfig,
        )

        stack = from_configs(
            redis_url="redis://localhost:6379",
            configs=[
                SemanticCacheConfig(ttl_seconds=3600),
                ToolCacheConfig(cacheable_tools=["search"]),
            ],
        )
        ```
    """
    middlewares = []
    for config in configs:
        middleware = _create_middleware_from_config(
            config, redis_client=redis_client, redis_url=redis_url
        )
        middlewares.append(middleware)

    return MiddlewareStack(middlewares)


def create_caching_stack(
    redis_client: Optional[AsyncRedis] = None,
    redis_url: Optional[str] = None,
    semantic_cache_name: str = "llmcache",
    semantic_cache_ttl: Optional[int] = None,
    tool_cache_name: str = "toolcache",
    tool_cache_ttl: Optional[int] = None,
    cacheable_tools: Optional[List[str]] = None,
    excluded_tools: Optional[List[str]] = None,
    distance_threshold: float = 0.1,
) -> MiddlewareStack:
    """Create a middleware stack with semantic and tool caching.

    A convenience function for the common pattern of caching both
    LLM responses and tool results.

    Args:
        redis_client: Optional Redis client to use (deprecated, use redis_url).
        redis_url: Redis URL to use.
        semantic_cache_name: Name for the semantic cache index.
        semantic_cache_ttl: TTL in seconds for semantic cache entries.
        tool_cache_name: Name for the tool cache index.
        tool_cache_ttl: TTL in seconds for tool cache entries.
        cacheable_tools: List of tool names to cache (None = all).
        excluded_tools: List of tool names to never cache.
        distance_threshold: Distance threshold for cache hits.

    Returns:
        A MiddlewareStack with semantic and tool caching.

    Example:
        ```python
        from langgraph.middleware.redis import create_caching_stack

        stack = create_caching_stack(
            redis_url="redis://localhost:6379",
            semantic_cache_ttl=3600,
            tool_cache_ttl=7200,
            cacheable_tools=["search", "calculate"],
        )
        ```
    """
    configs: List[MiddlewareConfigType] = [
        SemanticCacheConfig(
            name=semantic_cache_name,
            ttl_seconds=semantic_cache_ttl,
            distance_threshold=distance_threshold,
        ),
        ToolCacheConfig(
            name=tool_cache_name,
            ttl_seconds=tool_cache_ttl,
            distance_threshold=distance_threshold,
            cacheable_tools=cacheable_tools,
            excluded_tools=excluded_tools or [],
        ),
    ]

    return from_configs(
        configs=configs,
        redis_client=redis_client,
        redis_url=redis_url,
    )


class IntegratedRedisMiddleware:
    """Factory for creating middleware that shares Redis connections.

    This class provides static methods to create middleware stacks that
    connect to the same Redis server as existing checkpointers or stores.

    Note: The redisvl library components used by middleware require synchronous
    Redis connections. While the saver/store may use async clients, middleware
    will create their own sync connections to the same Redis URL.
    """

    @staticmethod
    def from_saver(
        saver: Any,
        configs: Sequence[MiddlewareConfigType],
    ) -> MiddlewareStack:
        """Create middleware stack connecting to same Redis as a checkpoint saver.

        Note: Middleware creates its own sync Redis connection to the same server.
        The redisvl library components require sync connections internally.

        Args:
            saver: A RedisSaver or AsyncRedisSaver instance.
            configs: List of middleware configurations.

        Returns:
            A MiddlewareStack connecting to the same Redis server.

        Example:
            ```python
            from langgraph.checkpoint.redis import AsyncRedisSaver
            from langgraph.middleware.redis import (
                IntegratedRedisMiddleware,
                SemanticCacheConfig,
            )

            saver = AsyncRedisSaver(redis_url="redis://localhost:6379")
            await saver.asetup()

            stack = IntegratedRedisMiddleware.from_saver(
                saver,
                [SemanticCacheConfig(ttl_seconds=3600)],
            )
            ```
        """
        # Try to get redis_url from the saver
        redis_url = getattr(saver, "_redis_url", None) or getattr(
            saver, "redis_url", None
        )

        if redis_url is None:
            # Try to extract from the Redis client
            redis_client = getattr(saver, "_redis", None)
            if redis_client is not None:
                # Try to reconstruct URL from connection pool
                pool = getattr(redis_client, "connection_pool", None)
                if pool is not None:
                    connection_kwargs = getattr(pool, "connection_kwargs", {})
                    host = connection_kwargs.get("host", "localhost")
                    port = connection_kwargs.get("port", 6379)
                    redis_url = f"redis://{host}:{port}"

        if redis_url is None:
            raise ValueError(
                "Could not determine Redis URL from saver. "
                "Please provide a redis_url in middleware configs."
            )

        middlewares = []
        for config in configs:
            # Set the redis_url for redisvl to create its own sync connection
            config.redis_url = redis_url
            config.redis_client = None

            middleware = _create_middleware_from_config(config)
            middlewares.append(middleware)

        return MiddlewareStack(middlewares)

    @staticmethod
    def from_store(
        store: Any,
        configs: Sequence[MiddlewareConfigType],
    ) -> MiddlewareStack:
        """Create middleware stack connecting to same Redis as a store.

        Note: Middleware creates its own sync Redis connection to the same server.
        The redisvl library components require sync connections internally.

        Args:
            store: A RedisStore or AsyncRedisStore instance.
            configs: List of middleware configurations.

        Returns:
            A MiddlewareStack connecting to the same Redis server.

        Example:
            ```python
            from langgraph.store.redis import AsyncRedisStore
            from langgraph.middleware.redis import (
                IntegratedRedisMiddleware,
                SemanticCacheConfig,
            )

            store = AsyncRedisStore(conn=redis_client)
            await store.asetup()

            stack = IntegratedRedisMiddleware.from_store(
                store,
                [SemanticCacheConfig(ttl_seconds=3600)],
            )
            ```
        """
        # Try to get redis_url from the store
        redis_url = getattr(store, "_redis_url", None) or getattr(
            store, "redis_url", None
        )

        if redis_url is None:
            # Try to extract from the Redis client
            redis_client = getattr(store, "_redis", None)
            if redis_client is not None:
                # Try to reconstruct URL from connection pool
                pool = getattr(redis_client, "connection_pool", None)
                if pool is not None:
                    connection_kwargs = getattr(pool, "connection_kwargs", {})
                    host = connection_kwargs.get("host", "localhost")
                    port = connection_kwargs.get("port", 6379)
                    redis_url = f"redis://{host}:{port}"

        if redis_url is None:
            raise ValueError(
                "Could not determine Redis URL from store. "
                "Please provide a redis_url in middleware configs."
            )

        middlewares = []
        for config in configs:
            # Set the redis_url for redisvl to create its own sync connection
            config.redis_url = redis_url
            config.redis_client = None

            middleware = _create_middleware_from_config(config)
            middlewares.append(middleware)

        return MiddlewareStack(middlewares)
