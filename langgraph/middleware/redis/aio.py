"""Async Redis middleware base class.

This module provides the async base class for Redis middleware that is
compatible with LangChain's AgentMiddleware protocol.
"""

import asyncio
import logging
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar, Union

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

from .types import MiddlewareConfig

logger = logging.getLogger(__name__)

AsyncRedisClientType = TypeVar("AsyncRedisClientType", bound=AsyncRedis)


class AsyncRedisMiddleware(AgentMiddleware, Generic[AsyncRedisClientType]):
    """Abstract base class for async Redis middleware.

    This class provides common functionality for all async Redis-based middleware:
    - Async Redis client lifecycle management
    - Lazy initialization with double-checked locking
    - Graceful degradation on Redis errors
    - Async context manager support
    - Default pass-through implementations for model/tool wrapping

    Subclasses must implement:
    - _setup_async(): Called once during initialization to set up resources

    Example:
        ```python
        class MyAsyncMiddleware(AsyncRedisMiddleware):
            async def _setup_async(self) -> None:
                # Initialize resources
                self._cache = SemanticCache(redis_client=self._redis)

        config = MiddlewareConfig(redis_url="redis://localhost:6379")
        async with MyAsyncMiddleware(config) as middleware:
            result = await middleware.awrap_model_call(request, handler)
        ```
    """

    _redis: AsyncRedisClientType
    _config: MiddlewareConfig
    _owns_client: bool
    _graceful_degradation: bool
    _initialized: bool
    _init_lock: asyncio.Lock

    def __init__(self, config: MiddlewareConfig) -> None:
        """Initialize the async middleware.

        Args:
            config: Middleware configuration with Redis connection details.

        Raises:
            ValueError: If neither redis_url nor redis_client is provided.
        """
        self._config = config
        self._graceful_degradation = config.graceful_degradation
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Set up Redis client
        if config.redis_client is not None:
            self._redis = config.redis_client
            self._owns_client = False
        elif config.redis_url is not None:
            connection_args = config.connection_args or {}
            self._redis = AsyncRedis.from_url(config.redis_url, **connection_args)
            self._owns_client = True
        else:
            raise ValueError("Either redis_url or redis_client must be provided")

    @abstractmethod
    async def _setup_async(self) -> None:
        """Set up middleware resources asynchronously.

        Called once during lazy initialization. Subclasses should override
        this to initialize caches, indices, or other resources.
        """
        pass

    async def _ensure_initialized_async(self) -> None:
        """Ensure middleware is initialized (async-safe).

        Uses double-checked locking pattern for async safety.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if not self._initialized:
                await self._setup_async()
                self._initialized = True

    async def aclose(self) -> None:
        """Close the Redis connection if owned by this middleware."""
        if self._owns_client and hasattr(self, "_redis"):
            try:
                await self._redis.aclose()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Error closing async Redis client: {e}")

    async def __aenter__(self) -> "AsyncRedisMiddleware[AsyncRedisClientType]":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit async context manager and close resources."""
        await self.aclose()

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap a model call with middleware logic.

        This method is part of the LangChain AgentMiddleware protocol.
        Default implementation passes through to the handler.
        Subclasses can override to add caching, logging, etc.

        Args:
            request: The model request (typically contains messages).
            handler: The async function to call the model.

        Returns:
            The model response (ModelResponse or AIMessage).
        """
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Wrap a tool call with middleware logic.

        This method is part of the LangChain AgentMiddleware protocol.
        Default implementation passes through to the handler.
        Subclasses can override to add caching, logging, etc.

        Args:
            request: The tool call request.
            handler: The async function to execute the tool.

        Returns:
            The tool result message or command.
        """
        return await handler(request)
