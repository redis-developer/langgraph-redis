"""Tool result cache middleware.

This module provides a middleware that caches tool call results
based on semantic similarity using Redis and vector embeddings.
Compatible with LangChain's AgentMiddleware protocol for use with create_agent.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Union

from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from redisvl.extensions.cache.llm import SemanticCache

from .aio import AsyncRedisMiddleware
from .types import ToolCacheConfig

logger = logging.getLogger(__name__)


class ToolResultCacheMiddleware(AsyncRedisMiddleware):
    """Middleware that caches tool call results based on semantic similarity.

    Uses redisvl.extensions.cache.llm.SemanticCache to store and retrieve
    cached tool results. When a tool call is semantically similar to a previous
    call (within the distance threshold), the cached result is returned
    without executing the tool.

    Tool caching is especially useful for expensive operations like:
    - API calls to external services
    - Database queries
    - Web searches
    - Complex calculations

    Example:
        ```python
        from langgraph.middleware.redis import (
            ToolResultCacheMiddleware,
            ToolCacheConfig,
        )

        config = ToolCacheConfig(
            redis_url="redis://localhost:6379",
            cacheable_tools=["search", "calculate"],
            excluded_tools=["random_number"],
            ttl_seconds=3600,
        )

        middleware = ToolResultCacheMiddleware(config)

        async def execute_tool(request):
            # Your tool execution here
            return result

        # Use middleware
        result = await middleware.awrap_tool_call(request, execute_tool)
        ```
    """

    _cache: SemanticCache
    _config: ToolCacheConfig

    def __init__(self, config: ToolCacheConfig) -> None:
        """Initialize the tool cache middleware.

        Args:
            config: Configuration for the tool cache.
        """
        super().__init__(config)
        self._config = config

    async def _setup_async(self) -> None:
        """Set up the SemanticCache instance for tool results.

        Note: SemanticCache from redisvl uses synchronous Redis operations
        internally, so we must provide redis_url and let it manage its own
        sync connection rather than passing our async client.
        """
        cache_kwargs: dict[str, Any] = {
            "name": self._config.name,
            "distance_threshold": self._config.distance_threshold,
        }

        # SemanticCache requires a sync Redis connection
        # Use redis_url to let it create its own connection
        if self._config.redis_url:
            cache_kwargs["redis_url"] = self._config.redis_url
        elif self._config.connection_args:
            cache_kwargs["connection_kwargs"] = self._config.connection_args

        if self._config.vectorizer is not None:
            cache_kwargs["vectorizer"] = self._config.vectorizer

        if self._config.ttl_seconds is not None:
            cache_kwargs["ttl"] = self._config.ttl_seconds

        self._cache = SemanticCache(**cache_kwargs)

    def _is_tool_cacheable_by_config(self, tool_name: str) -> bool:
        """Check if a tool's results should be cached based on config.

        Args:
            tool_name: The name of the tool.

        Returns:
            True if the tool's results should be cached, False otherwise.
        """
        # If cacheable_tools is set, only those tools are cached
        if self._config.cacheable_tools is not None:
            return tool_name in self._config.cacheable_tools

        # Otherwise, cache all tools except excluded ones
        return tool_name not in self._config.excluded_tools

    def _is_tool_cacheable(self, request: ToolCallRequest) -> bool:
        """Check if a tool's results should be cached.

        Checks both tool metadata (LangChain standard) and middleware config.
        Tool metadata takes precedence over config.

        The tool's metadata can specify {"cacheable": True/False} to override
        the middleware config. This follows LangChain's pattern for tool metadata.

        Args:
            request: The tool call request containing the tool object.

        Returns:
            True if the tool's results should be cached, False otherwise.
        """
        # Get tool name
        if isinstance(request, dict):
            tool_name = request.get("tool_name", "")
            tool = None
        else:
            tool_name = getattr(request, "name", "")
            tool = getattr(request, "tool", None)

        # Check tool metadata first (LangChain standard pattern)
        if tool is not None:
            metadata = getattr(tool, "metadata", None) or {}
            if "cacheable" in metadata:
                return bool(metadata["cacheable"])

        # Fall back to config-based check
        return self._is_tool_cacheable_by_config(tool_name)

    def _build_cache_key(self, request: ToolCallRequest) -> str:
        """Build a cache key from the tool request.

        Creates a string representation of the tool call that can be
        used for semantic similarity matching.

        Args:
            request: The tool call request (dict or LangChain type).

        Returns:
            A string key for caching.
        """
        # Support both dict-style and LangChain ToolCallRequest types
        if isinstance(request, dict):
            tool_name = request.get("tool_name", "")
            args = request.get("args", {})
        else:
            tool_name = getattr(request, "name", "")
            args = getattr(request, "args", {})

        # Create a human-readable representation for semantic matching
        args_str = json.dumps(args, sort_keys=True)
        return f"tool:{tool_name} args:{args_str}"

    def _serialize_tool_result(self, value: Any) -> str:
        """Serialize a tool result to a JSON string for caching.

        Supports LangChain ToolMessage/Command objects by converting
        them to JSON-compatible structures before encoding.

        Args:
            value: The tool result to serialize.

        Returns:
            A JSON string representation of the result.
        """
        # Handle known LangChain message/command types
        if isinstance(value, (LangChainToolMessage, Command)):
            to_json = getattr(value, "to_json", None)
            if callable(to_json):
                # Convert to plain data first, then dump
                return json.dumps(to_json())

        # Fallback: try direct JSON serialization
        try:
            return json.dumps(value)
        except TypeError:
            # Last resort: store string representation
            return json.dumps(str(value))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Wrap a tool call with caching.

        This method is part of the LangChain AgentMiddleware protocol.
        Checks the cache for a semantically similar tool call. If found,
        returns the cached result. Otherwise, calls the handler and
        caches the result.

        Args:
            request: The tool call request.
            handler: The async function to execute the tool.

        Returns:
            The tool result (from cache or handler).

        Raises:
            Exception: If graceful_degradation is False and cache operations fail.
        """
        tool_name = (
            getattr(request, "name", "") or request.get("tool_name", "")
            if isinstance(request, dict)
            else getattr(request, "name", "")
        )

        # If no tool name or tool is not cacheable, skip caching
        if not tool_name or not self._is_tool_cacheable(request):
            return await handler(request)

        await self._ensure_initialized_async()

        cache_key = self._build_cache_key(request)

        # Try to get from cache using async method
        try:
            cached = await self._cache.acheck(prompt=cache_key)
            if cached:
                # Cache hit - return cached result
                cached_response = cached[0].get("response")
                if cached_response:
                    # Parse the cached response if it's a string
                    if isinstance(cached_response, str):
                        try:
                            return json.loads(cached_response)
                        except json.JSONDecodeError:
                            return {"content": cached_response}
                    return cached_response
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Tool cache check failed, calling handler: {e}")

        # Cache miss - call handler
        result = await handler(request)

        # Store in cache using async method
        try:
            result_str = self._serialize_tool_result(result)
            await self._cache.astore(prompt=cache_key, response=result_str)
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Tool cache store failed: {e}")

        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Pass through model calls without caching.

        This method is part of the LangChain AgentMiddleware protocol.
        Tool cache middleware only caches tool calls, not model calls.

        Args:
            request: The model request.
            handler: The async function to call the model.

        Returns:
            The model response from the handler.
        """
        return await handler(request)
