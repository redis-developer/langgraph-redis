"""Tool result cache middleware.

This module provides a middleware that caches tool call results
using exact-match key-value lookup in Redis. Tool caching is
deterministic: same tool + same args = same result. This uses
direct Redis GET/SET instead of vector similarity.

Compatible with LangChain's AgentMiddleware protocol for use with create_agent.
"""

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Tuple, Union

from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from .aio import AsyncRedisMiddleware
from .types import ToolCacheConfig

logger = logging.getLogger(__name__)

DEFAULT_VOLATILE_ARG_NAMES: frozenset[str] = frozenset(
    {
        "timestamp",
        "current_time",
        "now",
        "date",
        "today",
        "current_date",
        "current_timestamp",
    }
)

DEFAULT_SIDE_EFFECT_PREFIXES: Tuple[str, ...] = (
    "send_",
    "delete_",
    "create_",
    "update_",
    "remove_",
    "write_",
    "post_",
    "put_",
    "patch_",
)


class ToolResultCacheMiddleware(AsyncRedisMiddleware):
    """Middleware that caches tool call results using exact-match lookup.

    Uses direct Redis GET/SET for deterministic tool result caching.
    When a tool is called with the same arguments as a previous call,
    the cached result is returned without executing the tool.

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

    _config: ToolCacheConfig

    def __init__(self, config: ToolCacheConfig) -> None:
        """Initialize the tool cache middleware.

        Args:
            config: Configuration for the tool cache.
        """
        super().__init__(config)
        self._config = config

    async def _setup_async(self) -> None:
        """Set up the tool cache.

        No additional setup needed — the tool cache uses the async Redis
        client from the base class directly for GET/SET operations.
        """
        pass

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

    def _has_volatile_args(self, args: Dict[str, Any]) -> bool:
        """Check if args contain volatile argument names at any nesting depth.

        Recurses into nested dicts and into dicts inside lists/tuples.

        Args:
            args: The tool arguments dict.

        Returns:
            True if any key in args (recursively) matches a configured
            volatile arg name, False otherwise.
        """
        volatile_names = self._config.volatile_arg_names
        if not volatile_names:
            return False
        for key, value in args.items():
            if key in volatile_names:
                return True
            if isinstance(value, dict):
                if self._has_volatile_args(value):
                    return True
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict) and self._has_volatile_args(item):
                        return True
        return False

    def _has_side_effect_prefix(self, tool_name: str) -> bool:
        """Check if tool name starts with a configured side-effect prefix.

        Args:
            tool_name: The name of the tool.

        Returns:
            True if the tool name matches a side-effect prefix.
        """
        prefixes = self._config.side_effect_prefixes
        if not prefixes:
            return False
        return tool_name.startswith(prefixes)

    def _strip_ignored_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of args with ignored names removed (top-level only).

        Args:
            args: The tool arguments dict.

        Returns:
            A new dict without the ignored keys, or the original if nothing
            to strip.
        """
        if not isinstance(args, dict):
            return {}
        ignored = self._config.ignored_arg_names
        if not ignored:
            return args
        return {k: v for k, v in args.items() if k not in ignored}

    @staticmethod
    def _extract_args(request: ToolCallRequest) -> Dict[str, Any]:
        """Extract args dict from a request (dict or ToolCallRequest).

        Args:
            request: The tool call request.

        Returns:
            The arguments dict.
        """
        if isinstance(request, dict):
            return request.get("args", {})
        tool_call = getattr(request, "tool_call", None)
        if isinstance(tool_call, dict):
            return tool_call.get("args", {})
        return {}

    def _is_tool_cacheable(self, request: ToolCallRequest) -> bool:
        """Check if a tool's results should be cached.

        Uses a priority chain inspired by SQL function volatility and
        MCP ToolAnnotations:

        1. ``metadata["cacheable"]`` — explicit override (highest priority)
        2. ``metadata["destructive"]`` — never cache destructive tools
        3. ``metadata["volatile"]`` — never cache volatile tools
        4. ``metadata["read_only"] and metadata["idempotent"]`` — cache
        5. Side-effect prefix match — never cache
        6. Volatile arg name in call args — never cache
        7. Config whitelist / blacklist — existing fallback

        Args:
            request: The tool call request containing the tool object.

        Returns:
            True if the tool's results should be cached, False otherwise.
        """
        # Extract tool name and tool object from ToolCallRequest
        if isinstance(request, dict):
            tool_name = request.get("tool_name", "")
            tool = None
        else:
            tool_call = getattr(request, "tool_call", None)
            tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
            tool = getattr(request, "tool", None)

        # --- Priority 1: explicit cacheable flag (highest) ---
        metadata: Dict[str, Any] = {}
        if tool is not None:
            metadata = getattr(tool, "metadata", None) or {}
            if "cacheable" in metadata:
                return bool(metadata["cacheable"])

        # --- Priority 2: destructive metadata → never cache ---
        if metadata.get("destructive") is True:
            return False

        # --- Priority 3: volatile metadata → never cache ---
        if metadata.get("volatile") is True:
            return False

        # --- Priority 4: read_only + idempotent → cache ---
        if metadata.get("read_only") is True and metadata.get("idempotent") is True:
            return True

        # --- Priority 5: side-effect prefix → never cache ---
        if self._has_side_effect_prefix(tool_name):
            return False

        # --- Priority 6: volatile arg names → never cache ---
        args = self._extract_args(request)
        if self._has_volatile_args(args):
            return False

        # --- Priority 7: config whitelist / blacklist ---
        return self._is_tool_cacheable_by_config(tool_name)

    def _build_cache_key(self, request: ToolCallRequest) -> str:
        """Build a deterministic cache key from the tool request.

        Creates an exact-match Redis key from the tool name and sorted
        JSON arguments. This ensures that identical tool calls always
        produce the same key, and different calls always produce different keys.

        Args:
            request: The tool call request (dict or LangChain type).

        Returns:
            A deterministic string key for Redis GET/SET.
        """
        # Support both dict-style and LangChain ToolCallRequest types
        if isinstance(request, dict):
            tool_name = request.get("tool_name", "")
            args = request.get("args", {})
        else:
            tool_call = getattr(request, "tool_call", None)
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                args = tool_call.get("args", {})
            else:
                tool_name = ""
                args = {}

        # Strip ignored args before building the key
        effective_args = self._strip_ignored_args(args)

        # Deterministic key: config name + tool name + sorted JSON args
        args_str = json.dumps(effective_args, sort_keys=True)
        return f"{self._config.name}:{tool_name}:{args_str}"

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

    def _deserialize_tool_result(
        self, cached_response: str, tool_name: str, tool_call_id: str
    ) -> LangChainToolMessage:
        """Deserialize a cached tool result into a ToolMessage.

        Converts the cached JSON string back into a proper LangChain
        ToolMessage so it conforms to the AgentMiddleware protocol.

        Args:
            cached_response: The cached JSON string.
            tool_name: The name of the tool that produced this result.
            tool_call_id: The ID of the tool call this result is for.

        Returns:
            A LangChainToolMessage containing the cached result.
        """
        # Parse the cached content
        if isinstance(cached_response, str):
            try:
                parsed = json.loads(cached_response)
            except json.JSONDecodeError:
                parsed = cached_response
        else:
            parsed = cached_response

        # Extract content from the parsed result
        if isinstance(parsed, dict):
            content = parsed.get("content", json.dumps(parsed))
        elif isinstance(parsed, str):
            content = parsed
        else:
            content = json.dumps(parsed)

        return LangChainToolMessage(
            content=content,
            name=tool_name,
            tool_call_id=tool_call_id or "",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Wrap a tool call with exact-match caching.

        This method is part of the LangChain AgentMiddleware protocol.
        Checks the cache for an exact tool+args match. If found,
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
        # Extract tool name from the request
        if isinstance(request, dict):
            tool_name = request.get("tool_name", "")
            tool_call_id = request.get("id", "")
        else:
            tool_call = getattr(request, "tool_call", None)
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                tool_call_id = tool_call.get("id", "")
            else:
                tool_name = ""
                tool_call_id = ""

        # If no tool name or tool is not cacheable, skip caching
        if not tool_name or not self._is_tool_cacheable(request):
            return await handler(request)

        await self._ensure_initialized_async()

        cache_key = self._build_cache_key(request)

        # Try to get from cache using exact-match Redis GET
        try:
            cached_response = await self._redis.get(cache_key)
            if cached_response is not None:
                # Decode bytes to string if needed
                if isinstance(cached_response, bytes):
                    cached_response = cached_response.decode("utf-8")
                logger.debug(f"Tool cache hit for key: {cache_key[:80]}")
                return self._deserialize_tool_result(
                    cached_response, tool_name, tool_call_id
                )
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Tool cache check failed, calling handler: {e}")

        # Cache miss - call handler
        result = await handler(request)

        # Store in cache using Redis SET with optional TTL
        try:
            result_str = self._serialize_tool_result(result)
            if self._config.ttl_seconds is not None:
                await self._redis.set(
                    cache_key, result_str, ex=self._config.ttl_seconds
                )
            else:
                await self._redis.set(cache_key, result_str)
            logger.debug(f"Tool cache stored for key: {cache_key[:80]}")
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
