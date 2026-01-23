"""Semantic cache middleware for LLM responses.

This module provides a middleware that caches LLM responses based on
semantic similarity using Redis and vector embeddings. Compatible with
LangChain's AgentMiddleware protocol for use with create_agent.
"""

import json
import logging
from typing import Any, Awaitable, Callable, List, Union

from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from redisvl.extensions.cache.llm import SemanticCache

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

from .aio import AsyncRedisMiddleware
from .types import SemanticCacheConfig

logger = logging.getLogger(__name__)

# Use the project's serializer for proper LangChain object handling
_serializer = JsonPlusRedisSerializer()


def _serialize_response(response: Any) -> str:
    """Serialize a model response for cache storage.

    Uses LangChain's to_json() for proper serialization of LangChain objects.

    Args:
        response: The response to serialize.

    Returns:
        JSON string representation of the response.
    """
    # Handle ModelResponse (has .result which is list of messages)
    if hasattr(response, "result") and isinstance(response.result, list):
        # Serialize the last AI message from the result list
        for msg in reversed(response.result):
            if hasattr(msg, "to_json"):
                # Use LangChain's built-in serialization
                return json.dumps(msg.to_json())
            elif hasattr(msg, "content"):
                # Fallback: extract content
                return json.dumps({"content": getattr(msg, "content", "")})
        return json.dumps({"content": ""})

    # Handle objects with to_json() (LangChain objects like AIMessage)
    if hasattr(response, "to_json"):
        return json.dumps(response.to_json())

    # Handle dict-style responses
    if isinstance(response, dict):
        return json.dumps(response)

    # Fallback: try to get content attribute
    content = getattr(response, "content", str(response))
    return json.dumps({"content": content})


def _deserialize_response(cached_str: str) -> ModelResponse:
    """Deserialize a cached response into a ModelResponse.

    Uses the project's JsonPlusRedisSerializer for proper LangChain object revival.
    Always returns a ModelResponse to maintain compatibility with agent routing.

    Args:
        cached_str: The cached JSON string.

    Returns:
        A ModelResponse containing the cached message.
    """
    try:
        data = json.loads(cached_str)
        if isinstance(data, dict):
            # Check if this is in LangChain constructor format
            if data.get("lc") in (1, 2) and data.get("type") == "constructor":
                # Use the project's serializer to properly revive
                revived = _serializer._revive_if_needed(data)
                if isinstance(revived, AIMessage):
                    return ModelResponse(result=[revived], structured_response=None)
                # If revived is not an AIMessage, wrap content in one
                content = getattr(revived, "content", str(revived))
                return ModelResponse(
                    result=[AIMessage(content=content)], structured_response=None
                )
            # Simple dict with content - wrap in ModelResponse
            content = data.get("content", "")
            return ModelResponse(
                result=[AIMessage(content=content)], structured_response=None
            )
        # Non-dict data - convert to string
        return ModelResponse(
            result=[AIMessage(content=str(data))], structured_response=None
        )
    except json.JSONDecodeError:
        # If not valid JSON, treat as plain content
        return ModelResponse(
            result=[AIMessage(content=cached_str)], structured_response=None
        )


class SemanticCacheMiddleware(AsyncRedisMiddleware):
    """Middleware that caches LLM responses based on semantic similarity.

    Uses redisvl.extensions.llmcache.SemanticCache to store and retrieve
    cached responses. When a request is semantically similar to a previous
    request (within the distance threshold), the cached response is returned
    without calling the LLM.

    By default, only "final" responses (those without tool_calls) are cached.
    This prevents caching intermediate responses that require tool execution.

    Example:
        ```python
        from langgraph.middleware.redis import (
            SemanticCacheMiddleware,
            SemanticCacheConfig,
        )

        config = SemanticCacheConfig(
            redis_url="redis://localhost:6379",
            distance_threshold=0.1,
            ttl_seconds=3600,
        )

        middleware = SemanticCacheMiddleware(config)

        async def call_llm(request):
            # Your LLM call here
            return response

        # Use middleware
        result = await middleware.awrap_model_call(request, call_llm)
        ```
    """

    _cache: SemanticCache
    _config: SemanticCacheConfig

    def __init__(self, config: SemanticCacheConfig) -> None:
        """Initialize the semantic cache middleware.

        Args:
            config: Configuration for the semantic cache.
        """
        super().__init__(config)
        self._config = config

    async def _setup_async(self) -> None:
        """Set up the SemanticCache instance.

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

    def _extract_prompt(self, messages: List[Union[dict[str, Any], Any]]) -> str:
        """Extract the prompt to use for cache lookup.

        Extracts the last user message content from the messages list.
        Handles both dict-style messages and LangChain message objects.

        Args:
            messages: List of messages from the request.

        Returns:
            The extracted prompt string.
        """
        if not messages:
            return ""

        # Find the last user message
        for message in reversed(messages):
            # Handle dict-style messages
            if isinstance(message, dict):
                role = message.get("role", "")
                if role == "user":
                    return message.get("content", "")
            else:
                # Handle LangChain-style message objects
                msg_type = getattr(message, "type", None) or getattr(
                    message, "role", None
                )
                if msg_type in ("user", "human"):
                    return getattr(message, "content", "")

        return ""

    def _is_final_response(self, response: Any) -> bool:
        """Check if the response is a final response (no tool calls).

        Args:
            response: The model response to check (dict or LangChain type).

        Returns:
            True if the response is final (should be cached), False otherwise.
        """
        # Support both dict-style and LangChain response types
        if isinstance(response, dict):
            tool_calls = response.get("tool_calls")
        else:
            # For ModelResponse, check result[0].tool_calls
            # ModelResponse itself doesn't have tool_calls attribute
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls is None and hasattr(response, "result"):
                result = response.result
                if result and len(result) > 0:
                    tool_calls = getattr(result[0], "tool_calls", None)
        # Response is final if there are no tool_calls or tool_calls is empty
        return not tool_calls

    def _get_tool_names_from_results(
        self, messages: List[Union[dict[str, Any], Any]]
    ) -> List[str]:
        """Extract tool names from tool result messages.

        Args:
            messages: List of messages from the request.

        Returns:
            List of tool names that have results in the messages.
        """
        tool_names = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "") or message.get("type", "")
                if role == "tool":
                    tool_names.append(message.get("name", ""))
            else:
                msg_type = getattr(message, "type", None) or getattr(
                    message, "role", None
                )
                if msg_type == "tool" or message.__class__.__name__ == "ToolMessage":
                    tool_names.append(getattr(message, "name", ""))
        return tool_names

    def _should_skip_cache_for_tool_results(
        self, messages: List[Union[dict[str, Any], Any]]
    ) -> bool:
        """Check if cache should be skipped due to tool results.

        When tool results are present, we check if ALL tools are in the
        deterministic_tools list. If so, caching is safe. Otherwise,
        we skip the cache to avoid returning stale responses.

        Args:
            messages: List of messages from the request.

        Returns:
            True if cache should be skipped, False if caching is OK.
        """
        tool_names = self._get_tool_names_from_results(messages)

        if not tool_names:
            # No tool results - caching is OK
            return False

        # If deterministic_tools is not configured, always skip cache
        # when tool results are present (safest default)
        if self._config.deterministic_tools is None:
            return True

        # Check if ALL tool results are from deterministic tools
        for tool_name in tool_names:
            if tool_name and tool_name not in self._config.deterministic_tools:
                # Found a non-deterministic tool result - skip cache
                return True

        # All tools are deterministic - caching is OK
        return False

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap a model call with semantic caching.

        Checks the cache for a semantically similar request. If found,
        returns the cached response. Otherwise, calls the handler and
        optionally caches the result.

        Args:
            request: The model request containing messages.
            handler: The async function to call the model.

        Returns:
            The model response (from cache or handler).

        Raises:
            Exception: If graceful_degradation is False and cache operations fail.
        """
        await self._ensure_initialized_async()

        # Support both dict-style and LangChain ModelRequest types
        if isinstance(request, dict):
            messages = request.get("messages", [])
        else:
            messages = getattr(request, "messages", [])
        prompt = self._extract_prompt(messages)

        if not prompt:
            # No prompt to cache, just call handler
            return await handler(request)

        # Skip cache lookup if request contains non-deterministic tool results
        # The model needs to process tool output to generate the final response
        if self._should_skip_cache_for_tool_results(messages):
            logger.debug(
                "Skipping cache - request contains non-deterministic tool results"
            )
            response = await handler(request)
            # Cache the final response after tool processing if tools are deterministic
            if not self._config.cache_final_only or self._is_final_response(response):
                try:
                    response_str = _serialize_response(response)
                    await self._cache.astore(prompt=prompt, response=response_str)
                except Exception as e:
                    if not self._graceful_degradation:
                        raise
                    logger.warning(f"Cache store failed: {e}")
            return response

        # Try to get from cache using async method
        try:
            cached = await self._cache.acheck(prompt=prompt)
            if cached:
                # Cache hit - return cached response
                cached_response = cached[0].get("response")
                if cached_response:
                    logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return _deserialize_response(cached_response)
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Cache check failed, calling handler: {e}")

        # Cache miss - call handler
        response = await handler(request)

        # Store in cache if appropriate
        should_cache = not self._config.cache_final_only or self._is_final_response(
            response
        )

        if should_cache:
            try:
                # Serialize response for storage using async method
                response_str = _serialize_response(response)
                await self._cache.astore(prompt=prompt, response=response_str)
                logger.debug(f"Cached response for prompt: {prompt[:50]}...")
            except Exception as e:
                if not self._graceful_degradation:
                    raise
                logger.warning(f"Cache store failed: {e}")

        return response
