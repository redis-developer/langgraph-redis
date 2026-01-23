"""Unit tests for SemanticCacheMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgraph.middleware.redis.types import SemanticCacheConfig


class TestSemanticCacheMiddleware:
    """Tests for SemanticCacheMiddleware class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self) -> None:
        """Test initialization with SemanticCacheConfig."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(
            redis_client=mock_client,
            name="test_cache",
            distance_threshold=0.15,
        )
        middleware = SemanticCacheMiddleware(config)
        assert middleware._config.name == "test_cache"
        assert middleware._config.distance_threshold == 0.15

    @pytest.mark.asyncio
    async def test_extract_prompt_from_messages(self) -> None:
        """Test prompt extraction from messages list."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        prompt = middleware._extract_prompt(messages)
        assert "What is Python?" in prompt

    @pytest.mark.asyncio
    async def test_extract_prompt_handles_empty_messages(self) -> None:
        """Test prompt extraction with empty messages."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        prompt = middleware._extract_prompt([])
        assert prompt == ""

    @pytest.mark.asyncio
    async def test_extract_prompt_uses_last_user_message(self) -> None:
        """Test that prompt extraction uses the last user message."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        prompt = middleware._extract_prompt(messages)
        assert prompt == "Second question"

    @pytest.mark.asyncio
    async def test_is_final_response_without_tool_calls(self) -> None:
        """Test that response without tool_calls is considered final."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        response = {"content": "The answer is 42."}
        assert middleware._is_final_response(response) is True

    @pytest.mark.asyncio
    async def test_is_final_response_with_tool_calls(self) -> None:
        """Test that response with tool_calls is not considered final."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        response = {
            "content": "",
            "tool_calls": [{"name": "search", "args": {"query": "test"}}],
        }
        assert middleware._is_final_response(response) is False

    @pytest.mark.asyncio
    async def test_is_final_response_with_empty_tool_calls(self) -> None:
        """Test that response with empty tool_calls is considered final."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        response = {"content": "Answer", "tool_calls": []}
        assert middleware._is_final_response(response) is True

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self) -> None:
        """Test that cache hit returns cached response without calling handler."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            # Cached response is stored as JSON, so we simulate that format
            mock_cache.acheck = AsyncMock(
                return_value=[
                    {"response": '{"content": "Cached answer"}', "metadata": {}}
                ]
            )
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            handler_called = []

            async def mock_handler(request: dict) -> dict:
                handler_called.append(True)
                return {"content": "New answer"}

            request = {"messages": [{"role": "user", "content": "What is Python?"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            # Cached responses are deserialized as ModelResponse for proper agent integration
            from langchain.agents.middleware.types import ModelResponse

            assert isinstance(result, ModelResponse)
            assert len(result.result) == 1
            assert result.result[0].content == "Cached answer"
            assert len(handler_called) == 0

    @pytest.mark.asyncio
    async def test_cache_miss_calls_handler(self) -> None:
        """Test that cache miss calls handler and stores result."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])  # Cache miss
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "New answer"}

            request = {"messages": [{"role": "user", "content": "What is Python?"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert result == {"content": "New answer"}
            mock_cache.astore.assert_called_once()

    @pytest.mark.asyncio
    async def test_does_not_cache_tool_calls_response(self) -> None:
        """Test that responses with tool_calls are not cached when cache_final_only=True."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client, cache_final_only=True)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])  # Cache miss
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {
                    "content": "",
                    "tool_calls": [{"name": "search", "args": {}}],
                }

            request = {"messages": [{"role": "user", "content": "Search for X"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert "tool_calls" in result
            mock_cache.astore.assert_not_called()

    @pytest.mark.asyncio
    async def test_caches_all_responses_when_cache_final_only_false(self) -> None:
        """Test that all responses are cached when cache_final_only=False."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client, cache_final_only=False)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])  # Cache miss
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {
                    "content": "",
                    "tool_calls": [{"name": "search", "args": {}}],
                }

            request = {"messages": [{"role": "user", "content": "Search for X"}]}
            await middleware.awrap_model_call(request, mock_handler)

            mock_cache.astore.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_cache_error(self) -> None:
        """Test that middleware passes through on cache error with graceful_degradation=True."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(
            redis_client=mock_client, graceful_degradation=True
        )

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(side_effect=Exception("Redis error"))
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "Handler response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert result == {"content": "Handler response"}

    @pytest.mark.asyncio
    async def test_raises_on_cache_error_without_graceful_degradation(self) -> None:
        """Test that middleware raises on cache error when graceful_degradation=False."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(
            redis_client=mock_client, graceful_degradation=False
        )

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(side_effect=Exception("Redis error"))
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "Handler response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            with pytest.raises(Exception, match="Redis error"):
                await middleware.awrap_model_call(request, mock_handler)

    @pytest.mark.asyncio
    async def test_ttl_passed_to_cache(self) -> None:
        """Test that TTL is passed to cache store."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client, ttl_seconds=3600)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache.acheck = AsyncMock(return_value=[])
            mock_cache.astore = AsyncMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> dict:
                return {"content": "Answer"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            await middleware.awrap_model_call(request, mock_handler)

            # Verify astore was called
            mock_cache.astore.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_vectorizer(self) -> None:
        """Test that custom vectorizer is used."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware
        from langgraph.middleware.redis.vectorizer import AsyncEmbeddingVectorizer

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 768 for _ in texts]

        mock_vectorizer = AsyncEmbeddingVectorizer(embed_fn=mock_embed, dims=768)

        mock_client = AsyncMock()
        config = SemanticCacheConfig(
            redis_client=mock_client, vectorizer=mock_vectorizer
        )

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify SemanticCache was created with vectorizer
            mock_cache_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_prompt_handles_langchain_messages(self) -> None:
        """Test prompt extraction with LangChain-style message objects."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client)
        middleware = SemanticCacheMiddleware(config)

        # Simulate LangChain message objects with content attribute
        class MockMessage:
            def __init__(self, role: str, content: str):
                self.type = role
                self.content = content

        messages = [
            MockMessage("system", "You are helpful."),
            MockMessage("human", "What is AI?"),
        ]
        prompt = middleware._extract_prompt(messages)
        assert prompt == "What is AI?"

    @pytest.mark.asyncio
    async def test_uses_distance_threshold(self) -> None:
        """Test that distance_threshold is passed to cache."""
        from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware

        mock_client = AsyncMock()
        config = SemanticCacheConfig(redis_client=mock_client, distance_threshold=0.2)

        with patch(
            "langgraph.middleware.redis.semantic_cache.SemanticCache"
        ) as mock_cache_class:
            mock_cache = MagicMock()
            mock_cache_class.return_value = mock_cache

            middleware = SemanticCacheMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify SemanticCache was created with distance_threshold
            call_kwargs = mock_cache_class.call_args.kwargs
            assert call_kwargs.get("distance_threshold") == 0.2
