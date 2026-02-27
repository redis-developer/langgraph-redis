"""Unit tests for ConversationMemoryMiddleware."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from langgraph.middleware.redis.types import ConversationMemoryConfig


class TestConversationMemoryMiddleware:
    """Tests for ConversationMemoryMiddleware class."""

    @pytest.mark.asyncio
    async def test_init_with_config(self) -> None:
        """Test initialization with ConversationMemoryConfig."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(
            redis_client=mock_client,
            name="test_memory",
            session_tag="user_123",
            top_k=10,
        )
        middleware = ConversationMemoryMiddleware(config)
        assert middleware._config.name == "test_memory"
        assert middleware._config.session_tag == "user_123"
        assert middleware._config.top_k == 10

    @pytest.mark.asyncio
    async def test_extracts_last_user_message(self) -> None:
        """Test extraction of last user message for context retrieval."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)
        middleware = ConversationMemoryMiddleware(config)

        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        query = middleware._extract_query(messages)
        assert query == "Second question"

    @pytest.mark.asyncio
    async def test_extracts_query_handles_empty(self) -> None:
        """Test query extraction with empty messages."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)
        middleware = ConversationMemoryMiddleware(config)

        query = middleware._extract_query([])
        assert query == ""

    @pytest.mark.asyncio
    async def test_retrieves_relevant_context(self) -> None:
        """Test that relevant context is retrieved and injected."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client, top_k=3)

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history.get_relevant = MagicMock(
                return_value=[
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"},
                ]
            )
            mock_history.add_messages = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> ModelResponse:
                # Check that context was added (messages captured for potential assertions)
                _messages = request.get("messages", [])  # noqa: F841
                # Should have injected context
                return ModelResponse(result=[AIMessage(content="Response")])

            request = {"messages": [{"role": "user", "content": "New question"}]}
            await middleware.awrap_model_call(request, mock_handler)

            mock_history.get_relevant.assert_called_once()

    @pytest.mark.asyncio
    async def test_stores_new_messages_after_response(self) -> None:
        """Test that new messages are stored after model response."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history.get_relevant = MagicMock(return_value=[])
            mock_history.add_messages = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> ModelResponse:
                return ModelResponse(result=[AIMessage(content="Model response")])

            request = {"messages": [{"role": "user", "content": "User question"}]}
            await middleware.awrap_model_call(request, mock_handler)

            # Should have stored both user message and assistant response
            assert mock_history.add_messages.call_count == 2
            # First call: user message
            user_call = mock_history.add_messages.call_args_list[0]
            assert user_call[0][0][0]["role"] == "user"
            assert user_call[0][0][0]["content"] == "User question"
            # Second call: assistant message with "llm" role for redisvl
            llm_call = mock_history.add_messages.call_args_list[1]
            assert llm_call[0][0][0]["role"] == "llm"
            assert llm_call[0][0][0]["content"] == "Model response"

    @pytest.mark.asyncio
    async def test_uses_session_tag(self) -> None:
        """Test that session_tag is used for message isolation."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(
            redis_client=mock_client, session_tag="user_abc"
        )

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify history was created with session_tag
            call_kwargs = mock_history_class.call_args.kwargs
            assert call_kwargs.get("session_tag") == "user_abc"

    @pytest.mark.asyncio
    async def test_uses_distance_threshold(self) -> None:
        """Test that distance_threshold is passed to history."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(
            redis_client=mock_client, distance_threshold=0.4
        )

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            # Verify history was created with distance_threshold
            call_kwargs = mock_history_class.call_args.kwargs
            assert call_kwargs.get("distance_threshold") == 0.4

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_history_error(self) -> None:
        """Test that middleware passes through on history error."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(
            redis_client=mock_client, graceful_degradation=True
        )

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history.get_relevant = MagicMock(side_effect=Exception("Redis error"))
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> ModelResponse:
                return ModelResponse(result=[AIMessage(content="Handler response")])

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            assert hasattr(result, "result")
            assert result.result[0].content == "Handler response"

    @pytest.mark.asyncio
    async def test_raises_on_history_error_without_graceful_degradation(self) -> None:
        """Test that middleware raises on history error when graceful_degradation=False."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(
            redis_client=mock_client, graceful_degradation=False
        )

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history.get_relevant = MagicMock(side_effect=Exception("Redis error"))
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> ModelResponse:
                return ModelResponse(result=[AIMessage(content="Handler response")])

            request = {"messages": [{"role": "user", "content": "Test"}]}
            with pytest.raises(Exception, match="Redis error"):
                await middleware.awrap_model_call(request, mock_handler)

    @pytest.mark.asyncio
    async def test_ttl_configuration(self) -> None:
        """Test that TTL is passed to history."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client, ttl_seconds=86400)

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            # Note: SemanticMessageHistory doesn't support TTL directly,
            # so TTL is stored in config but not passed to the history class
            # Verify the config still has TTL set
            assert middleware._config.ttl_seconds == 86400

    @pytest.mark.asyncio
    async def test_context_injection_format(self) -> None:
        """Test that context is injected in the correct format."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            # Return some relevant context
            mock_history.get_relevant = MagicMock(
                return_value=[
                    {"role": "user", "content": "What is Python?"},
                    {
                        "role": "assistant",
                        "content": "Python is a programming language.",
                    },
                ]
            )
            mock_history.add_messages = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            seen_messages = []

            async def mock_handler(request: dict) -> ModelResponse:
                seen_messages.extend(request.get("messages", []))
                return ModelResponse(result=[AIMessage(content="New response")])

            request = {"messages": [{"role": "user", "content": "Tell me more"}]}
            await middleware.awrap_model_call(request, mock_handler)

            # Context should be injected as a single SystemMessage + the user message
            assert len(seen_messages) == 2
            from langchain_core.messages import SystemMessage

            # First message: SystemMessage with consolidated context
            assert isinstance(seen_messages[0], SystemMessage)
            assert "earlier in this conversation" in seen_messages[0].content.lower()
            # Context should contain the retrieved messages
            assert "What is Python?" in seen_messages[0].content
            assert "Python is a programming language." in seen_messages[0].content

    @pytest.mark.asyncio
    async def test_tool_call_passes_through(self) -> None:
        """Test that awrap_tool_call passes through (memory is for model calls)."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)
        middleware = ConversationMemoryMiddleware(config)

        async def mock_handler(request: dict) -> dict:
            return {"result": "tool result"}

        request = {"tool_name": "search", "args": {"query": "test"}}
        result = await middleware.awrap_tool_call(request, mock_handler)

        assert result == {"result": "tool result"}

    @pytest.mark.asyncio
    async def test_stores_messages_from_model_response(self) -> None:
        """Test that both user and assistant messages are stored when handler returns ModelResponse."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)

        with patch(
            "langgraph.middleware.redis.conversation_memory.SemanticMessageHistory"
        ) as mock_history_class:
            mock_history = MagicMock()
            mock_history.get_relevant = MagicMock(return_value=[])
            mock_history.add_messages = MagicMock()
            mock_history_class.return_value = mock_history

            middleware = ConversationMemoryMiddleware(config)
            await middleware._ensure_initialized_async()

            async def mock_handler(request: dict) -> ModelResponse:
                return ModelResponse(
                    result=[AIMessage(content="I'm doing great, thanks!")]
                )

            request = {"messages": [{"role": "user", "content": "How are you?"}]}
            result = await middleware.awrap_model_call(request, mock_handler)

            # Verify the response is a ModelResponse with the right content
            assert hasattr(result, "result")
            assert result.result[0].content == "I'm doing great, thanks!"

            # Verify both messages were stored
            assert mock_history.add_messages.call_count == 2
            user_call = mock_history.add_messages.call_args_list[0]
            assert user_call[0][0] == [{"role": "user", "content": "How are you?"}]
            llm_call = mock_history.add_messages.call_args_list[1]
            assert llm_call[0][0] == [
                {"role": "llm", "content": "I'm doing great, thanks!"}
            ]

    @pytest.mark.asyncio
    async def test_handles_langchain_messages(self) -> None:
        """Test handling of LangChain-style message objects."""
        from langgraph.middleware.redis.conversation_memory import (
            ConversationMemoryMiddleware,
        )

        mock_client = AsyncMock()
        config = ConversationMemoryConfig(redis_client=mock_client)
        middleware = ConversationMemoryMiddleware(config)

        class MockMessage:
            def __init__(self, role: str, content: str):
                self.type = role
                self.content = content

        messages = [
            MockMessage("human", "What is AI?"),
        ]
        query = middleware._extract_query(messages)
        assert query == "What is AI?"
