"""Conversation memory middleware for semantic message history.

This module provides a middleware that retrieves relevant past messages
based on semantic similarity and injects them into the conversation context.
Compatible with LangChain's AgentMiddleware protocol for use with create_agent.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Union

from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage as LangChainToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from redisvl.extensions.message_history import SemanticMessageHistory

from .aio import AsyncRedisMiddleware
from .types import ConversationMemoryConfig

logger = logging.getLogger(__name__)


def _content_to_str(content: Any) -> str:
    """Convert message content to a plain string for storage.

    When using the OpenAI Responses API, AIMessage.content is a list of
    content blocks (dicts with 'type' and 'text' keys) rather than a plain
    string. SemanticMessageHistory requires string content, so we extract
    and join the text from all blocks.

    Args:
        content: Message content — either a string or a list of content blocks.

    Returns:
        A plain string suitable for storage and embedding.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts) if parts else ""
    return str(content) if content else ""


class ConversationMemoryMiddleware(AsyncRedisMiddleware):
    """Middleware that injects relevant past messages into context.

    Uses redisvl.extensions.message_history.SemanticMessageHistory to store
    conversation history and retrieve semantically relevant past messages.
    This enables long-term memory for conversational agents by:
    - Storing all messages in Redis with vector embeddings
    - Retrieving relevant past context based on the current query
    - Injecting context to help the model maintain coherent conversations

    Example:
        ```python
        from langgraph.middleware.redis import (
            ConversationMemoryMiddleware,
            ConversationMemoryConfig,
        )

        config = ConversationMemoryConfig(
            redis_url="redis://localhost:6379",
            session_tag="user_123",
            top_k=5,
            distance_threshold=0.7,
        )

        middleware = ConversationMemoryMiddleware(config)

        # Use with your model calls
        result = await middleware.awrap_model_call(request, call_model)
        ```
    """

    _history: SemanticMessageHistory
    _config: ConversationMemoryConfig

    def __init__(self, config: ConversationMemoryConfig) -> None:
        """Initialize the conversation memory middleware.

        Args:
            config: Configuration for the conversation memory.
        """
        super().__init__(config)
        self._config = config

    async def _setup_async(self) -> None:
        """Set up the SemanticMessageHistory instance.

        Note: SemanticMessageHistory from redisvl uses synchronous Redis operations
        internally, so we must provide redis_url and let it manage its own
        sync connection rather than passing our async client.
        """
        history_kwargs: dict[str, Any] = {
            "name": self._config.name,
            "distance_threshold": self._config.distance_threshold,
        }

        # SemanticMessageHistory requires a sync Redis connection
        # Use redis_url to let it create its own connection
        if self._config.redis_url:
            history_kwargs["redis_url"] = self._config.redis_url
        elif self._config.connection_args:
            history_kwargs["connection_kwargs"] = self._config.connection_args

        if self._config.session_tag is not None:
            history_kwargs["session_tag"] = self._config.session_tag

        if self._config.vectorizer is not None:
            history_kwargs["vectorizer"] = self._config.vectorizer

        # Note: SemanticMessageHistory doesn't support TTL directly
        # TTL configuration in config is stored but not used by this implementation

        self._history = SemanticMessageHistory(**history_kwargs)

    def _extract_query(self, messages: List[Union[dict[str, Any], Any]]) -> str:
        """Extract the query to use for context retrieval.

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

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Wrap a model call with conversation memory.

        This method is part of the LangChain AgentMiddleware protocol.
        Retrieves relevant past messages based on the current query,
        injects them into the context, and stores the new exchange.

        Args:
            request: The model request containing messages.
            handler: The async function to call the model.

        Returns:
            The model response.

        Raises:
            Exception: If graceful_degradation is False and history operations fail.
        """
        await self._ensure_initialized_async()

        # Support both dict-style and LangChain ModelRequest types
        if isinstance(request, dict):
            messages = request.get("messages", [])
        else:
            messages = getattr(request, "messages", [])
        query = self._extract_query(messages)

        # Try to retrieve relevant context
        context_messages: List[Dict[str, Any]] = []
        if query:
            try:
                context_messages = self._history.get_relevant(
                    prompt=query,
                    top_k=self._config.top_k,
                )
            except Exception as e:
                if not self._graceful_degradation:
                    raise
                logger.warning(f"Failed to retrieve context: {e}")

        # Inject context into messages if found
        if context_messages:
            # Build a single system message with the retrieved context.
            # Packaging context into one SystemMessage (rather than injecting
            # separate HumanMessage/AIMessage objects) avoids confusing the LLM
            # about which messages belong to the current turn vs. history.
            context_lines = []
            for msg in context_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "human"):
                    context_lines.append(f"User: {content}")
                elif role in ("llm", "ai", "assistant"):
                    context_lines.append(f"Assistant: {content}")
                else:
                    context_lines.append(f"{role.title()}: {content}")
            context_block = "\n".join(context_lines)
            context_note = SystemMessage(
                content=(
                    "The following is relevant context from earlier in this "
                    "conversation. Use it to inform your response and maintain "
                    "continuity:\n\n"
                    f"{context_block}"
                )
            )
            enhanced_messages = [context_note] + list(messages)
            # Support both dict-style and LangChain ModelRequest types
            if isinstance(request, dict):
                request = {**request, "messages": enhanced_messages}
            else:
                request = request.override(messages=enhanced_messages)

        # Call the model
        response = await handler(request)

        # Store the new exchange
        try:
            # Get the user message
            user_content = query
            # Get the assistant response (support ModelResponse, dict, and
            # other LangChain types).
            # Note: content may be a list of blocks (Responses API) or a
            # plain string (Chat Completions). We normalize to string for
            # SemanticMessageHistory which requires string content.
            if hasattr(response, "result") and isinstance(response.result, list):
                # ModelResponse: result is list[BaseMessage]
                if response.result:
                    raw_content = getattr(response.result[-1], "content", "")
                    assistant_content = _content_to_str(raw_content)
                else:
                    assistant_content = ""
            elif isinstance(response, dict):
                assistant_content = _content_to_str(response.get("content", ""))
            else:
                assistant_content = _content_to_str(getattr(response, "content", ""))

            if user_content:
                self._history.add_messages(
                    [
                        {"role": "user", "content": user_content},
                    ]
                )
            if assistant_content:
                self._history.add_messages(
                    [
                        {"role": "llm", "content": assistant_content},
                    ]
                )
        except Exception as e:
            if not self._graceful_degradation:
                raise
            logger.warning(f"Failed to store messages: {e}")

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[Union[LangChainToolMessage, Command]]
        ],
    ) -> Union[LangChainToolMessage, Command]:
        """Pass through tool calls without memory operations.

        This method is part of the LangChain AgentMiddleware protocol.
        Conversation memory only applies to model calls, not tool calls.

        Args:
            request: The tool call request.
            handler: The async function to execute the tool.

        Returns:
            The tool result from the handler.
        """
        return await handler(request)
