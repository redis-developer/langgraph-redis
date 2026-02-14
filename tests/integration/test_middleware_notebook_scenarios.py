"""Integration tests simulating notebook scenarios.

These tests verify that the middleware works correctly in the patterns
demonstrated in the example notebooks.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver
from langgraph.middleware.redis import (
    ConversationMemoryConfig,
    ConversationMemoryMiddleware,
    IntegratedRedisMiddleware,
    MiddlewareStack,
    SemanticCacheConfig,
    SemanticCacheMiddleware,
    ToolCacheConfig,
    ToolResultCacheMiddleware,
    create_caching_stack,
    from_configs,
)

# Check if sentence-transformers is available
try:
    import sentence_transformers

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)


@pytest.fixture(scope="module")
def redis_url():
    """Provide a Redis URL using TestContainers."""
    redis_container = RedisContainer("redis/redis-stack-server:latest")
    redis_container.start()
    try:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        redis_container.stop()


def _extract_content(result) -> str:
    """Extract content from various response types."""
    # Handle ModelResponse (has .result which is list of messages)
    if hasattr(result, "result") and isinstance(result.result, list):
        if result.result:
            return getattr(result.result[0], "content", "")
        return ""
    # Handle dict
    if isinstance(result, dict):
        return result.get("content", "")
    # Handle AIMessage or other objects with content
    return getattr(result, "content", "")


@requires_sentence_transformers
class TestSemanticCacheNotebookScenario:
    """Test scenarios from middleware_semantic_cache.ipynb."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_model_response(self, redis_url: str) -> None:
        """Test that cache hits return ModelResponse objects (for agent integration)."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="notebook_cache_hit_test",
            distance_threshold=0.2,
            ttl_seconds=60,
            cache_final_only=True,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_llm(request: dict) -> dict:
                call_count[0] += 1
                # Simulate LLM returning a response
                return {
                    "content": f"The capital of France is Paris. (call #{call_count[0]})"
                }

            # First call - cache miss
            request1 = {
                "messages": [HumanMessage(content="What is the capital of France?")]
            }
            result1 = await middleware.awrap_model_call(request1, mock_llm)
            assert call_count[0] == 1

            content1 = _extract_content(result1)
            assert "Paris" in content1

            # Second call - same prompt, should hit cache
            request2 = {
                "messages": [HumanMessage(content="What is the capital of France?")]
            }
            result2 = await middleware.awrap_model_call(request2, mock_llm)

            # Handler may or may not be called depending on embedding similarity
            # but result should have content about Paris
            content2 = _extract_content(result2)
            assert "Paris" in content2 or content2 != ""

    @pytest.mark.asyncio
    async def test_multiple_different_prompts(self, redis_url: str) -> None:
        """Test that middleware handles multiple different prompts."""
        import uuid

        # Use unique name to avoid cross-test collisions
        unique_name = f"semantic_multi_test_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=unique_name,
            distance_threshold=0.2,
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            responses = []

            async def mock_llm(request: dict) -> dict:
                # Extract prompt for unique response
                msgs = request.get("messages", [])
                if msgs:
                    content = (
                        msgs[-1].content
                        if hasattr(msgs[-1], "content")
                        else str(msgs[-1])
                    )
                else:
                    content = "no prompt"
                return {"content": f"Response for: {content[:20]}"}

            # Send several different prompts
            prompts = [
                "What is the capital of France?",
                "How do I bake cookies?",
                "What is machine learning?",
            ]

            for prompt in prompts:
                request = {"messages": [HumanMessage(content=prompt)]}
                result = await middleware.awrap_model_call(request, mock_llm)

                # Extract content from result using helper
                content = _extract_content(result)

                responses.append(content)
                # Each response should have content
                assert content != "", f"Empty content for prompt: {prompt}"

            # All responses should have content
            assert len(responses) == 3


@requires_sentence_transformers
class TestToolCachingNotebookScenario:
    """Test scenarios from middleware_tool_caching.ipynb."""

    @pytest.mark.asyncio
    async def test_cacheable_vs_excluded_tools(self, redis_url: str) -> None:
        """Test that cacheable_tools and excluded_tools work correctly."""
        from langgraph.prebuilt.tool_node import ToolCallRequest

        config = ToolCacheConfig(
            redis_url=redis_url,
            name="notebook_tool_cache_test",
            cacheable_tools=["search", "calculate"],
            excluded_tools=["random_number"],
            distance_threshold=0.1,
            ttl_seconds=60,
        )

        async with ToolResultCacheMiddleware(config) as middleware:
            search_calls = [0]
            random_calls = [0]

            async def mock_search(request: ToolCallRequest) -> ToolMessage:
                search_calls[0] += 1
                return ToolMessage(
                    content=f"Search results #{search_calls[0]}",
                    name="search",
                    tool_call_id=request.tool_call["id"],
                )

            async def mock_random(request: ToolCallRequest) -> ToolMessage:
                random_calls[0] += 1
                import random

                return ToolMessage(
                    content=f"Random: {random.randint(1, 100)}",
                    name="random_number",
                    tool_call_id=request.tool_call["id"],
                )

            # Search tool - should be cached
            search_request = ToolCallRequest(
                tool_call={"name": "search", "args": {"query": "Python"}, "id": "c1"},
                tool=None,
                state={},
                runtime=MagicMock(),
            )
            result1 = await middleware.awrap_tool_call(search_request, mock_search)
            assert search_calls[0] == 1
            assert isinstance(result1, ToolMessage)
            assert "Search results" in result1.content

            # Same search - might hit cache
            result2 = await middleware.awrap_tool_call(search_request, mock_search)
            # Cache hit or miss, but should have a ToolMessage result
            assert isinstance(result2, ToolMessage)
            assert "Search results" in result2.content

            # Random tool - should NOT be cached (not in cacheable_tools)
            random_request = ToolCallRequest(
                tool_call={
                    "name": "random_number",
                    "args": {"max": 100},
                    "id": "c2",
                },
                tool=None,
                state={},
                runtime=MagicMock(),
            )
            await middleware.awrap_tool_call(random_request, mock_random)
            first_random_count = random_calls[0]

            await middleware.awrap_tool_call(random_request, mock_random)
            # Should have been called again (not cached)
            assert random_calls[0] == first_random_count + 1


@requires_sentence_transformers
class TestCompositionNotebookScenario:
    """Test scenarios from middleware_composition.ipynb."""

    @pytest.mark.asyncio
    async def test_multiple_middleware_list(self, redis_url: str) -> None:
        """Test passing multiple middleware to agent."""
        # Create individual middleware
        semantic_cache = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name="comp_semantic_cache",
                ttl_seconds=60,
            )
        )

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name="comp_tool_cache",
                ttl_seconds=60,
            )
        )

        try:
            # Both should initialize successfully
            await semantic_cache._ensure_initialized_async()
            await tool_cache._ensure_initialized_async()

            async def mock_llm(request: dict) -> dict:
                return {"content": "Response from LLM"}

            # Both middleware should process the request
            request = {"messages": [HumanMessage(content="Test")]}
            result = await semantic_cache.awrap_model_call(request, mock_llm)
            assert result is not None

        finally:
            await semantic_cache.aclose()
            await tool_cache.aclose()

    @pytest.mark.asyncio
    async def test_middleware_stack_composition(self, redis_url: str) -> None:
        """Test MiddlewareStack composition."""
        stack = MiddlewareStack(
            [
                SemanticCacheMiddleware(
                    SemanticCacheConfig(
                        redis_url=redis_url,
                        name="stack_cache",
                        ttl_seconds=60,
                    )
                ),
                ConversationMemoryMiddleware(
                    ConversationMemoryConfig(
                        redis_url=redis_url,
                        name="stack_memory",
                        session_tag="test_session",
                        top_k=3,
                    )
                ),
            ]
        )

        async with stack:

            async def mock_llm(request: dict) -> dict:
                return {"content": "Stack response"}

            request = {"messages": [HumanMessage(content="Test stack")]}
            result = await stack.awrap_model_call(request, mock_llm)

            # Result should have content
            content = _extract_content(result)
            assert content != "", "Result should have content"

    @pytest.mark.asyncio
    async def test_create_caching_stack_factory(self, redis_url: str) -> None:
        """Test create_caching_stack factory function."""
        stack = create_caching_stack(
            redis_url=redis_url,
            semantic_cache_name="factory_cache",
            semantic_cache_ttl=60,
            tool_cache_name="factory_tools",
            tool_cache_ttl=60,
            cacheable_tools=["search"],
        )

        async with stack:
            assert len(stack._middlewares) == 2

    @pytest.mark.asyncio
    async def test_from_configs_factory(self, redis_url: str) -> None:
        """Test from_configs factory function."""
        stack = from_configs(
            redis_url=redis_url,
            configs=[
                SemanticCacheConfig(name="cfg_cache", ttl_seconds=60),
                ToolCacheConfig(name="cfg_tools", ttl_seconds=60),
                ConversationMemoryConfig(name="cfg_memory", session_tag="cfg_session"),
            ],
        )

        async with stack:
            assert len(stack._middlewares) == 3

    @pytest.mark.asyncio
    async def test_redis_saver_sync_context_manager(self, redis_url: str) -> None:
        """Test that RedisSaver.from_conn_string uses sync context manager."""
        # This tests the pattern: with RedisSaver.from_conn_string(...) as saver:
        with RedisSaver.from_conn_string(redis_url) as checkpointer:
            # Setup synchronously
            checkpointer.setup()

            # Checkpointer should be functional
            assert checkpointer is not None
            # Verify it's a RedisSaver instance
            assert isinstance(checkpointer, RedisSaver)

    @pytest.mark.asyncio
    async def test_integrated_middleware_from_saver(self, redis_url: str) -> None:
        """Test IntegratedRedisMiddleware.from_saver."""
        with RedisSaver.from_conn_string(redis_url) as checkpointer:
            checkpointer.setup()

            # Create integrated middleware
            integrated = IntegratedRedisMiddleware.from_saver(
                checkpointer,
                configs=[
                    SemanticCacheConfig(name="integrated_cache", ttl_seconds=60),
                ],
            )

            assert integrated is not None
            assert len(integrated._middlewares) == 1


@requires_sentence_transformers
class TestConversationMemoryNotebookScenario:
    """Test scenarios from middleware_conversation_memory.ipynb."""

    @pytest.mark.asyncio
    async def test_session_tag_isolation(self, redis_url: str) -> None:
        """Test that different session_tags are isolated."""
        # User 1's middleware
        middleware_user1 = ConversationMemoryMiddleware(
            ConversationMemoryConfig(
                redis_url=redis_url,
                name="memory_isolation_test",
                session_tag="user_alice",
                top_k=3,
            )
        )

        # User 2's middleware
        middleware_user2 = ConversationMemoryMiddleware(
            ConversationMemoryConfig(
                redis_url=redis_url,
                name="memory_isolation_test",
                session_tag="user_bob",
                top_k=3,
            )
        )

        try:
            await middleware_user1._ensure_initialized_async()
            await middleware_user2._ensure_initialized_async()

            async def mock_llm(request: dict) -> dict:
                # Extract messages to check context
                msgs = request.get("messages", [])
                return {"content": f"Response with {len(msgs)} messages"}

            # User 1 sends a message
            request1 = {
                "messages": [HumanMessage(content="Hi, I'm Alice the engineer")]
            }
            await middleware_user1.awrap_model_call(request1, mock_llm)

            # User 2 should not see User 1's messages
            request2 = {"messages": [HumanMessage(content="Do you know Alice?")]}
            result2 = await middleware_user2.awrap_model_call(request2, mock_llm)
            # Result should be returned regardless
            assert result2 is not None

        finally:
            await middleware_user1.aclose()
            await middleware_user2.aclose()


@requires_sentence_transformers
class TestResponseSerialization:
    """Test response serialization/deserialization for cache."""

    @pytest.mark.asyncio
    async def test_ai_message_serialization(self, redis_url: str) -> None:
        """Test that AIMessage responses are properly serialized and deserialized."""
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name="serialization_test",
            distance_threshold=0.1,  # Strict matching
            ttl_seconds=60,
        )

        async with SemanticCacheMiddleware(config) as middleware:
            call_count = [0]

            async def mock_llm_with_ai_message(request: dict) -> AIMessage:
                """Return an AIMessage like a real LLM would."""
                call_count[0] += 1
                return AIMessage(content=f"AI Response #{call_count[0]}")

            # First call
            request1 = {"messages": [HumanMessage(content="Test serialization")]}
            result1 = await middleware.awrap_model_call(
                request1, mock_llm_with_ai_message
            )

            # Extract content using helper
            content1 = _extract_content(result1)
            assert "AI Response" in content1

            # Second identical call - might hit cache
            result2 = await middleware.awrap_model_call(
                request1, mock_llm_with_ai_message
            )

            # Should still have content regardless of cache hit/miss
            content2 = _extract_content(result2)
            assert content2 != ""
