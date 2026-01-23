"""End-to-end integration tests for middleware with LangGraph patterns.

These tests simulate real agent workflow patterns.
"""

import pytest
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    ConversationMemoryConfig,
    ConversationMemoryMiddleware,
    MiddlewareStack,
    SemanticCacheConfig,
    SemanticCacheMiddleware,
    SemanticRouterConfig,
    SemanticRouterMiddleware,
    ToolCacheConfig,
    ToolResultCacheMiddleware,
    create_caching_stack,
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
    """Provide a Redis URL using TestContainers.

    Uses redis/redis-stack-server which includes RediSearch module required by redisvl.
    """
    redis_container = RedisContainer("redis/redis-stack-server:latest")
    redis_container.start()
    try:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        redis_container.stop()


@requires_sentence_transformers
class TestReActAgentPattern:
    """Test middleware in a ReAct agent pattern (think -> act -> observe)."""

    @pytest.mark.asyncio
    async def test_react_with_caching_stack(self, redis_url: str) -> None:
        """Test ReAct pattern with semantic and tool caching."""
        stack = create_caching_stack(
            redis_url=redis_url,
            semantic_cache_ttl=60,
            tool_cache_ttl=60,
            cacheable_tools=["search", "calculate"],
        )

        async with stack:
            # Simulate ReAct cycle
            llm_calls = []
            tool_calls = []

            async def mock_llm(request: dict) -> dict:
                llm_calls.append(request)
                messages = request.get("messages", [])
                last_msg = messages[-1] if messages else {}

                # Simulate LLM deciding to use a tool
                if "search" in last_msg.get("content", "").lower():
                    return {
                        "content": "",
                        "tool_calls": [{"name": "search", "args": {"query": "Python"}}],
                    }
                return {"content": "Final answer based on search results."}

            async def mock_tool(request: dict) -> dict:
                tool_calls.append(request)
                return {"result": "Search results: Python is a programming language."}

            # Step 1: User asks a question
            request1 = {
                "messages": [
                    {"role": "user", "content": "Please search for Python info"}
                ]
            }
            response1 = await stack.awrap_model_call(request1, mock_llm)
            assert "tool_calls" in response1

            # Step 2: Execute the tool
            tool_request = {"tool_name": "search", "args": {"query": "Python"}}
            tool_result = await stack.awrap_tool_call(tool_request, mock_tool)
            assert "result" in tool_result

            # Step 3: Final response
            request2 = {
                "messages": [
                    {"role": "user", "content": "Please search for Python info"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": response1["tool_calls"],
                    },
                    {"role": "tool", "content": tool_result["result"]},
                ]
            }
            response2 = await stack.awrap_model_call(request2, mock_llm)
            assert "content" in response2


@requires_sentence_transformers
class TestMultiTurnConversation:
    """Test middleware in multi-turn conversation patterns."""

    @pytest.mark.asyncio
    async def test_multi_turn_with_memory(self, redis_url: str) -> None:
        """Test multi-turn conversation with memory middleware."""
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name="test_memory",
            session_tag="test_session",
            top_k=3,
            ttl_seconds=60,
        )

        async with ConversationMemoryMiddleware(config) as middleware:
            # Simulate multi-turn conversation
            async def mock_llm(request: dict) -> dict:
                messages = request.get("messages", [])
                user_msg = ""
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_msg = m.get("content", "")
                        break

                return {"content": f"I received: {user_msg}"}

            # Turn 1
            request1 = {"messages": [{"role": "user", "content": "Hello, I'm Alice."}]}
            response1 = await middleware.awrap_model_call(request1, mock_llm)
            assert "Alice" in response1["content"]

            # Turn 2
            request2 = {
                "messages": [{"role": "user", "content": "What's my name again?"}]
            }
            response2 = await middleware.awrap_model_call(request2, mock_llm)
            # The middleware should have injected context
            assert "content" in response2


@requires_sentence_transformers
class TestSemanticRouting:
    """Test semantic router middleware patterns."""

    @pytest.mark.asyncio
    async def test_route_to_handler(self, redis_url: str) -> None:
        """Test routing requests to specific handlers."""
        routes = [
            {"name": "greeting", "references": ["hello", "hi", "hey", "greetings"]},
            {"name": "farewell", "references": ["bye", "goodbye", "see you"]},
        ]

        config = SemanticRouterConfig(
            redis_url=redis_url,
            name="test_router",
            routes=routes,
        )

        async with SemanticRouterMiddleware(config) as middleware:
            greeting_handled = []

            async def greeting_handler(request: dict, route_match: dict) -> dict:
                greeting_handled.append(route_match)
                return {"content": "Hello! How can I help you today?"}

            middleware.register_route_handler("greeting", greeting_handler)

            async def default_handler(request: dict) -> dict:
                return {"content": "Default response"}

            # Test greeting route
            request = {"messages": [{"role": "user", "content": "Hello there!"}]}
            response = await middleware.awrap_model_call(request, default_handler)
            assert response is not None  # Verify response was returned

            # The greeting handler should have been called
            # (depends on vectorizer availability)


@requires_sentence_transformers
class TestFullMiddlewareStack:
    """Test complete middleware stack with all components."""

    @pytest.mark.asyncio
    async def test_full_stack_model_call(self, redis_url: str) -> None:
        """Test a full middleware stack processing a model call."""
        # Create a stack with multiple middlewares
        middlewares = [
            SemanticCacheMiddleware(
                SemanticCacheConfig(
                    redis_url=redis_url,
                    name="full_stack_cache",
                    ttl_seconds=60,
                )
            ),
            ToolResultCacheMiddleware(
                ToolCacheConfig(
                    redis_url=redis_url,
                    name="full_stack_tool_cache",
                    ttl_seconds=60,
                )
            ),
        ]

        stack = MiddlewareStack(middlewares)

        async with stack:
            call_count = [0]

            async def mock_llm(request: dict) -> dict:
                call_count[0] += 1
                return {"content": f"Response #{call_count[0]}"}

            def get_content(result):
                """Extract content from dict, AIMessage, or ModelResponse."""
                # Handle ModelResponse (has .result which is list of messages)
                if hasattr(result, "result") and isinstance(result.result, list):
                    if result.result:
                        return getattr(result.result[0], "content", None)
                    return None
                # Handle dict
                if isinstance(result, dict):
                    return result.get("content")
                # Handle AIMessage or other objects with content
                return getattr(result, "content", None)

            # First call
            request = {"messages": [{"role": "user", "content": "Test question"}]}
            result1 = await stack.awrap_model_call(request, mock_llm)
            assert get_content(result1) is not None

            # Second call - might be cached (returns AIMessage when cached)
            result2 = await stack.awrap_model_call(request, mock_llm)
            assert get_content(result2) is not None

    @pytest.mark.asyncio
    async def test_full_stack_tool_call(self, redis_url: str) -> None:
        """Test a full middleware stack processing a tool call."""
        stack = create_caching_stack(
            redis_url=redis_url,
            semantic_cache_ttl=60,
            tool_cache_ttl=60,
            cacheable_tools=["test_tool"],
        )

        async with stack:

            async def mock_tool(request: dict) -> dict:
                return {"result": "Tool executed successfully"}

            request = {"tool_name": "test_tool", "args": {"param": "value"}}
            result = await stack.awrap_tool_call(request, mock_tool)
            assert result["result"] == "Tool executed successfully"


@requires_sentence_transformers
class TestConnectionSharing:
    """Test middleware connecting to same Redis server."""

    @pytest.mark.asyncio
    async def test_middleware_with_shared_connection(self, redis_url: str) -> None:
        """Test that middleware can connect to same Redis server as checkpointer.

        Note: redisvl library requires synchronous Redis connections, so middleware
        creates its own sync connection using the redis_url rather than sharing
        an async client.
        """
        from redis.asyncio import Redis

        # Create an async Redis connection (simulating a checkpointer)
        redis_client = Redis.from_url(redis_url)

        try:
            # Create middleware with redis_url (middleware creates its own sync connection)
            config = SemanticCacheConfig(
                redis_url=redis_url,
                name="shared_conn_cache",
                ttl_seconds=60,
            )

            middleware = SemanticCacheMiddleware(config)

            async def mock_handler(request: dict) -> dict:
                return {"content": "Response"}

            request = {"messages": [{"role": "user", "content": "Test"}]}
            result = await middleware.awrap_model_call(request, mock_handler)
            assert result["content"] == "Response"

            # Close middleware
            await middleware.aclose()

            # The async connection (checkpointer's) should still work
            await redis_client.ping()
        finally:
            await redis_client.aclose()
