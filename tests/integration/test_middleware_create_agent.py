"""Integration tests for middleware with langchain.agents.create_agent.

These tests verify that our middleware works correctly with the actual
LangChain agent creation API, not just in isolation.
"""

import os

import pytest
from testcontainers.redis import RedisContainer

# Skip if sentence_transformers not available
pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for the test module."""
    container = RedisContainer("redis:8")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def redis_url(redis_container):
    """Get Redis URL from container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


class TestCreateAgentWithMiddleware:
    """Test middleware integration with langchain.agents.create_agent."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_semantic_cache_with_real_agent(self, redis_url: str):
        """Test SemanticCacheMiddleware with a real LangChain agent."""
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        # Create middleware with unique cache name
        import uuid

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
            )
        )

        try:
            # Create agent with middleware
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[get_weather],
                middleware=[middleware],
            )

            # First call - should be cache miss
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What's the weather in Paris?")]}
            )

            assert "messages" in result1
            assert len(result1["messages"]) > 0

            # Second call with same query - should be cache hit
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What's the weather in Paris?")]}
            )

            assert "messages" in result2
            assert len(result2["messages"]) > 0

            print(f"First result: {result1['messages'][-1].content[:100]}...")
            print(f"Second result: {result2['messages'][-1].content[:100]}...")

        finally:
            await middleware.aclose()

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_middleware_without_llm_call(self, redis_url: str):
        """Test that middleware correctly wraps model calls.

        This test pre-populates the cache so no LLM call is needed.
        Note: Still requires OPENAI_API_KEY for create_agent initialization.
        """
        import ast
        import json
        import operator as op
        import uuid

        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.tools import tool
        from redis.asyncio import Redis
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import HFTextVectorizer

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        # Safe math evaluator
        safe_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](
                    _eval_node(node.left), _eval_node(node.right)
                )
            elif isinstance(node, ast.UnaryOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](_eval_node(node.operand))
            raise ValueError("Unsupported expression")

        def safe_eval(expr: str) -> float:
            return _eval_node(ast.parse(expr, mode="eval").body)

        @tool
        def calculator(expression: str) -> str:
            """Calculate a math expression."""
            return str(safe_eval(expression))

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        # Use the SAME vectorizer for both pre-population and middleware
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        # Pre-populate cache with a response
        test_prompt = "What is 2 + 2?"
        # Serialize as our middleware does
        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "The answer is 4.",
                    "type": "ai",
                    "tool_calls": [],
                },
            }
        )
        cache.store(prompt=test_prompt, response=cached_response)

        # Now create middleware with the SAME vectorizer
        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
                vectorizer=vectorizer,  # Use same vectorizer!
            )
        )

        try:
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[calculator],
                middleware=[middleware],
            )

            # This should hit the cache and NOT call the LLM
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 2 + 2?")]}
            )

            # Verify we got a response
            assert "messages" in result
            last_message = result["messages"][-1]
            print(f"Result type: {type(last_message)}")
            print(f"Result: {last_message}")

            # The response should contain our cached answer
            assert "4" in last_message.content

        finally:
            await middleware.aclose()

    @pytest.mark.asyncio
    async def test_cache_miss_handler_return(self, redis_url: str):
        """Test what the handler returns on cache miss.

        This verifies the handler return type that our middleware must handle.
        """
        import uuid

        from langchain.agents.middleware.types import ModelRequest, ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
            )
        )

        await middleware._ensure_initialized_async()

        # Track what handler receives and returns
        handler_calls = []

        async def tracking_handler(request):
            """Handler that returns a ModelResponse like the real agent does."""
            handler_calls.append(request)
            # Return what the real agent's model node returns
            return ModelResponse(
                result=[AIMessage(content="Handler response")],
                structured_response=None,
            )

        # Create a simple request
        request = {
            "messages": [HumanMessage(content="Test question")],
        }

        try:
            result = await middleware.awrap_model_call(request, tracking_handler)

            print(f"Handler was called: {len(handler_calls)} times")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")

            # On cache miss, handler should be called
            assert len(handler_calls) == 1

            # Result should be a ModelResponse (what handler returned)
            assert isinstance(result, ModelResponse)
            assert len(result.result) == 1
            assert isinstance(result.result[0], AIMessage)
            assert result.result[0].content == "Handler response"

        finally:
            await middleware.aclose()

    @pytest.mark.asyncio
    async def test_cache_hit_return_type(self, redis_url: str):
        """Test that cache hit returns ModelResponse, not AIMessage."""
        import json
        import uuid

        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import HFTextVectorizer

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        # Use the SAME vectorizer for both pre-population and middleware
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        # Pre-populate with cached response
        test_prompt = "What is the capital of France?"
        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Paris is the capital of France.",
                    "type": "ai",
                    "tool_calls": [],
                },
            }
        )
        cache.store(prompt=test_prompt, response=cached_response)

        # Create middleware with the SAME vectorizer
        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
                vectorizer=vectorizer,  # Use same vectorizer!
            )
        )

        await middleware._ensure_initialized_async()

        handler_calls = []

        async def tracking_handler(request):
            handler_calls.append(request)
            return ModelResponse(
                result=[AIMessage(content="Should not be called")],
                structured_response=None,
            )

        request = {
            "messages": [HumanMessage(content="What is the capital of France?")],
        }

        try:
            result = await middleware.awrap_model_call(request, tracking_handler)

            print(f"Handler was called: {len(handler_calls)} times")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")

            # Cache hit - handler should NOT be called
            assert len(handler_calls) == 0, "Handler should not be called on cache hit"

            # CRITICAL: Result MUST be ModelResponse for agent routing to work
            assert isinstance(
                result, ModelResponse
            ), f"Expected ModelResponse, got {type(result)}"
            assert len(result.result) == 1
            assert isinstance(result.result[0], AIMessage)
            assert "Paris" in result.result[0].content

        finally:
            await middleware.aclose()

    @pytest.mark.asyncio
    async def test_multiple_middleware_cache_miss(self, redis_url: str):
        """Test multiple middleware together - simulating notebook scenario.

        This is the exact pattern used in middleware_composition.ipynb
        """
        import uuid

        from langchain.agents.middleware.types import ModelRequest, ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
            ToolCacheConfig,
            ToolResultCacheMiddleware,
        )

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"
        tool_cache_name = f"test_tool_cache_{uuid.uuid4().hex[:8]}"

        # Create both middleware like notebook does
        semantic_cache = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
            )
        )

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name=tool_cache_name,
                cacheable_tools=["search"],
                ttl_seconds=60,
            )
        )

        await semantic_cache._ensure_initialized_async()
        await tool_cache._ensure_initialized_async()

        # Track calls
        handler_calls = []

        async def tracking_handler(request):
            """Handler that returns a ModelResponse like the real agent does."""
            handler_calls.append(request)
            return ModelResponse(
                result=[AIMessage(content="Handler response from model")],
                structured_response=None,
            )

        # Create request like agent does
        request = {
            "messages": [HumanMessage(content="Search for Python tutorials")],
        }

        try:
            # Chain middleware like create_agent does
            async def chained_handler(req):
                # Tool cache wraps the handler
                return await tool_cache.awrap_model_call(req, tracking_handler)

            # Semantic cache wraps tool cache
            result = await semantic_cache.awrap_model_call(request, chained_handler)

            print(f"Handler was called: {len(handler_calls)} times")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")

            # Handler should be called (cache miss)
            assert len(handler_calls) == 1

            # Result should be ModelResponse
            assert isinstance(
                result, ModelResponse
            ), f"Expected ModelResponse, got {type(result)}"
            assert len(result.result) == 1
            assert isinstance(result.result[0], AIMessage)

        finally:
            await semantic_cache.aclose()
            await tool_cache.aclose()

    @pytest.mark.asyncio
    async def test_multiple_middleware_with_tool_calls(self, redis_url: str):
        """Test middleware with a response that HAS tool_calls.

        The routing issue happens when the model wants to call a tool.
        """
        import uuid

        from langchain.agents.middleware.types import ModelRequest, ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                cache_final_only=True,  # Should NOT cache tool call responses
            )
        )

        await middleware._ensure_initialized_async()

        handler_calls = []

        async def tracking_handler(request):
            """Handler that returns a response WITH tool_calls."""
            handler_calls.append(request)
            # This simulates a model response that wants to call a tool
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_123",
                                "name": "search",
                                "args": {"query": "Python tutorials"},
                            }
                        ],
                    )
                ],
                structured_response=None,
            )

        request = {
            "messages": [HumanMessage(content="Search for Python tutorials")],
        }

        try:
            result = await middleware.awrap_model_call(request, tracking_handler)

            print(f"Handler was called: {len(handler_calls)} times")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
            print(f"Tool calls: {result.result[0].tool_calls}")

            # Handler should be called
            assert len(handler_calls) == 1

            # Result should be ModelResponse with tool_calls
            assert isinstance(result, ModelResponse)
            assert len(result.result) == 1
            assert isinstance(result.result[0], AIMessage)
            assert len(result.result[0].tool_calls) == 1
            assert result.result[0].tool_calls[0]["name"] == "search"

        finally:
            await middleware.aclose()

    @pytest.mark.asyncio
    async def test_create_agent_with_middleware_cache_hit(self, redis_url: str):
        """Test create_agent with pre-populated cache - NO LLM call needed.

        This is THE test that proves the notebook scenario works.
        """
        import json
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.tools import tool
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import HFTextVectorizer

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        # Use same vectorizer for pre-population and middleware
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

        # Pre-populate cache with a FINAL response (no tool_calls)
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        test_prompt = "What is Python?"
        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Python is a programming language.",
                    "type": "ai",
                    "tool_calls": [],  # No tool calls = final response
                },
            }
        )
        cache.store(prompt=test_prompt, response=cached_response)

        # Create middleware with same vectorizer
        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                distance_threshold=0.1,
                vectorizer=vectorizer,
            )
        )

        try:
            # Set fake API key since we expect cache hit - no LLM call
            original_key = os.environ.get("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = "fake-key-cache-hit-test"

            try:
                # Create agent WITH middleware
                agent = create_agent(
                    model="gpt-4o-mini",
                    tools=[search],
                    middleware=[middleware],
                )

                print("Agent created, invoking...")

                # This should HIT cache and return without calling LLM
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content="What is Python?")]}
                )

                print(f"Result: {result}")
                print(f"Last message: {result['messages'][-1]}")

                # Verify we got the cached response
                assert "messages" in result
                last_message = result["messages"][-1]
                assert "Python" in last_message.content

                print("SUCCESS! create_agent with middleware and cache hit works!")

            finally:
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)

        finally:
            await middleware.aclose()

    @pytest.mark.asyncio
    async def test_tool_calling_flow_skips_cache_on_tool_results(self, redis_url: str):
        """Test that cache is skipped when request contains tool results.

        This is the bug that caused KeyError: 'model' in notebooks.
        """
        import uuid

        from langchain.agents.middleware.types import ModelResponse
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        cache_name = f"test_cache_{uuid.uuid4().hex[:8]}"

        middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                ttl_seconds=60,
                cache_final_only=True,
            )
        )

        await middleware._ensure_initialized_async()

        call_count = 0

        async def multi_turn_handler(request):
            """Simulate multi-turn: first call returns tool_calls, second returns final."""
            nonlocal call_count
            call_count += 1

            messages = (
                request.get("messages", [])
                if isinstance(request, dict)
                else getattr(request, "messages", [])
            )

            # Check if there's a tool result in messages
            has_tool_result = any(
                isinstance(m, ToolMessage) or (hasattr(m, "type") and m.type == "tool")
                for m in messages
            )

            if has_tool_result:
                # Second call - return final response
                return ModelResponse(
                    result=[AIMessage(content="The weather in Tokyo is sunny, 75°F")],
                    structured_response=None,
                )
            else:
                # First call - return tool call request
                return ModelResponse(
                    result=[
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_abc123",
                                    "name": "get_weather",
                                    "args": {"location": "Tokyo"},
                                }
                            ],
                        )
                    ],
                    structured_response=None,
                )

        try:
            # First model call - should return tool_calls
            request1 = {
                "messages": [HumanMessage(content="What's the weather in Tokyo?")]
            }
            result1 = await middleware.awrap_model_call(request1, multi_turn_handler)

            assert isinstance(result1, ModelResponse)
            assert len(result1.result[0].tool_calls) == 1
            assert call_count == 1

            # Second model call with tool results - MUST call handler, NOT return cached
            request2 = {
                "messages": [
                    HumanMessage(content="What's the weather in Tokyo?"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_abc123",
                                "name": "get_weather",
                                "args": {"location": "Tokyo"},
                            }
                        ],
                    ),
                    ToolMessage(
                        content="Tokyo: Sunny, 75°F", tool_call_id="call_abc123"
                    ),
                ],
            }
            result2 = await middleware.awrap_model_call(request2, multi_turn_handler)

            # CRITICAL: Handler MUST be called for tool result processing
            assert call_count == 2, "Handler should be called when tool results present"
            assert isinstance(result2, ModelResponse)
            assert result2.result[0].content == "The weather in Tokyo is sunny, 75°F"
            assert not result2.result[0].tool_calls

        finally:
            await middleware.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
