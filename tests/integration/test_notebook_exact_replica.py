"""Exact replicas of notebook code as integration tests.

These tests replicate the EXACT code from the notebooks to ensure
they work correctly before deploying to the notebook environment.
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


class TestSemanticCacheNotebook:
    """Exact replica of middleware_semantic_cache.ipynb"""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_semantic_cache_notebook_exact(self, redis_url: str):
        """EXACT replica of middleware_semantic_cache.ipynb cells."""
        import time

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        # === Cell: define-tools ===
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Search results for: {query}"

        tools = [get_weather, search]

        # === Cell: create-middleware ===
        import uuid

        cache_name = f"demo_semantic_cache_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
            )
        )

        print("SemanticCacheMiddleware created successfully!")

        try:
            # === Cell: create-agent ===
            agent = create_agent(
                model="gpt-4o-mini",
                tools=tools,
                middleware=[cache_middleware],
            )

            print("Agent created with SemanticCacheMiddleware!")

            # === Cell: first-query ===
            print("Query 1: 'What is the capital of France?'")
            print("=" * 50)

            start = time.time()
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of France?")]}
            )
            elapsed1 = time.time() - start

            print(f"Response: {result1['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed1:.2f}s (cache miss - LLM call)")

            assert "messages" in result1
            assert len(result1["messages"]) >= 2  # Human + AI

            # === Cell: second-query ===
            print("\nQuery 2: 'Tell me France's capital city'")
            print("=" * 50)

            start = time.time()
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Tell me France's capital city")]}
            )
            elapsed2 = time.time() - start

            print(f"Response: {result2['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed2:.2f}s (expected: cache hit - much faster!)")

            assert "messages" in result2
            # Cache hit should be faster (but not guaranteed, so no assertion)

            # === Cell: third-query ===
            print("\nQuery 3: 'What is the capital of Germany?'")
            print("=" * 50)

            start = time.time()
            result3 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of Germany?")]}
            )
            elapsed3 = time.time() - start

            print(f"Response: {result3['messages'][-1].content[:200]}...")
            print(f"Time: {elapsed3:.2f}s (cache miss - different topic)")

            assert "messages" in result3

            print("\n" + "=" * 50)
            print("SUCCESS! middleware_semantic_cache.ipynb replica passed!")

        finally:
            await cache_middleware.aclose()


class TestCompositionNotebook:
    """Exact replica of middleware_composition.ipynb"""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_composition_notebook_multiple_middleware(self, redis_url: str):
        """EXACT replica of middleware_composition.ipynb cells."""
        import ast
        import operator as op
        import time
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
            ToolCacheConfig,
            ToolResultCacheMiddleware,
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

        # Track tool calls
        tool_calls = []

        @tool
        def search(query: str) -> str:
            """Search for information."""
            tool_calls.append(("search", query))
            return f"Results for: {query}"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            tool_calls.append(("calculate", expression))
            return str(safe_eval(expression))

        tools = [search, calculate]

        # === Cell: create-individual-middleware ===
        llm_cache_name = f"composition_llm_cache_{uuid.uuid4().hex[:8]}"
        tool_cache_name = f"composition_tool_cache_{uuid.uuid4().hex[:8]}"

        semantic_cache = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=llm_cache_name,
                ttl_seconds=3600,
            )
        )

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name=tool_cache_name,
                cacheable_tools=["search", "calculate"],
                ttl_seconds=1800,
            )
        )

        print("Created individual middleware:")
        print("- SemanticCacheMiddleware for LLM responses")
        print("- ToolResultCacheMiddleware for tool results")

        try:
            # === Cell: create-agent-multiple ===
            agent = create_agent(
                model="gpt-4o-mini",
                tools=tools,
                middleware=[semantic_cache, tool_cache],
            )

            print("Agent created with both SemanticCache and ToolCache middleware!")

            # === Cell: test-multiple ===
            print("\nTest 1: Search query")
            print("=" * 50)

            start = time.time()
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Search for Python tutorials")]}
            )
            elapsed1 = time.time() - start

            print(f"Response: {result1['messages'][-1].content[:100]}...")
            print(f"Tool calls: {tool_calls}")
            print(f"Time: {elapsed1:.2f}s")

            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            print("\n" + "=" * 50)
            print("SUCCESS! middleware_composition.ipynb replica passed!")

        finally:
            await semantic_cache.aclose()
            await tool_cache.aclose()


class TestMiddlewareStack:
    """Test MiddlewareStack composition."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_middleware_stack_with_create_agent(self, redis_url: str):
        """Test MiddlewareStack with create_agent."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
            MiddlewareStack,
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        cache_name = f"stack_cache_{uuid.uuid4().hex[:8]}"
        memory_name = f"stack_memory_{uuid.uuid4().hex[:8]}"

        # Create middleware stack
        stack = MiddlewareStack(
            [
                SemanticCacheMiddleware(
                    SemanticCacheConfig(
                        redis_url=redis_url,
                        name=cache_name,
                        ttl_seconds=3600,
                    )
                ),
                ConversationMemoryMiddleware(
                    ConversationMemoryConfig(
                        redis_url=redis_url,
                        name=memory_name,
                        session_tag="test_session",
                        top_k=3,
                    )
                ),
            ]
        )

        try:
            # Create agent with stack
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[search],
                middleware=[stack],
            )

            print("Agent created with MiddlewareStack!")

            result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="Hi, I'm testing the middleware stack!")
                    ]
                }
            )

            print(f"Response: {result['messages'][-1].content}")
            assert "messages" in result

            print("SUCCESS! MiddlewareStack with create_agent works!")

        finally:
            await stack.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
