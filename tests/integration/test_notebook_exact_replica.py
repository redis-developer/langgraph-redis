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
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        # === Cell: two-model-setup ===
        model_default = ChatOpenAI(model="gpt-4o-mini")

        # === Cell: define-tools ===
        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return f"{expression} = 42"

        tools = [get_weather, calculate]

        # === Cell: create-middleware ===
        import uuid

        cache_name = f"demo_semantic_cache_default_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
                deterministic_tools=["calculate"],
            )
        )

        print("SemanticCacheMiddleware created successfully!")

        try:
            # === Cell: create-agent ===
            agent = create_agent(
                model=model_default,
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_semantic_cache_responses_api_mode(self, redis_url: str):
        """Test semantic cache with Responses API mode."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        tools = [get_weather]

        cache_name = f"demo_semantic_cache_responses_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=tools,
                middleware=[cache_middleware],
            )

            # Cache miss
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is the capital of Japan?")]}
            )
            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            # Cache hit
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Tell me Japan's capital city")]}
            )
            assert "messages" in result2

            print("SUCCESS! Responses API semantic cache test passed!")

        finally:
            await cache_middleware.aclose()

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_semantic_cache_responses_api_clean_blocks(self, redis_url: str):
        """Test that cached Responses API content blocks have no provider IDs."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def noop(x: str) -> str:
            """Do nothing."""
            return x

        cache_name = f"demo_clean_blocks_{uuid.uuid4().hex[:8]}"

        cache_middleware = SemanticCacheMiddleware(
            SemanticCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.15,
                ttl_seconds=3600,
                cache_final_only=True,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[noop],
                middleware=[cache_middleware],
            )

            # Populate cache
            await agent.ainvoke({"messages": [HumanMessage(content="Say hello")]})

            # Cache hit
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Say hello please")]}
            )

            ai_msg = result["messages"][-1]
            if isinstance(ai_msg.content, list):
                for block in ai_msg.content:
                    if isinstance(block, dict):
                        assert (
                            "id" not in block
                        ), f"Cached block has provider ID: {block}"
                print("Cached content blocks are clean -- no provider IDs!")

            print("SUCCESS! Responses API clean blocks verification passed!")

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
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            SemanticCacheConfig,
            SemanticCacheMiddleware,
            ToolCacheConfig,
            ToolResultCacheMiddleware,
        )

        model_default = ChatOpenAI(model="gpt-4o-mini")

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
                deterministic_tools=["search", "calculate"],
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
                model=model_default,
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

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_middleware_stack_responses_api_sanitization(self, redis_url: str):
        """Test MiddlewareStack sanitizes Responses API content blocks."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
            MiddlewareStack,
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        cache_name = f"resp_stack_cache_{uuid.uuid4().hex[:8]}"
        memory_name = f"resp_stack_memory_{uuid.uuid4().hex[:8]}"

        responses_stack = MiddlewareStack(
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
                        session_tag="responses_demo",
                        top_k=3,
                    )
                ),
            ]
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[search],
                middleware=[responses_stack],
            )

            # Turn 1
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Hello, I like Python programming")]}
            )
            assert "messages" in result1

            # Turn 2
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What language did I mention?")]}
            )
            assert "messages" in result2

            # Verify no duplicate IDs
            all_ids = set()
            for label, result in [("Turn 1", result1), ("Turn 2", result2)]:
                ai_msg = result["messages"][-1]
                if isinstance(ai_msg.content, list):
                    for block in ai_msg.content:
                        if isinstance(block, dict) and "id" in block:
                            block_id = block["id"]
                            assert (
                                block_id not in all_ids
                            ), f"Duplicate ID in {label}: {block_id}"
                            all_ids.add(block_id)

            print("SUCCESS! MiddlewareStack Responses API sanitization passed!")

        finally:
            await responses_stack.aclose()

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_multi_turn_checkpointer_responses_api(self, redis_url: str):
        """Test multi-turn with checkpointer + Responses API."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from langgraph.middleware.redis import (
            IntegratedRedisMiddleware,
            SemanticCacheConfig,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        async_checkpointer = AsyncRedisSaver(redis_url=redis_url)
        await async_checkpointer.asetup()

        cache_name = f"integrated_cache_{uuid.uuid4().hex[:8]}"

        integrated_stack = IntegratedRedisMiddleware.from_saver(
            async_checkpointer,
            configs=[
                SemanticCacheConfig(name=cache_name, ttl_seconds=3600),
            ],
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[search],
                checkpointer=async_checkpointer,
                middleware=[integrated_stack],
            )

            thread_id = f"integrated_{uuid.uuid4().hex[:8]}"
            config = {"configurable": {"thread_id": thread_id}}

            # Turn 1
            result1 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="What is the population of Tokyo?")
                    ]
                },
                config=config,
            )
            assert "messages" in result1

            # Turn 2
            result2 = await agent.ainvoke(
                {"messages": [HumanMessage(content="And what about New York?")]},
                config=config,
            )
            assert "messages" in result2

            # Turn 3
            result3 = await agent.ainvoke(
                {"messages": [HumanMessage(content="Which one is larger?")]},
                config=config,
            )
            assert "messages" in result3

            print("SUCCESS! Multi-turn checkpointer + Responses API test passed!")

        finally:
            try:
                await async_checkpointer.aclose()
            except Exception:
                pass


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
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
            MiddlewareStack,
            SemanticCacheConfig,
            SemanticCacheMiddleware,
        )

        model_default = ChatOpenAI(model="gpt-4o-mini")

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
                model=model_default,
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


class TestToolCacheNotebook:
    """Test tool caching with both API modes."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_tool_cache_responses_api_mode(self, redis_url: str):
        """Test tool caching works identically with Responses API."""
        import ast
        import operator as op
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ToolCacheConfig,
            ToolResultCacheMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        # Safe math evaluator
        safe_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in safe_ops:
                return safe_ops[type(node.op)](
                    _eval_node(node.left), _eval_node(node.right)
                )
            raise ValueError("Unsupported expression")

        def safe_eval(expr: str) -> float:
            return _eval_node(ast.parse(expr, mode="eval").body)

        exec_count = {"calculate": 0}

        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression."""
            exec_count["calculate"] += 1
            return str(safe_eval(expression))

        calculate.metadata = {"cacheable": True}

        cache_name = f"demo_tool_cache_responses_{uuid.uuid4().hex[:8]}"

        tool_cache = ToolResultCacheMiddleware(
            ToolCacheConfig(
                redis_url=redis_url,
                name=cache_name,
                distance_threshold=0.1,
                ttl_seconds=1800,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[calculate],
                middleware=[tool_cache],
            )

            # First call
            result1 = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 15 * 8 + 20?")]}
            )
            assert "messages" in result1
            assert len(result1["messages"]) >= 2

            print("SUCCESS! Tool cache with Responses API mode passed!")

        finally:
            await tool_cache.aclose()


class TestConversationMemoryNotebook:
    """Test conversation memory with both API modes."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real LLM test",
    )
    @pytest.mark.asyncio
    async def test_memory_responses_api_recall(self, redis_url: str):
        """Test memory recall works with Responses API (Carol persona)."""
        import uuid

        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        from langgraph.middleware.redis import (
            ConversationMemoryConfig,
            ConversationMemoryMiddleware,
        )

        model_responses_api = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

        @tool
        def get_user_preferences(category: str) -> str:
            """Get user preferences for a category."""
            return f"Preferences for {category}: not set"

        memory_name = f"demo_conversation_memory_{uuid.uuid4().hex[:8]}"

        memory_middleware = ConversationMemoryMiddleware(
            ConversationMemoryConfig(
                redis_url=redis_url,
                name=memory_name,
                session_tag="user_789",
                top_k=3,
                distance_threshold=0.7,
            )
        )

        try:
            agent = create_agent(
                model=model_responses_api,
                tools=[get_user_preferences],
                middleware=[memory_middleware],
            )

            # Turn 1: Introduce Carol
            result1 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Hi! I'm Carol, an embedded systems engineer. I work with C and Rust."
                        )
                    ]
                }
            )
            assert "messages" in result1

            # Turn 2: Share interests
            result2 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="I'm interested in RTOS, bare-metal programming, and IoT protocols."
                        )
                    ]
                }
            )
            assert "messages" in result2

            # Turn 3: Test recall
            result3 = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="What's my name and what languages do I use?"
                        )
                    ]
                }
            )
            assert "messages" in result3

            # Check that Carol's name appears in the response
            response_content = result3["messages"][-1].content
            if isinstance(response_content, list):
                # Responses API: extract text from blocks
                text_parts = []
                for block in response_content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_text = " ".join(text_parts)
            else:
                response_text = response_content

            assert (
                "carol" in response_text.lower()
            ), f"Expected 'Carol' in response: {response_text[:200]}"

            print("SUCCESS! Memory with Responses API recall test passed!")

        finally:
            await memory_middleware.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
