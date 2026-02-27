"""Integration tests for ConversationMemoryMiddleware with real Redis.

These tests verify the ACTUAL behavior of the middleware using real Redis
and real SemanticMessageHistory — no mocks allowed.
"""

import time
import uuid

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from redisvl.extensions.message_history import SemanticMessageHistory
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    ConversationMemoryConfig,
    ConversationMemoryMiddleware,
)

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

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


@requires_sentence_transformers
class TestSemanticMessageHistoryDirect:
    """Test redisvl SemanticMessageHistory directly to establish baseline."""

    def test_store_and_retrieve_messages(self, redis_url: str) -> None:
        """Verify that SemanticMessageHistory can store and retrieve messages."""
        name = f"direct_test_{uuid.uuid4().hex[:8]}"
        history = SemanticMessageHistory(
            name=name,
            session_tag="test_session",
            redis_url=redis_url,
            distance_threshold=0.9,
        )
        try:
            # Store messages
            history.add_messages(
                [
                    {
                        "role": "user",
                        "content": "My name is Alice and I'm a software engineer.",
                    },
                ]
            )
            history.add_messages(
                [
                    {"role": "llm", "content": "Nice to meet you Alice!"},
                ]
            )

            # Small delay for index to update
            time.sleep(0.5)

            # Retrieve - asking about name should find the introduction
            results = history.get_relevant(
                prompt="What is my name?",
                top_k=5,
            )
            assert len(results) > 0, (
                f"Expected to find stored messages but got empty results. "
                f"All messages: {history.messages}"
            )

            # Check that we got back relevant content
            contents = [r.get("content", "") for r in results]
            found_alice = any("Alice" in c for c in contents)
            assert (
                found_alice
            ), f"Expected to find 'Alice' in retrieved messages. Got: {contents}"
        finally:
            history.delete()

    def test_store_method(self, redis_url: str) -> None:
        """Test the convenience store() method."""
        name = f"store_test_{uuid.uuid4().hex[:8]}"
        history = SemanticMessageHistory(
            name=name,
            session_tag="test_session",
            redis_url=redis_url,
            distance_threshold=0.9,
        )
        try:
            history.store(
                prompt="I love Python programming",
                response="Python is a great language!",
            )
            time.sleep(0.5)

            results = history.get_relevant(
                prompt="Tell me about Python",
                top_k=5,
            )
            assert len(results) > 0, "store() should have persisted messages"
        finally:
            history.delete()


@requires_sentence_transformers
class TestConversationMemoryMiddlewareIntegration:
    """Integration tests for ConversationMemoryMiddleware with real Redis."""

    @pytest.mark.asyncio
    async def test_messages_are_stored_after_model_call(self, redis_url: str) -> None:
        """Verify that the middleware actually stores messages in Redis."""
        name = f"store_test_{uuid.uuid4().hex[:8]}"
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name=name,
            session_tag="store_test_session",
            top_k=5,
            distance_threshold=0.9,
            graceful_degradation=False,  # Surface all errors
        )

        async with ConversationMemoryMiddleware(config) as middleware:

            async def mock_llm(request: dict) -> ModelResponse:
                return ModelResponse(
                    result=[AIMessage(content="Nice to meet you Alice!")]
                )

            request = {
                "messages": [
                    HumanMessage(
                        content="Hi, my name is Alice and I'm a software engineer."
                    )
                ]
            }
            await middleware.awrap_model_call(request, mock_llm)

            # Give Redis index time to update
            time.sleep(0.5)

            # Directly check the underlying history
            all_messages = middleware._history.messages
            assert len(all_messages) >= 2, (
                f"Expected at least 2 stored messages (user + llm) "
                f"but got {len(all_messages)}: {all_messages}"
            )

            # Check roles
            roles = [m.get("role") for m in all_messages]
            assert "user" in roles, f"Expected 'user' role in {roles}"
            assert "llm" in roles, f"Expected 'llm' role in {roles}"

    @pytest.mark.asyncio
    async def test_context_injected_on_second_turn(self, redis_url: str) -> None:
        """Verify that stored messages are retrieved and injected into subsequent calls."""
        name = f"inject_test_{uuid.uuid4().hex[:8]}"
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name=name,
            session_tag="inject_test_session",
            top_k=5,
            distance_threshold=0.9,  # Very permissive for testing
            graceful_degradation=False,
        )

        async with ConversationMemoryMiddleware(config) as middleware:
            seen_messages_per_turn: list[list] = []

            async def mock_llm(request: dict) -> ModelResponse:
                msgs = request.get("messages", [])
                seen_messages_per_turn.append(list(msgs))
                return ModelResponse(result=[AIMessage(content="Got it, thanks!")])

            # Turn 1: introduce topic
            request1 = {
                "messages": [
                    HumanMessage(
                        content="I love Python programming and machine learning"
                    )
                ]
            }
            await middleware.awrap_model_call(request1, mock_llm)

            # Give index time to update
            time.sleep(0.5)

            # Turn 2: ask about the same topic (semantically similar)
            request2 = {
                "messages": [
                    HumanMessage(
                        content="Tell me more about Python and machine learning"
                    )
                ]
            }
            await middleware.awrap_model_call(request2, mock_llm)

            # Turn 1 should have just the user message (no prior context)
            assert (
                len(seen_messages_per_turn[0]) == 1
            ), f"Turn 1 should have 1 message, got {len(seen_messages_per_turn[0])}"

            # Turn 2 should have context injected: [SystemMessage, HumanMessage]
            turn2_messages = seen_messages_per_turn[1]
            assert len(turn2_messages) > 1, (
                f"Turn 2 should have injected context from turn 1 but got "
                f"only {len(turn2_messages)} message(s): {turn2_messages}"
            )
            # First message should be a SystemMessage with context
            assert isinstance(
                turn2_messages[0], SystemMessage
            ), f"Expected SystemMessage with context, got {type(turn2_messages[0])}"
            assert "earlier in this conversation" in turn2_messages[0].content, (
                f"Context should reference earlier conversation, "
                f"got: {turn2_messages[0].content[:200]}"
            )

    @pytest.mark.asyncio
    async def test_notebook_scenario_diverse_prompts(self, redis_url: str) -> None:
        """Reproduce the exact notebook scenario with diverse prompts.

        This is the scenario that was failing: the user introduces themselves,
        talks about interests, then asks questions that should recall earlier context.
        Uses distance_threshold=0.7 (the new default) to match the notebook config.
        """
        name = f"notebook_test_{uuid.uuid4().hex[:8]}"
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name=name,
            session_tag="alice_session",
            top_k=5,
            distance_threshold=0.7,  # New default - must work for diverse prompts
            graceful_degradation=False,
        )

        async with ConversationMemoryMiddleware(config) as middleware:
            seen_messages_per_turn: list[list] = []

            async def mock_llm(request: dict) -> ModelResponse:
                msgs = request.get("messages", [])
                seen_messages_per_turn.append(list(msgs))
                return ModelResponse(
                    result=[AIMessage(content="Response from assistant")]
                )

            # Turn 1: Introduction
            await middleware.awrap_model_call(
                {
                    "messages": [
                        HumanMessage(
                            content="Hi! My name is Alice and I'm a software engineer."
                        )
                    ]
                },
                mock_llm,
            )
            time.sleep(0.5)

            # Turn 2: Share interests
            await middleware.awrap_model_call(
                {
                    "messages": [
                        HumanMessage(
                            content="I'm really interested in machine learning and I work with Python."
                        )
                    ]
                },
                mock_llm,
            )
            time.sleep(0.5)

            # Turn 3: Ask about recommendations (should recall ML/Python context)
            await middleware.awrap_model_call(
                {
                    "messages": [
                        HumanMessage(
                            content="What Python libraries would be most useful for me?"
                        )
                    ]
                },
                mock_llm,
            )
            time.sleep(0.5)

            # Turn 4: Ask about identity (should recall name/job)
            await middleware.awrap_model_call(
                {
                    "messages": [
                        HumanMessage(
                            content="What's my name and what do I do for work?"
                        )
                    ]
                },
                mock_llm,
            )

            # Verify all messages were stored
            all_messages = middleware._history.messages
            assert len(all_messages) >= 8, (
                f"Expected at least 8 stored messages (4 user + 4 llm) "
                f"but got {len(all_messages)}: {all_messages}"
            )

            # Turn 3 should have context injected as a SystemMessage
            turn3_messages = seen_messages_per_turn[2]
            assert len(turn3_messages) > 1, (
                f"Turn 3 should have injected context but only got "
                f"{len(turn3_messages)} message(s)"
            )
            # First message should be the context SystemMessage
            assert isinstance(
                turn3_messages[0], SystemMessage
            ), f"Expected SystemMessage with context, got {type(turn3_messages[0])}"
            context_content = turn3_messages[0].content
            assert "earlier in this conversation" in context_content, (
                f"Context SystemMessage should reference earlier conversation, "
                f"got: {context_content[:200]}"
            )

            # Turn 4 should have context from earlier turns
            turn4_messages = seen_messages_per_turn[3]
            assert len(turn4_messages) > 1, (
                f"Turn 4 should have injected context but only got "
                f"{len(turn4_messages)} message(s)"
            )
            assert isinstance(
                turn4_messages[0], SystemMessage
            ), f"Expected SystemMessage with context, got {type(turn4_messages[0])}"
            # Turn 4 asks about name — context should contain "Alice"
            turn4_context = turn4_messages[0].content
            assert (
                "Alice" in turn4_context
            ), f"Turn 4 context should contain 'Alice' but got: {turn4_context[:300]}"

    @pytest.mark.asyncio
    async def test_with_langchain_message_objects(self, redis_url: str) -> None:
        """Test that the middleware works when messages are LangChain objects
        (as they would be when coming from create_agent/ModelRequest)."""
        name = f"langchain_msg_test_{uuid.uuid4().hex[:8]}"
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name=name,
            session_tag="langchain_test",
            top_k=5,
            distance_threshold=0.9,
            graceful_degradation=False,
        )

        async with ConversationMemoryMiddleware(config) as middleware:
            seen_messages_per_turn: list[list] = []

            async def mock_llm(request: dict) -> ModelResponse:
                msgs = request.get("messages", [])
                seen_messages_per_turn.append(list(msgs))
                return ModelResponse(
                    result=[AIMessage(content="Hello! How can I help?")]
                )

            # Use HumanMessage objects (not dicts) - this is what create_agent does
            await middleware.awrap_model_call(
                {
                    "messages": [
                        HumanMessage(content="I am a data scientist who loves R")
                    ]
                },
                mock_llm,
            )
            time.sleep(0.5)

            await middleware.awrap_model_call(
                {"messages": [HumanMessage(content="What kind of scientist am I?")]},
                mock_llm,
            )

            # Second turn should have context from first
            assert len(seen_messages_per_turn[1]) > 1, (
                f"Expected context injection but got only "
                f"{len(seen_messages_per_turn[1])} message(s)"
            )
