"""Tests for streaming message ID fix.

Issue: Cached responses not appearing in frontend

Problem:
When SemanticCacheMiddleware returns a cached response, it reuses the same
message ID from the cached AIMessage. This causes the frontend to deduplicate
messages, so cached responses don't appear.

Solution:
Generate a NEW UUID for each cached response message. This ensures:
1. The frontend sees it as a NEW message (not a duplicate)
2. The checkpoint captures a unique message
3. stream.messages shows the cached response

These tests verify that:
1. Each cache hit returns a message with a unique ID
2. Multiple cache hits return different IDs
3. Cached messages are properly marked
"""

import json
import uuid

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    SemanticCacheConfig,
    SemanticCacheMiddleware,
)

pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for the test module."""
    container = RedisContainer("redis/redis-stack-server:latest")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def redis_url(redis_container):
    """Get Redis URL from container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


class TestCachedMessageIds:
    """Test that cached responses get unique message IDs for streaming."""

    @pytest.mark.asyncio
    async def test_cache_hit_has_message_id(self, redis_url: str):
        """Test that a cache hit returns a message with a non-None ID."""
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        cache_name = f"id_test_{uuid.uuid4().hex[:8]}"

        # Pre-populate cache
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
                    "tool_calls": [],
                },
            }
        )
        cache.store(prompt=test_prompt, response=cached_response)

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            ttl_seconds=60,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called on cache hit")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="What is Python?")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            # Verify result structure
            assert isinstance(result, ModelResponse)
            assert len(result.result) == 1
            assert isinstance(result.result[0], AIMessage)

            # CRITICAL: Message must have an ID for streaming to work
            msg_id = result.result[0].id
            assert msg_id is not None, "Cached message must have an ID for streaming"
            print(f"Cache hit message ID: {msg_id}")

    @pytest.mark.asyncio
    async def test_repeated_cache_hits_have_different_ids(self, redis_url: str):
        """Test that each cache hit returns a message with a DIFFERENT ID.

        This is THE key test for the streaming fix. Without unique IDs,
        the frontend deduplicates messages and doesn't show cached responses.
        """
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        cache_name = f"multi_id_test_{uuid.uuid4().hex[:8]}"

        # Pre-populate cache
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )
        cache.store(
            prompt="Hello",
            response=json.dumps(
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "Hi there!",
                        "type": "ai",
                        "tool_calls": [],
                    },
                }
            ),
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called on cache hit")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="Hello")]}

            # Make three identical requests
            result1 = await middleware.awrap_model_call(request, should_not_be_called)
            result2 = await middleware.awrap_model_call(request, should_not_be_called)
            result3 = await middleware.awrap_model_call(request, should_not_be_called)

            # Extract message IDs
            id1 = result1.result[0].id
            id2 = result2.result[0].id
            id3 = result3.result[0].id

            print(f"Cache hit IDs: {id1}, {id2}, {id3}")

            # CRITICAL: All IDs must be DIFFERENT for streaming to work
            assert id1 != id2, f"IDs 1 and 2 must differ: {id1} == {id2}"
            assert id2 != id3, f"IDs 2 and 3 must differ: {id2} == {id3}"
            assert id1 != id3, f"IDs 1 and 3 must differ: {id1} == {id3}"

            # Verify all are valid UUIDs
            for msg_id in [id1, id2, id3]:
                try:
                    uuid.UUID(msg_id)
                except (ValueError, TypeError):
                    pytest.fail(f"Message ID should be valid UUID: {msg_id}")

    @pytest.mark.asyncio
    async def test_cache_hit_marked_as_cached(self, redis_url: str):
        """Test that cached responses are marked with additional_kwargs."""
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        cache_name = f"marked_test_{uuid.uuid4().hex[:8]}"

        # Pre-populate cache
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )
        cache.store(
            prompt="Test prompt",
            response=json.dumps({"content": "Cached response"}),
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="Test prompt")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            # Check that response is marked as cached
            msg = result.result[0]
            cached_marker = msg.additional_kwargs.get("cached")
            assert (
                cached_marker is True
            ), "Cached response should be marked with cached=True"


class TestStreamingScenario:
    """Test realistic streaming scenarios with cache."""

    @pytest.mark.asyncio
    async def test_conversation_with_repeated_questions(self, redis_url: str):
        """Simulate a conversation where the same question is asked multiple times.

        In a real frontend, each response should appear separately.
        This requires unique message IDs for each cache hit.
        """
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        cache_name = f"convo_test_{uuid.uuid4().hex[:8]}"

        # Pre-populate with a greeting response
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )
        cache.store(
            prompt="Hello, how are you?",
            response=json.dumps(
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "Hello! I'm doing great, thanks for asking!",
                        "type": "ai",
                        "tool_calls": [],
                    },
                }
            ),
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            ttl_seconds=60,
            vectorizer=vectorizer,
        )

        # Track all messages as they would appear in checkpoint state
        conversation_messages = []

        async def should_not_be_called(request):
            raise AssertionError("Should use cache")

        async with SemanticCacheMiddleware(config) as middleware:
            # Simulate 3 users asking the same greeting
            for user_num in range(3):
                user_msg = HumanMessage(content="Hello, how are you?")
                conversation_messages.append(user_msg)

                request = {"messages": conversation_messages.copy()}
                result = await middleware.awrap_model_call(
                    request, should_not_be_called
                )

                ai_msg = result.result[0]
                conversation_messages.append(ai_msg)

                print(f"Turn {user_num + 1}: AI message ID = {ai_msg.id}")

            # Extract all AI message IDs
            ai_messages = [
                msg for msg in conversation_messages if isinstance(msg, AIMessage)
            ]
            ai_ids = [msg.id for msg in ai_messages]

            # All IDs must be unique for proper streaming
            assert len(set(ai_ids)) == len(
                ai_ids
            ), f"All AI message IDs must be unique: {ai_ids}"

            print(f"\nAll {len(ai_ids)} cached responses have unique IDs!")
            print("Frontend will correctly show each response separately.")


class TestCacheMissStillWorks:
    """Verify cache miss behavior is unchanged."""

    @pytest.mark.asyncio
    async def test_cache_miss_calls_handler(self, redis_url: str):
        """Test that cache miss still calls the handler properly."""
        vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        cache_name = f"miss_test_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            ttl_seconds=60,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def counting_handler(request):
            call_count[0] += 1
            return ModelResponse(
                result=[AIMessage(content=f"Fresh response #{call_count[0]}")],
                structured_response=None,
            )

        async with SemanticCacheMiddleware(config) as middleware:
            # First call - cache miss
            request = {"messages": [HumanMessage(content="What is Redis?")]}
            result1 = await middleware.awrap_model_call(request, counting_handler)

            assert call_count[0] == 1
            assert "Fresh response #1" in result1.result[0].content

            # Second call - should be cache hit (handler not called)
            result2 = await middleware.awrap_model_call(request, counting_handler)
            assert call_count[0] == 1, "Handler should not be called on cache hit"

            # But the IDs should be different
            assert result1.result[0].id != result2.result[0].id
