"""Tests for LangChain embeddings integration with redisvl.

Issue 1: No support for custom vectorization model from langchain.embeddings

Problem:
SemanticCache from redisvl requires a BaseVectorizer subclass, but users want
to use LangChain embeddings (OpenAI, Azure OpenAI, etc.) which have a different
interface (embed_documents, aembed_documents).

Solution:
Create a LangChainVectorizer that:
1. Extends redisvl's BaseVectorizer (Pydantic model)
2. Wraps any LangChain Embeddings object
3. Works seamlessly with SemanticCache and SemanticCacheMiddleware

These tests verify the integration works end-to-end.
"""

import json
import os
import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize.base import BaseVectorizer
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    SemanticCacheConfig,
    SemanticCacheMiddleware,
)


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


class MockLangChainEmbeddings(Embeddings):
    """Mock LangChain embeddings for testing.

    Returns deterministic 8-dimensional vectors for testing.
    """

    def __init__(self, dims: int = 8):
        self._dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self._embed_single(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._embed_single(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query."""
        return self.embed_query(text)

    def _embed_single(self, text: str) -> List[float]:
        """Create a deterministic embedding based on text hash."""
        # Simple deterministic embedding for testing
        h = hash(text) % 1000000
        return [(h >> i & 0xFF) / 255.0 for i in range(0, self._dims * 8, 8)]


class TestLangChainVectorizerCreation:
    """Test creating a vectorizer from LangChain embeddings."""

    def test_create_vectorizer_from_langchain_embeddings(self):
        """Test that we can create a BaseVectorizer from LangChain embeddings."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        # Must be a BaseVectorizer subclass for redisvl compatibility
        assert isinstance(vectorizer, BaseVectorizer)
        assert vectorizer.dims == 8

    def test_vectorizer_embed_method(self):
        """Test the embed method works correctly."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        # Test single text embedding
        result = vectorizer.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) == 8  # Embedding dimensions
        assert all(isinstance(x, float) for x in result)

    def test_vectorizer_embed_many_method(self):
        """Test the embed_many method works correctly."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        # Test batch embedding
        texts = ["Hello", "World", "Test"]
        result = vectorizer.embed_many(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 8

    @pytest.mark.asyncio
    async def test_vectorizer_async_embed(self):
        """Test async embedding methods."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        # Test async single embedding
        result = await vectorizer.aembed("Hello async")
        assert isinstance(result, list)
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_vectorizer_async_embed_many(self):
        """Test async batch embedding methods."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        # Test async batch embedding
        texts = ["Async", "Batch", "Test"]
        result = await vectorizer.aembed_many(texts)
        assert isinstance(result, list)
        assert len(result) == 3


class TestLangChainVectorizerWithSemanticCache:
    """Test LangChainVectorizer works with redisvl's SemanticCache."""

    def test_semantic_cache_accepts_langchain_vectorizer(self, redis_url: str):
        """Test that SemanticCache accepts our LangChainVectorizer."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        cache_name = f"lc_cache_{uuid.uuid4().hex[:8]}"

        # This should NOT raise TypeError about invalid vectorizer
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.2,
        )

        assert cache is not None

    def test_semantic_cache_store_and_check(self, redis_url: str):
        """Test storing and retrieving from SemanticCache with LangChain vectorizer."""
        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        cache_name = f"lc_store_{uuid.uuid4().hex[:8]}"
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.2,
        )

        # Store a response
        cache.store(
            prompt="What is Python?", response="Python is a programming language."
        )

        # Check - should find the stored response
        result = cache.check(prompt="What is Python?")
        assert result is not None
        assert len(result) > 0
        assert result[0].get("response") == "Python is a programming language."


class TestLangChainVectorizerWithMiddleware:
    """Test LangChainVectorizer works with SemanticCacheMiddleware."""

    @pytest.mark.asyncio
    async def test_middleware_accepts_langchain_vectorizer(self, redis_url: str):
        """Test that SemanticCacheMiddleware works with LangChainVectorizer."""
        from langchain.agents.middleware.types import ModelResponse

        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        cache_name = f"lc_middleware_{uuid.uuid4().hex[:8]}"
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.2,
            ttl_seconds=60,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def mock_handler(request):
            call_count[0] += 1
            return ModelResponse(
                result=[AIMessage(content=f"Response #{call_count[0]}")],
                structured_response=None,
            )

        async with SemanticCacheMiddleware(config) as middleware:
            # First call - cache miss
            request1 = {"messages": [HumanMessage(content="Tell me about Redis")]}
            result1 = await middleware.awrap_model_call(request1, mock_handler)
            assert call_count[0] == 1

            # Second call - should hit cache
            request2 = {"messages": [HumanMessage(content="Tell me about Redis")]}
            result2 = await middleware.awrap_model_call(request2, mock_handler)
            assert call_count[0] == 1, "Handler should not be called on cache hit"

            # Verify cached response is marked
            assert result2.result[0].additional_kwargs.get("cached") is True

    @pytest.mark.asyncio
    async def test_middleware_cache_hit_with_langchain_embeddings(self, redis_url: str):
        """Test full cache hit scenario with LangChain vectorizer."""
        from langchain.agents.middleware.types import ModelResponse

        from langgraph.middleware.redis.vectorizer import LangChainVectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = LangChainVectorizer(
            embeddings=mock_embeddings,
            dims=8,
        )

        cache_name = f"lc_hit_{uuid.uuid4().hex[:8]}"

        # Pre-populate cache using redisvl directly
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.2,
        )
        cache.store(
            prompt="What is the weather?",
            response=json.dumps(
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "I don't have access to weather data.",
                        "type": "ai",
                        "tool_calls": [],
                    },
                }
            ),
        )

        # Now use middleware with same cache
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.2,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called on cache hit")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="What is the weather?")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            # Verify we got the cached response
            assert "weather" in result.result[0].content.lower()


class TestHelperFunction:
    """Test the create_langchain_vectorizer helper function."""

    def test_create_langchain_vectorizer_helper(self):
        """Test helper function creates proper vectorizer."""
        from langgraph.middleware.redis.vectorizer import create_langchain_vectorizer

        mock_embeddings = MockLangChainEmbeddings(dims=8)
        vectorizer = create_langchain_vectorizer(mock_embeddings, dims=8)

        assert isinstance(vectorizer, BaseVectorizer)
        assert vectorizer.dims == 8

    def test_create_langchain_vectorizer_infers_dims(self):
        """Test helper can infer dimensions from embeddings object."""
        from langgraph.middleware.redis.vectorizer import create_langchain_vectorizer

        # Create embeddings with a dimensions attribute
        mock_embeddings = MockLangChainEmbeddings(dims=8)
        mock_embeddings.dimensions = 8  # Add attribute for inference

        vectorizer = create_langchain_vectorizer(mock_embeddings)
        assert vectorizer.dims == 8


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable",
)
class TestOpenAIEmbeddingsIntegration:
    """Integration tests with real OpenAI embeddings (requires API key)."""

    @pytest.mark.asyncio
    async def test_openai_embeddings_with_middleware(self, redis_url: str):
        """Test real OpenAI embeddings work with the middleware."""
        from langchain.agents.middleware.types import ModelResponse
        from langchain_openai import OpenAIEmbeddings

        from langgraph.middleware.redis.vectorizer import create_langchain_vectorizer

        # Create OpenAI embeddings
        openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorizer = create_langchain_vectorizer(openai_embeddings, dims=1536)

        cache_name = f"openai_test_{uuid.uuid4().hex[:8]}"
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.2,
            ttl_seconds=60,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def mock_handler(request):
            call_count[0] += 1
            return ModelResponse(
                result=[AIMessage(content=f"Response #{call_count[0]}")],
                structured_response=None,
            )

        async with SemanticCacheMiddleware(config) as middleware:
            # First call - cache miss
            request = {"messages": [HumanMessage(content="Explain machine learning")]}
            await middleware.awrap_model_call(request, mock_handler)
            assert call_count[0] == 1

            # Second call - cache hit
            await middleware.awrap_model_call(request, mock_handler)
            assert call_count[0] == 1, "OpenAI embeddings should enable cache hit"
