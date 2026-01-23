"""Redis middleware package for LangGraph agent workflows.

This package provides Redis/RedisVL-native middleware for LangGraph agents:
- SemanticCacheMiddleware: Cache LLM responses by semantic similarity
- ToolResultCacheMiddleware: Cache tool call results
- SemanticRouterMiddleware: Route requests based on semantic matching
- ConversationMemoryMiddleware: Inject relevant past messages into context
"""

from langgraph.middleware.redis.aio import AsyncRedisMiddleware
from langgraph.middleware.redis.base import BaseRedisMiddleware
from langgraph.middleware.redis.composition import (
    IntegratedRedisMiddleware,
    MiddlewareStack,
    create_caching_stack,
    from_configs,
)
from langgraph.middleware.redis.conversation_memory import ConversationMemoryMiddleware
from langgraph.middleware.redis.semantic_cache import SemanticCacheMiddleware
from langgraph.middleware.redis.semantic_router import SemanticRouterMiddleware
from langgraph.middleware.redis.tool_cache import ToolResultCacheMiddleware
from langgraph.middleware.redis.types import (
    ConversationMemoryConfig,
    MiddlewareConfig,
    SemanticCacheConfig,
    SemanticRouterConfig,
    ToolCacheConfig,
)
from langgraph.middleware.redis.vectorizer import (
    AsyncEmbeddingVectorizer,
    create_vectorizer_from_langchain,
)

__all__ = [
    # Base classes
    "BaseRedisMiddleware",
    "AsyncRedisMiddleware",
    # Middleware implementations
    "SemanticCacheMiddleware",
    "ToolResultCacheMiddleware",
    "SemanticRouterMiddleware",
    "ConversationMemoryMiddleware",
    # Configuration classes
    "MiddlewareConfig",
    "SemanticCacheConfig",
    "ToolCacheConfig",
    "SemanticRouterConfig",
    "ConversationMemoryConfig",
    # Composition utilities
    "MiddlewareStack",
    "from_configs",
    "create_caching_stack",
    "IntegratedRedisMiddleware",
    # Vectorizer utilities
    "AsyncEmbeddingVectorizer",
    "create_vectorizer_from_langchain",
]
