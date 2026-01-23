"""Type definitions and configuration dataclasses for Redis middleware."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

# Type variable for Redis clients
RedisClientType = TypeVar(
    "RedisClientType", bound=Union[Redis, AsyncRedis, RedisCluster, AsyncRedisCluster]
)

# Type alias for async model call handler
ModelCallHandler = Callable[..., Any]
ToolCallHandler = Callable[..., Any]


@dataclass
class MiddlewareConfig:
    """Base configuration for all Redis middleware.

    Attributes:
        redis_url: Redis connection URL. If not provided, redis_client must be set.
        redis_client: Existing Redis client instance to use.
        connection_args: Additional arguments for Redis connection.
        graceful_degradation: If True, middleware passes through on Redis errors.
    """

    redis_url: Optional[str] = None
    redis_client: Optional[Union[Redis, AsyncRedis]] = None
    connection_args: Optional[Dict[str, Any]] = None
    graceful_degradation: bool = True


@dataclass
class SemanticCacheConfig(MiddlewareConfig):
    """Configuration for SemanticCacheMiddleware.

    Uses redisvl.extensions.llmcache.SemanticCache for semantic similarity caching.

    Attributes:
        name: Index name for the semantic cache.
        distance_threshold: Maximum distance for cache hits (lower = stricter).
        ttl_seconds: Time-to-live for cache entries in seconds.
        vectorizer: Optional vectorizer for embeddings. If not provided,
            uses default from redisvl.
        cache_final_only: If True, only cache responses without tool_calls.
        deterministic_tools: List of tool names whose results are deterministic.
            When a request contains tool results, cache lookup is only performed
            if ALL tool results are from tools in this list. If None, cache is
            always skipped when tool results are present (safest default).
    """

    name: str = "llmcache"
    distance_threshold: float = 0.1
    ttl_seconds: Optional[int] = None
    vectorizer: Optional[Any] = None
    cache_final_only: bool = True
    deterministic_tools: Optional[List[str]] = None


@dataclass
class ToolCacheConfig(MiddlewareConfig):
    """Configuration for ToolResultCacheMiddleware.

    Uses SemanticCache with tool_name filtering for tool result caching.

    Attributes:
        name: Index name for the tool cache.
        distance_threshold: Maximum distance for cache hits.
        ttl_seconds: Time-to-live for cache entries in seconds.
        vectorizer: Optional vectorizer for embeddings.
        cacheable_tools: List of tool names to cache. If None, all tools
            except excluded_tools are cached.
        excluded_tools: List of tool names to never cache.
    """

    name: str = "toolcache"
    distance_threshold: float = 0.1
    ttl_seconds: Optional[int] = None
    vectorizer: Optional[Any] = None
    cacheable_tools: Optional[List[str]] = None
    excluded_tools: List[str] = field(default_factory=list)


@dataclass
class SemanticRouterConfig(MiddlewareConfig):
    """Configuration for SemanticRouterMiddleware.

    Uses redisvl.extensions.router.SemanticRouter for intent-based routing.

    Attributes:
        name: Index name for the router.
        routes: List of route configurations. Each route should have:
            - name: Route identifier
            - references: List of example phrases for this route
            - distance_threshold: Optional distance threshold for this route
        vectorizer: Optional vectorizer for embeddings.
        max_k: Maximum number of routes to consider.
        aggregation_method: Method to aggregate route scores.
    """

    name: str = "semantic_router"
    routes: List[Dict[str, Any]] = field(default_factory=list)
    vectorizer: Optional[Any] = None
    max_k: int = 3
    aggregation_method: str = "avg"


@dataclass
class ConversationMemoryConfig(MiddlewareConfig):
    """Configuration for ConversationMemoryMiddleware.

    Uses redisvl.extensions.session_manager.SemanticSessionManager for
    semantic message history.

    Attributes:
        name: Index name for message history.
        session_tag: Tag to identify the conversation session.
        top_k: Number of relevant messages to retrieve.
        distance_threshold: Maximum distance for relevant messages.
        vectorizer: Optional vectorizer for embeddings.
        ttl_seconds: Time-to-live for messages in seconds.
    """

    name: str = "conversation_memory"
    session_tag: Optional[str] = None
    top_k: int = 5
    distance_threshold: float = 0.3
    vectorizer: Optional[Any] = None
    ttl_seconds: Optional[int] = None


# Type alias for middleware configuration types
MiddlewareConfigType = Union[
    SemanticCacheConfig,
    ToolCacheConfig,
    SemanticRouterConfig,
    ConversationMemoryConfig,
]
