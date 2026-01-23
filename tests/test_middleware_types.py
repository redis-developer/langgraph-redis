"""Unit tests for middleware types and configurations."""

from langgraph.middleware.redis.types import (
    ConversationMemoryConfig,
    MiddlewareConfig,
    SemanticCacheConfig,
    SemanticRouterConfig,
    ToolCacheConfig,
)


class TestMiddlewareConfig:
    """Tests for the base MiddlewareConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that MiddlewareConfig has correct default values."""
        config = MiddlewareConfig()
        assert config.redis_url is None
        assert config.redis_client is None
        assert config.connection_args is None
        assert config.graceful_degradation is True

    def test_custom_values(self) -> None:
        """Test MiddlewareConfig with custom values."""
        config = MiddlewareConfig(
            redis_url="redis://localhost:6379",
            graceful_degradation=False,
            connection_args={"decode_responses": True},
        )
        assert config.redis_url == "redis://localhost:6379"
        assert config.graceful_degradation is False
        assert config.connection_args == {"decode_responses": True}

    def test_redis_client_parameter(self) -> None:
        """Test MiddlewareConfig accepts redis_client."""
        # Mock Redis client
        mock_client = object()
        config = MiddlewareConfig(redis_client=mock_client)
        assert config.redis_client is mock_client


class TestSemanticCacheConfig:
    """Tests for SemanticCacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that SemanticCacheConfig has correct default values."""
        config = SemanticCacheConfig()
        assert config.name == "llmcache"
        assert config.distance_threshold == 0.1
        assert config.ttl_seconds is None
        assert config.vectorizer is None
        assert config.cache_final_only is True
        # Inherited from MiddlewareConfig
        assert config.graceful_degradation is True

    def test_custom_values(self) -> None:
        """Test SemanticCacheConfig with custom values."""
        mock_vectorizer = object()
        config = SemanticCacheConfig(
            redis_url="redis://localhost:6379",
            name="custom_cache",
            distance_threshold=0.2,
            ttl_seconds=3600,
            vectorizer=mock_vectorizer,
            cache_final_only=False,
        )
        assert config.name == "custom_cache"
        assert config.distance_threshold == 0.2
        assert config.ttl_seconds == 3600
        assert config.vectorizer is mock_vectorizer
        assert config.cache_final_only is False

    def test_inherits_from_middleware_config(self) -> None:
        """Test that SemanticCacheConfig inherits from MiddlewareConfig."""
        config = SemanticCacheConfig()
        assert isinstance(config, MiddlewareConfig)


class TestToolCacheConfig:
    """Tests for ToolCacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that ToolCacheConfig has correct default values."""
        config = ToolCacheConfig()
        assert config.name == "toolcache"
        assert config.distance_threshold == 0.1
        assert config.ttl_seconds is None
        assert config.cacheable_tools is None
        assert config.excluded_tools == []

    def test_custom_values(self) -> None:
        """Test ToolCacheConfig with custom values."""
        config = ToolCacheConfig(
            name="my_tool_cache",
            cacheable_tools=["search", "calculate"],
            excluded_tools=["random_tool"],
        )
        assert config.name == "my_tool_cache"
        assert config.cacheable_tools == ["search", "calculate"]
        assert config.excluded_tools == ["random_tool"]

    def test_mutable_default_excluded_tools(self) -> None:
        """Test that excluded_tools default is not shared between instances."""
        config1 = ToolCacheConfig()
        config2 = ToolCacheConfig()
        config1.excluded_tools.append("test")
        assert "test" not in config2.excluded_tools


class TestSemanticRouterConfig:
    """Tests for SemanticRouterConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that SemanticRouterConfig has correct default values."""
        config = SemanticRouterConfig()
        assert config.name == "semantic_router"
        assert config.routes == []
        assert config.max_k == 3
        assert config.aggregation_method == "avg"

    def test_custom_routes(self) -> None:
        """Test SemanticRouterConfig with custom routes."""
        routes = [
            {"name": "greeting", "references": ["hello", "hi", "hey"]},
            {"name": "farewell", "references": ["bye", "goodbye"]},
        ]
        config = SemanticRouterConfig(routes=routes)
        assert len(config.routes) == 2
        assert config.routes[0]["name"] == "greeting"

    def test_mutable_default_routes(self) -> None:
        """Test that routes default is not shared between instances."""
        config1 = SemanticRouterConfig()
        config2 = SemanticRouterConfig()
        config1.routes.append({"name": "test"})
        assert len(config2.routes) == 0


class TestConversationMemoryConfig:
    """Tests for ConversationMemoryConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that ConversationMemoryConfig has correct default values."""
        config = ConversationMemoryConfig()
        assert config.name == "conversation_memory"
        assert config.session_tag is None
        assert config.top_k == 5
        assert config.distance_threshold == 0.3
        assert config.ttl_seconds is None

    def test_custom_values(self) -> None:
        """Test ConversationMemoryConfig with custom values."""
        config = ConversationMemoryConfig(
            session_tag="user_123",
            top_k=10,
            distance_threshold=0.5,
            ttl_seconds=86400,
        )
        assert config.session_tag == "user_123"
        assert config.top_k == 10
        assert config.distance_threshold == 0.5
        assert config.ttl_seconds == 86400
