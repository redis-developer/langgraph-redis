"""Base synchronous Redis middleware class."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from redis import Redis

from .types import MiddlewareConfig

logger = logging.getLogger(__name__)

RedisClientType = TypeVar("RedisClientType", bound=Redis)


class BaseRedisMiddleware(ABC, Generic[RedisClientType]):
    """Abstract base class for synchronous Redis middleware.

    This class provides common functionality for all Redis-based middleware:
    - Redis client lifecycle management
    - Lazy initialization with thread-safe setup
    - Graceful degradation on Redis errors
    - Context manager support

    Subclasses must implement:
    - _setup_sync(): Called once during initialization to set up resources

    Example:
        ```python
        class MyMiddleware(BaseRedisMiddleware):
            def _setup_sync(self) -> None:
                # Initialize resources
                self._cache = SemanticCache(redis_client=self._redis)

        config = MiddlewareConfig(redis_url="redis://localhost:6379")
        with MyMiddleware(config) as middleware:
            # Use middleware
            pass
        ```
    """

    _redis: RedisClientType
    _config: MiddlewareConfig
    _owns_client: bool
    _graceful_degradation: bool
    _initialized: bool
    _init_lock: threading.Lock

    def __init__(self, config: MiddlewareConfig) -> None:
        """Initialize the middleware.

        Args:
            config: Middleware configuration with Redis connection details.

        Raises:
            ValueError: If neither redis_url nor redis_client is provided.
        """
        self._config = config
        self._graceful_degradation = config.graceful_degradation
        self._initialized = False
        self._init_lock = threading.Lock()

        # Set up Redis client
        if config.redis_client is not None:
            self._redis = config.redis_client
            self._owns_client = False
        elif config.redis_url is not None:
            connection_args = config.connection_args or {}
            self._redis = Redis.from_url(config.redis_url, **connection_args)
            self._owns_client = True
        else:
            raise ValueError("Either redis_url or redis_client must be provided")

    @abstractmethod
    def _setup_sync(self) -> None:
        """Set up middleware resources.

        Called once during lazy initialization. Subclasses should override
        this to initialize caches, indices, or other resources.
        """
        pass

    def _ensure_initialized_sync(self) -> None:
        """Ensure middleware is initialized (thread-safe).

        Uses double-checked locking pattern for thread safety.
        """
        if self._initialized:
            return

        with self._init_lock:
            if not self._initialized:
                self._setup_sync()
                self._initialized = True

    def close(self) -> None:
        """Close the Redis connection if owned by this middleware."""
        if self._owns_client and hasattr(self, "_redis"):
            try:
                self._redis.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")

    def __enter__(self) -> "BaseRedisMiddleware[RedisClientType]":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit context manager and close resources."""
        self.close()
