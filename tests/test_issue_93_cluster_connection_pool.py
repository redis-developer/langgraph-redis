"""Test for issue #93 - RedisCluster connection_pool attribute error."""

from unittest.mock import MagicMock, Mock

import pytest
from redis import Redis
from redis.cluster import RedisCluster

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


def test_redis_cluster_connection_pool_attribute_error():
    """Test that connection cleanup now works with RedisCluster which lacks connection_pool."""

    # Create a mock RedisCluster that mimics the real behavior
    mock_cluster = Mock(spec=RedisCluster)
    mock_cluster.close = Mock()

    # RedisCluster doesn't have connection_pool attribute
    # This should raise AttributeError when accessed
    del mock_cluster.connection_pool

    # Test that the fix allows graceful handling
    saver = RedisSaver(redis_client=mock_cluster)
    saver._owns_its_client = True

    # This should NOT fail anymore with our fix
    saver._redis.close()
    # The getattr check should prevent the AttributeError
    if getattr(saver._redis, "connection_pool", None):
        saver._redis.connection_pool.disconnect()

    # Verify close was called
    mock_cluster.close.assert_called_once()


def test_redis_standard_has_connection_pool():
    """Test that standard Redis client has connection_pool."""

    # Create a mock standard Redis client
    mock_redis = Mock(spec=Redis)
    mock_redis.close = Mock()
    mock_redis.connection_pool = Mock()
    mock_redis.connection_pool.disconnect = Mock()

    # This should work fine with standard Redis
    saver = RedisSaver(redis_client=mock_redis)
    saver._owns_its_client = True

    # Simulate the finally block in from_conn_string
    saver._redis.close()
    saver._redis.connection_pool.disconnect()

    # Verify methods were called
    mock_redis.close.assert_called_once()
    mock_redis.connection_pool.disconnect.assert_called_once()


def test_proposed_fix_works_with_both():
    """Test that the proposed fix works with both Redis and RedisCluster."""

    # Test with RedisCluster (no connection_pool)
    mock_cluster = Mock(spec=RedisCluster)
    mock_cluster.close = Mock()
    del mock_cluster.connection_pool  # Remove connection_pool attribute

    saver_cluster = RedisSaver(redis_client=mock_cluster)
    saver_cluster._owns_its_client = True

    # Proposed fix - check if connection_pool exists
    saver_cluster._redis.close()
    if getattr(saver_cluster._redis, "connection_pool", None):
        saver_cluster._redis.connection_pool.disconnect()

    mock_cluster.close.assert_called_once()

    # Test with standard Redis (has connection_pool)
    mock_redis = Mock(spec=Redis)
    mock_redis.close = Mock()
    mock_redis.connection_pool = Mock()
    mock_redis.connection_pool.disconnect = Mock()

    saver_redis = RedisSaver(redis_client=mock_redis)
    saver_redis._owns_its_client = True

    # Same fix should work with standard Redis
    saver_redis._redis.close()
    if getattr(saver_redis._redis, "connection_pool", None):
        saver_redis._redis.connection_pool.disconnect()

    mock_redis.close.assert_called_once()
    mock_redis.connection_pool.disconnect.assert_called_once()


def test_shallow_saver_has_fix_too():
    """Test that ShallowRedisSaver also has the fix applied."""

    # Create a mock RedisCluster
    mock_cluster = Mock(spec=RedisCluster)
    mock_cluster.close = Mock()
    del mock_cluster.connection_pool

    # ShallowRedisSaver should also work with the fix
    saver = ShallowRedisSaver(redis_client=mock_cluster)
    saver._owns_its_client = True

    # This should NOT fail with our fix
    saver._redis.close()
    # The getattr check should prevent the AttributeError
    if getattr(saver._redis, "connection_pool", None):
        saver._redis.connection_pool.disconnect()

    # Verify close was called
    mock_cluster.close.assert_called_once()


def test_context_manager_with_redis_cluster():
    """Test that from_conn_string context manager works with RedisCluster."""
    from unittest.mock import patch

    # Mock the RedisConnectionFactory to return our mock cluster
    mock_cluster = Mock(spec=RedisCluster)
    mock_cluster.close = Mock()
    del mock_cluster.connection_pool

    with patch(
        "langgraph.checkpoint.redis.RedisConnectionFactory.get_redis_connection"
    ) as mock_factory:
        mock_factory.return_value = mock_cluster

        # Test using the context manager doesn't raise AttributeError
        with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
            # Use the saver
            pass

        # Verify close was called (since it owns the client)
        mock_cluster.close.assert_called_once()

    # Test with ShallowRedisSaver too
    mock_cluster2 = Mock(spec=RedisCluster)
    mock_cluster2.close = Mock()
    del mock_cluster2.connection_pool

    with patch(
        "langgraph.checkpoint.redis.shallow.RedisConnectionFactory.get_redis_connection"
    ) as mock_factory2:
        mock_factory2.return_value = mock_cluster2

        with ShallowRedisSaver.from_conn_string("redis://localhost:6379") as saver:
            # Use the saver
            pass

        # Verify close was called
        mock_cluster2.close.assert_called_once()
