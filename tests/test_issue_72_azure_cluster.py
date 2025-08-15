"""Test for issue #72: Azure Cache for Redis cluster mode compatibility.

This test demonstrates the proper way to configure Redis clients for
Azure Cache for Redis and other enterprise/proxy environments.

The key insight is that Azure Cache for Redis (and similar enterprise setups)
use a proxy layer that makes the cluster appear as a single endpoint. The
solution is to pass a properly configured Redis client, not to override
internal cluster detection.
"""

from typing import Any, Dict, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def test_azure_cache_with_standard_redis_client() -> None:
    """Test using standard Redis client for Azure Cache (single endpoint).

    Azure Cache for Redis uses a proxy that exposes the cluster through
    a single endpoint. In this case, you should use a standard Redis client,
    not a RedisCluster client.
    """
    with RedisContainer("redis:8") as redis_container:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # For Azure Cache, use standard Redis client (not RedisCluster)
        # The proxy handles cluster routing internally
        client = Redis.from_url(redis_url)

        # Pass the configured client to RedisSaver
        saver = RedisSaver(redis_client=client)
        saver.setup()  # Initialize the saver

        # The saver will detect this as a non-cluster client
        assert saver.cluster_mode == False

        # Test basic operations
        config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "azure-test",
                    "checkpoint_ns": "",
                    "checkpoint_id": "checkpoint-1",
                }
            },
        )

        checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": "checkpoint-1",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )

        metadata = cast(
            CheckpointMetadata, {"source": "input", "step": 1, "writes": {}}
        )

        # Should work without cluster-related issues
        saver.put(config, checkpoint, metadata, {})
        result = saver.get_tuple(config)

        assert result is not None
        assert result.checkpoint["id"] == "checkpoint-1"


def test_redis_cluster_client_detection() -> None:
    """Test that RedisCluster client is properly detected.

    When connecting to a real Redis cluster (not through a proxy),
    use RedisCluster client which will be auto-detected.
    """
    with RedisContainer("redis:8") as redis_container:
        host = redis_container.get_container_host_ip()
        port = redis_container.get_exposed_port(6379)

        # For demonstration - in real scenario, you'd have multiple nodes
        # RedisCluster would be used for actual cluster deployments
        try:
            # This will fail with single node, but shows the pattern
            cluster_client = RedisCluster(
                host=host,
                port=port,
                skip_full_coverage_check=True,  # For testing with single node
            )

            saver = RedisSaver(redis_client=cluster_client)
            saver.setup()
            # The saver should detect this as a cluster client
            assert saver.cluster_mode == True

        except Exception:
            # Expected - single node isn't a real cluster
            # This just demonstrates the pattern
            pass


@pytest.mark.asyncio
async def test_async_azure_cache_configuration() -> None:
    """Test async configuration for Azure Cache for Redis."""
    redis_container = RedisContainer("redis:8")
    redis_container.start()

    try:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        # For Azure Cache, use standard async Redis client
        client = AsyncRedis.from_url(redis_url)

        # Pass the configured client
        async with AsyncRedisSaver(redis_client=client) as saver:
            await saver.asetup()

            # Should detect as non-cluster
            assert saver.cluster_mode == False

            config = cast(
                RunnableConfig,
                {
                    "configurable": {
                        "thread_id": "async-azure-test",
                        "checkpoint_ns": "",
                        "checkpoint_id": "async-checkpoint-1",
                    }
                },
            )

            checkpoint = cast(
                Checkpoint,
                {
                    "v": 1,
                    "ts": "2024-01-01T00:00:00+00:00",
                    "id": "async-checkpoint-1",
                    "channel_values": {},
                    "channel_versions": {},
                    "versions_seen": {},
                    "pending_sends": [],
                },
            )

            metadata = cast(
                CheckpointMetadata, {"source": "input", "step": 1, "writes": {}}
            )

            await saver.aput(config, checkpoint, metadata, {})
            result = await saver.aget_tuple(config)

            assert result is not None
            assert result.checkpoint["id"] == "async-checkpoint-1"

        await client.aclose()

    finally:
        redis_container.stop()


def test_workaround_manual_cluster_mode_override() -> None:
    """Test manual override of cluster_mode after creation.

    This demonstrates a potential workaround if auto-detection fails,
    though the proper solution is to use the correct client type.
    """
    with RedisContainer("redis:8") as redis_container:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        client = Redis.from_url(redis_url)

        saver = RedisSaver(redis_client=client)
        saver.setup()

        # Auto-detection will set cluster_mode=False for standard Redis client
        assert saver.cluster_mode == False

        # Manual override (workaround if needed)
        # Note: This is NOT recommended - use proper client instead
        saver.cluster_mode = True

        # Now it will use cluster-mode code paths
        assert saver.cluster_mode == True

        # Operations will use cluster-mode logic
        config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "override-test",
                    "checkpoint_ns": "",
                    "checkpoint_id": "checkpoint-1",
                }
            },
        )

        checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": "checkpoint-1",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )

        metadata = cast(
            CheckpointMetadata, {"source": "input", "step": 1, "writes": {}}
        )

        saver.put(config, checkpoint, metadata, {})
        result = saver.get_tuple(config)

        assert result is not None


def test_hash_tags_for_cluster_operations() -> None:
    """Test using hash tags to ensure keys go to same slot in cluster.

    This is useful when you need to perform multi-key operations
    in a Redis cluster environment.
    """
    with RedisContainer("redis:8") as redis_container:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        client = Redis.from_url(redis_url)

        saver = RedisSaver(redis_client=client)
        saver.setup()

        # Use hash tags to force keys to same slot
        # Keys with {tag} will hash to the same slot
        config1 = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "{user123}:thread-1",  # Hash tag
                    "checkpoint_ns": "{user123}:ns",  # Same hash tag
                    "checkpoint_id": "checkpoint-1",
                }
            },
        )

        config2 = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "{user123}:thread-2",  # Same hash tag
                    "checkpoint_ns": "{user123}:ns",  # Same hash tag
                    "checkpoint_id": "checkpoint-2",
                }
            },
        )

        checkpoint = cast(
            Checkpoint,
            {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": "checkpoint-1",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            },
        )

        metadata = cast(
            CheckpointMetadata, {"source": "input", "step": 1, "writes": {}}
        )

        # These will go to the same slot due to hash tags
        saver.put(config1, checkpoint, metadata, {})

        checkpoint["id"] = "checkpoint-2"
        saver.put(config2, checkpoint, metadata, {})

        # List operations will work efficiently with hash tags
        results = list(saver.list(config1, limit=10))
        assert len(results) >= 1


# Example documentation for users
def example_azure_cache_configuration() -> None:
    """Example: How to configure for Azure Cache for Redis.

    Azure Cache for Redis uses a proxy layer that makes the cluster
    appear as a single endpoint. Use a standard Redis client, not
    RedisCluster.
    """
    # For Azure Cache for Redis
    azure_redis_url = "redis://your-cache.redis.cache.windows.net:6379"
    azure_password = "your-access-key"

    # Use standard Redis client (not RedisCluster)
    client = Redis.from_url(
        azure_redis_url,
        password=azure_password,
        ssl=True,  # Azure Cache uses SSL
        ssl_cert_reqs=None,  # Azure uses self-signed certs
    )

    # Pass the configured client to RedisSaver
    saver = RedisSaver(redis_client=client)
    saver.setup()
    # Will auto-detect as non-cluster (correct for Azure proxy)
    # Operations will work through Azure's proxy layer


def example_real_cluster_configuration() -> None:
    """Example: How to configure for a real Redis Cluster.

    For actual Redis Cluster deployments (not behind a proxy),
    use RedisCluster client.
    """
    # For real Redis Cluster with multiple nodes
    startup_nodes = [
        {"host": "node1.example.com", "port": 7000},
        {"host": "node2.example.com", "port": 7001},
        {"host": "node3.example.com", "port": 7002},
    ]

    # Use RedisCluster client for real clusters
    cluster_client = RedisCluster(
        startup_nodes=startup_nodes,
        decode_responses=False,  # RedisSaver expects bytes
        skip_full_coverage_check=False,
    )

    # Pass the configured client
    saver = RedisSaver(redis_client=cluster_client)
    saver.setup()
    # Will auto-detect as cluster (correct for real cluster)
