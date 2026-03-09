"""Integration tests for Redis Sentinel support.

These tests require --run-sentinel-tests flag and spin up Redis master + Sentinel
containers via tests/sentinel/docker-compose.yml.

Note on Docker networking: When running sentinel in Docker on macOS, sentinel
reports Docker-internal IPs for the master. These are not reachable from the host.
We work around this by using redis.sentinel.Sentinel with force_master_ip to
redirect client connections to the host-mapped port (localhost:6399).
"""

from __future__ import annotations

from typing import Tuple
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.sentinel import Sentinel as AsyncSentinel
from redis.sentinel import Sentinel

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from langgraph.store.redis import RedisStore
from langgraph.store.redis.aio import AsyncRedisStore

pytestmark = pytest.mark.sentinel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(thread_id: str | None = None) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id or str(uuid4()),
            "checkpoint_ns": "",
        }
    }


def _make_checkpoint() -> tuple[Checkpoint, CheckpointMetadata]:
    chkpnt = empty_checkpoint()
    metadata: CheckpointMetadata = {
        "source": "input",
        "step": 1,
        "writes": {},
        "score": 1,
    }
    return chkpnt, metadata


class _FixedPortSentinel(Sentinel):
    """Sentinel subclass that overrides the discovered master port.

    In Docker-on-macOS, sentinel reports the container-internal port (6379)
    but the host-mapped port is different (6399). This subclass forces the
    correct host-reachable address.
    """

    def __init__(
        self,
        sentinels,
        master_host: str = "127.0.0.1",
        master_port: int = 6399,
        **kwargs,
    ):
        super().__init__(sentinels, force_master_ip=master_host, **kwargs)
        self._master_port = master_port

    def discover_master(self, service_name):
        ip, _port = super().discover_master(service_name)
        return ip, self._master_port


class _AsyncFixedPortSentinel(AsyncSentinel):
    """Async version of _FixedPortSentinel."""

    def __init__(
        self,
        sentinels,
        master_host: str = "127.0.0.1",
        master_port: int = 6399,
        **kwargs,
    ):
        super().__init__(sentinels, force_master_ip=master_host, **kwargs)
        self._master_port = master_port

    async def discover_master(self, service_name):
        ip, _port = await super().discover_master(service_name)
        return ip, self._master_port


def _make_sync_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> Redis:
    """Create a sync Redis client via Sentinel with correct host-mapped port."""
    sentinel_host, sentinel_port, master_host, master_port = sentinel_info
    sentinel = _FixedPortSentinel(
        [(sentinel_host, sentinel_port)],
        master_host=master_host,
        master_port=master_port,
    )
    client = sentinel.master_for("mymaster", redis_class=Redis)
    return client


def _make_async_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> AsyncRedis:
    """Create an async Redis client via Sentinel with correct host-mapped port."""
    sentinel_host, sentinel_port, master_host, master_port = sentinel_info
    sentinel = _AsyncFixedPortSentinel(
        [(sentinel_host, sentinel_port)],
        master_host=master_host,
        master_port=master_port,
    )
    client = sentinel.master_for("mymaster", redis_class=AsyncRedis)
    return client


# ---------------------------------------------------------------------------
# 1. Sentinel client smoke tests
# ---------------------------------------------------------------------------


def test_sentinel_sync_client_ping(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """Sync Sentinel client can ping the master."""
    client = _make_sync_sentinel_client(sentinel_info)
    try:
        assert client.ping()
    finally:
        client.close()


@pytest.mark.asyncio
async def test_sentinel_async_client_ping(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """Async Sentinel client can ping the master."""
    client = _make_async_sentinel_client(sentinel_info)
    try:
        assert await client.ping()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 2. RedisSaver (sync) with sentinel client
# ---------------------------------------------------------------------------


def test_redis_saver_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """Full checkpoint lifecycle through Sentinel-discovered master (sync)."""
    client = _make_sync_sentinel_client(sentinel_info)
    try:
        saver = RedisSaver(redis_client=client)
        saver.setup()

        config = _make_config()
        chkpnt, metadata = _make_checkpoint()

        saved_config = saver.put(config, chkpnt, metadata, {})
        result = saver.get_tuple(saved_config)

        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 3. AsyncRedisSaver with sentinel client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_redis_saver_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """Full checkpoint lifecycle through Sentinel (async) — the main fix test."""
    client = _make_async_sentinel_client(sentinel_info)
    try:
        async with AsyncRedisSaver(redis_client=client) as saver:
            config = _make_config()
            chkpnt, metadata = _make_checkpoint()

            saved_config = await saver.aput(config, chkpnt, metadata, {})
            result = await saver.aget_tuple(saved_config)

            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 4. ShallowRedisSaver (sync) with sentinel client
# ---------------------------------------------------------------------------


def test_shallow_saver_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """ShallowRedisSaver works with Sentinel-discovered client."""
    client = _make_sync_sentinel_client(sentinel_info)
    try:
        saver = ShallowRedisSaver(redis_client=client)
        saver.setup()

        config = _make_config()
        chkpnt, metadata = _make_checkpoint()

        saved_config = saver.put(config, chkpnt, metadata, {})
        result = saver.get_tuple(saved_config)

        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 5. AsyncShallowRedisSaver with sentinel client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_shallow_saver_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """AsyncShallowRedisSaver works with Sentinel client."""
    client = _make_async_sentinel_client(sentinel_info)
    try:
        async with AsyncShallowRedisSaver(redis_client=client) as saver:
            config = _make_config()
            chkpnt, metadata = _make_checkpoint()

            saved_config = await saver.aput(config, chkpnt, metadata, {})
            result = await saver.aget_tuple(saved_config)

            assert result is not None
            assert result.checkpoint["id"] == chkpnt["id"]
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 6. RedisStore (sync) with sentinel client
# ---------------------------------------------------------------------------


def test_redis_store_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """RedisStore works with a Sentinel-discovered client."""
    client = _make_sync_sentinel_client(sentinel_info)
    try:
        store = RedisStore(conn=client)
        store.setup()

        store.put(("test", "ns"), "key1", {"data": "value1"})
        items = store.get(("test", "ns"), "key1")

        assert items is not None
        assert items.value == {"data": "value1"}
    finally:
        client.close()


# ---------------------------------------------------------------------------
# 7. AsyncRedisStore with sentinel client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_redis_store_with_sentinel_client(
    sentinel_info: Tuple[str, int, str, int],
) -> None:
    """AsyncRedisStore works with Sentinel client."""
    client = _make_async_sentinel_client(sentinel_info)
    try:
        store = AsyncRedisStore(redis_client=client)
        await store.setup()

        await store.aput(("test", "ns"), "key1", {"data": "value1"})
        items = await store.aget(("test", "ns"), "key1")

        assert items is not None
        assert items.value == {"data": "value1"}
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 8. Data written via Sentinel matches direct connection
# ---------------------------------------------------------------------------


def test_sentinel_data_matches_direct_connection(
    sentinel_info: Tuple[str, int, str, int],
    sentinel_master_url: str,
) -> None:
    """Data written via Sentinel is readable via direct connection to master."""
    from redisvl.redis.connection import RedisConnectionFactory

    thread_id = str(uuid4())
    config = _make_config(thread_id)
    chkpnt, metadata = _make_checkpoint()

    # Write via Sentinel, then read via direct connection to same Redis
    sentinel_client = _make_sync_sentinel_client(sentinel_info)
    direct_client = RedisConnectionFactory.get_redis_connection(sentinel_master_url)
    try:
        saver_sentinel = RedisSaver(redis_client=sentinel_client)
        saver_sentinel.setup()
        saved_config = saver_sentinel.put(config, chkpnt, metadata, {})

        # Read via direct connection using a different saver instance
        saver_direct = RedisSaver(redis_client=direct_client)
        saver_direct.setup()
        result = saver_direct.get_tuple(saved_config)

        assert result is not None
        assert result.checkpoint["id"] == chkpnt["id"]
    finally:
        sentinel_client.close()
        direct_client.close()


# ---------------------------------------------------------------------------
# 9. Verify configure_client uses RedisConnectionFactory
# ---------------------------------------------------------------------------


def test_async_saver_configure_client_uses_factory(
    sentinel_master_url: str,
) -> None:
    """Verify AsyncRedisSaver.configure_client() uses RedisConnectionFactory.

    This validates the core code change: async configure_client() now delegates
    to RedisConnectionFactory.get_async_redis_connection() which supports
    sentinel URLs, instead of AsyncRedis.from_url() which does not.
    """
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock(spec=AsyncRedis)

    with patch("langgraph.checkpoint.redis.aio.RedisConnectionFactory") as mock_factory:
        mock_factory.get_async_redis_connection.return_value = mock_client

        saver = AsyncRedisSaver.__new__(AsyncRedisSaver)
        saver.configure_client(redis_url="redis+sentinel://localhost:26379/mymaster/0")

        mock_factory.get_async_redis_connection.assert_called_once_with(
            "redis+sentinel://localhost:26379/mymaster/0"
        )
        assert saver._redis is mock_client
        assert saver._owns_its_client is True


def test_async_shallow_configure_client_uses_factory() -> None:
    """Verify AsyncShallowRedisSaver.configure_client() uses RedisConnectionFactory."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock(spec=AsyncRedis)

    with patch(
        "langgraph.checkpoint.redis.ashallow.RedisConnectionFactory"
    ) as mock_factory:
        mock_factory.get_async_redis_connection.return_value = mock_client

        saver = AsyncShallowRedisSaver.__new__(AsyncShallowRedisSaver)
        saver.configure_client(redis_url="redis+sentinel://localhost:26379/mymaster/0")

        mock_factory.get_async_redis_connection.assert_called_once_with(
            "redis+sentinel://localhost:26379/mymaster/0"
        )
        assert saver._redis is mock_client


def test_async_store_configure_client_uses_factory() -> None:
    """Verify AsyncRedisStore.configure_client() uses RedisConnectionFactory."""
    from unittest.mock import MagicMock, patch

    mock_client = MagicMock(spec=AsyncRedis)

    with patch("langgraph.store.redis.aio.RedisConnectionFactory") as mock_factory:
        mock_factory.get_async_redis_connection.return_value = mock_client

        store = AsyncRedisStore.__new__(AsyncRedisStore)
        store.configure_client(redis_url="redis+sentinel://localhost:26379/mymaster/0")

        mock_factory.get_async_redis_connection.assert_called_once_with(
            "redis+sentinel://localhost:26379/mymaster/0"
        )
        assert store._redis is mock_client
