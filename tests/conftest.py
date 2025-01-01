import pytest
from redis.asyncio import Redis

DEFAULT_REDIS_URI = "redis://localhost:6379"


@pytest.fixture(autouse=True)
async def clear_redis() -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(DEFAULT_REDIS_URI)
    await client.flushall()
    await client.aclose()  # type: ignore[attr-defined]
