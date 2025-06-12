import asyncio
import os
import socket
import time

import pytest
from redis.asyncio import Redis
from redisvl.redis.connection import RedisConnectionFactory
from testcontainers.compose import DockerCompose

VECTOR_TYPES = ["vector", "halfvec"]


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Disable tokenizers parallelism in tests to avoid deadlocks"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """
    If using xdist, create a unique Compose project for each xdist worker by
    setting COMPOSE_PROJECT_NAME. That prevents collisions on container/volume
    names.
    """
    # In xdist, the config has "workerid" in workerinput
    workerinput = getattr(request.config, "workerinput", {})
    worker_id = workerinput.get("workerid", "master")

    # Set the Compose project name so containers do not clash across workers
    os.environ["COMPOSE_PROJECT_NAME"] = f"redis_test_{worker_id}"
    os.environ.setdefault("REDIS_VERSION", "latest")
    os.environ.setdefault("REDIS_IMAGE", "redis/redis-stack-server:latest")

    compose = DockerCompose(
        context="tests",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    try:
        compose.start()
    except Exception:
        # Ignore compose startup errors (e.g., existing containers)
        pass

    yield compose

    try:
        compose.stop()
    except Exception:
        # Ignore compose stop errors
        pass


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)

    # Wait up to 15 seconds for the container to accept TCP connections.
    deadline = time.time() + 15
    while True:
        try:
            with socket.create_connection((host, int(port)), timeout=1):
                break  # Redis is accepting connections
        except OSError:
            if time.time() > deadline:
                pytest.skip(
                    "Redis container failed to become ready for this worker – skipping tests."
                )
            time.sleep(0.5)

    return f"redis://{host}:{port}"


@pytest.fixture
async def async_client(redis_url):
    """
    An async Redis client that uses the dynamic `redis_url`.
    """
    async with await RedisConnectionFactory._get_aredis_connection(redis_url) as client:
        yield client


@pytest.fixture
def client(redis_url):
    """
    A sync Redis client that uses the dynamic `redis_url`.
    """
    conn = RedisConnectionFactory.get_redis_connection(redis_url)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
async def clear_redis(redis_url: str) -> None:
    """Clear Redis before each test."""
    # Add a small delay to allow container to stabilize between tests
    await asyncio.sleep(0.1)
    try:
        client = Redis.from_url(redis_url)
        await client.flushall()
        await client.aclose()
    except Exception:
        # Ignore clear_redis errors when Redis container is unavailable
        pass


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-api-tests"):
        return
    skip_api = pytest.mark.skip(
        reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
    )
    for item in items:
        if item.get_closest_marker("requires_api_keys"):
            item.add_marker(skip_api)
