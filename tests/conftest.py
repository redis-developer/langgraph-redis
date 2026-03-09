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
    os.environ.setdefault("REDIS_VERSION", "8")
    os.environ.setdefault("REDIS_IMAGE", "redis:8")

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


@pytest.fixture(scope="session")
def sentinel_container(request):
    """Start Redis master + Sentinel via Docker Compose for sentinel tests."""
    if not request.config.getoption("--run-sentinel-tests"):
        pytest.skip("Sentinel tests require --run-sentinel-tests flag")

    compose = DockerCompose(
        context="tests/sentinel",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    try:
        compose.start()
    except Exception as exc:
        pytest.fail(f"Failed to start Sentinel containers: {exc}")

    yield compose

    try:
        compose.stop()
    except Exception:
        pass


@pytest.fixture(scope="session")
def sentinel_master_url(sentinel_container):
    """Direct connection URL to the Sentinel-monitored Redis master."""
    # The master is exposed on fixed port 6399
    host = "localhost"
    port = 6399

    deadline = time.time() + 15
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            if time.time() > deadline:
                pytest.skip("Redis master for Sentinel tests failed to become ready.")
            time.sleep(0.5)

    return f"redis://{host}:{port}"


@pytest.fixture(scope="session")
def sentinel_info(sentinel_container):
    """Return sentinel host/port after waiting for readiness.

    Returns a tuple of (sentinel_host, sentinel_port, master_host, master_port)
    where master_host/port are the host-reachable mapped ports.
    """
    sentinel_host = "localhost"
    sentinel_port = 26399
    # The master is port-mapped to localhost:6399
    master_host = "127.0.0.1"
    master_port = 6399

    # Poll sentinel until it has discovered the master
    from redis import Redis as SyncRedis

    deadline = time.time() + 30
    while True:
        try:
            client = SyncRedis(host=sentinel_host, port=sentinel_port)
            result = client.execute_command(
                "SENTINEL", "get-master-addr-by-name", "mymaster"
            )
            client.close()
            if result is not None:
                break
        except Exception:
            pass
        if time.time() > deadline:
            pytest.skip("Redis Sentinel failed to discover master.")
        time.sleep(0.5)

    return sentinel_host, sentinel_port, master_host, master_port


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys",
    )
    parser.addoption(
        "--run-sentinel-tests",
        action="store_true",
        default=False,
        help="Run tests that require Redis Sentinel (extra containers)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "sentinel: mark test as requiring Redis Sentinel"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-api-tests"):
        skip_api = pytest.mark.skip(
            reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
        )
        for item in items:
            if item.get_closest_marker("requires_api_keys"):
                item.add_marker(skip_api)

    if not config.getoption("--run-sentinel-tests"):
        skip_sentinel = pytest.mark.skip(
            reason="Skipping sentinel test. Use --run-sentinel-tests to run."
        )
        for item in items:
            if item.get_closest_marker("sentinel"):
                item.add_marker(skip_sentinel)
