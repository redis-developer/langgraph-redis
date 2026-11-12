import asyncio
import os
import socket
import time
import uuid

import pytest
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from redis.exceptions import RedisError
from redisvl.redis.connection import RedisConnectionFactory
from testcontainers.compose import DockerCompose

VECTOR_TYPES = ["vector", "halfvec"]

# Set this to pin the Compose project prefix, for example to reuse one Redis
# container across runs. Left unset, every run gets a prefix of its own.
PROJECT_PREFIX_ENV = "REDIS_TEST_PROJECT_PREFIX"


def _compose_project_name(config: pytest.Config) -> str:
    """Return the Compose project name for this pytest process.

    Each xdist worker gets its own project, and by default so does each run,
    so two suites sharing a Docker host never recreate or tear down each
    other's Redis. Workers of one xdist run share its ``testrun_uid``.
    """
    workerinput = getattr(config, "workerinput", {})
    worker_id = workerinput.get("workerid", "master")
    prefix = os.environ.get(PROJECT_PREFIX_ENV)
    if not prefix:
        run_id = workerinput.get("testrun_uid") or uuid.uuid4().hex
        prefix = f"redis_test_{run_id[:12]}"
    return f"{prefix}_{worker_id}".lower()


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Disable tokenizers parallelism in tests to avoid deadlocks"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """Start Redis through Docker Compose under a project of this run's own.

    See ``_compose_project_name`` for how the project is named. A failed
    start fails the tests that need Redis, rather than letting them run
    against a missing or foreign container.
    """
    project_name = _compose_project_name(request.config)
    os.environ["COMPOSE_PROJECT_NAME"] = project_name
    os.environ.setdefault("REDIS_VERSION", "8")
    os.environ.setdefault("REDIS_IMAGE", "redis:8")

    compose = DockerCompose(
        context="tests",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    try:
        compose.start()
    except Exception as exc:
        stderr = getattr(exc, "stderr", None)
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = f"\n{stderr.strip()}" if stderr else ""
        pytest.fail(
            f"Could not start the Redis test container (Compose project "
            f"{project_name!r}, image {os.environ['REDIS_IMAGE']!r}): {exc}{detail}",
            pytrace=False,
        )

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
    url = f"redis://{host}:{port}"

    # Wait up to 15 seconds for Redis to answer PING. A published port can
    # accept TCP connections before the server behind it is ready.
    deadline = time.time() + 15
    while True:
        try:
            with SyncRedis.from_url(url, socket_timeout=1) as client:
                if client.ping():
                    break
        except (RedisError, OSError) as exc:
            if time.time() > deadline:
                pytest.fail(
                    f"Redis test container at {url} did not answer PING within "
                    f"15 seconds (Compose project "
                    f"{os.environ.get('COMPOSE_PROJECT_NAME')!r}): {exc}",
                    pytrace=False,
                )
        time.sleep(0.5)

    return url


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
    client = Redis.from_url(redis_url)
    try:
        await client.flushall()
    except (RedisError, OSError) as exc:
        pytest.fail(
            f"Could not flush Redis at {redis_url} before the test; the test "
            f"container may have stopped or been removed: {exc}",
            pytrace=False,
        )
    finally:
        await client.aclose()


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
