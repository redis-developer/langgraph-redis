import os
import subprocess

import pytest
from redis.asyncio import Redis
from redisvl.redis.connection import RedisConnectionFactory
from testcontainers.compose import DockerCompose

VECTOR_TYPES = ["vector", "halfvec"]

# try:
#     from testcontainers.compose import DockerCompose

#     TESTCONTAINERS_AVAILABLE = True
# except ImportError:
#     TESTCONTAINERS_AVAILABLE = False

# if TESTCONTAINERS_AVAILABLE:


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism():
    """Disable tokenizers parallelism in tests to avoid deadlocks"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # @pytest.fixture(scope="session", autouse=True)
    # def redis_container() -> DockerCompose:
    #     # Set the default Redis version if not already set
    #     os.environ.setdefault("REDIS_VERSION", "edge")

    #     try:
    #         compose = DockerCompose(
    #             "tests", compose_file_name="docker-compose.yml", pull=True
    #         )
    #         compose.start()

    #         redis_host, redis_port = compose.get_service_host_and_port("redis", 6379)
    #         redis_url = f"redis://{redis_host}:{redis_port}"
    #         os.environ["DEFAULT_REDIS_URI"] = redis_url

    #         yield compose

    #         compose.stop()
    #     except subprocess.CalledProcessError:
    #         yield None


@pytest.fixture(scope="session", autouse=True)
def redis_container(request):
    """
    Create a unique Compose project for each xdist worker by setting
    COMPOSE_PROJECT_NAME. That prevents collisions on container/volume names.
    """
    # In xdist, the config has "workerid" in workerinput
    worker_id = request.config.workerinput.get("workerid", "master")

    # Set the Compose project name so containers do not clash across workers
    os.environ["COMPOSE_PROJECT_NAME"] = f"redis_test_{worker_id}"
    os.environ.setdefault("REDIS_VERSION", "edge")

    compose = DockerCompose(
        context="tests",
        compose_file_name="docker-compose.yml",
        pull=True,
    )
    compose.start()

    yield compose

    compose.stop()


# @pytest.fixture(scope="session")
# def redis_url() -> str:
#     return os.getenv("DEFAULT_REDIS_URI", "redis://localhost:6379")


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Use the `DockerCompose` fixture to get host/port of the 'redis' service
    on container port 6379 (mapped to an ephemeral port on the host).
    """
    host, port = redis_container.get_service_host_and_port("redis", 6379)
    return f"redis://{host}:{port}"


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
    client = Redis.from_url(redis_url)
    await client.flushall()
    await client.aclose()  # type: ignore[attr-defined]


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
