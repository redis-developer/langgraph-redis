from contextlib import contextmanager
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


@pytest.fixture
def test_data() -> dict[str, list[Any]]:
    """Test data fixture."""
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "thread_ts": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {
        "source": "input",
        "step": 2,
        "writes": {},
        "score": 1,
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
        "score": None,
    }
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.fixture(autouse=True)
async def clear_test_redis(redis_url: str) -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(redis_url)
    try:
        client.flushall()
    finally:
        client.close()


@contextmanager
def _saver(redis_url: str) -> Any:
    """Fixture for shallow saver testing."""
    saver = ShallowRedisSaver(redis_url)
    saver.setup()
    try:
        yield saver
    finally:
        pass


def test_only_latest_checkpoint(
    test_data: dict[str, list[Any]], redis_url: str
) -> None:
    """Test that only latest checkpoint is stored."""
    with _saver(redis_url) as saver:
        thread_id = "test-thread"
        checkpoint_ns = ""

        # Create initial checkpoint
        config_1 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        checkpoint_1 = test_data["checkpoints"][0]
        saver.put(config_1, checkpoint_1, test_data["metadata"][0], {})

        # Create second checkpoint
        config_2 = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        checkpoint_2 = test_data["checkpoints"][1]
        saver.put(config_2, checkpoint_2, test_data["metadata"][1], {})

        # Verify only latest checkpoint exists
        results = list(saver.list(None))
        assert len(results) == 1
        assert results[0].config["configurable"]["checkpoint_id"] == checkpoint_2["id"]


@pytest.mark.parametrize(
    "query, expected_count",
    [
        ({"source": "input"}, 1),  # Matches metadata.source
        ({"step": 1}, 1),  # Matches metadata.step
        ({}, 2),  # Retrieve latest checkpoints (one per thread)
        ({"source": "update", "step": 1}, 0),  # No matches
    ],
)
def test_search(
    query: dict[str, Any],
    expected_count: int,
    test_data: dict[str, list[Any]],
    redis_url: str,
) -> None:
    """Test search functionality."""
    with _saver(redis_url) as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        # Store checkpoints with different thread IDs
        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})

        search_results = list(saver.list(None, filter=query))
        assert len(search_results) == expected_count


def test_overwrite_writes(test_data: dict[str, list[Any]], redis_url: str) -> None:
    """Test that writes are overwritten, not appended."""
    with _saver(redis_url) as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # Store initial checkpoint
        saved_config = saver.put(config, checkpoint, metadata, {})

        # Add initial writes
        initial_writes = [("channel1", "value1")]
        saver.put_writes(saved_config, initial_writes, "task1")

        # Add more writes
        new_writes = [("channel2", "value2")]
        saver.put_writes(saved_config, new_writes, "task1")

        # Verify only latest writes exist
        result = saver.get_tuple(saved_config)
        assert result is not None
        assert len(result.pending_writes) == 1
        assert result.pending_writes[0] == ("task1", "channel2", "value2")


@pytest.mark.parametrize(
    "key, value",
    [
        ("my_key", "\x00abc"),  # Null character in value
    ],
)
def test_null_chars(
    key: str, value: str, test_data: dict[str, list[Any]], redis_url: str
) -> None:
    """Test handling of null characters."""
    with _saver(redis_url) as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {key: value},
            {},
        )

        result = saver.get_tuple(config)
        assert result is not None
        sanitized_key = key.replace("\x00", "")
        sanitized_value = value.replace("\x00", "")
        print(f"sanitized_key = {sanitized_key}, sanitized_value = {sanitized_value}")
        print(f"result.metadata ==> {result.metadata}   ")
        assert result.metadata[sanitized_key] == sanitized_value


def test_from_conn_string_with_url(redis_url: str) -> None:
    """Test creating ShallowRedisSaver with connection URL."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        saver._redis.set("test_key", "test_value")
        assert saver._redis.get("test_key") == b"test_value"


def test_from_conn_string_with_client(redis_url: str) -> None:
    """Test creating ShallowRedisSaver with existing client."""
    client = Redis.from_url(redis_url)
    try:
        with ShallowRedisSaver.from_conn_string(redis_client=client) as saver:
            saver.setup()
            saver._redis.set("test_key2", "test_value")
            assert saver._redis.get("test_key2") == b"test_value"
    finally:
        client.close()


def test_from_conn_string_with_connection_args(redis_url: str) -> None:
    """Test creating ShallowRedisSaver with connection arguments."""
    with ShallowRedisSaver.from_conn_string(
        redis_url=redis_url, connection_args={"decode_responses": True}
    ) as saver:
        saver.setup()
        assert saver._redis.connection_pool.connection_kwargs["decode_responses"]

        saver._redis.set("test_key", "test_value")
        value = saver._redis.get("test_key")
        assert isinstance(value, str)


def test_from_conn_string_errors(redis_url: str) -> None:
    """Test proper cleanup of Redis connections."""
    with ShallowRedisSaver.from_conn_string(redis_url) as s:
        saver_redis = s._redis
        assert saver_redis.ping()

    client = Redis.from_url(redis_url)
    try:
        with ShallowRedisSaver.from_conn_string(redis_client=client) as saver:
            assert saver._redis is client
            assert saver._redis.ping()
        assert client.ping()
    finally:
        client.close()

    """Test error conditions for from_conn_string."""
    # Test with neither URL nor client provided
    with pytest.raises(
        ValueError, match="Either redis_url or redis_client must be provided"
    ):
        with ShallowRedisSaver.from_conn_string() as _:
            pass

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        with ShallowRedisSaver.from_conn_string("redis://nonexistent:6379") as _:
            pass

    # Test with non-responding client
    client = Redis(host="nonexistent", port=6379)
    with pytest.raises(RedisConnectionError):
        with ShallowRedisSaver.from_conn_string(redis_client=client) as _:
            pass

    # Test with empty URL
    with pytest.raises(ValueError, match="REDIS_URL env var not set"):
        with ShallowRedisSaver.from_conn_string("") as _:
            pass
