import json
from concurrent.futures import ThreadPoolExecutor
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
from langgraph.checkpoint.redis import RedisSaver
from tests.conftest import DEFAULT_REDIS_URI


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
async def clear_test_redis() -> None:
    """Clear Redis before each test."""
    client = Redis.from_url("redis://localhost:6379")
    try:
        client.flushall()
    finally:
        client.close()


@contextmanager
def _saver() -> Any:
    """Fixture for regular saver testing."""
    saver = RedisSaver(DEFAULT_REDIS_URI)
    saver.setup()
    try:
        yield saver
    finally:
        pass


@pytest.mark.parametrize(
    "query, expected_count",
    [
        ({"source": "input"}, 1),  # Matches metadata.source alias
        ({"step": 1}, 1),  # Matches metadata.step alias
        ({}, 3),  # Retrieve all checkpoints
        ({"source": "update", "step": 1}, 0),  # No matches
    ],
)
def test_search(
    query: dict[str, Any], expected_count: int, test_data: dict[str, list[Any]]
) -> None:
    with _saver() as saver:
        configs = test_data["configs"]
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        saver.put(configs[0], checkpoints[0], metadata[0], {})
        saver.put(configs[1], checkpoints[1], metadata[1], {})
        saver.put(configs[2], checkpoints[2], metadata[2], {})

        search_results = list(saver.list(None, filter=query))
        assert len(search_results) == expected_count


@pytest.mark.parametrize(
    "key, value",
    [
        ("my_key", "\x00abc"),  # Null character in value
    ],
)
def test_null_chars(key: str, value: str, test_data: dict[str, list[Any]]) -> None:
    with _saver() as saver:
        config = saver.put(
            test_data["configs"][0],
            test_data["checkpoints"][0],
            {key: value},
            {},
        )

        result = saver.get_tuple(config)
        assert result is not None, "Checkpoint not found in Redis"
        sanitized_key = key.replace("\x00", "")
        sanitized_value = value.replace("\x00", "")
        assert result.metadata[sanitized_key] == sanitized_value


def test_put_writes(test_data: dict[str, list[Any]]) -> None:
    """Test storing writes in Redis."""
    with _saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # First store a checkpoint
        saved_config = saver.put(config, checkpoint, metadata, {})

        # Test regular writes
        writes = [("channel1", "value1"), ("channel2", "value2")]
        task_id = "task1"
        saver.put_writes(saved_config, writes, task_id)

        # Test special writes (using WRITES_IDX_MAP)
        special_writes = [("__error__", "error_value"), ("channel3", "value3")]
        task_id2 = "task2"
        saver.put_writes(saved_config, special_writes, task_id2)

        # Verify writes through get_tuple
        result = saver.get_tuple(saved_config)
        assert result is not None
        assert len(result.pending_writes) > 0

        # Verify regular writes
        found_writes = {(w[1], w[2]) for w in result.pending_writes}
        assert ("channel1", "value1") in found_writes
        assert ("channel2", "value2") in found_writes

        # Verify special writes
        assert ("__error__", "error_value") in found_writes


def test_put_writes_json_structure(test_data: dict[str, list[Any]]) -> None:
    """Test that writes are properly stored in Redis JSON format."""
    with _saver() as saver:
        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # First store a checkpoint to get proper config
        saved_config = saver.put(config, checkpoint, metadata, {})

        writes = [("channel1", "value1")]
        task_id = "task1"

        # Store write
        saver.put_writes(saved_config, writes, task_id)

        # Verify JSON structure directly
        write_key = f"{saver.index_prefix}:writes:{saved_config['configurable']['thread_id']}:{saved_config['configurable'].get('checkpoint_ns', '')}:{saved_config['configurable']['checkpoint_id']}"

        # Get raw JSON
        json_data = saver._redis.json().get(write_key)

        # Verify structure
        assert json_data["thread_id"] == saved_config["configurable"]["thread_id"]
        assert len(json_data["writes"]) == 1
        assert json_data["writes"][0]["channel"] == "channel1"
        assert json_data["writes"][0]["task_id"] == task_id


def test_search_writes() -> None:
    """Test searching writes using Redis Search."""
    with _saver() as saver:
        # Set up some test data
        # Create initial config with checkpoint and metadata
        config = {"configurable": {"thread_id": "thread1", "checkpoint_ns": "ns1"}}
        checkpoint = empty_checkpoint()  # Need to import this
        metadata = {"source": "test", "step": 1, "writes": {}}

        # Store checkpoint to get proper config
        saved_config = saver.put(config, checkpoint, metadata, {})

        # Add writes for multiple channels
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        saver.put_writes(saved_config, writes1, "task1")

        writes2 = [("channel1", "value3")]
        saver.put_writes(saved_config, writes2, "task2")

        # Search by channel
        query = "(@channel:{channel1})"
        results = saver.writes_index.search(query)
        assert len(results.docs) == 1  # One document containing channel1 writes
        doc_data = json.loads(results.docs[0].json)
        channel1_writes = [w for w in doc_data["writes"] if w["channel"] == "channel1"]
        assert len(channel1_writes) == 2  # Should find both writes for channel1

        # Search by task
        query = "(@task_id:{task1})"
        results = saver.writes_index.search(query)
        assert len(results.docs) == 1  # One document containing task1's writes
        doc_data = json.loads(results.docs[0].json)
        task1_writes = [w for w in doc_data["writes"] if w["task_id"] == "task1"]
        assert len(task1_writes) == 2  # task1 made two writes

        # Search by thread/namespace
        query = "(@thread_id:{thread1} @checkpoint_ns:{ns1})"
        results = saver.writes_index.search(query)
        assert len(results.docs) == 1  # One document for this thread/ns
        doc_data = json.loads(results.docs[0].json)
        assert len(doc_data["writes"]) == 3  # Contains all three writes


def test_from_conn_string_with_url() -> None:
    """Test creating a RedisSaver with a connection URL."""
    with RedisSaver.from_conn_string(
        "redis://localhost:6379", index_prefix="test_prefix"
    ) as saver:
        saver.setup()
        # Verify connection works by creating and checking a key
        saver._redis.set("test_key", "test_value")
        assert saver._redis.get("test_key") == b"test_value"

        # Verify prefix was set
        assert saver.index_prefix == "test_prefix"


def test_from_conn_string_with_client() -> None:
    """Test creating a RedisSaver with an existing Redis client."""
    client = Redis.from_url("redis://localhost:6379")
    try:
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            saver.setup()
            # Verify connection works
            saver._redis.set("test_key2", "test_value")
            assert saver._redis.get("test_key2") == b"test_value"

            # Verify default prefix
            assert saver.index_prefix == "checkpoint"
    finally:
        client.close()


def test_from_conn_string_with_connection_args() -> None:
    """Test creating a RedisSaver with connection arguments."""
    # Test that decode_responses is propagated to Redis
    with RedisSaver.from_conn_string(
        redis_url="redis://localhost:6379", connection_args={"decode_responses": True}
    ) as saver:
        saver.setup()
        # Check the connection parameter was passed through
        assert saver._redis.connection_pool.connection_kwargs["decode_responses"]

        # Functional test - we should get str not bytes back
        saver._redis.set("test_key", "test_value")
        value = saver._redis.get("test_key")
        assert isinstance(value, str)  # not bytes


def test_from_conn_string_cleanup() -> None:
    """Test proper cleanup of Redis connections."""
    # Test with auto-created client
    with RedisSaver.from_conn_string("redis://localhost:6379") as s:
        saver_redis = s._redis
        # Verify it works during context
        assert saver_redis.ping()

    # Test with provided client
    client = Redis.from_url("redis://localhost:6379")
    try:
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            # Verify both use the same client
            assert saver._redis is client
            assert saver._redis.ping()
        # Verify client still works after context
        assert client.ping()
    finally:
        client.close()


def test_from_conn_string_errors() -> None:
    """Test error conditions for from_conn_string."""
    # Test with neither URL nor client provided
    with pytest.raises(
        ValueError, match="Either redis_url or redis_client must be provided"
    ):
        with RedisSaver.from_conn_string() as _:
            pass

    # Test with empty URL
    with pytest.raises(ValueError, match="REDIS_URL env var not set"):
        with RedisSaver.from_conn_string("") as _:
            pass

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        with RedisSaver.from_conn_string("redis://nonexistent:6379") as _:
            pass

    # Test with non-responding client
    client = Redis(host="nonexistent", port=6379)
    with pytest.raises(RedisConnectionError):
        with RedisSaver.from_conn_string(redis_client=client) as _:
            pass


def test_large_batches(test_data: dict[str, Any]) -> None:
    """Test handling large numbers of operations with thread pool."""
    with RedisSaver.from_conn_string(DEFAULT_REDIS_URI) as saver:
        saver.setup()

        N = 10  # Number of operations per batch
        M = 5  # Number of batches

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Store configs and futures for debugging
            stored_configs = []
            futures = []

            for m in range(M):
                for i in range(N):
                    test_config: RunnableConfig = {
                        "configurable": {
                            "thread_id": f"thread-{m}-{i}",
                            "checkpoint_ns": "",
                            "checkpoint_id": f"checkpoint-{i}",
                        }
                    }
                    stored_configs.append(test_config)
                    futures.append(
                        executor.submit(
                            saver.put,
                            test_config,
                            test_data["checkpoints"][0],
                            test_data["metadata"][0],
                            {},
                        )
                    )

            # Get results from puts
            put_results = [future.result() for future in futures]
            assert len(put_results) == M * N

            # Verify using configs returned from put
            verify_futures = []
            for result_config in put_results:
                verify_futures.append(executor.submit(saver.get_tuple, result_config))

            # Get verification results and add debug output
            verify_results = [future.result() for future in verify_futures]

            # Debug output for failures
            for i, result in enumerate(verify_results):
                if result is None:
                    print(f"Failed to retrieve checkpoint {i}")
                    print(f"Put config: {stored_configs[i]}")
                    print(f"Result config: {put_results[i]}")

            assert len(verify_results) == M * N
            assert all(r is not None for r in verify_results)
