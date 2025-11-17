import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, List, Literal
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from langgraph.checkpoint.redis import BaseRedisSaver, RedisSaver


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
    """Fixture for regular saver testing."""
    saver = RedisSaver(redis_url)
    saver.setup()
    try:
        yield saver
    finally:
        pass


@pytest.mark.parametrize(
    "query, expected_count",
    [
        ({"source": "input"}, 1),  # search by 1 key
        ({"step": 1}, 1),  # search by multiple keys - , "writes": {"foo": "bar"}
        ({}, 3),  # search by no keys, return all checkpoints
        ({"source": "update", "step": 1}, 0),  # no matches
    ],
)
def test_search(
    query: dict[str, Any],
    expected_count: int,
    test_data: dict[str, list[Any]],
    redis_url: str,
) -> None:
    with _saver(redis_url) as saver:
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
def test_null_chars(
    key: str, value: str, test_data: dict[str, list[Any]], redis_url: str
) -> None:
    with _saver(redis_url) as saver:
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


def test_put_writes(test_data: dict[str, list[Any]], redis_url: str) -> None:
    """Test storing writes in Redis."""
    with _saver(redis_url) as saver:
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


def test_put_writes_json_structure(
    test_data: dict[str, list[Any]], redis_url: str
) -> None:
    """Test that writes are properly stored in Redis JSON format."""
    with _saver(redis_url) as saver:
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
        thread_id = saved_config["configurable"]["thread_id"]
        checkpoint_ns = saved_config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]

        write_key = BaseRedisSaver._make_redis_checkpoint_writes_key(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            WRITES_IDX_MAP.get("channel1", 0),
        )

        # Get raw JSON
        json_data = saver._redis.json().get(write_key)

        # Verify structure
        assert json_data["thread_id"] == saved_config["configurable"]["thread_id"]
        assert json_data["channel"] == "channel1"
        assert json_data["task_id"] == task_id


def test_search_writes(redis_url: str) -> None:
    """Test searching writes using Redis Search."""
    with _saver(redis_url) as saver:
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
        results = saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 2  # One document per channel1 writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)

        assert doc1["channel"] == doc2["channel"] == "channel1"

        # Search by task
        query = "(@task_id:{task1})"
        results = saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 2  # One document per channel1 writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)

        assert doc1["task_id"] == doc2["task_id"] == "task1"

        # Search by thread/namespace
        query = "(@thread_id:{thread1} @checkpoint_ns:{ns1})"
        results = saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 3  # Contains all three writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)
        doc3 = json.loads(results.docs[2].json)

        # Blobs are now base64-encoded in Redis (checkpoint 3.0)
        assert base64.b64decode(doc1["blob"]).decode() == '"value1"'
        assert base64.b64decode(doc2["blob"]).decode() == '"value2"'
        assert base64.b64decode(doc3["blob"]).decode() == '"value3"'


def test_from_conn_string_with_url(redis_url: str) -> None:
    """Test creating a RedisSaver with a connection URL."""
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        # Verify connection works by creating and checking a key
        saver._redis.set("test_key", "test_value")
        assert saver._redis.get("test_key") == b"test_value"


def test_from_conn_string_with_client(redis_url: str) -> None:
    """Test creating a RedisSaver with an existing Redis client."""
    client = Redis.from_url(redis_url)
    try:
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            saver.setup()
            # Verify connection works
            saver._redis.set("test_key2", "test_value")
            assert saver._redis.get("test_key2") == b"test_value"
    finally:
        client.close()


def test_from_conn_string_with_connection_args(redis_url: str) -> None:
    """Test creating a RedisSaver with connection arguments."""
    # Test that decode_responses is propagated to Redis
    with RedisSaver.from_conn_string(
        redis_url=redis_url, connection_args={"decode_responses": True}
    ) as saver:
        saver.setup()
        # Check the connection parameter was passed through
        assert saver._redis.connection_pool.connection_kwargs["decode_responses"]

        # Functional test - we should get str not bytes back
        saver._redis.set("test_key", "test_value")
        value = saver._redis.get("test_key")
        assert isinstance(value, str)  # not bytes


def test_from_conn_string_cleanup(redis_url: str) -> None:
    """Test proper cleanup of Redis connections."""
    # Test with auto-created client
    with RedisSaver.from_conn_string(redis_url) as s:
        saver_redis = s._redis
        # Verify it works during context
        assert saver_redis.ping()

    # Test with provided client
    client = Redis.from_url(redis_url)
    try:
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            # Verify both use the same client
            assert saver._redis is client
            assert saver._redis.ping()
        # Verify client still works after context
        assert client.ping()
    finally:
        client.close()


def test_client_info_setting(redis_url: str, monkeypatch) -> None:
    """Test that client_setinfo is called with correct library information."""
    from langgraph.checkpoint.redis.version import __full_lib_name__

    # Create a mock to track if client_setinfo was called with our library name
    client_info_called = False
    lib_calls = []
    original_client_setinfo = Redis.client_setinfo

    def mock_client_setinfo(self, key, value):
        nonlocal client_info_called, lib_calls
        if key == "LIB-NAME":
            lib_calls.append(value)
            # Note: RedisVL might call this with its own lib name first
            # We only track calls with our lib name
            if __full_lib_name__ in value:
                client_info_called = True
        return original_client_setinfo(self, key, value)

    # Apply the mock
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)

    # Test client info setting when creating a new saver
    with RedisSaver.from_conn_string(redis_url) as saver:
        # Call set_client_info directly to ensure it's called
        saver.set_client_info()

    # Print debug info
    print(f"Library name values seen: {lib_calls}")

    # Verify client_setinfo was called with our library info
    assert client_info_called, "client_setinfo was not called with our library name"


def test_client_info_fallback_to_echo(redis_url: str, monkeypatch) -> None:
    """Test that when client_setinfo is not available, fall back to echo."""
    from redis.exceptions import ResponseError

    from langgraph.checkpoint.redis.version import __full_lib_name__

    # Create a Redis client directly first - this bypasses RedisVL validation
    client = Redis.from_url(redis_url)

    # Remove client_setinfo to simulate older Redis version
    def mock_client_setinfo(self, key, value):
        raise ResponseError("ERR unknown command")

    # Track if echo was called with our lib name
    echo_called = False
    echo_messages = []
    original_echo = Redis.echo

    def mock_echo(self, message):
        nonlocal echo_called, echo_messages
        echo_messages.append(message)
        if __full_lib_name__ in message:
            echo_called = True
        return original_echo(self, message)

    # Apply the mocks
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    monkeypatch.setattr(Redis, "echo", mock_echo)

    try:
        # Test direct fallback without RedisVL interference
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            # Force another call to set_client_info
            saver.set_client_info()

        # Print debug info
        print(f"Echo messages seen: {echo_messages}")

        # Verify echo was called as fallback with our library info
        assert echo_called, "echo was not called as fallback with our library name"
    finally:
        client.close()


def test_client_info_graceful_failure(redis_url: str, monkeypatch) -> None:
    """Test graceful failure when both client_setinfo and echo fail."""
    from redis.exceptions import ResponseError

    # Create a Redis client directly first - this bypasses RedisVL validation
    client = Redis.from_url(redis_url)

    # Simulate failures for both methods
    def mock_client_setinfo(self, key, value):
        raise ResponseError("ERR unknown command")

    def mock_echo(self, message):
        raise ResponseError("ERR broken connection")

    # Apply the mocks
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    monkeypatch.setattr(Redis, "echo", mock_echo)

    try:
        # Should not raise any exceptions when both methods fail
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            # Explicitly call set_client_info which should handle the errors
            saver.set_client_info()
            pass
    except Exception as e:
        assert False, f"set_client_info did not handle failure gracefully: {e}"
    finally:
        client.close()


def test_from_conn_string_errors() -> None:
    """Test error conditions for from_conn_string."""
    # Test with neither URL nor client provided
    with pytest.raises(
        ValueError, match="Either redis_url or redis_client must be provided"
    ):
        with RedisSaver.from_conn_string() as saver:
            saver.setup()

    # Test with empty URL
    # Handle both old and new RedisVL error message formats
    with pytest.raises(
        ValueError, match="REDIS_URL (env var|environment variable) not set"
    ):
        with RedisSaver.from_conn_string("") as saver:
            saver.setup()

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        with RedisSaver.from_conn_string("redis://nonexistent:6379") as saver:
            saver.setup()

    # Test with non-responding client
    client = Redis(host="nonexistent", port=6379)
    with pytest.raises(RedisConnectionError):
        with RedisSaver.from_conn_string(redis_client=client) as saver:
            saver.setup()


def test_large_batches(test_data: dict[str, Any], redis_url: str) -> None:
    """Test handling large numbers of operations with thread pool."""
    with RedisSaver.from_conn_string(redis_url) as saver:
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


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


@pytest.fixture
def tools() -> List[BaseTool]:
    return [get_weather]


@pytest.fixture
def mock_llm() -> Any:
    """Create a mock LLM for testing without requiring API keys."""
    from unittest.mock import MagicMock

    # Create a mock that can be used in place of a real LLM
    mock = MagicMock()
    mock.invoke.return_value = "This is a mock response from the LLM"
    return mock


@pytest.fixture
def mock_agent() -> Any:
    """Create a mock agent that creates checkpoints without requiring a real LLM."""
    from unittest.mock import MagicMock

    # Create a mock agent that returns a dummy response
    mock = MagicMock()

    # Set the invoke method to also create a fake chat session
    def mock_invoke(messages, config):
        # Return a dummy response that mimics a chat conversation
        return {
            "messages": [
                (
                    "human",
                    messages.get("messages", [("human", "default message")])[0][1],
                ),
                ("ai", "I'll help you with that"),
                ("tool", "get_weather"),
                ("ai", "The weather looks good"),
            ]
        }

    mock.invoke = mock_invoke
    return mock


def test_sync_redis_checkpointer(
    tools: list[BaseTool], mock_agent: Any, redis_url: str
) -> None:
    """Test RedisSaver checkpoint functionality using a mock agent."""
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        # Use the mock agent instead of creating a real one
        graph = mock_agent

        # Use a unique thread_id
        thread_id = f"test-{uuid4()}"

        # Test initial query
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "",
            }
        }

        # Create a checkpoint manually to simulate what would happen during agent execution
        checkpoint = {
            "id": str(uuid4()),
            "ts": str(int(time.time())),
            "v": 1,
            "channel_values": {
                "messages": [
                    ("human", "what's the weather in sf?"),
                    ("ai", "I'll check the weather for you"),
                    ("tool", "get_weather(city='sf')"),
                    ("ai", "It's always sunny in sf"),
                ]
            },
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        # Store the checkpoint
        next_config = checkpointer.put(
            config, checkpoint, {"source": "test", "step": 1}, {"messages": "1"}
        )

        # Verify next_config has the right structure
        assert "configurable" in next_config
        assert "thread_id" in next_config["configurable"]

        # Test checkpoint retrieval
        latest = checkpointer.get(config)

        assert latest is not None
        assert all(
            k in latest
            for k in [
                "v",
                "ts",
                "id",
                "channel_values",
                "channel_versions",
                "versions_seen",
            ]
        )
        assert "messages" in latest["channel_values"]
        assert isinstance(latest["channel_values"]["messages"], list)

        # Test checkpoint tuple
        tuple_result = checkpointer.get_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint["id"] == latest["id"]

        # Test listing checkpoints
        checkpoints = list(checkpointer.list(config))
        assert len(checkpoints) > 0
        assert checkpoints[-1].checkpoint["id"] == latest["id"]


def test_root_graph_checkpoint(
    tools: list[BaseTool], mock_agent: Any, redis_url: str
) -> None:
    """
    A regression test for a bug where queries for checkpoints from the
    root graph were failing to find valid checkpoints. When called from
    a root graph, the `checkpoint_id` and `checkpoint_ns` keys are not
    in the config object.
    """
    with RedisSaver.from_conn_string(redis_url) as checkpointer:
        checkpointer.setup()

        # Use a unique thread_id
        thread_id = f"root-graph-{uuid4()}"

        # Create a config with checkpoint_id and checkpoint_ns
        # For a root graph test, we need to add an empty checkpoint_ns
        # since that's how real root graphs work
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",  # Empty string is valid
            }
        }

        # Create a checkpoint manually to simulate what would happen during agent execution
        checkpoint = {
            "id": str(uuid4()),
            "ts": str(int(time.time())),
            "v": 1,
            "channel_values": {
                "messages": [
                    ("human", "what's the weather in sf?"),
                    ("ai", "I'll check the weather for you"),
                    ("tool", "get_weather(city='sf')"),
                    ("ai", "It's always sunny in sf"),
                ]
            },
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        # Store the checkpoint
        next_config = checkpointer.put(
            config, checkpoint, {"source": "test", "step": 1}, {"messages": "1"}
        )

        # Verify the checkpoint was stored
        assert next_config is not None

        # Test retrieving the checkpoint with a root graph config
        # that doesn't have checkpoint_id or checkpoint_ns
        latest = checkpointer.get(config)

        # This is the key test - verify we can retrieve checkpoints
        # when called from a root graph configuration
        assert latest is not None
        assert all(
            k in latest
            for k in [
                "v",
                "ts",
                "id",
                "channel_values",
                "channel_versions",
                "versions_seen",
            ]
        )
        assert "messages" in latest["channel_values"]
        assert (
            len(latest["channel_values"]["messages"]) == 4
        )  # Initial + LLM + Tool + Final

        # Test checkpoint tuple
        tuple_result = checkpointer.get_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint == latest

        # Test listing checkpoints
        checkpoints = list(checkpointer.list(config))
        assert len(checkpoints) > 0
        assert checkpoints[-1].checkpoint["id"] == latest["id"]
