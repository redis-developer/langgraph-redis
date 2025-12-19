from contextlib import contextmanager
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from langgraph.checkpoint.redis.base import (
    CHECKPOINT_BLOB_PREFIX,
    CHECKPOINT_WRITE_PREFIX,
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
        with ShallowRedisSaver.from_conn_string() as saver:
            saver.setup()

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        with ShallowRedisSaver.from_conn_string("redis://nonexistent:6379") as saver:
            saver.setup()

    # Test with non-responding client
    client = Redis(host="nonexistent", port=6379)
    with pytest.raises(RedisConnectionError):
        with ShallowRedisSaver.from_conn_string(redis_client=client) as saver:
            saver.setup()

    # Test with empty URL
    # Handle both old and new RedisVL error message formats
    with pytest.raises(
        ValueError, match="REDIS_URL (env var|environment variable) not set"
    ):
        with ShallowRedisSaver.from_conn_string("") as saver:
            saver.setup()


def test_shallow_client_info_setting(redis_url: str, monkeypatch) -> None:
    """Test that ShallowRedisSaver sets client info correctly."""

    from langgraph.checkpoint.redis.version import __full_lib_name__

    # Create a mock to track if client_setinfo was called with our library name
    client_info_called = False
    original_client_setinfo = Redis.client_setinfo

    def mock_client_setinfo(self, key, value):
        nonlocal client_info_called
        # Note: RedisVL might call this with its own lib name first
        # We only track calls with our full lib name
        if key == "LIB-NAME" and __full_lib_name__ in value:
            client_info_called = True
        return original_client_setinfo(self, key, value)

    # Apply the mock
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)

    # Test client info setting when creating a new shallow saver
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        pass

    # Verify client_setinfo was called with our library info
    assert client_info_called, "client_setinfo was not called with our library name"


def test_shallow_client_info_fallback(redis_url: str, monkeypatch) -> None:
    """Test that ShallowRedisSaver falls back to echo when client_setinfo is not available."""
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
        with ShallowRedisSaver.from_conn_string(redis_client=client) as saver:
            # Force another call to set_client_info
            saver.set_client_info()

        # Print debug info
        print(f"Echo messages seen: {echo_messages}")

        # Verify echo was called as fallback with our library info
        assert echo_called, "echo was not called as fallback with our library name"
    finally:
        client.close()


def test_key_generation_inconsistency(redis_url: str) -> None:
    """Test for Key generation consistency between base and shallow savers.

    This test verifies that both the base class and shallow saver cleanup patterns
    use the same storage-safe transformations, ensuring cleanup works correctly.
    """
    thread_id = "test_thread"
    checkpoint_ns = ""  # Empty namespace - the problematic case
    channel = "test_channel"
    version = "1"

    # Create a saver instance to test key generation
    with _saver(redis_url) as saver:
        # Test blob key generation
        base_blob_key = saver._make_redis_checkpoint_blob_key(
            thread_id, checkpoint_ns, channel, version
        )
        shallow_blob_pattern = (
            ShallowRedisSaver._make_shallow_redis_checkpoint_blob_key_pattern(
                thread_id, checkpoint_ns
            )
        )

        # The base key uses storage-safe transformations
        expected_base_key = (
            f"{CHECKPOINT_BLOB_PREFIX}:test_thread:__empty__:test_channel:1"
        )
        assert base_blob_key == expected_base_key

        # The shallow pattern now uses storage-safe transformations (fixed!)
        expected_pattern = f"{CHECKPOINT_BLOB_PREFIX}:test_thread:__empty__:*"
        assert shallow_blob_pattern == expected_pattern

        # Both base key and pattern now consistently use "__empty__" (fix confirmed!)
        assert "__empty__" in base_blob_key
        assert "__empty__" in shallow_blob_pattern

        # Test writes key generation
        checkpoint_id = "test_checkpoint"
        task_id = "test_task"

        base_writes_key = saver._make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, task_id, 0
        )
        shallow_writes_pattern = (
            ShallowRedisSaver._make_shallow_redis_checkpoint_writes_key_pattern(
                thread_id, checkpoint_ns
            )
        )

        # The base key uses storage-safe transformations
        expected_base_key = f"{CHECKPOINT_WRITE_PREFIX}:test_thread:__empty__:test_checkpoint:test_task:0"
        assert base_writes_key == expected_base_key

        # The shallow pattern now uses storage-safe transformations (fixed!)
        expected_pattern = f"{CHECKPOINT_WRITE_PREFIX}:test_thread:__empty__:*"
        assert shallow_writes_pattern == expected_pattern

        # Both base key and pattern now consistently use "__empty__" (fix confirmed!)
        assert "__empty__" in base_writes_key
        assert "__empty__" in shallow_writes_pattern


def test_shallow_saver_inline_storage(redis_url: str) -> None:
    """Test that ShallowRedisSaver stores channel values inline.

    This test verifies that the shallow saver stores channel values
    inline within the checkpoint document rather than as separate blob keys.
    """
    import uuid

    from langgraph.checkpoint.redis.base import (
        CHECKPOINT_BLOB_PREFIX,
        CHECKPOINT_PREFIX,
    )

    with _saver(redis_url) as saver:
        # Create test data
        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""  # Empty namespace - problematic case
        checkpoint_id1 = str(uuid.uuid4())
        checkpoint_id2 = str(uuid.uuid4())

        config1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id1,
            }
        }

        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id2,
            }
        }

        # Create first checkpoint
        checkpoint1 = {
            "v": 1,
            "ts": "2023-01-01T00:00:00Z",
            "id": checkpoint_id1,
            "channel_values": {
                "test_channel": ["test_value"],
                "another_channel": {"key": "value"},
            },
            "channel_versions": {"test_channel": "1", "another_channel": "2"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata1: CheckpointMetadata = {"source": "input", "step": 1}
        versions1 = {"test_channel": "1", "another_channel": "2"}

        saver.put(config1, checkpoint1, metadata1, versions1)

        # Check keys after first checkpoint
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        try:
            all_keys = redis_client.keys("*")
            test_keys = [k for k in all_keys if thread_id in k]
            blob_keys_after_first = [
                k for k in test_keys if k.startswith(CHECKPOINT_BLOB_PREFIX)
            ]

            # Should have exactly one checkpoint key and no blob keys
            checkpoint_keys = [k for k in test_keys if k.startswith(CHECKPOINT_PREFIX)]
            assert (
                len(checkpoint_keys) == 1
            ), f"Expected 1 checkpoint key, got {len(checkpoint_keys)}"
            assert (
                len(blob_keys_after_first) == 0
            ), f"Expected 0 blob keys, got {len(blob_keys_after_first)}"

            # Verify channel values are stored inline
            checkpoint_data = redis_client.json().get(checkpoint_keys[0])
            assert "checkpoint" in checkpoint_data
            assert "channel_values" in checkpoint_data["checkpoint"]
            assert "test_channel" in checkpoint_data["checkpoint"]["channel_values"]
            assert "another_channel" in checkpoint_data["checkpoint"]["channel_values"]

            # Create second checkpoint with different channel versions
            checkpoint2 = {
                "v": 1,
                "ts": "2023-01-01T00:00:01Z",
                "id": checkpoint_id2,
                "channel_values": {
                    "test_channel": ["test_value_new"],
                    "another_channel": {"key": "value_new"},
                },
                "channel_versions": {"test_channel": "3", "another_channel": "4"},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata2: CheckpointMetadata = {"source": "loop", "step": 2}
            versions2 = {
                "test_channel": "3",
                "another_channel": "4",
            }  # Different versions

            saver.put(config2, checkpoint2, metadata2, versions2)

            # Check keys after second checkpoint
            all_keys = redis_client.keys("*")
            test_keys = [k for k in all_keys if thread_id in k]
            blob_keys_after_second = [
                k for k in test_keys if k.startswith(CHECKPOINT_BLOB_PREFIX)
            ]

            # Still should have exactly one checkpoint key (overwritten) and no blob keys
            checkpoint_keys_after_second = [
                k for k in test_keys if k.startswith(CHECKPOINT_PREFIX)
            ]
            assert (
                len(checkpoint_keys_after_second) == 1
            ), f"Expected 1 checkpoint key, got {len(checkpoint_keys_after_second)}"
            assert (
                len(blob_keys_after_second) == 0
            ), f"Expected 0 blob keys, got {len(blob_keys_after_second)}"

            # Verify the checkpoint contains the new data
            checkpoint_data = redis_client.json().get(checkpoint_keys_after_second[0])
            assert checkpoint_data["checkpoint_id"] == checkpoint_id2
            assert "test_value_new" in str(
                checkpoint_data["checkpoint"]["channel_values"]["test_channel"]
            )

        finally:
            redis_client.close()


def test_pr37_writes_persist_for_hitl_support(redis_url: str) -> None:
    """Test for PR #37 updated for Issue #133: Writes persist across checkpoints for HITL.

    This test verifies that writes are NOT cleaned up when new checkpoints are saved.
    This is necessary to support Human-in-the-Loop (HITL) workflows where interrupt
    writes are saved BEFORE the new checkpoint is created.

    Writes are cleaned up via:
    1. delete_thread - explicitly cleans up all data for a thread
    2. TTL expiration - if configured
    3. Overwrite - when put_writes is called with the same task_id and idx

    See Issue #133 for details on why this behavior is required.
    """
    import uuid

    from langgraph.checkpoint.redis.base import CHECKPOINT_WRITE_PREFIX

    with _saver(redis_url) as saver:
        # Create test data
        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""  # Empty namespace
        checkpoint_id1 = str(uuid.uuid4())
        checkpoint_id2 = str(uuid.uuid4())

        config1: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id1,
            }
        }

        config2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id2,
            }
        }

        # Create first checkpoint
        checkpoint1 = {
            "v": 1,
            "ts": "2023-01-01T00:00:00Z",
            "id": checkpoint_id1,
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata1: CheckpointMetadata = {"source": "input", "step": 1}

        saver.put(config1, checkpoint1, metadata1, {})

        # Add writes for first checkpoint
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        saver.put_writes(config1, writes1, "task1")

        # Check writes after first checkpoint
        redis_client = Redis.from_url(redis_url, decode_responses=True)
        try:
            all_keys = redis_client.keys("*")
            test_keys = [k for k in all_keys if thread_id in k]
            write_keys_after_first = [
                k for k in test_keys if k.startswith(CHECKPOINT_WRITE_PREFIX)
            ]

            # Should have 2 write keys
            assert len(write_keys_after_first) == 2

            # Create second checkpoint
            checkpoint2 = {
                "v": 1,
                "ts": "2023-01-01T00:00:01Z",
                "id": checkpoint_id2,
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            metadata2: CheckpointMetadata = {"source": "loop", "step": 2}

            saver.put(config2, checkpoint2, metadata2, {})

            # Add writes for second checkpoint
            writes2 = [("channel3", "value3"), ("channel4", "value4")]
            saver.put_writes(config2, writes2, "task2")

            # Check writes after second checkpoint
            all_keys = redis_client.keys("*")
            test_keys = [k for k in all_keys if thread_id in k]
            write_keys_after_second = [
                k for k in test_keys if k.startswith(CHECKPOINT_WRITE_PREFIX)
            ]

            print(f"Write keys after first checkpoint: {len(write_keys_after_first)}")
            print(f"Write keys after second checkpoint: {len(write_keys_after_second)}")
            print("Write keys after second checkpoint:")
            for key in sorted(write_keys_after_second):
                print(f"  {key}")

            # Issue #133 fix: Writes now persist across checkpoint updates to support HITL.
            # We expect all 4 writes to still exist (2 from checkpoint1 + 2 from checkpoint2)
            assert len(write_keys_after_second) == 4, (
                f"Writes should persist across checkpoints for HITL support. "
                f"Expected 4, got {len(write_keys_after_second)}"
            )

            # Verify that delete_thread properly cleans up writes
            saver.delete_thread(thread_id)

            all_keys = redis_client.keys("*")
            test_keys = [k for k in all_keys if thread_id in k]
            write_keys_after_delete = [
                k for k in test_keys if k.startswith(CHECKPOINT_WRITE_PREFIX)
            ]

            assert len(write_keys_after_delete) == 0, (
                f"delete_thread should clean up all writes. "
                f"Expected 0, got {len(write_keys_after_delete)}"
            )

        finally:
            redis_client.close()
