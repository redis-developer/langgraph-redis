from typing import Any, AsyncGenerator, Dict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver


@pytest.fixture
async def test_data() -> Dict[str, Any]:
    """Test data fixture."""
    config_1: RunnableConfig = RunnableConfig(
        {
            "configurable": {
                "thread_id": "thread-1",
                "thread_ts": "1",
                "checkpoint_ns": "",
            }
        }
    )
    config_2: RunnableConfig = RunnableConfig(
        {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
    )
    config_3: RunnableConfig = RunnableConfig(
        {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }
    )

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


@pytest.fixture
async def saver(redis_url: str) -> AsyncGenerator[AsyncShallowRedisSaver, None]:
    """AsyncShallowRedisSaver fixture."""
    saver = AsyncShallowRedisSaver(redis_url)
    await saver.asetup()
    yield saver


@pytest.mark.asyncio
async def test_only_latest_checkpoint(
    saver: AsyncShallowRedisSaver, test_data: Dict[str, Any]
) -> None:
    """Test that only the latest checkpoint is stored."""
    thread_id = "test-thread"
    checkpoint_ns = ""

    # Create initial checkpoint
    config_1 = RunnableConfig(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
    )
    checkpoint_1 = test_data["checkpoints"][0]
    await saver.aput(config_1, checkpoint_1, test_data["metadata"][0], {})

    # Create second checkpoint
    config_2 = RunnableConfig(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
    )
    checkpoint_2 = test_data["checkpoints"][1]
    await saver.aput(config_2, checkpoint_2, test_data["metadata"][1], {})

    # Verify only latest checkpoint exists
    results = [c async for c in saver.alist(None)]
    assert len(results) == 1
    assert results[0].config["configurable"]["checkpoint_id"] == checkpoint_2["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_count",
    [
        ({"source": "input"}, 1),  # Matches metadata.source
        ({"step": 1}, 1),  # Matches metadata.step
        ({}, 3),  # Retrieve all checkpoints
        ({"source": "update"}, 0),  # No matches
    ],
)
async def test_search(
    saver: AsyncShallowRedisSaver,
    test_data: Dict[str, Any],
    query: Dict[str, Any],
    expected_count: int,
) -> None:
    """Test search functionality."""
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    await saver.aput(configs[0], checkpoints[0], metadata[0], {})
    await saver.aput(configs[1], checkpoints[1], metadata[1], {})
    await saver.aput(configs[2], checkpoints[2], metadata[2], {})

    results = [c async for c in saver.alist(None, filter=query)]
    assert len(results) == expected_count


@pytest.mark.asyncio
async def test_null_chars(
    saver: AsyncShallowRedisSaver, test_data: Dict[str, Any]
) -> None:
    """Test handling of null characters."""
    config = await saver.aput(
        test_data["configs"][0],
        test_data["checkpoints"][0],
        {"source": "\x00value"},
        {},
    )

    result = await saver.aget_tuple(config)
    assert result is not None

    sanitized_value = "\x00value".replace("\x00", "")
    assert result.metadata["source"] == sanitized_value


@pytest.mark.asyncio
async def test_put_writes(
    saver: AsyncShallowRedisSaver, test_data: Dict[str, Any]
) -> None:
    """Test storing writes asynchronously."""
    config = test_data["configs"][0]
    checkpoint = test_data["checkpoints"][0]
    metadata = test_data["metadata"][0]

    saved_config = await saver.aput(config, checkpoint, metadata, {})

    writes = [("channel1", "value1"), ("channel2", "value2")]
    await saver.aput_writes(saved_config, writes, "task1")

    result = await saver.aget_tuple(saved_config)
    assert result is not None
    found_writes = {(w[1], w[2]) for w in result.pending_writes or []}
    assert ("channel1", "value1") in found_writes
    assert ("channel2", "value2") in found_writes


@pytest.mark.asyncio
async def test_sequential_writes(
    saver: AsyncShallowRedisSaver, test_data: Dict[str, Any]
) -> None:
    """Test sequential writes for consistent overwrite behavior."""
    config = test_data["configs"][0]
    checkpoint = test_data["checkpoints"][0]
    metadata = test_data["metadata"][0]

    saved_config = await saver.aput(config, checkpoint, metadata, {})

    # Add initial writes
    initial_writes = [("channel1", "value1")]
    await saver.aput_writes(saved_config, initial_writes, "task1")

    # Add more writes
    new_writes = [("channel2", "value2")]
    await saver.aput_writes(saved_config, new_writes, "task1")

    # Verify only latest writes exist
    result = await saver.aget_tuple(saved_config)
    assert result is not None
    assert result.pending_writes is not None
    assert len(result.pending_writes) == 1
    assert result.pending_writes[0] == ("task1", "channel2", "value2")


@pytest.mark.asyncio
async def test_from_conn_string_errors(redis_url: str) -> None:
    """Test proper cleanup of Redis connections."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as s:
        saver_redis = s._redis
        assert await saver_redis.ping()

    client = Redis.from_url(redis_url)
    try:
        async with AsyncShallowRedisSaver.from_conn_string(
            redis_client=client
        ) as saver:
            assert saver._redis is client
            assert await saver._redis.ping()
        assert await client.ping()
    finally:
        await client.aclose()

    """Test error conditions for from_conn_string."""
    # Test with neither URL nor client provided
    with pytest.raises(
        ValueError, match="Either redis_url or redis_client must be provided"
    ):
        async with AsyncShallowRedisSaver.from_conn_string() as saver:
            await saver.asetup()

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        async with AsyncShallowRedisSaver.from_conn_string(
            "redis://nonexistent:6379"
        ) as saver:
            await saver.asetup()

    # Test with non-responding client
    client = Redis(host="nonexistent", port=6379)
    with pytest.raises(RedisConnectionError):
        async with AsyncShallowRedisSaver.from_conn_string(
            redis_client=client
        ) as saver:
            await saver.asetup()

    # Test with empty URL
    with pytest.raises(ValueError, match="REDIS_URL env var not set"):
        async with AsyncShallowRedisSaver.from_conn_string("") as saver:
            await saver.asetup()
            
            
@pytest.mark.asyncio
async def test_async_shallow_client_info_setting(redis_url: str, monkeypatch) -> None:
    """Test that client_setinfo is called with correct library information in AsyncShallowRedisSaver."""
    from langgraph.checkpoint.redis.version import __full_lib_name__
    
    # Track if client_setinfo was called with the right parameters
    client_info_called = False
    
    # Store the original method
    original_client_setinfo = Redis.client_setinfo
    
    # Create a mock function for client_setinfo
    async def mock_client_setinfo(self, key, value):
        nonlocal client_info_called
        # Note: RedisVL might call this with its own lib name first
        # We only track calls with our full lib name
        if key == "LIB-NAME" and __full_lib_name__ in value:
            client_info_called = True
        # Call original method to ensure normal function
        return await original_client_setinfo(self, key, value)
    
    # Apply the mock
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    
    # Test client info setting when creating a new async shallow saver
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
    
    # Verify client_setinfo was called with our library info
    assert client_info_called, "client_setinfo was not called with our library name"


@pytest.mark.asyncio
async def test_async_shallow_client_info_fallback(redis_url: str, monkeypatch) -> None:
    """Test that AsyncShallowRedisSaver falls back to echo when client_setinfo is not available."""
    from langgraph.checkpoint.redis.version import __full_lib_name__
    from redis.exceptions import ResponseError
    from redis.asyncio import Redis
    
    # Create a Redis client directly first - this bypasses RedisVL validation
    client = Redis.from_url(redis_url)
    
    # Remove client_setinfo to simulate older Redis version
    async def mock_client_setinfo(self, key, value):
        raise ResponseError("ERR unknown command")
    
    # Track if echo was called with our lib name
    echo_called = False
    echo_messages = []
    original_echo = Redis.echo
    
    # Create mock for echo
    async def mock_echo(self, message):
        nonlocal echo_called, echo_messages
        echo_messages.append(message)
        if __full_lib_name__ in message:
            echo_called = True
        return await original_echo(self, message) if hasattr(original_echo, "__await__") else None
    
    # Apply the mocks
    monkeypatch.setattr(Redis, "client_setinfo", mock_client_setinfo)
    monkeypatch.setattr(Redis, "echo", mock_echo)
    
    try:
        # Test direct fallback without RedisVL interference
        async with AsyncShallowRedisSaver.from_conn_string(redis_client=client) as saver:
            # Force another call to set_client_info
            await saver.aset_client_info()
            
        # Print debug info
        print(f"Echo messages seen: {echo_messages}")
        
        # Verify echo was called as fallback with our library info
        assert echo_called, "echo was not called as fallback with our library name"
    finally:
        await client.aclose()
