"""Tests for AsyncRedisSaver."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Literal

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.redis import BaseRedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.prebuilt import create_react_agent


@pytest.fixture
async def test_data() -> Dict[str, Any]:
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
    }
    metadata_2: CheckpointMetadata = {
        "source": "loop",
        "step": 1,
        "writes": {"foo": "bar"},
    }
    metadata_3: CheckpointMetadata = {
        "source": "",
        "step": 0,
        "writes": {},
    }

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


@pytest.fixture
async def saver(redis_url: str) -> AsyncGenerator[AsyncRedisSaver, None]:
    """Async saver fixture."""
    saver = AsyncRedisSaver(redis_url)
    await saver.asetup()
    yield saver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected_count",
    [
        ({"source": "input"}, 1),  # Matches metadata.source
        ({"step": 1}, 1),  # Matches metadata.step
        ({}, 3),  # Retrieve all checkpoints
        ({"source": "update"}, 0),  # No matches
    ],
)
async def test_search(
    saver: AsyncRedisSaver,
    test_data: Dict[str, Any],
    query: Dict[str, Any],
    expected_count: int,
) -> None:
    """Test search functionality with different queries."""
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    # Save test data
    await saver.aput(configs[0], checkpoints[0], metadata[0], {})
    await saver.aput(configs[1], checkpoints[1], metadata[1], {})
    await saver.aput(configs[2], checkpoints[2], metadata[2], {})

    # Execute search
    search_results = [c async for c in saver.alist(None, filter=query)]
    assert len(search_results) == expected_count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key, value",
    [
        ("source", "\x00abc"),  # Null character in value
    ],
)
async def test_null_chars(
    saver: AsyncRedisSaver,
    key: str,
    value: str,
    test_data: dict[str, list[Any]],
) -> None:
    config = await saver.aput(
        test_data["configs"][0],
        test_data["checkpoints"][0],
        {key: value},  # type: ignore[misc]
        {},
    )

    result = await saver.aget_tuple(config)
    assert result is not None, "Checkpoint not found in Redis"
    sanitized_value = value.replace("\x00", "")
    assert result.metadata[key] == sanitized_value  # type: ignore[literal-required]


@pytest.mark.asyncio
async def test_put_writes_async(redis_url: str, test_data: Dict[str, Any]) -> None:
    """Test storing writes in Redis asynchronously."""
    async with AsyncRedisSaver(redis_url) as saver:
        await saver.asetup()

        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # First store a checkpoint
        saved_config = await saver.aput(config, checkpoint, metadata, {})

        # Test regular writes
        writes = [("channel1", "value1"), ("channel2", "value2")]
        task_id = "task1"
        await saver.aput_writes(saved_config, writes, task_id)

        # Test special writes (using WRITES_IDX_MAP)
        special_writes = [("__error__", "error_value"), ("channel3", "value3")]
        task_id2 = "task2"
        await saver.aput_writes(saved_config, special_writes, task_id2)

        # Verify writes through get_tuple
        result = await saver.aget_tuple(saved_config)
        assert result is not None, "Expected to find checkpoint"
        pending_writes = result.pending_writes
        assert pending_writes is not None, "Expected pending writes"
        assert len(pending_writes) > 0, "Expected at least one write"

        # Verify regular writes
        found_writes = {(w[1], w[2]) for w in pending_writes}
        assert ("channel1", "value1") in found_writes
        assert ("channel2", "value2") in found_writes

        # Verify special writes
        assert ("__error__", "error_value") in found_writes


@pytest.mark.asyncio
async def test_concurrent_writes_async(
    redis_url: str, test_data: Dict[str, Any]
) -> None:
    """Test concurrent writes to Redis."""
    async with AsyncRedisSaver(redis_url) as saver:
        await saver.asetup()

        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # First store a checkpoint
        saved_config = await saver.aput(config, checkpoint, metadata, {})

        # Create multiple write operations
        tasks = []
        for i in range(5):
            writes = [(f"channel{i}", f"value{i}")]
            tasks.append(saver.aput_writes(saved_config, writes, f"task{i}"))

        # Execute writes concurrently
        await asyncio.gather(*tasks)

        # Verify all writes were stored
        result = await saver.aget_tuple(saved_config)
        assert result is not None, "Expected to find checkpoint"
        pending_writes = result.pending_writes
        assert pending_writes is not None, "Expected pending writes"

        found_writes = {(w[1], w[2]) for w in pending_writes}

        # Verify each write was stored
        for i in range(5):
            assert (f"channel{i}", f"value{i}") in found_writes


@pytest.mark.asyncio
async def test_from_conn_string_with_url(redis_url: str) -> None:
    """Test creating an AsyncRedisSaver with a connection URL."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        # Verify connection works by creating and checking a key
        await saver._redis.set("test_key", "test_value")
        value = await saver._redis.get("test_key")
        assert value == b"test_value"


@pytest.mark.asyncio
async def test_from_conn_string_with_client(redis_url: str) -> None:
    """Test creating an AsyncRedisSaver with an existing Redis client."""
    client = Redis.from_url(redis_url)
    try:
        async with AsyncRedisSaver.from_conn_string(redis_client=client) as saver:
            await saver.asetup()
            # Verify connection works
            await saver._redis.set("test_key2", "test_value")
            value = await saver._redis.get("test_key2")
            assert value == b"test_value"
    finally:
        await client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_from_conn_string_with_connection_args(redis_url: str) -> None:
    """Test creating an AsyncRedisSaver with connection arguments."""
    async with AsyncRedisSaver.from_conn_string(
        redis_url, connection_args={"decode_responses": True}
    ) as saver:
        await saver.asetup()
        # Test with decode_responses=True, so we get str instead of bytes
        await saver._redis.set("test_key3", "test_value")
        value = await saver._redis.get("test_key3")
        assert isinstance(value, str)  # not bytes
        assert value == "test_value"


@pytest.mark.asyncio
async def test_from_conn_string_cleanup(redis_url: str) -> None:
    """Test proper cleanup of Redis connections."""
    # When creating from URL
    client = None
    saver = None
    async with AsyncRedisSaver.from_conn_string(redis_url) as s:
        saver = s
        client = s._redis
        assert await client.ping()  # Connection works

    # When passing external client, should not close it
    ext_client = Redis.from_url(redis_url)
    try:
        async with AsyncRedisSaver.from_conn_string(redis_client=ext_client) as saver:
            client = saver._redis
            assert await client == ext_client
        assert await ext_client.ping()  # Should still work
    finally:
        await ext_client.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_from_conn_string_errors() -> None:
    """Test error conditions for from_conn_string."""
    # Test with neither URL nor client provided
    with pytest.raises(
        ValueError, match="Either redis_url or redis_client must be provided"
    ):
        async with AsyncRedisSaver.from_conn_string() as _:
            pass

    # Test with empty URL - should fail
    with pytest.raises(ValueError, match="REDIS_URL env var not set"):
        async with AsyncRedisSaver.from_conn_string("") as _:
            pass

    # Test with invalid connection URL
    with pytest.raises(RedisConnectionError):
        async with AsyncRedisSaver.from_conn_string("redis://nonexistent:6379") as _:
            await _.asetup()  # Force connection attempt

    # Test with non-functional client
    client = Redis.from_url("redis://nonexistent:6379")
    with pytest.raises(RedisConnectionError):
        async with AsyncRedisSaver.from_conn_string(redis_client=client) as _:
            await _.asetup()  # Force connection attempt


@pytest.mark.asyncio
async def test_put_writes_json_structure_async(
    redis_url: str, test_data: Dict[str, Any]
) -> None:
    """Test that writes are properly stored in Redis JSON format."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()

        config = test_data["configs"][0]
        checkpoint = test_data["checkpoints"][0]
        metadata = test_data["metadata"][0]

        # First store a checkpoint to get proper config
        saved_config = await saver.aput(config, checkpoint, metadata, {})

        writes = [("channel1", "value1")]
        task_id = "task1"

        # Store write
        await saver.aput_writes(saved_config, writes, task_id)

        # Verify JSON structure directly
        thread_id = saved_config["configurable"]["thread_id"]
        checkpoint_ns = saved_config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = saved_config["configurable"]["checkpoint_id"]

        # Verify JSON structure directly
        write_key = BaseRedisSaver._make_redis_checkpoint_writes_key(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            WRITES_IDX_MAP.get("channel1", 0),
        )

        # Get raw JSON
        json_data = await saver._redis.json().get(write_key)

        # Verify structure
        assert json_data["thread_id"] == saved_config["configurable"]["thread_id"]
        assert json_data["channel"] == "channel1"
        assert json_data["task_id"] == task_id


@pytest.mark.asyncio
async def test_search_writes_async(redis_url: str) -> None:
    """Test searching writes using Redis Search."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()

        # Set up test data with proper typing
        config: RunnableConfig = {
            "configurable": {"thread_id": "thread1", "checkpoint_ns": "ns1"}
        }
        checkpoint = empty_checkpoint()
        metadata: CheckpointMetadata = {"source": "test", "step": 1, "writes": {}}

        # Store checkpoint to get proper config
        saved_config = await saver.aput(config, checkpoint, metadata, {})

        # Add writes for multiple channels
        writes1 = [("channel1", "value1"), ("channel2", "value2")]
        await saver.aput_writes(saved_config, writes1, "task1")

        writes2 = [("channel1", "value3")]
        await saver.aput_writes(saved_config, writes2, "task2")

        # Search by channel
        query = "(@channel:{channel1})"
        results = await saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 2  # One document per channel1 writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)

        assert doc1["channel"] == doc2["channel"] == "channel1"

        # Search by task
        query = "(@task_id:{task1})"
        results = await saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 2  # One document per channel1 writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)

        # Search by thread/namespace
        query = "(@thread_id:{thread1} @checkpoint_ns:{ns1})"
        results = await saver.checkpoint_writes_index.search(query)
        assert len(results.docs) == 3  # Contains all three writes

        doc1 = json.loads(results.docs[0].json)
        doc2 = json.loads(results.docs[1].json)
        doc3 = json.loads(results.docs[2].json)

        assert doc1["blob"] == '"value1"'
        assert doc2["blob"] == '"value2"'
        assert doc3["blob"] == '"value3"'


@pytest.mark.asyncio
async def test_no_running_loop(redis_url: str, test_data: dict[str, Any]) -> None:
    """Test that sync operations raise error when called from async loop."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saver.loop = asyncio.get_running_loop()  # Set the loop explicitly

        # Test get_tuple which has the explicit check
        with pytest.raises(asyncio.InvalidStateError):
            saver.get_tuple(test_data["configs"][0])

        # Other sync operations from background threads should work
        with ThreadPoolExecutor() as pool:
            future = pool.submit(
                saver.put,
                test_data["configs"][0],
                test_data["checkpoints"][0],
                test_data["metadata"][0],
                {},
            )
            result = await asyncio.wrap_future(future)
            assert result is not None


@pytest.mark.asyncio
async def test_large_batches(redis_url: str, test_data: dict[str, Any]) -> None:
    """Test handling large numbers of operations with thread pool."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        saver.loop = asyncio.get_running_loop()

        N = 1000
        M = 5

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for m in range(M):
                for i in range(N):
                    # Create config and cast to RunnableConfig type
                    test_config: RunnableConfig = {
                        "configurable": {
                            "thread_id": f"thread-{m}-{i}",
                            "checkpoint_ns": "",
                            "checkpoint_id": f"checkpoint-{i}",
                        }
                    }
                    futures.append(
                        executor.submit(
                            saver.put,
                            test_config,
                            test_data["checkpoints"][0],
                            test_data["metadata"][0],
                            {},
                        )
                    )

            results = await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures)
            )
            assert len(results) == M * N


@pytest.mark.asyncio
async def test_large_batches_async(redis_url: str, test_data: dict[str, Any]) -> None:
    """Test handling large numbers of async operations."""
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()

        N = 1000
        M = 2

        # Store configs and their responses
        stored_configs = []
        coros = []

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
                coros.append(
                    saver.aput(
                        test_config,
                        test_data["checkpoints"][0],
                        test_data["metadata"][0],
                        {},
                    )
                )

        try:
            put_results = await asyncio.gather(*coros)
            assert len(put_results) == M * N

            # Verify we can retrieve using the configs returned from put
            verify_coros = []
            for result_config in put_results:
                verify_coros.append(saver.aget_tuple(result_config))

            verify_results = await asyncio.gather(*verify_coros)

            assert len(verify_results) == M * N
            assert all(r is not None for r in verify_results)

        except Exception as e:
            pytest.fail(f"Failed to process async batch: {str(e)}")


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
def model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)


@pytest.mark.requires_api_keys
@pytest.mark.asyncio
async def test_async_redis_checkpointer(
    redis_url: str, tools: List[BaseTool], model: ChatOpenAI
) -> None:
    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        await checkpointer.asetup()
        # Create agent with checkpointer
        graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)

        # Test initial query
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "test1",
                "checkpoint_ns": "",
                "checkpoint_id": "",
            }
        }
        res = await graph.ainvoke(
            {"messages": [("human", "what's the weather in sf")]}, config
        )

        assert res is not None

        # Test checkpoint retrieval
        latest = await checkpointer.aget(config)

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
        tuple_result = await checkpointer.aget_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint == latest

        # Test listing checkpoints
        checkpoints = [c async for c in checkpointer.alist(config)]
        assert len(checkpoints) > 0
        assert checkpoints[-1].checkpoint["id"] == latest["id"]
