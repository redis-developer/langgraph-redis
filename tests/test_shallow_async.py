from typing import Any, AsyncGenerator, Dict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

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
