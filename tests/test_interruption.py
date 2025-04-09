"""Tests for interruption handling in Redis checkpointers."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
)
from redis.asyncio import Redis

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver


class InterruptionError(Exception):
    """Error used to simulate an interruption during checkpoint operations."""

    pass


class MockRedis:
    """Mock Redis class that can simulate interruptions during operations."""

    def __init__(self, real_redis: Redis, interrupt_on: str = None) -> None:
        """Initialize with a real Redis client and optional interruption point.

        Args:
            real_redis: The real Redis client to delegate to
            interrupt_on: Operation name to interrupt on (e.g., 'json().set', 'Pipeline.execute')
        """
        self.real_redis = real_redis
        self.interrupt_on = interrupt_on
        self.operations_count = {}
        self.interrupt_after_count = {}

    def __getattr__(self, name):
        """Proxy attribute access to the real Redis client, but track operations."""
        attr = getattr(self.real_redis, name)

        # For methods we want to potentially interrupt
        if callable(attr) and name == self.interrupt_on:
            # Initialize counter for this operation if not exist
            if name not in self.operations_count:
                self.operations_count[name] = 0

            async def wrapper(*args, **kwargs):
                # Increment operation count
                self.operations_count[name] += 1

                # Check if we should interrupt
                if (
                    name in self.interrupt_after_count
                    and self.operations_count[name] >= self.interrupt_after_count[name]
                ):
                    raise InterruptionError(
                        f"Simulated interruption during {name} operation"
                    )

                # Otherwise, call the real method
                return await attr(*args, **kwargs)

            return wrapper

        # Special handling for pipeline to ensure we can intercept pipeline.execute()
        elif name == "pipeline":
            original_method = attr

            def pipeline_wrapper(*args, **kwargs):
                pipeline = original_method(*args, **kwargs)
                return MockRedisSubsystem(pipeline, self)

            return pipeline_wrapper

        # For Redis subsystems (like json())
        elif name in ["json"]:
            original_method = attr

            if callable(original_method):

                def subsystem_wrapper(*args, **kwargs):
                    subsystem = original_method(*args, **kwargs)
                    return MockRedisSubsystem(subsystem, self)

                return subsystem_wrapper
            else:
                return MockRedisSubsystem(attr, self)

        # For other attributes, return as is
        return attr


class MockRedisSubsystem:
    """Mock Redis subsystem (like json()) that can simulate interruptions."""

    def __init__(self, real_subsystem, parent_mock):
        self.real_subsystem = real_subsystem
        self.parent_mock = parent_mock

    def __getattr__(self, name):
        attr = getattr(self.real_subsystem, name)

        # For methods we want to potentially interrupt
        operation_name = f"{self.real_subsystem.__class__.__name__}.{name}"
        if callable(attr) and operation_name == self.parent_mock.interrupt_on:
            # Initialize counter for this operation if not exist
            if operation_name not in self.parent_mock.operations_count:
                self.parent_mock.operations_count[operation_name] = 0

            async def wrapper(*args, **kwargs):
                # Increment operation count
                self.parent_mock.operations_count[operation_name] += 1

                # Check if we should interrupt
                if (
                    operation_name in self.parent_mock.interrupt_after_count
                    and self.parent_mock.operations_count[operation_name]
                    >= self.parent_mock.interrupt_after_count[operation_name]
                ):
                    raise InterruptionError(
                        f"Simulated interruption during {operation_name} operation"
                    )

                # Otherwise, call the real method
                return await attr(*args, **kwargs)

            if asyncio.iscoroutinefunction(attr):
                return wrapper
            else:
                # For non-async methods
                def sync_wrapper(*args, **kwargs):
                    # Increment operation count
                    self.parent_mock.operations_count[operation_name] += 1

                    # Check if we should interrupt
                    if (
                        operation_name in self.parent_mock.interrupt_after_count
                        and self.parent_mock.operations_count[operation_name]
                        >= self.parent_mock.interrupt_after_count[operation_name]
                    ):
                        raise InterruptionError(
                            f"Simulated interruption during {operation_name} operation"
                        )

                    # Otherwise, call the real method
                    return attr(*args, **kwargs)

                return sync_wrapper

        # Special handling for pipeline method to track operations within the pipeline
        elif name == "execute" and hasattr(self.real_subsystem, "execute"):
            # This is likely a pipeline execute method
            async def execute_wrapper(*args, **kwargs):
                # Check if we should interrupt pipeline execution
                if self.parent_mock.interrupt_on == "Pipeline.execute":
                    if "Pipeline.execute" not in self.parent_mock.operations_count:
                        self.parent_mock.operations_count["Pipeline.execute"] = 0

                    self.parent_mock.operations_count["Pipeline.execute"] += 1

                    if (
                        "Pipeline.execute" in self.parent_mock.interrupt_after_count
                        and self.parent_mock.operations_count["Pipeline.execute"]
                        >= self.parent_mock.interrupt_after_count["Pipeline.execute"]
                    ):
                        raise InterruptionError(
                            f"Simulated interruption during Pipeline.execute operation"
                        )

                # Otherwise call the real execute
                return await attr(*args, **kwargs)

            if asyncio.iscoroutinefunction(attr):
                return execute_wrapper
            else:
                return attr

        # For other attributes, return as is
        return attr


@asynccontextmanager
async def create_interruptible_saver(
    redis_url: str,
    saver_class,
    interrupt_on: str = None,
    interrupt_after_count: int = 1,
) -> AsyncGenerator:
    """Create a saver with a mock Redis client that can simulate interruptions.

    Args:
        redis_url: Redis connection URL
        saver_class: The saver class to instantiate (AsyncRedisSaver or AsyncShallowRedisSaver)
        interrupt_on: Operation to interrupt on
        interrupt_after_count: Number of operations to allow before interrupting

    Yields:
        A configured saver instance with interruptible Redis client
    """
    # Create real Redis client
    real_redis = Redis.from_url(redis_url)

    # Create mock Redis client that will interrupt on specified operation
    mock_redis = MockRedis(real_redis, interrupt_on)
    if interrupt_on:
        mock_redis.interrupt_after_count[interrupt_on] = interrupt_after_count

    # Create saver with mock Redis
    saver = saver_class(redis_client=mock_redis)

    try:
        await saver.asetup()
        yield saver
    finally:
        # Close Redis client
        if hasattr(saver, "__aexit__"):
            await saver.__aexit__(None, None, None)
        else:
            # Cleanup manually if __aexit__ doesn't exist
            if saver._owns_its_client:
                await real_redis.aclose()
                await real_redis.connection_pool.disconnect()


def create_test_checkpoint() -> (
    tuple[RunnableConfig, Checkpoint, CheckpointMetadata, Dict[str, str]]
):
    """Create test checkpoint data for the tests."""
    thread_id = f"test-{uuid.uuid4()}"
    checkpoint_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "",
        }
    }

    checkpoint = {
        "id": checkpoint_id,
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {
            "messages": [
                ("human", "What's the weather?"),
                ("ai", "I'll check for you."),
                ("tool", "get_weather()"),
                ("ai", "It's sunny."),
            ]
        },
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }

    metadata = {
        "source": "test",
        "step": 1,
        "writes": {},
    }

    new_versions = {"messages": "1"}

    return config, checkpoint, metadata, new_versions


def verify_checkpoint_state(
    redis_client: Redis,
    thread_id: str,
    checkpoint_id: str,
    expected_present: bool = True,
) -> None:
    """Verify whether checkpoint data exists in Redis as expected."""
    # Check if checkpoint data exists in Redis
    keys = redis_client.keys(f"*{thread_id}*")
    assert (
        len(keys) > 0
    ) == expected_present, (
        f"Expected checkpoint data {'to exist' if expected_present else 'to not exist'}"
    )

    if expected_present:
        # Check if specific checkpoint ID exists
        assert any(
            checkpoint_id.encode() in key or checkpoint_id in key.decode()
            for key in keys
        ), f"Checkpoint ID {checkpoint_id} not found in Redis"


@pytest.mark.asyncio
async def test_aput_interruption_regular_saver(redis_url: str) -> None:
    """Test interruption during AsyncRedisSaver.aput operation."""
    # Create test data
    config, checkpoint, metadata, new_versions = create_test_checkpoint()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_id = checkpoint["id"]

    # Create saver with interruption during pipeline execute
    async with create_interruptible_saver(
        redis_url,
        AsyncRedisSaver,
        interrupt_on="Pipeline.execute",
        interrupt_after_count=1,
    ) as saver:
        # Try to save checkpoint, expect interruption
        with pytest.raises(InterruptionError):
            await saver.aput(config, checkpoint, metadata, new_versions)

        # Verify that the checkpoint data is incomplete or inconsistent
        real_redis = Redis.from_url(redis_url)
        try:
            # Attempt to retrieve the checkpoint
            result = await saver.aget(config)
            # Either the result should be None or contain incomplete data
            if result is not None:
                assert (
                    result != checkpoint
                ), "Checkpoint should not be completely saved after interruption"
        finally:
            await real_redis.flushall()
            await real_redis.aclose()


@pytest.mark.asyncio
async def test_aput_interruption_shallow_saver(redis_url: str) -> None:
    """Test interruption during AsyncShallowRedisSaver.aput operation."""
    # Create test data
    config, checkpoint, metadata, new_versions = create_test_checkpoint()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_id = checkpoint["id"]

    # Create saver with interruption during pipeline execute
    async with create_interruptible_saver(
        redis_url,
        AsyncShallowRedisSaver,
        interrupt_on="Pipeline.execute",
        interrupt_after_count=1,
    ) as saver:
        # Try to save checkpoint, expect interruption
        with pytest.raises(InterruptionError):
            await saver.aput(config, checkpoint, metadata, new_versions)

        # Verify that the checkpoint data is incomplete or inconsistent
        real_redis = Redis.from_url(redis_url)
        try:
            # Attempt to retrieve the checkpoint
            result = await saver.aget(config)
            # Either the result should be None or contain incomplete data
            if result is not None:
                assert (
                    result != checkpoint
                ), "Checkpoint should not be completely saved after interruption"
        finally:
            await real_redis.flushall()
            await real_redis.aclose()


@pytest.mark.asyncio
async def test_aput_writes_interruption(redis_url: str) -> None:
    """Test interruption during aput_writes operation."""
    # Create test data
    config, checkpoint, metadata, new_versions = create_test_checkpoint()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_id = checkpoint["id"]

    # Successfully save a checkpoint first
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        next_config = await saver.aput(config, checkpoint, metadata, new_versions)

        # Now create a saver that will interrupt during pipeline execution
        mock_redis = MockRedis(saver._redis, "Pipeline.execute")
        mock_redis.interrupt_after_count["Pipeline.execute"] = 1

        # Replace the Redis client with our mock
        original_redis = saver._redis
        saver._redis = mock_redis

        try:
            # Try to save writes, expect interruption
            with pytest.raises(InterruptionError):
                await saver.aput_writes(
                    next_config,
                    [("channel1", "value1"), ("channel2", "value2")],
                    "task_id_1",
                )

            # Restore original Redis client to verify state
            saver._redis = original_redis

            # Verify that no writes were saved due to transaction abort
            checkpoint_tuple = await saver.aget_tuple(next_config)

            # Either there are no pending writes or they are not the ones we tried to save
            if checkpoint_tuple and checkpoint_tuple.pending_writes:
                for write in checkpoint_tuple.pending_writes:
                    assert write.channel not in [
                        "channel1",
                        "channel2",
                    ], "Transaction should have been rolled back"
        finally:
            # Cleanup
            saver._redis = original_redis


@pytest.mark.asyncio
async def test_recovery_after_interruption(redis_url: str) -> None:
    """Test whether checkpoint operations can recover after an interruption."""
    # Create test data
    config, checkpoint, metadata, new_versions = create_test_checkpoint()
    thread_id = config["configurable"]["thread_id"]
    checkpoint_id = checkpoint["id"]

    # Step 1: Try to save with interruption
    async with create_interruptible_saver(
        redis_url,
        AsyncRedisSaver,
        interrupt_on="Pipeline.execute",
        interrupt_after_count=1,
    ) as saver:
        # Try to save checkpoint, expect interruption
        with pytest.raises(InterruptionError):
            await saver.aput(config, checkpoint, metadata, new_versions)

    # Step 2: Try to save again with a new saver (simulate process restart after interruption)
    async with AsyncRedisSaver.from_conn_string(redis_url) as new_saver:
        # Try to save the same checkpoint again
        next_config = await new_saver.aput(config, checkpoint, metadata, new_versions)

        # Verify the checkpoint was saved successfully
        result = await new_saver.aget(config)
        assert result is not None
        assert result["id"] == checkpoint["id"]

        # Clean up
        real_redis = Redis.from_url(redis_url)
        await real_redis.flushall()
        await real_redis.aclose()


@pytest.mark.asyncio
async def test_graph_simulation_with_interruption(redis_url: str) -> None:
    """Test a more complete scenario simulating a graph execution with interruption."""
    # Create a mock graph execution
    thread_id = f"test-{uuid.uuid4()}"

    # Config without checkpoint_id to simulate first run
    initial_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }

    # Create initial checkpoint
    initial_checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {"messages": []},
        "channel_versions": {"messages": "initial"},
        "versions_seen": {},
        "pending_sends": [],
    }

    # First save the initial checkpoint
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        next_config = await saver.aput(
            initial_config,
            initial_checkpoint,
            {"source": "initial", "step": 0},
            {"messages": "initial"},
        )

        # Verify initial checkpoint was saved
        initial_result = await saver.aget(initial_config)
        assert initial_result is not None

        # Now prepare update with interruption
        second_checkpoint = {
            "id": str(uuid.uuid4()),
            "ts": str(int(time.time())),
            "v": 1,
            "channel_values": {"messages": [("human", "What's the weather?")]},
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }

        # Replace Redis client with mock that will interrupt
        original_redis = saver._redis
        mock_redis = MockRedis(original_redis, "Pipeline.execute")
        mock_redis.interrupt_after_count["Pipeline.execute"] = 1
        saver._redis = mock_redis

        # Try to update, expect interruption
        with pytest.raises(InterruptionError):
            await saver.aput(
                next_config,
                second_checkpoint,
                {"source": "update", "step": 1},
                {"messages": "1"},
            )

        # Restore original Redis for verification
        saver._redis = original_redis

        # Check checkpoint state - with transaction handling, we expect to see the initial checkpoint
        # since the transaction should have been rolled back
        current = await saver.aget(next_config)

        # With transaction handling, we should still see the initial checkpoint
        assert (
            current and current["id"] == initial_checkpoint["id"]
        ), "Should still have initial checkpoint after transaction abort"

        # Clean up
        await original_redis.flushall()
