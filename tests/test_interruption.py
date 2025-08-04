"""Tests for interruption handling in Redis checkpointers."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

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


class MockRedis(Redis):
    """Mock Redis class that can simulate interruptions during operations."""

    def __init__(self, real_redis: Redis, interrupt_on: str = None) -> None:
        """Initialize with a real Redis client and optional interruption point.

        Args:
            real_redis: The real Redis client to delegate to
            interrupt_on: Operation name to interrupt on (e.g., 'json().set', 'Pipeline.execute')
        """
        # Copy connection info from real_redis to satisfy Redis base class
        super().__init__(
            connection_pool=real_redis.connection_pool,
            single_connection_client=real_redis.single_connection_client,
        )
        self.real_redis = real_redis
        self.interrupt_on = interrupt_on
        self.operations_count = {}
        self.interrupt_after_count = {}

    def __getattribute__(self, name):
        """Proxy attribute access to the real Redis client, but track operations."""
        # For special attributes we've set in __init__, use the parent implementation
        if name in [
            "real_redis",
            "interrupt_on",
            "operations_count",
            "interrupt_after_count",
        ]:
            return super().__getattribute__(name)

        # For Redis base class attributes
        if name in ["connection_pool", "single_connection_client", "_parser", "_lock"]:
            return super().__getattribute__(name)

        try:
            attr = getattr(self.real_redis, name)
        except AttributeError:
            # Fall back to parent class
            return super().__getattribute__(name)

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

    async def __aenter__(self):
        """Support async context manager protocol."""
        if hasattr(self.real_subsystem, "__aenter__"):
            await self.real_subsystem.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager protocol."""
        if hasattr(self.real_subsystem, "__aexit__"):
            return await self.real_subsystem.__aexit__(exc_type, exc_val, exc_tb)
        return False

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
        # The InterruptionError is wrapped by RedisVL as RedisVLError
        from redisvl.exceptions import RedisVLError

        with pytest.raises(RedisVLError) as exc_info:
            await saver.aput(config, checkpoint, metadata, new_versions)

        # Verify it was actually an interruption
        assert "Simulated interruption during Pipeline.execute" in str(exc_info.value)

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
        # Shallow saver uses direct pipeline operations, so InterruptionError isn't wrapped
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
                print(checkpoint_tuple.pending_writes, flush=True)
                for write in checkpoint_tuple.pending_writes:
                    assert write[1] not in [
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
        # The InterruptionError is wrapped by RedisVL as RedisVLError
        from redisvl.exceptions import RedisVLError

        with pytest.raises(RedisVLError) as exc_info:
            await saver.aput(config, checkpoint, metadata, new_versions)

        # Verify it was actually an interruption
        assert "Simulated interruption during Pipeline.execute" in str(exc_info.value)

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
    """Test a realistic graph execution scenario with interruption during checkpoint updates.

    This simulates a LangGraph workflow where:
    1. An initial checkpoint is saved
    2. The graph starts processing and tries to save an updated checkpoint
    3. An interruption occurs during the update
    4. The system recovers and completes the checkpoint save
    """
    # Create test thread ID
    thread_id = f"test-graph-{uuid.uuid4()}"

    # Step 1: Save initial checkpoint (graph start state)
    initial_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }

    initial_checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {"messages": [], "state": {"status": "initialized"}},
        "channel_versions": {"messages": "0", "state": "0"},
        "versions_seen": {},
        "pending_sends": [],
    }

    # Save initial checkpoint successfully
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        first_config = await saver.aput(
            initial_config,
            initial_checkpoint,
            {"source": "initialize", "step": 0},
            initial_checkpoint["channel_versions"],
        )

        # Verify initial state
        saved_initial = await saver.aget(initial_config)
        assert saved_initial is not None
        assert saved_initial["id"] == initial_checkpoint["id"]
        assert saved_initial["channel_values"]["state"]["status"] == "initialized"

    # Step 2: Simulate graph processing with interruption
    # Create an updated checkpoint after user input
    user_checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {
            "messages": [("human", "What's the weather in SF?")],
            "state": {"status": "processing", "location": "San Francisco"},
        },
        "channel_versions": {"messages": "1", "state": "1"},
        "versions_seen": initial_checkpoint["channel_versions"],
        "pending_sends": [],
    }

    # Try to save with interruption
    async with create_interruptible_saver(
        redis_url,
        AsyncRedisSaver,
        interrupt_on="Pipeline.execute",
        interrupt_after_count=1,
    ) as interrupted_saver:
        from redisvl.exceptions import RedisVLError

        # Attempt to save checkpoint - should be interrupted
        with pytest.raises(RedisVLError) as exc_info:
            await interrupted_saver.aput(
                first_config,
                user_checkpoint,
                {"source": "user_input", "step": 1},
                user_checkpoint["channel_versions"],
            )

        assert "Simulated interruption" in str(exc_info.value)

        # Reset the interruption counter so we can verify the state
        # The mock interceptor counts pipeline operations cumulatively,
        # and increments before checking, so we need to set to -10 to ensure
        # the next few operations won't trigger interruption
        if hasattr(interrupted_saver._redis, "operations_count"):
            interrupted_saver._redis.operations_count["Pipeline.execute"] = -10

        # Verify the checkpoint was NOT saved (still have initial state)
        check_result = await interrupted_saver.aget(first_config)
        assert check_result is not None
        assert (
            check_result["id"] == initial_checkpoint["id"]
        )  # Still the initial checkpoint
        assert check_result["channel_values"]["state"]["status"] == "initialized"

    # Step 3: Simulate recovery after interruption
    # In a real scenario, this would be after the process restarts
    async with AsyncRedisSaver.from_conn_string(redis_url) as recovery_saver:
        # Need to get the actual first config that includes checkpoint_id
        # First, get the saved initial checkpoint to get its ID
        initial_result = await recovery_saver.aget(initial_config)
        first_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": initial_result["id"],
            }
        }

        # Retry saving the user checkpoint
        second_config = await recovery_saver.aput(
            first_config,
            user_checkpoint,
            {"source": "user_input", "step": 1},
            user_checkpoint["channel_versions"],
        )

        # Verify the checkpoint was saved correctly this time
        # Note: aget returns the latest checkpoint, not a specific one by ID
        saved_user = await recovery_saver.aget(initial_config)
        assert saved_user is not None
        assert saved_user["id"] == user_checkpoint["id"]
        assert saved_user["channel_values"]["state"]["status"] == "processing"
        assert len(saved_user["channel_values"]["messages"]) == 1

        # Step 4: Continue with AI response (complete workflow)
        ai_checkpoint = {
            "id": str(uuid.uuid4()),
            "ts": str(int(time.time())),
            "v": 1,
            "channel_values": {
                "messages": [
                    ("human", "What's the weather in SF?"),
                    ("ai", "I'll check the weather in San Francisco for you."),
                    ("tool", "weather_api.get(location='San Francisco')"),
                    ("ai", "The weather in San Francisco is currently 68°F and sunny."),
                ],
                "state": {
                    "status": "completed",
                    "location": "San Francisco",
                    "weather": "68°F, sunny",
                },
            },
            "channel_versions": {"messages": "2", "state": "2"},
            "versions_seen": user_checkpoint["channel_versions"],
            "pending_sends": [],
        }

        # Save final state
        final_config = await recovery_saver.aput(
            second_config,
            ai_checkpoint,
            {"source": "ai_response", "step": 2},
            ai_checkpoint["channel_versions"],
        )

        # Verify complete workflow state
        final_result = await recovery_saver.aget(initial_config)
        assert final_result is not None
        assert final_result["id"] == ai_checkpoint["id"]
        assert final_result["channel_values"]["state"]["status"] == "completed"
        assert len(final_result["channel_values"]["messages"]) == 4

        # Test listing checkpoints - should have all 3 checkpoints
        list_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        checkpoints = []
        async for checkpoint_tuple in recovery_saver.alist(list_config):
            checkpoints.append(checkpoint_tuple)

        # Redis should store all checkpoints like Postgres/MongoDB
        assert len(checkpoints) == 3

        # Get checkpoint IDs for verification
        checkpoint_ids = {cp.checkpoint["id"] for cp in checkpoints}
        assert ai_checkpoint["id"] in checkpoint_ids
        assert user_checkpoint["id"] in checkpoint_ids
        assert initial_checkpoint["id"] in checkpoint_ids

        # Find the final checkpoint
        final_checkpoint = None
        for cp in checkpoints:
            if cp.checkpoint["id"] == ai_checkpoint["id"]:
                final_checkpoint = cp
                break

        assert final_checkpoint is not None
        assert (
            final_checkpoint.checkpoint["channel_values"]["state"]["status"]
            == "completed"
        )
        assert len(final_checkpoint.checkpoint["channel_values"]["messages"]) == 4
        assert final_checkpoint.metadata["step"] == 2
        assert final_checkpoint.metadata["source"] == "ai_response"


@pytest.mark.asyncio
async def test_graph_simulation_with_interruption_shallow(redis_url: str) -> None:
    """Test a realistic graph execution scenario with interruption for shallow checkpointers.

    Shallow checkpointers only keep the most recent checkpoint, so we test
    that the latest state is preserved correctly after interruption and recovery.
    """
    # Create test thread ID
    thread_id = f"test-graph-shallow-{uuid.uuid4()}"

    # Step 1: Save initial checkpoint (graph start state)
    initial_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }

    initial_checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {"messages": [], "state": {"status": "initialized"}},
        "channel_versions": {"messages": "0", "state": "0"},
        "versions_seen": {},
        "pending_sends": [],
    }

    # Save initial checkpoint successfully
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        first_config = await saver.aput(
            initial_config,
            initial_checkpoint,
            {"source": "initialize", "step": 0},
            initial_checkpoint["channel_versions"],
        )

        # Verify initial state
        saved_initial = await saver.aget(initial_config)
        assert saved_initial is not None
        assert saved_initial["id"] == initial_checkpoint["id"]
        assert saved_initial["channel_values"]["state"]["status"] == "initialized"

    # Step 2: Simulate graph processing with interruption
    # Create an updated checkpoint after user input
    user_checkpoint = {
        "id": str(uuid.uuid4()),
        "ts": str(int(time.time())),
        "v": 1,
        "channel_values": {
            "messages": [("human", "What's the weather in SF?")],
            "state": {"status": "processing", "location": "San Francisco"},
        },
        "channel_versions": {"messages": "1", "state": "1"},
        "versions_seen": initial_checkpoint["channel_versions"],
        "pending_sends": [],
    }

    # Try to save with interruption
    async with create_interruptible_saver(
        redis_url,
        AsyncShallowRedisSaver,
        interrupt_on="Pipeline.execute",
        interrupt_after_count=1,
    ) as interrupted_saver:
        # Attempt to save checkpoint - should be interrupted
        with pytest.raises(InterruptionError) as exc_info:
            await interrupted_saver.aput(
                first_config,
                user_checkpoint,
                {"source": "user_input", "step": 1},
                user_checkpoint["channel_versions"],
            )

        assert "Simulated interruption" in str(exc_info.value)

        # Reset the interruption counter so we can verify the state
        # The mock interceptor counts pipeline operations cumulatively,
        # and increments before checking, so we need to set to -10 to ensure
        # the next few operations won't trigger interruption
        if hasattr(interrupted_saver._redis, "operations_count"):
            interrupted_saver._redis.operations_count["Pipeline.execute"] = -10

        # Verify the checkpoint was NOT saved (still have initial state)
        check_result = await interrupted_saver.aget(first_config)
        assert check_result is not None
        assert (
            check_result["id"] == initial_checkpoint["id"]
        )  # Still the initial checkpoint
        assert check_result["channel_values"]["state"]["status"] == "initialized"

    # Step 3: Simulate recovery after interruption
    # In a real scenario, this would be after the process restarts
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as recovery_saver:
        # Retry saving the user checkpoint
        second_config = await recovery_saver.aput(
            first_config,
            user_checkpoint,
            {"source": "user_input", "step": 1},
            user_checkpoint["channel_versions"],
        )

        # Verify the checkpoint was saved correctly this time
        # Note: aget returns the latest checkpoint, not a specific one by ID
        saved_user = await recovery_saver.aget(initial_config)
        assert saved_user is not None
        assert saved_user["id"] == user_checkpoint["id"]
        assert saved_user["channel_values"]["state"]["status"] == "processing"
        assert len(saved_user["channel_values"]["messages"]) == 1

        # Step 4: Continue with AI response (complete workflow)
        ai_checkpoint = {
            "id": str(uuid.uuid4()),
            "ts": str(int(time.time())),
            "v": 1,
            "channel_values": {
                "messages": [
                    ("human", "What's the weather in SF?"),
                    ("ai", "I'll check the weather in San Francisco for you."),
                    ("tool", "weather_api.get(location='San Francisco')"),
                    ("ai", "The weather in San Francisco is currently 68°F and sunny."),
                ],
                "state": {
                    "status": "completed",
                    "location": "San Francisco",
                    "weather": "68°F, sunny",
                },
            },
            "channel_versions": {"messages": "2", "state": "2"},
            "versions_seen": user_checkpoint["channel_versions"],
            "pending_sends": [],
        }

        # Save final state
        final_config = await recovery_saver.aput(
            second_config,
            ai_checkpoint,
            {"source": "ai_response", "step": 2},
            ai_checkpoint["channel_versions"],
        )

        # Verify complete workflow state
        final_result = await recovery_saver.aget(initial_config)
        assert final_result is not None
        assert final_result["id"] == ai_checkpoint["id"]
        assert final_result["channel_values"]["state"]["status"] == "completed"
        assert len(final_result["channel_values"]["messages"]) == 4

        # Test listing checkpoints - shallow saver only keeps the latest
        # Use config without checkpoint_id to list all checkpoints for the thread
        list_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
        checkpoints = []
        async for checkpoint_tuple in recovery_saver.alist(list_config):
            checkpoints.append(checkpoint_tuple)

        assert len(checkpoints) == 1
        # Should only have the latest checkpoint
        assert checkpoints[0].checkpoint["id"] == ai_checkpoint["id"]
        assert checkpoints[0].metadata["step"] == 2
