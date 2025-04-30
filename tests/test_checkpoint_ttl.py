"""Tests for TTL functionality with RedisSaver."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, Iterator, Optional, TypedDict, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.graph import END, StateGraph
from redis import Redis

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.util import to_storage_safe_id


class State(TypedDict):
    """Simple state with count."""

    count: int


@pytest.fixture(scope="function")
def redis_url(redis_container) -> str:
    """Get the Redis URL from the container."""
    host, port = redis_container.get_service_host_and_port("redis", 6379)
    return f"redis://{host}:{port}"


@pytest.fixture(scope="function")
def redis_client(redis_url: str) -> Generator[Redis, None, None]:
    """Create a Redis client for testing."""
    client = Redis.from_url(redis_url)
    try:
        yield client
    finally:
        # Clean up any test keys
        keys = client.keys("checkpoint:test_ttl*")
        if keys:
            client.delete(*keys)
        client.close()


@pytest.fixture(scope="function")
def ttl_checkpoint_saver(redis_client: Redis) -> Generator[RedisSaver, None, None]:
    """Create a RedisSaver instance with TTL support."""
    saver = RedisSaver(
        redis_client=redis_client,
        ttl={
            "default_ttl": 0.1,
            "refresh_on_read": True,
        },  # 0.1 minutes = 6 seconds TTL
    )
    saver.setup()
    yield saver


def test_ttl_config_in_constructor(redis_client: Redis) -> None:
    """Test that TTL config can be passed through constructor."""
    saver = RedisSaver(
        redis_client=redis_client,
        ttl={"default_ttl": 10, "refresh_on_read": True},
    )
    assert saver.ttl_config is not None
    assert saver.ttl_config.get("default_ttl") == 10
    assert saver.ttl_config.get("refresh_on_read") is True


def test_checkpoint_expires(redis_client: Redis) -> None:
    """Test that a checkpoint expires after the TTL period."""
    try:
        # Create unique identifiers to avoid test collisions
        unique_prefix = f"expires_test_{int(time.time())}"

        # Create a saver with TTL
        ttl_checkpoint_saver = RedisSaver(
            redis_client=redis_client,
            ttl={
                "default_ttl": 0.1,  # 0.1 minutes = 6 seconds TTL
                "refresh_on_read": True,
            },
        )
        ttl_checkpoint_saver.setup()

        # Create a checkpoint with unique thread ID
        thread_id = f"{unique_prefix}_thread"
        checkpoint_ns = f"{unique_prefix}_ns"
        checkpoint_id = f"{unique_prefix}_checkpoint"

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "id": checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1.0"},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
        }

        # Save the checkpoint (with default TTL of 0.1 minutes = 6 seconds)
        ttl_checkpoint_saver.put(config, checkpoint, metadata, {"test_channel": "1.0"})

        # Verify checkpoint exists immediately after creation
        initial_result = ttl_checkpoint_saver.get_tuple(config)
        assert initial_result is not None, "Checkpoint should exist after creation"

        # Wait for TTL to expire (plus a small buffer)
        time.sleep(7)  # 7 seconds > 6 seconds TTL

        # Verify checkpoint no longer exists
        result = ttl_checkpoint_saver.get_tuple(config)
        assert result is None, "Checkpoint with TTL should expire"
    finally:
        # Clean up
        keys = redis_client.keys(f"checkpoint:*{thread_id}*")
        if keys:
            redis_client.delete(*keys)
        # Do not close the client as it's provided by the fixture


def test_ttl_refresh_on_read(redis_client: Redis) -> None:
    """Test that TTL is refreshed when reading a checkpoint if refresh_on_read is enabled."""
    try:
        # Create unique identifiers to avoid test collisions
        unique_prefix = f"refresh_test_{int(time.time())}"

        # Create a saver with TTL and refresh_on_read enabled
        ttl_checkpoint_saver = RedisSaver(
            redis_client=redis_client,
            ttl={
                "default_ttl": 0.1,  # 0.1 minutes = 6 seconds TTL
                "refresh_on_read": True,
            },
        )
        ttl_checkpoint_saver.setup()

        # Create a checkpoint with unique thread ID
        thread_id = f"{unique_prefix}_thread"
        checkpoint_ns = f"{unique_prefix}_ns"
        checkpoint_id = f"{unique_prefix}_checkpoint"

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "id": checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1.0"},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
        }

        # Save the checkpoint (with default TTL of 0.1 minutes = 6 seconds)
        ttl_checkpoint_saver.put(config, checkpoint, metadata, {"test_channel": "1.0"})

        # Verify checkpoint exists immediately after creation
        initial_result = ttl_checkpoint_saver.get_tuple(config)
        assert initial_result is not None, "Checkpoint should exist after creation"

        # Wait for 3 seconds (less than TTL)
        time.sleep(3)

        # Read the checkpoint (should refresh TTL)
        ttl_checkpoint_saver.get_tuple(config)

        # Wait another 2 seconds (would be 5 seconds total, less than original TTL)
        time.sleep(2)

        # Wait extra time to account for any test delays
        time.sleep(1)

        # Checkpoint should still exist because TTL was refreshed
        result = ttl_checkpoint_saver.get_tuple(config)
        assert result is not None, "Checkpoint should still exist after TTL refresh"

        # Wait for TTL to expire again
        time.sleep(7)

        # Verify checkpoint no longer exists
        result = ttl_checkpoint_saver.get_tuple(config)
        assert result is None, "Checkpoint should expire after refreshed TTL"
    finally:
        # Clean up
        keys = redis_client.keys(f"checkpoint:*{thread_id}*")
        if keys:
            redis_client.delete(*keys)
        # Do not close the client as it's provided by the fixture


def test_put_writes_with_ttl(redis_client: Redis) -> None:
    """Test that writes also expire with TTL."""
    try:
        # Create unique identifiers to avoid test collisions
        unique_prefix = f"writes_test_{int(time.time())}"

        # Create a saver with TTL
        ttl_checkpoint_saver = RedisSaver(
            redis_client=redis_client,
            ttl={
                "default_ttl": 0.1,  # 0.1 minutes = 6 seconds TTL
                "refresh_on_read": True,
            },
        )
        ttl_checkpoint_saver.setup()

        # Create a checkpoint with unique thread ID
        thread_id = f"{unique_prefix}_thread"
        checkpoint_ns = f"{unique_prefix}_ns"
        checkpoint_id = f"{unique_prefix}_checkpoint"

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Create some writes
        ttl_checkpoint_saver.put_writes(
            config, [("test_channel", "test_value")], "test_task_id"
        )

        # Verify writes exist immediately after creation
        initial_writes = ttl_checkpoint_saver._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        assert len(initial_writes) > 0, "Writes should exist after creation"

        # Wait for TTL to expire
        time.sleep(7)  # 7 seconds > 6 seconds TTL

        # Verify writes no longer exist
        writes = ttl_checkpoint_saver._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        assert len(writes) == 0, "Writes with TTL should expire"
    finally:
        # Clean up
        keys = redis_client.keys(f"checkpoint:*{thread_id}*")
        if keys:
            redis_client.delete(*keys)
        # Do not close the client as it's provided by the fixture


def test_no_ttl_when_not_configured(redis_client: Redis) -> None:
    """Test that keys don't expire when TTL is not configured."""
    try:
        # Create unique identifiers to avoid test collisions
        unique_prefix = f"no_ttl_test_{int(time.time())}"

        # Create a saver without TTL
        saver = RedisSaver(redis_client=redis_client)
        saver.setup()

        # Create a checkpoint with unique thread ID
        thread_id = f"{unique_prefix}_thread"
        checkpoint_ns = f"{unique_prefix}_ns"
        checkpoint_id = f"{unique_prefix}_checkpoint"

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "id": checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1.0"},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
        }

        # Save the checkpoint (no TTL configured)
        saver.put(config, checkpoint, metadata, {"test_channel": "1.0"})

        # Verify checkpoint exists immediately after creation
        initial_result = saver.get_tuple(config)
        assert initial_result is not None, "Checkpoint should exist after creation"

        # Wait for the same amount of time that would cause TTL expiration
        time.sleep(7)

        # Verify checkpoint still exists
        result = saver.get_tuple(config)
        assert result is not None, "Checkpoint without TTL should not expire"
    finally:
        # Clean up
        keys = redis_client.keys(f"checkpoint:*{thread_id}*")
        if keys:
            redis_client.delete(*keys)
        # Do not close the client as it's provided by the fixture


def test_simple_graph_with_ttl(redis_client: Redis) -> None:
    """Test a simple graph with TTL configuration."""
    # Use an isolated Redis client to prevent interference from parallel tests
    unique_prefix = f"graph_test_{int(time.time())}"
    thread_id = f"{unique_prefix}_thread"

    def add_one(state):
        """Add one to the state."""
        state["count"] = state.get("count", 0) + 1
        return state

    # Define a simple graph
    builder = StateGraph(State)
    builder.add_node("add_one", add_one)
    builder.set_entry_point("add_one")
    builder.set_finish_point("add_one")

    try:
        # Create a checkpointer with TTL
        with RedisSaver.from_conn_string(
            redis_client=redis_client,
            ttl={"default_ttl": 0.1, "refresh_on_read": True},  # 6 seconds TTL
        ) as checkpointer:
            checkpointer.setup()

            # Compile the graph with the checkpointer
            graph = builder.compile(checkpointer=checkpointer)

            # Use the graph with a specific thread_id
            config = {"configurable": {"thread_id": thread_id}}

            # Initial run
            result = graph.invoke({"count": 0}, config=config)
            assert result["count"] == 1, "Initial count should be 1"

            # Run again immediately - should continue from checkpoint
            result = graph.invoke({}, config=config)
            assert result["count"] == 2, "Count should increment to 2 from checkpoint"

            # Wait for TTL to expire
            time.sleep(7)  # Wait longer than the 6 second TTL

            # Run again - should start from beginning since checkpoint expired
            result = graph.invoke({}, config=config)
            assert (
                result["count"] == 1
            ), "Count should reset to 1 after checkpoint expired"
    finally:
        # Clean up
        keys = redis_client.keys(f"checkpoint:*{thread_id}*")
        if keys:
            redis_client.delete(*keys)
        # Do not close the client as it's provided by the fixture
