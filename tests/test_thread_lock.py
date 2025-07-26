import concurrent.futures

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.redis import RedisSaver


# Helper function to increment a value in a thread-safe manner
def _increment(
    checkpointer: RedisSaver, config: RunnableConfig, iterations: int
) -> None:
    for _ in range(iterations):  # Loop for the specified number of iterations
        # Acquire a thread lock for the given thread ID
        with checkpointer.thread_lock(config["configurable"]["thread_id"]):
            # Retrieve the current checkpoint tuple
            tup = checkpointer.get_tuple(config)
            # Use an empty checkpoint if none exists
            cp = tup.checkpoint if tup else empty_checkpoint()
            # Initialize or update the "count" in channel_values and channel_versions
            cp.setdefault("channel_values", {}).setdefault("count", 0)
            cp.setdefault("channel_versions", {}).setdefault("count", "0")
            cp["channel_values"]["count"] += 1  # Increment the count
            cp["channel_versions"]["count"] = str(
                cp["channel_values"]["count"]
            )  # Update version
            # Save the updated checkpoint
            checkpointer.put(config, cp, {}, {"count": cp["channel_versions"]["count"]})


# Test function to verify thread-safe serialization of checkpoints
def test_thread_lock_serialization(redis_url: str) -> None:
    # Create a RedisSaver instance using the provided Redis URL
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()  # Perform any necessary setup
        # Define a configuration for the checkpoint
        config: RunnableConfig = {
            "configurable": {"thread_id": "t", "checkpoint_ns": ""}
        }
        # Initialize the checkpoint with an empty state
        saver.put(config, empty_checkpoint(), {}, {})

        # Use a ThreadPoolExecutor to simulate concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit two tasks to increment the checkpoint concurrently
            futures = [executor.submit(_increment, saver, config, 5) for _ in range(2)]
            # Wait for all tasks to complete
            for f in futures:
                f.result()

        # Retrieve the final checkpoint
        final = saver.get_tuple(config)
        assert final is not None  # Ensure the checkpoint exists
        # Verify that the count has been incremented correctly
        assert final.checkpoint["channel_values"]["count"] == 10
