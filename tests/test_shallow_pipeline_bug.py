"""Test to reproduce the pipeline error in shallow checkpoint implementation."""

import pytest
from langgraph.checkpoint.base import Checkpoint
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


def test_put_writes_before_checkpoint_exists():
    """Test that put_writes doesn't fail when checkpoint doesn't exist yet.

    This reproduces the error:
    "Command # 6 (JSON.SET...) of pipeline caused error: new objects must be created at the root"

    The issue occurs in Fanout Graph benchmarks where put_writes may be called
    before a checkpoint has been created via put().
    """
    with RedisContainer("redis:8") as redis_container:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        with ShallowRedisSaver.from_conn_string(redis_url) as checkpointer:
            # Setup indices
            checkpointer.setup()

            # Create a config for a checkpoint that doesn't exist yet
            config = {
                "configurable": {
                    "thread_id": "test_thread",
                    "checkpoint_ns": "test_ns",
                    "checkpoint_id": "test_checkpoint_id",
                }
            }

            # Try to put writes without creating the checkpoint first
            # This should reproduce the pipeline error
            writes = [("channel1", "value1"), ("channel2", "value2")]

            # This should fail with the pipeline error if the bug exists
            # or succeed if the bug is fixed
            try:
                checkpointer.put_writes(config, writes, task_id="test_task")
                # If we get here, the bug is fixed or didn't occur
                print("put_writes succeeded without error")
            except Exception as e:
                # Check if it's the expected error
                error_msg = str(e)
                assert (
                    "new objects must be created at the root" in error_msg
                ), f"Unexpected error: {error_msg}"
                pytest.fail(f"Bug reproduced: {error_msg}")


def test_concurrent_puts_and_writes():
    """Test concurrent puts and writes that might trigger the pipeline error.

    This simulates the Fanout Graph scenario where multiple parallel operations
    might cause put_writes to be called before put().
    """
    with RedisContainer("redis:8") as redis_container:
        redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"

        with ShallowRedisSaver.from_conn_string(redis_url) as checkpointer:
            # Setup indices
            checkpointer.setup()

            thread_id = "fanout_thread"
            checkpoint_ns = "fanout_ns"

            # Simulate multiple parallel operations
            for i in range(5):
                checkpoint_id = f"checkpoint_{i}"

                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }

                # Simulate race condition: put_writes before put
                if i % 2 == 0:
                    # Even iterations: writes before checkpoint
                    writes = [(f"channel_{i}", f"value_{i}")]
                    try:
                        checkpointer.put_writes(config, writes, task_id=f"task_{i}")
                    except Exception as e:
                        error_msg = str(e)
                        if "new objects must be created at the root" in error_msg:
                            pytest.fail(f"Bug reproduced in iteration {i}: {error_msg}")
                        else:
                            raise

                    # Now create the checkpoint
                    checkpoint = Checkpoint(
                        v=1,
                        id=checkpoint_id,
                        ts="2024-01-01T00:00:00+00:00",
                        channel_values={},
                        channel_versions={},
                        versions_seen={},
                    )
                    checkpointer.put(config, checkpoint, {}, {})
                else:
                    # Odd iterations: checkpoint before writes (normal order)
                    checkpoint = Checkpoint(
                        v=1,
                        id=checkpoint_id,
                        ts="2024-01-01T00:00:00+00:00",
                        channel_values={},
                        channel_versions={},
                        versions_seen={},
                    )
                    checkpointer.put(config, checkpoint, {}, {})

                    writes = [(f"channel_{i}", f"value_{i}")]
                    checkpointer.put_writes(config, writes, task_id=f"task_{i}")


if __name__ == "__main__":
    # Run the tests directly
    print("Testing put_writes before checkpoint exists...")
    test_put_writes_before_checkpoint_exists()
    print("\nTesting concurrent puts and writes...")
    test_concurrent_puts_and_writes()
    print("\nAll tests passed!")
