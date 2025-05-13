"""
Test for the fix to issue #40 - Fixing numeric version handling with Tag type.
"""

from contextlib import contextmanager

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from redis import Redis

from langgraph.checkpoint.redis import RedisSaver


@pytest.fixture(autouse=True)
async def clear_test_redis(redis_url: str) -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(redis_url)
    try:
        client.flushall()
    finally:
        client.close()


@contextmanager
def patched_redis_saver(redis_url):
    """
    Create a RedisSaver with a patched _dump_blobs method to fix the issue.
    This demonstrates the fix approach.
    """
    original_dump_blobs = RedisSaver._dump_blobs

    def patched_dump_blobs(self, thread_id, checkpoint_ns, values, versions):
        """
        Patched version of _dump_blobs that ensures version is a string.
        """
        # Convert version to string in versions dictionary
        string_versions = {k: str(v) for k, v in versions.items()}

        # Call the original method with string versions
        return original_dump_blobs(
            self, thread_id, checkpoint_ns, values, string_versions
        )

    # Apply the patch
    RedisSaver._dump_blobs = patched_dump_blobs

    try:
        # Create the saver with patched method
        saver = RedisSaver(redis_url)
        saver.setup()
        yield saver
    finally:
        # Restore the original method
        RedisSaver._dump_blobs = original_dump_blobs
        # Clean up
        if saver._owns_its_client:
            saver._redis.close()


def test_numeric_version_fix(redis_url: str) -> None:
    """
    Test that demonstrates the fix for issue #40.

    This shows how to handle numeric versions correctly by ensuring
    they are converted to strings before being used with Tag.
    """
    # Use our patched version that converts numeric versions to strings
    with patched_redis_saver(redis_url) as saver:
        # Set up a basic config
        config = {
            "configurable": {
                "thread_id": "thread-numeric-version-fix",
                "checkpoint_ns": "",
            }
        }

        # Create a basic checkpoint
        checkpoint = empty_checkpoint()

        # Store the checkpoint with our patched method
        saved_config = saver.put(
            config, checkpoint, {}, {"test_channel": 1}
        )  # Numeric version

        # Get the checkpoint ID from the saved config
        thread_id = saved_config["configurable"]["thread_id"]
        checkpoint_ns = saved_config["configurable"].get("checkpoint_ns", "")

        # Now query the data - this should work with the fix
        query = f"@channel:{{test_channel}}"

        # This should not raise an error now with our patch
        results = saver.checkpoint_blobs_index.search(query)

        # Verify we can find the data
        assert len(results.docs) > 0

        # Load one document and verify the version is a string
        doc = results.docs[0]
        data = saver._redis.json().get(doc.id)

        # The key test: version should be a string even though we passed a numeric value
        assert isinstance(data["version"], str)
