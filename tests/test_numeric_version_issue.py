"""
Test for issue #40 - Error when comparing numeric version with Tag type.
"""

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from redis import Redis
from redisvl.query.filter import Tag

from langgraph.checkpoint.redis import RedisSaver


@pytest.fixture(autouse=True)
async def clear_test_redis(redis_url: str) -> None:
    """Clear Redis before each test."""
    client = Redis.from_url(redis_url)
    try:
        client.flushall()
    finally:
        client.close()


def test_numeric_version_issue(redis_url: str) -> None:
    """
    Test reproduction for issue #40.

    This test explicitly creates a scenario where a numeric version field
    is compared with a Tag type, which should cause a TypeError.
    """
    # Create a Redis saver with default configuration
    saver = RedisSaver(redis_url)
    saver.setup()

    try:
        # Here we'll directly test the specific problem from issue #40
        # In the real app, the version field is stored as a number in Redis
        # Then when the code in _dump_blobs tries to pass that numeric version
        # to the Tag filter, it causes a TypeError

        # First create a fixed test with direct Tag usage to demonstrate the issue
        tag_filter = Tag("version")

        with pytest.raises(TypeError) as excinfo:
            # This will trigger the error because we're comparing Tag with integer
            result = tag_filter == 1  # Integer instead of string

        # Verify the specific error message related to Tag comparison
        assert "Right side argument passed to operator" in str(excinfo.value)
        assert "Tag must be of type" in str(excinfo.value)

        # Another approach would be a direct test of our _dump_blobs method
        # by creating a fake numeric version and then trying to create a Tag query
        # based on it
        channel_name = "test_channel"
        numeric_version = 1  # This is the root issue - numeric version not string

        # This mimics the code in _dump_blobs that would fail
        versions = {channel_name: numeric_version}

        # We can't directly patch the method, but we can verify the same type issue
        # Here we simulate what happens when a numeric version is passed to Tag filter
        tag_filter = Tag("version")
        with pytest.raises(TypeError) as excinfo2:
            # This fails because we're comparing a Tag with a numeric value directly
            result = tag_filter == versions[channel_name]  # Numeric version

        # Check the error message
        assert "must be of type" in str(excinfo2.value)

    finally:
        # Clean up
        if saver._owns_its_client:
            saver._redis.close()
