"""Test for GitHub Issue #136: Warning spam when checkpoint_id is UUIDv6 instead of ULID.

LangGraph generates checkpoint IDs using uuid6() format (36 characters) but the shallow
savers try to parse them as ULID (26 characters), causing warning spam in production logs.
"""

import logging
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
)
from redis import Redis

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


@contextmanager
def _saver(redis_url: str) -> Generator[ShallowRedisSaver, None, None]:
    """Fixture for shallow saver testing."""
    saver = ShallowRedisSaver(redis_url)
    saver.setup()
    try:
        yield saver
    finally:
        pass


@asynccontextmanager
async def _async_saver(redis_url: str) -> AsyncGenerator[AsyncShallowRedisSaver, None]:
    """Fixture for async shallow saver testing."""
    async with AsyncShallowRedisSaver.from_conn_string(redis_url) as saver:
        yield saver


def test_uuid6_checkpoint_id_no_warning(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that UUIDv6 checkpoint IDs don't produce warning spam.

    This test verifies the fix for GitHub Issue #136:
    https://github.com/redis-developer/langgraph-redis/issues/136

    LangGraph generates checkpoint IDs using uuid6() format (36 characters like
    '1f0be35a-360e-6154-8002-cb3ee66bf299'), but the shallow savers were trying
    to parse them as ULID (26 characters), causing warning spam.
    """
    with _saver(redis_url) as saver:
        # Create a UUIDv6-style checkpoint ID (36 characters with dashes)
        # This is the format that LangGraph uses
        uuid6_checkpoint_id = "1f0be35a-360e-6154-8002-cb3ee66bf299"

        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": uuid6_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": uuid6_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}

        # Capture log output during put operation
        with caplog.at_level(logging.WARNING):
            saver.put(config, checkpoint, metadata, {})

        # Verify NO warning about "Invalid ULID" was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings but got {len(invalid_ulid_warnings)}: "
            f"{invalid_ulid_warnings}"
        )


def test_uuid4_checkpoint_id_no_warning(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that standard UUID4 checkpoint IDs don't produce warning spam.

    Even regular UUID4 format IDs should not produce warnings - they should
    silently fall back to using the checkpoint's timestamp field.
    """
    with _saver(redis_url) as saver:
        # Create a standard UUID4 checkpoint ID
        uuid4_checkpoint_id = str(uuid.uuid4())

        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": uuid4_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": uuid4_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}

        # Capture log output during put operation
        with caplog.at_level(logging.WARNING):
            saver.put(config, checkpoint, metadata, {})

        # Verify NO warning about "Invalid ULID" was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings but got {len(invalid_ulid_warnings)}: "
            f"{invalid_ulid_warnings}"
        )


def test_ulid_checkpoint_id_still_works(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that ULID checkpoint IDs still work correctly.

    This ensures the fix doesn't break existing ULID support.
    """
    from ulid import ULID

    with _saver(redis_url) as saver:
        # Create a valid ULID checkpoint ID
        ulid_checkpoint_id = str(ULID())

        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ulid_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": ulid_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}

        # Capture log output during put operation
        with caplog.at_level(logging.WARNING):
            saver.put(config, checkpoint, metadata, {})

        # Verify NO warning was logged for valid ULID
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings for valid ULID but got: "
            f"{invalid_ulid_warnings}"
        )

        # Verify the checkpoint was stored and can be retrieved
        result = saver.get_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == ulid_checkpoint_id


def test_list_before_with_uuid6_no_warning(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that list() with 'before' filter using UUIDv6 doesn't produce warnings.

    The list() method also tries to parse checkpoint IDs as ULID for the 'before'
    filter, which should also handle UUIDv6 gracefully.
    """
    with _saver(redis_url) as saver:
        # First create a checkpoint
        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""
        uuid6_checkpoint_id = "1f0be35a-360e-6154-8002-cb3ee66bf299"

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": uuid6_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": uuid6_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}
        saver.put(config, checkpoint, metadata, {})

        # Now try to list with a 'before' filter using UUIDv6
        before_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": "1f0be35a-360e-6154-8002-cb3ee66bf300",
            }
        }

        # Clear previous log records
        caplog.clear()

        # Capture log output during list operation
        with caplog.at_level(logging.WARNING):
            list(saver.list(config, before=before_config))

        # Verify NO warning about "Invalid ULID" was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        # Note: The current implementation silently ignores non-ULID 'before' IDs
        # without warning, so this should pass. But we document it here for clarity.
        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings but got {len(invalid_ulid_warnings)}: "
            f"{invalid_ulid_warnings}"
        )


# ===================== Async Tests =====================


@pytest.mark.asyncio
async def test_async_uuid6_checkpoint_id_no_warning(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that async saver handles UUIDv6 checkpoint IDs without warning.

    This test verifies the fix for GitHub Issue #136 for the async implementation.
    """
    async with _async_saver(redis_url) as saver:
        # Create a UUIDv6-style checkpoint ID (36 characters with dashes)
        uuid6_checkpoint_id = "1f0be35a-360e-6154-8002-cb3ee66bf299"

        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": uuid6_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": uuid6_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}

        # Capture log output during put operation
        with caplog.at_level(logging.WARNING):
            await saver.aput(config, checkpoint, metadata, {})

        # Verify NO warning about "Invalid ULID" was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings but got {len(invalid_ulid_warnings)}: "
            f"{invalid_ulid_warnings}"
        )

        # Verify the checkpoint was stored and can be retrieved
        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == uuid6_checkpoint_id


@pytest.mark.asyncio
async def test_async_ulid_checkpoint_id_still_works(
    redis_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that async saver still handles ULID checkpoint IDs correctly.

    This ensures the fix doesn't break existing ULID support.
    """
    from ulid import ULID

    async with _async_saver(redis_url) as saver:
        # Create a valid ULID checkpoint ID
        ulid_checkpoint_id = str(ULID())

        thread_id = f"test_thread_{uuid.uuid4()}"
        checkpoint_ns = ""

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ulid_checkpoint_id,
            }
        }

        checkpoint: Checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00Z",
            "id": ulid_checkpoint_id,
            "channel_values": {"test_channel": "test_value"},
            "channel_versions": {"test_channel": "1"},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata: CheckpointMetadata = {"source": "input", "step": 1}

        # Capture log output during put operation
        with caplog.at_level(logging.WARNING):
            await saver.aput(config, checkpoint, metadata, {})

        # Verify NO warning was logged for valid ULID
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]

        invalid_ulid_warnings = [
            msg for msg in warning_messages if "Invalid ULID" in msg
        ]

        assert len(invalid_ulid_warnings) == 0, (
            f"Expected no 'Invalid ULID' warnings for valid ULID but got: "
            f"{invalid_ulid_warnings}"
        )

        # Verify the checkpoint was stored and can be retrieved
        result = await saver.aget_tuple(config)
        assert result is not None
        assert result.checkpoint["id"] == ulid_checkpoint_id
