"""Test for handling missing blob attribute in shallow checkpoint savers.

This test verifies the fix for GitHub issue #80 where AsyncShallowRedisSaver
and ShallowRedisSaver would raise AttributeError when encountering documents
without a blob attribute.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.constants import TASKS
from testcontainers.redis import RedisContainer

from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver


class MockDocument:
    """Mock document that simulates missing blob attribute."""

    def __init__(self, data: Dict[str, Any]):
        self.type = data.get("type", "")
        self.task_path = data.get("task_path", "")
        self.task_id = data.get("task_id", "")
        self.idx = data.get("idx", 0)
        # Intentionally NOT setting blob attribute to simulate the issue
        # In some cases, we'll set it to None to test that case too
        if "blob" in data:
            self.blob = data["blob"]


@pytest.fixture
def redis_url():
    """Create a Redis container for testing."""
    redis_container = RedisContainer("redis:8")
    redis_container.start()
    yield f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"
    redis_container.stop()


@pytest.mark.asyncio
async def test_async_shallow_missing_blob_attribute(redis_url: str) -> None:
    """Test that AsyncShallowRedisSaver handles documents without blob attribute."""
    async with AsyncShallowRedisSaver(redis_url=redis_url) as saver:
        # Create mock search result with documents missing blob attribute
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "type": "test_type1",
                    "task_path": "path1",
                    "task_id": "task1",
                    "idx": 0,
                }
            ),
            MockDocument(
                {
                    "type": "test_type2",
                    "task_path": "path2",
                    "task_id": "task2",
                    "idx": 1,
                }
            ),
            MockDocument(
                {
                    "type": "test_type3",
                    "task_path": "path3",
                    "task_id": "task3",
                    "idx": 2,
                    "blob": None,
                }
            ),
        ]

        # Mock the search method
        original_search = saver.checkpoint_writes_index.search

        async def mock_search(_):
            return mock_search_result

        saver.checkpoint_writes_index.search = mock_search

        try:
            # This should NOT raise AttributeError with the fix
            result = await saver._aload_pending_sends("test_thread", "test_ns")

            # Result should be empty list since all docs have missing or None blob
            assert (
                result == []
            ), f"Expected empty list when blob is missing/None, got {result}"

        finally:
            # Restore original search method
            saver.checkpoint_writes_index.search = original_search


@pytest.mark.asyncio
async def test_async_shallow_with_valid_blob(redis_url: str) -> None:
    """Test that AsyncShallowRedisSaver correctly processes documents with valid blob."""
    async with AsyncShallowRedisSaver(redis_url=redis_url) as saver:
        # Create mock search result with documents having valid blob data
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "type": "test_type1",
                    "task_path": "path1",
                    "task_id": "task1",
                    "idx": 0,
                    "blob": b"data1",
                }
            ),
            MockDocument(
                {
                    "type": "test_type2",
                    "task_path": "path2",
                    "task_id": "task2",
                    "idx": 1,
                    "blob": b"data2",
                }
            ),
            MockDocument(
                {
                    "type": "test_type3",
                    "task_path": "path3",
                    "task_id": "task3",
                    "idx": 2,
                    "blob": None,
                }
            ),
            MockDocument(
                {
                    "type": "test_type4",
                    "task_path": "path4",
                    "task_id": "task4",
                    "idx": 3,
                }
            ),  # Missing blob
        ]

        # Mock the search method
        original_search = saver.checkpoint_writes_index.search

        async def mock_search(_):
            return mock_search_result

        saver.checkpoint_writes_index.search = mock_search

        try:
            result = await saver._aload_pending_sends("test_thread", "test_ns")

            # Should only include documents with valid (non-None) blob
            assert (
                len(result) == 2
            ), f"Expected 2 results with valid blobs, got {len(result)}"
            assert result[0] == ("test_type1", b"data1")
            assert result[1] == ("test_type2", b"data2")

        finally:
            saver.checkpoint_writes_index.search = original_search


def test_sync_shallow_missing_blob_attribute(redis_url: str) -> None:
    """Test that ShallowRedisSaver handles documents without blob attribute."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        # Create mock search result with documents missing blob attribute
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "type": "test_type1",
                    "task_path": "path1",
                    "task_id": "task1",
                    "idx": 0,
                }
            ),
            MockDocument(
                {
                    "type": "test_type2",
                    "task_path": "path2",
                    "task_id": "task2",
                    "idx": 1,
                }
            ),
            MockDocument(
                {
                    "type": "test_type3",
                    "task_path": "path3",
                    "task_id": "task3",
                    "idx": 2,
                    "blob": None,
                }
            ),
        ]

        # Mock the search method
        original_search = saver.checkpoint_writes_index.search
        saver.checkpoint_writes_index.search = lambda _: mock_search_result

        try:
            # This should NOT raise AttributeError with the fix
            result = saver._load_pending_sends("test_thread", "test_ns")

            # Result should be empty list since all docs have missing or None blob
            assert (
                result == []
            ), f"Expected empty list when blob is missing/None, got {result}"

        finally:
            # Restore original search method
            saver.checkpoint_writes_index.search = original_search


def test_sync_shallow_with_valid_blob(redis_url: str) -> None:
    """Test that ShallowRedisSaver correctly processes documents with valid blob."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        # Create mock search result with documents having valid blob data
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "type": "test_type1",
                    "task_path": "path1",
                    "task_id": "task1",
                    "idx": 0,
                    "blob": b"data1",
                }
            ),
            MockDocument(
                {
                    "type": "test_type2",
                    "task_path": "path2",
                    "task_id": "task2",
                    "idx": 1,
                    "blob": b"data2",
                }
            ),
            MockDocument(
                {
                    "type": "test_type3",
                    "task_path": "path3",
                    "task_id": "task3",
                    "idx": 2,
                    "blob": None,
                }
            ),
            MockDocument(
                {
                    "type": "test_type4",
                    "task_path": "path4",
                    "task_id": "task4",
                    "idx": 3,
                }
            ),  # Missing blob
        ]

        # Mock the search method
        original_search = saver.checkpoint_writes_index.search
        saver.checkpoint_writes_index.search = lambda _: mock_search_result

        try:
            result = saver._load_pending_sends("test_thread", "test_ns")

            # Should only include documents with valid (non-None) blob
            assert (
                len(result) == 2
            ), f"Expected 2 results with valid blobs, got {len(result)}"
            assert result[0] == ("test_type1", b"data1")
            assert result[1] == ("test_type2", b"data2")

        finally:
            saver.checkpoint_writes_index.search = original_search


@pytest.mark.asyncio
async def test_async_shallow_json_path_blob_attribute(redis_url: str) -> None:
    """Test handling of $.blob JSON path attribute in AsyncShallowRedisSaver."""
    async with AsyncShallowRedisSaver(redis_url=redis_url) as saver:
        # Create mock documents with $.blob attribute (JSON path syntax)
        class MockDocumentWithJsonPath:
            def __init__(self, data: Dict[str, Any]):
                self.type = data.get("type", "")
                self.task_path = data.get("task_path", "")
                self.task_id = data.get("task_id", "")
                self.idx = data.get("idx", 0)
                # Set $.blob attribute (JSON path syntax from Redis)
                if "json_blob" in data:
                    setattr(self, "$.blob", data["json_blob"])

        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocumentWithJsonPath(
                {
                    "type": "test_type1",
                    "task_path": "p1",
                    "task_id": "t1",
                    "idx": 0,
                    "json_blob": b"json_data1",
                }
            ),
            MockDocumentWithJsonPath(
                {"type": "test_type2", "task_path": "p2", "task_id": "t2", "idx": 1}
            ),  # No blob at all
            MockDocumentWithJsonPath(
                {
                    "type": "test_type3",
                    "task_path": "p3",
                    "task_id": "t3",
                    "idx": 2,
                    "json_blob": None,
                }
            ),
        ]

        original_search = saver.checkpoint_writes_index.search

        async def mock_search(_):
            return mock_search_result

        saver.checkpoint_writes_index.search = mock_search

        try:
            result = await saver._aload_pending_sends("test_thread", "test_ns")

            # Should only include the document with valid $.blob
            assert (
                len(result) == 1
            ), f"Expected 1 result with valid $.blob, got {len(result)}"
            assert result[0] == ("test_type1", b"json_data1")

        finally:
            saver.checkpoint_writes_index.search = original_search


def test_sync_shallow_json_path_blob_attribute(redis_url: str) -> None:
    """Test handling of $.blob JSON path attribute in ShallowRedisSaver."""
    with ShallowRedisSaver.from_conn_string(redis_url) as saver:
        # Create mock documents with $.blob attribute (JSON path syntax)
        class MockDocumentWithJsonPath:
            def __init__(self, data: Dict[str, Any]):
                self.type = data.get("type", "")
                self.task_path = data.get("task_path", "")
                self.task_id = data.get("task_id", "")
                self.idx = data.get("idx", 0)
                # Set $.blob attribute (JSON path syntax from Redis)
                if "json_blob" in data:
                    setattr(self, "$.blob", data["json_blob"])

        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocumentWithJsonPath(
                {
                    "type": "test_type1",
                    "task_path": "p1",
                    "task_id": "t1",
                    "idx": 0,
                    "json_blob": b"json_data1",
                }
            ),
            MockDocumentWithJsonPath(
                {"type": "test_type2", "task_path": "p2", "task_id": "t2", "idx": 1}
            ),  # No blob at all
            MockDocumentWithJsonPath(
                {
                    "type": "test_type3",
                    "task_path": "p3",
                    "task_id": "t3",
                    "idx": 2,
                    "json_blob": None,
                }
            ),
        ]

        original_search = saver.checkpoint_writes_index.search
        saver.checkpoint_writes_index.search = lambda _: mock_search_result

        try:
            result = saver._load_pending_sends("test_thread", "test_ns")

            # Should only include the document with valid $.blob
            assert (
                len(result) == 1
            ), f"Expected 1 result with valid $.blob, got {len(result)}"
            assert result[0] == ("test_type1", b"json_data1")

        finally:
            saver.checkpoint_writes_index.search = original_search
