"""
Regression test for Issue #116: AsyncRedisSaver AttributeError when calling aget_state_history()

This test verifies that the async implementation correctly handles blob access
when using _abatch_load_pending_sends with the JSON path syntax ($.blob).

The bug manifested as:
    AttributeError: 'Document' object has no attribute 'blob'

This was caused by a mismatch between:
1. The return_fields specification ("blob" instead of "$.blob")
2. The attribute access pattern (direct access d.blob instead of getattr(d, "$.blob", ...))

The fix aligns the async implementation with the sync version by:
1. Using "$.blob" in return_fields
2. Using getattr(doc, "$.blob", getattr(doc, "blob", b"")) for access
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from langgraph.checkpoint.redis.aio import AsyncRedisSaver


class MockDocument:
    """Mock document that simulates Redis JSON path attribute behavior."""

    def __init__(self, data: Dict[str, Any]):
        self.checkpoint_id = data.get("checkpoint_id", "")
        self.type = data.get("type", "")
        self.task_path = data.get("task_path", "")
        self.task_id = data.get("task_id", "")
        self.idx = data.get("idx", 0)
        # When using "$.blob" in return_fields, Redis returns it as "$.blob" attribute
        if "json_blob" in data:
            setattr(self, "$.blob", data["json_blob"])


@pytest.mark.asyncio
async def test_abatch_load_pending_sends_with_json_path_blob(redis_url: str) -> None:
    """
    Test that _abatch_load_pending_sends correctly handles $.blob JSON path attribute.

    This is a unit test with mocked Redis responses that directly tests the bug fix.
    Before the fix, accessing d.blob would raise AttributeError because Redis returns
    the attribute as "$.blob" (not "blob") when you specify "$.blob" in return_fields.
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()

        # Create mock search result with documents using $.blob (JSON path syntax)
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "checkpoint_id": "checkpoint_1",
                    "type": "test_type1",
                    "task_path": "path1",
                    "task_id": "task1",
                    "idx": 0,
                    "json_blob": b"data1",  # This becomes $.blob attribute
                }
            ),
            MockDocument(
                {
                    "checkpoint_id": "checkpoint_1",
                    "type": "test_type2",
                    "task_path": "path2",
                    "task_id": "task2",
                    "idx": 1,
                    "json_blob": b"data2",  # This becomes $.blob attribute
                }
            ),
            MockDocument(
                {
                    "checkpoint_id": "checkpoint_2",
                    "type": "test_type3",
                    "task_path": "path3",
                    "task_id": "task3",
                    "idx": 0,
                    "json_blob": b"data3",  # This becomes $.blob attribute
                }
            ),
        ]

        # Mock the search method to return our mock documents
        original_search = saver.checkpoint_writes_index.search

        async def mock_search(_: Any) -> MagicMock:
            return mock_search_result

        saver.checkpoint_writes_index.search = mock_search

        try:
            # Call the method that was failing before the fix
            # This internally tries to access d.blob which would fail without the fix
            result = await saver._abatch_load_pending_sends(
                [
                    ("test_thread", "test_ns", "checkpoint_1"),
                    ("test_thread", "test_ns", "checkpoint_2"),
                ]
            )

            # Verify results are correctly extracted
            assert ("test_thread", "test_ns", "checkpoint_1") in result
            assert ("test_thread", "test_ns", "checkpoint_2") in result

            # Verify the blob data was correctly accessed via $.blob
            checkpoint_1_data = result[("test_thread", "test_ns", "checkpoint_1")]
            assert len(checkpoint_1_data) == 2
            assert checkpoint_1_data[0] == ("test_type1", b"data1")
            assert checkpoint_1_data[1] == ("test_type2", b"data2")

            checkpoint_2_data = result[("test_thread", "test_ns", "checkpoint_2")]
            assert len(checkpoint_2_data) == 1
            assert checkpoint_2_data[0] == ("test_type3", b"data3")

        finally:
            # Restore original search method
            saver.checkpoint_writes_index.search = original_search


@pytest.mark.asyncio
async def test_abatch_load_pending_sends_handles_missing_blob(redis_url: str) -> None:
    """
    Test that _abatch_load_pending_sends gracefully handles missing blob attributes.

    This tests the fallback logic: getattr(doc, "$.blob", getattr(doc, "blob", b""))
    """
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()

        # Create mock documents - some with $.blob, some without
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MockDocument(
                {
                    "checkpoint_id": "checkpoint_1",
                    "type": "test_type1",
                    "task_path": "p1",
                    "task_id": "t1",
                    "idx": 0,
                    "json_blob": b"data1",
                }
            ),
            MockDocument(
                {
                    "checkpoint_id": "checkpoint_1",
                    "type": "test_type2",
                    "task_path": "p2",
                    "task_id": "t2",
                    "idx": 1,
                    # No json_blob - this simulates missing $.blob attribute
                }
            ),
        ]

        original_search = saver.checkpoint_writes_index.search

        async def mock_search(_: Any) -> MagicMock:
            return mock_search_result

        saver.checkpoint_writes_index.search = mock_search

        try:
            result = await saver._abatch_load_pending_sends(
                [("test_thread", "test_ns", "checkpoint_1")]
            )

            # Should handle the missing blob gracefully with empty bytes fallback
            checkpoint_data = result[("test_thread", "test_ns", "checkpoint_1")]
            assert len(checkpoint_data) == 2
            assert checkpoint_data[0] == ("test_type1", b"data1")
            assert checkpoint_data[1] == ("test_type2", b"")  # Fallback to b""

        finally:
            saver.checkpoint_writes_index.search = original_search
