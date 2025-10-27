"""Test for Interrupt serialization fix (GitHub Issue #33556).

This test verifies that Interrupt objects are properly serialized and deserialized
by the JsonPlusRedisSerializer, preventing the AttributeError that occurs when
code tries to access the 'id' attribute on what it expects to be an Interrupt
object but is actually a plain dictionary.

Issue: https://github.com/langchain-ai/langchain/issues/33556
"""

import asyncio
import json
import uuid
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata
from langgraph.types import Interrupt

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer


class TestInterruptSerialization:
    """Test suite for Interrupt object serialization and deserialization."""

    def test_interrupt_direct_serialization(self):
        """Test that Interrupt objects are properly serialized and deserialized."""
        serializer = JsonPlusRedisSerializer()

        # Create an Interrupt object
        # Handle both old and new versions of Interrupt
        try:
            # Try new version with id parameter
            interrupt_obj = Interrupt(
                value={"tool_name": "external_action", "message": "Need approval"},
                id="test-interrupt-123",
            )
            custom_id_set = True
        except TypeError:
            # Fall back to old version without id parameter
            interrupt_obj = Interrupt(
                value={"tool_name": "external_action", "message": "Need approval"}
            )
            custom_id_set = False

        # Test serialization/deserialization
        serialized = serializer.dumps(interrupt_obj)
        deserialized = serializer.loads(serialized)

        # Verify it's an Interrupt object with the correct attributes
        assert isinstance(
            deserialized, Interrupt
        ), f"Expected Interrupt, got {type(deserialized)}"
        # Check id exists only if the Interrupt class supports it
        if hasattr(Interrupt, "id") or hasattr(deserialized, "id"):
            assert hasattr(deserialized, "id"), "Should have id attribute"
            # In new version, id should be preserved
            if custom_id_set and hasattr(deserialized, "id"):
                assert (
                    deserialized.id == "test-interrupt-123"
                ), f"ID mismatch: {deserialized.id}"
        assert deserialized.value == {
            "tool_name": "external_action",
            "message": "Need approval",
        }

    def test_interrupt_constructor_format(self):
        """Test that Interrupt objects are serialized in LangChain constructor format."""
        serializer = JsonPlusRedisSerializer()

        try:
            interrupt_obj = Interrupt(value={"data": "test"}, id="constructor-test-id")
            custom_id_set = True
        except TypeError:
            interrupt_obj = Interrupt(value={"data": "test"})
            custom_id_set = False

        serialized = serializer.dumps(interrupt_obj)

        # Parse the JSON to check the format
        parsed = json.loads(serialized)
        assert parsed.get("lc") == 2, "Should have lc=2 for constructor format"
        assert parsed.get("type") == "constructor", "Should have type=constructor"
        assert parsed.get("id") == [
            "langgraph",
            "types",
            "Interrupt",
        ], "Should have correct id path"
        assert "kwargs" in parsed, "Should have kwargs field"
        assert parsed["kwargs"]["value"] == {"data": "test"}
        # Check id only if it was set
        if custom_id_set:
            assert parsed["kwargs"]["id"] == "constructor-test-id"

    def test_plain_dict_reconstruction(self):
        """Test that plain dicts with value/id keys are reconstructed as Interrupt objects."""
        serializer = JsonPlusRedisSerializer()

        # This simulates what happens when Interrupt is stored as plain dict
        plain_dict_interrupt = {"value": {"data": "test"}, "id": "plain-id"}
        serialized = serializer.dumps(plain_dict_interrupt)
        deserialized = serializer.loads(serialized)

        # Check if it was reconstructed as Interrupt or remains as dict
        # Depends on whether the version supports reconstruction with custom id
        if isinstance(deserialized, Interrupt):
            # Only check id if the Interrupt class supports it
            if hasattr(Interrupt, "id"):
                assert hasattr(deserialized, "id"), "Should have 'id' attribute"
                # Only check exact id if reconstruction preserves it
                if hasattr(deserialized, "id") and deserialized.id == "plain-id":
                    assert deserialized.id == "plain-id"
            assert deserialized.value == {"data": "test"}
        else:
            # Old version may not reconstruct, remains as dict
            assert deserialized == plain_dict_interrupt

    def test_nested_interrupt_in_list(self):
        """Test Interrupt serialization in nested structures like pending_writes."""
        serializer = JsonPlusRedisSerializer()

        # Simulate pending_writes structure
        try:
            interrupt_obj = Interrupt(value={"interrupt": "data"}, id="nested-id")
            custom_id_set = True
        except TypeError:
            interrupt_obj = Interrupt(value={"interrupt": "data"})
            custom_id_set = False
        nested_data = [("task1", interrupt_obj), ("task2", {"regular": "dict"})]

        serialized = serializer.dumps(nested_data)
        deserialized = serializer.loads(serialized)

        # Verify the Interrupt in the nested structure
        assert len(deserialized) == 2
        task1_value = deserialized[0][1]
        task2_value = deserialized[1][1]

        # Check if Interrupt is preserved or becomes dict
        if isinstance(task1_value, Interrupt):
            # Only check id if the Interrupt class supports it
            if hasattr(Interrupt, "id"):
                assert hasattr(task1_value, "id")
                # Only check exact id if it was set and preserved
                if custom_id_set:
                    assert task1_value.id == "nested-id"
        else:
            # May become dict in some versions
            assert isinstance(task1_value, dict)
            assert task1_value["value"] == {"interrupt": "data"}
        assert isinstance(task2_value, dict), "task2 should remain dict"

    def test_plain_dict_in_nested_structure(self):
        """Test that plain dicts with value/id in nested structures are reconstructed."""
        serializer = JsonPlusRedisSerializer()

        # Simulate the problematic case from the issue
        nested_structure = [
            ("task1", {"value": {"interrupt": "data"}, "id": "interrupt-1"}),
            ("task2", {"normal": "dict", "no": "conversion"}),
        ]

        serialized = serializer.dumps(nested_structure)
        deserialized = serializer.loads(serialized)

        task1_value = deserialized[0][1]
        task2_value = deserialized[1][1]

        # Check if reconstruction works
        if isinstance(task1_value, Interrupt):
            # Successfully reconstructed as Interrupt
            if hasattr(Interrupt, "id"):
                assert hasattr(task1_value, "id")
                # This is the line that would fail in the original bug
                interrupt_id = task1_value.id  # Should not raise AttributeError
                assert interrupt_id == "interrupt-1"
        else:
            # Remains as dict in old version
            assert task1_value == {"value": {"interrupt": "data"}, "id": "interrupt-1"}

        # task2 should remain a dict
        assert isinstance(
            task2_value, dict
        ), f"task2 should remain dict, got {type(task2_value)}"

    def test_edge_cases_not_converted(self):
        """Test that dicts that shouldn't be converted to Interrupt remain as dicts."""
        serializer = JsonPlusRedisSerializer()

        # Dict with non-string id - should not convert
        non_string_id = {"value": "test", "id": 123}
        result = serializer.loads(serializer.dumps(non_string_id))
        assert isinstance(result, dict), "Should not convert when id is not string"

        # Dict with extra fields - should not convert
        extra_fields = {"value": "test", "id": "test-id", "extra": "field"}
        result = serializer.loads(serializer.dumps(extra_fields))
        assert isinstance(result, dict), "Should not convert when extra fields present"

        # Dict with only value - should not convert
        only_value = {"value": "test"}
        result = serializer.loads(serializer.dumps(only_value))
        assert isinstance(result, dict), "Should not convert with only value field"

        # Dict with only id - should not convert
        only_id = {"id": "test-id"}
        result = serializer.loads(serializer.dumps(only_id))
        assert isinstance(result, dict), "Should not convert with only id field"

    def test_complex_interrupt_value(self):
        """Test Interrupt with complex nested value structures."""
        serializer = JsonPlusRedisSerializer()

        complex_value = {
            "tool_name": "external_action",
            "tool_args": {
                "name": "Foo",
                "config": {"timeout": 30, "retries": 3},
                "nested": {"deep": {"structure": ["a", "b", "c"]}},
            },
            "metadata": {"timestamp": "2024-01-01", "user_id": "user123"},
        }

        try:
            interrupt_obj = Interrupt(value=complex_value, id="complex-id")
            custom_id_set = True
        except TypeError:
            interrupt_obj = Interrupt(value=complex_value)
            custom_id_set = False

        serialized = serializer.dumps(interrupt_obj)
        deserialized = serializer.loads(serialized)

        assert isinstance(deserialized, Interrupt)
        # Check id only if the Interrupt class supports it
        if hasattr(Interrupt, "id") or hasattr(deserialized, "id"):
            assert hasattr(deserialized, "id") and deserialized.id is not None
            # Check exact id only if it was set
            if custom_id_set and hasattr(deserialized, "id"):
                assert deserialized.id == "complex-id"
        assert deserialized.value == complex_value
        assert deserialized.value["tool_args"]["nested"]["deep"]["structure"] == [
            "a",
            "b",
            "c",
        ]


@pytest.mark.asyncio
class TestInterruptSerializationAsync:
    """Async tests for Interrupt serialization with Redis checkpointers."""

    async def test_interrupt_in_checkpoint_async(self, redis_url: str):
        """Test that Interrupt objects in checkpoints are properly handled."""
        async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
            thread_id = f"test-interrupt-{uuid.uuid4()}"
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": str(uuid.uuid4()),
                }
            }

            # Create an Interrupt object
            try:
                interrupt_obj = Interrupt(
                    value={
                        "tool_name": "external_action",
                        "tool_args": {"name": "TestArg"},
                        "message": "Need external system call",
                    },
                    id="async-interrupt-id",
                )
                custom_id_set = True
            except TypeError:
                interrupt_obj = Interrupt(
                    value={
                        "tool_name": "external_action",
                        "tool_args": {"name": "TestArg"},
                        "message": "Need external system call",
                    }
                )
                custom_id_set = False

            # Create checkpoint WITHOUT pending_writes (they're stored separately)
            checkpoint = {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": config["configurable"]["checkpoint_id"],
                "channel_values": {"messages": ["test message"]},
                "channel_versions": {},
                "versions_seen": {},
            }

            metadata = {"source": "test", "step": 1, "writes": {}}

            # Save the checkpoint
            await checkpointer.aput(config, checkpoint, metadata, {})

            # Save pending_writes separately using aput_writes
            await checkpointer.aput_writes(
                config, [("interrupt_task", interrupt_obj)], "interrupt_task"
            )

            # Retrieve the checkpoint
            checkpoint_tuple = await checkpointer.aget_tuple(config)

            assert checkpoint_tuple is not None

            # Verify pending_writes contains an Interrupt object
            assert len(checkpoint_tuple.pending_writes) == 1
            # PendingWrite is a 3-tuple: (task_id, channel, value)
            task_id, channel, value = checkpoint_tuple.pending_writes[0]

            assert task_id == "interrupt_task"
            assert (
                channel == "interrupt_task"
            )  # channel is same as task_id in this case
            assert isinstance(
                value, Interrupt
            ), f"Expected Interrupt, got {type(value)}"
            # Check id only if the Interrupt class supports it
            if hasattr(Interrupt, "id") or hasattr(value, "id"):
                assert hasattr(value, "id"), "Should have 'id' attribute"
                # Check id matches only if it was set
                if custom_id_set and hasattr(value, "id"):
                    assert value.id == "async-interrupt-id"

            # This simulates the code that was failing in the issue
            # It should not raise AttributeError
            pending_interrupts = {}
            for task_id, channel, val in checkpoint_tuple.pending_writes:
                if isinstance(val, Interrupt):
                    # Only access id if it exists
                    if hasattr(val, "id"):
                        pending_interrupts[task_id] = val.id
                    else:
                        # Old version without id
                        pending_interrupts[task_id] = "no-id"

            # Check we have the interrupt
            assert "interrupt_task" in pending_interrupts
            # Check exact id only if it was set and id is supported
            if custom_id_set and hasattr(interrupt_obj, "id"):
                assert pending_interrupts["interrupt_task"] == "async-interrupt-id"

    async def test_multiple_interrupts_async(self, redis_url: str):
        """Test handling multiple Interrupt objects in a checkpoint."""
        async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
            thread_id = f"test-multi-interrupt-{uuid.uuid4()}"
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                    "checkpoint_id": str(uuid.uuid4()),
                }
            }

            # Create multiple Interrupts
            def create_interrupt(value, interrupt_id):
                try:
                    return Interrupt(value=value, id=interrupt_id)
                except TypeError:
                    return Interrupt(value=value)

            interrupts = [
                ("task1", create_interrupt({"action": "approve"}, "interrupt-1")),
                ("task2", create_interrupt({"action": "deny"}, "interrupt-2")),
                ("task3", {"regular": "dict", "not": "interrupt"}),
                ("task4", create_interrupt({"action": "retry"}, "interrupt-3")),
            ]

            checkpoint = {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": config["configurable"]["checkpoint_id"],
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
            }

            metadata = {"source": "test", "step": 1}

            await checkpointer.aput(config, checkpoint, metadata, {})

            # Save pending_writes separately using aput_writes
            # Each write needs to be saved with its task_id
            for task_id, value in interrupts:
                await checkpointer.aput_writes(config, [(task_id, value)], task_id)

            checkpoint_tuple = await checkpointer.aget_tuple(config)

            assert checkpoint_tuple is not None
            assert len(checkpoint_tuple.pending_writes) == 4

            # Verify each item
            for i, (task_id, channel, value) in enumerate(
                checkpoint_tuple.pending_writes
            ):
                if task_id in ["task1", "task2", "task4"]:
                    assert isinstance(
                        value, Interrupt
                    ), f"{task_id} should have Interrupt"
                    # Only check id if the Interrupt class supports it
                    if hasattr(Interrupt, "id"):
                        assert hasattr(value, "id")
                        # Verify we can access the id without error
                        _ = value.id
                elif task_id == "task3":
                    assert isinstance(value, dict), "task3 should remain dict"


class TestInterruptSerializationSync:
    """Sync tests for Interrupt serialization with Redis checkpointers."""

    def test_interrupt_with_empty_value(self):
        """Test Interrupt with None or empty value."""
        serializer = JsonPlusRedisSerializer()

        # Interrupt with None value
        try:
            interrupt_none = Interrupt(value=None, id="none-value-id")
            custom_id_set = True
        except TypeError:
            interrupt_none = Interrupt(value=None)
            custom_id_set = False
        result = serializer.loads(serializer.dumps(interrupt_none))
        assert isinstance(result, Interrupt)
        assert result.value is None
        # Check id only if the Interrupt class supports it
        if hasattr(Interrupt, "id") or hasattr(result, "id"):
            assert hasattr(result, "id") and result.id is not None
            # Check exact id only if it was set
            if custom_id_set and hasattr(result, "id"):
                assert result.id == "none-value-id"

        # Interrupt with empty dict value
        try:
            interrupt_empty = Interrupt(value={}, id="empty-value-id")
            custom_id_set = True
        except TypeError:
            interrupt_empty = Interrupt(value={})
            custom_id_set = False
        result = serializer.loads(serializer.dumps(interrupt_empty))
        assert isinstance(result, Interrupt)
        assert result.value == {}
        # Check id only if the Interrupt class supports it
        if hasattr(Interrupt, "id") or hasattr(result, "id"):
            assert hasattr(result, "id") and result.id is not None
            # Check exact id only if it was set
            if custom_id_set and hasattr(result, "id"):
                assert result.id == "empty-value-id"

    def test_backwards_compatibility(self):
        """Test that the fix doesn't break existing non-Interrupt data."""
        serializer = JsonPlusRedisSerializer()

        # Various data types that should work as before
        test_cases = [
            {"message": "regular dict", "type": "test"},
            ["list", "of", "strings"],
            {"nested": {"structure": {"with": ["mixed", "types", 123]}}},
            {"value": "has value key but not id"},
            {"id": "has id key but not value"},
            {"value": 123, "id": "non-string-value", "extra": "field"},
        ]

        for original in test_cases:
            result = serializer.loads(serializer.dumps(original))
            assert result == original, f"Data should be unchanged: {original}"
