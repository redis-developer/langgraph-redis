"""Integration tests for checkpoint blob serialization and pending write operations.

This file tests:
- Blob serialization/deserialization (_load_blobs)
- Checkpoint metadata conversion methods
- Pending write operations and edge cases
- Write operation serialization (_dump_writes)
- Version number generation for checkpoints
- TTL (Time To Live) operations on checkpoint writes
"""

from contextlib import contextmanager
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.redis import RedisSaver


@contextmanager
def _saver(redis_url: str):
    """Create a checkpoint saver with proper setup and teardown."""
    saver = RedisSaver(redis_url)
    saver.setup()
    try:
        yield saver
    finally:
        # Don't flush - let tests be isolated by unique thread IDs
        del saver


def test_issue_83_command_resume_no_warning(redis_url: str) -> None:
    """Test that Command(resume={...}) doesn't cause 'invalid packet type' warning (issue #83).

    The user reported that Command(resume={'interrupt_id': {'some': 'result'}})
    caused warning: "Ignoring invalid packet type <class 'dict'> in pending sends"
    This test verifies our fix prevents that warning.
    """
    import warnings

    from langgraph.constants import TASKS

    with _saver(redis_url) as saver:
        # Create interrupted checkpoint
        interrupted_config = {
            "configurable": {
                "thread_id": "test-thread-83",
                "checkpoint_ns": "",
                "checkpoint_id": "interrupted-checkpoint",
            }
        }

        interrupted_checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:00:00+00:00",
            "id": "interrupted-checkpoint",
            "channel_values": {"messages": ["before interrupt"]},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        metadata = {"source": "loop", "step": 1, "writes": {}}

        # Save the interrupted checkpoint
        saver.put(interrupted_config, interrupted_checkpoint, metadata, {})

        # Simulate Command(resume={'interrupt_id': {'some': 'result'}})
        resume_data = {"interrupt_id": {"some": "result"}}
        saver.put_writes(
            interrupted_config,
            [(TASKS, resume_data)],  # This puts a dict into TASKS
            task_id="resume_task",
        )

        # Create resumed checkpoint with parent reference
        resumed_config = {
            "configurable": {
                "thread_id": "test-thread-83",
                "checkpoint_ns": "",
                "checkpoint_id": "resumed-checkpoint",
            }
        }

        resumed_checkpoint = {
            "v": 1,
            "ts": "2024-01-01T00:01:00+00:00",
            "id": "resumed-checkpoint",
            "channel_values": {"messages": ["after resume"]},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }

        resumed_metadata = {
            "source": "loop",
            "step": 2,
            "writes": {},
            "parent_config": interrupted_config,
        }

        saver.put(resumed_config, resumed_checkpoint, resumed_metadata, {})

        # Load resumed checkpoint - check for warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = saver.get_tuple(resumed_config)

            # Check if we get the warning about invalid packet type
            dict_warnings = [
                warning
                for warning in w
                if "Ignoring invalid packet type" in str(warning.message)
                and "dict" in str(warning.message)
            ]

            # Our fix should prevent this warning
            assert len(dict_warnings) == 0, f"Got warning: {dict_warnings}"

        assert result is not None
        assert result.checkpoint["id"] == "resumed-checkpoint"


def test_issue_83_pending_sends_type_compatibility(redis_url: str) -> None:
    """Test that pending_sends work with string blobs from Redis JSON (issue #83).

    Issue #83 was caused by type mismatch where _load_pending_sends returned
    List[Tuple[str, Union[str, bytes]]] but was annotated as List[Tuple[str, bytes]].
    This test verifies the fix works correctly.
    """
    with _saver(redis_url) as saver:
        checkpoint_dict = {
            "v": 1,
            "ts": "2024-01-01T00:00:00+00:00",
            "id": "test-checkpoint",
            "channel_versions": {},
            "versions_seen": {},
        }

        channel_values = {}

        # Test with string blobs (what Redis JSON returns)
        pending_sends_with_strings = [
            ("json", '{"test": "value"}'),  # String blob from Redis JSON
        ]

        # This should work without type errors
        result = saver._load_checkpoint(
            checkpoint_dict, channel_values, pending_sends_with_strings
        )

        assert "pending_sends" in result
        assert len(result["pending_sends"]) == 1
        assert result["pending_sends"][0] == {"test": "value"}

        # Test JsonPlusRedisSerializer compatibility
        test_data = {"some": "result", "user_input": "continue"}

        # Serialize
        type_str, blob = saver.serde.dumps_typed(test_data)
        assert isinstance(type_str, str)
        assert isinstance(blob, str)  # JsonPlusRedisSerializer returns strings

        # Deserialize - should work with both string and bytes
        result1 = saver.serde.loads_typed((type_str, blob))
        result2 = saver.serde.loads_typed((type_str, blob.encode()))  # bytes version

        assert result1 == test_data
        assert result2 == test_data


def test_load_blobs_method(redis_url: str) -> None:
    """Test _load_blobs method with various scenarios.

    This covers lines 297-299 in base.py
    """
    with _saver(redis_url) as saver:
        # Test 1: Empty blob_values
        result = saver._load_blobs({})
        assert result == {}

        # Test 2: None blob_values
        result = saver._load_blobs(None)  # type: ignore
        assert result == {}

        # Test 3: Blob values with different types
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        blob_values = {}

        for key, value in test_data.items():
            type_, blob = saver.serde.dumps_typed(value)
            blob_values[key] = {"type": type_, "blob": blob}

        # Add an "empty" type that should be filtered out
        blob_values["empty_key"] = {"type": "empty", "blob": b""}

        # Load blobs
        loaded = saver._load_blobs(blob_values)

        # Verify all non-empty values are loaded correctly
        assert len(loaded) == 3  # Should not include empty_key
        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        assert loaded["list"] == [1, 2, 3]
        assert "empty_key" not in loaded


def test_metadata_conversion_methods(redis_url: str) -> None:
    """Test _load_metadata and _dump_metadata methods.

    This covers lines 338 and 351 in base.py
    """
    with _saver(redis_url) as saver:
        # Test 1: Simple metadata
        metadata: CheckpointMetadata = {
            "source": "test",
            "step": 1,
            "writes": {"task1": "value1"},
            "parents": {},
        }

        # Dump metadata
        dumped = saver._dump_metadata(metadata)
        assert isinstance(dumped, str)
        assert "\\u0000" not in dumped  # Null chars should be removed

        # Test 2: Metadata with null characters
        metadata_with_nulls: CheckpointMetadata = {
            "source": "test\x00with\x00nulls",
            "step": 2,
            "writes": {"key": "value\x00null"},
            "parents": {},
        }

        dumped_nulls = saver._dump_metadata(metadata_with_nulls)
        assert isinstance(dumped_nulls, str)
        assert "\x00" not in dumped_nulls
        assert "\\u0000" not in dumped_nulls

        # Test 3: Load metadata
        test_metadata_dict = {
            "source": "loaded",
            "step": 3,
            "writes": {"loaded": "data"},
            "parents": {"parent1": {"checkpoint_id": "123"}},
        }

        loaded = saver._load_metadata(test_metadata_dict)
        assert loaded["source"] == "loaded"
        assert loaded["step"] == 3
        assert loaded["writes"]["loaded"] == "data"
        assert "parent1" in loaded["parents"]


def test_get_next_version_more_cases(redis_url: str) -> None:
    """Test get_next_version with additional edge cases.

    This provides more coverage for the get_next_version method.
    """
    with _saver(redis_url) as saver:
        # Mock channel
        class MockChannel:
            pass

        channel = MockChannel()

        # Test with string version that has decimal part
        version_with_decimal = "00000000000000000000000000000042.9876543210"
        next_version = saver.get_next_version(version_with_decimal, channel)
        assert next_version.startswith("00000000000000000000000000000043.")

        # Test incrementing from 0
        version_zero = saver.get_next_version(None, channel)
        assert version_zero.startswith("00000000000000000000000000000001.")

        # Test with current as string "0"
        version_str_zero = saver.get_next_version("0", channel)
        assert version_str_zero.startswith("00000000000000000000000000000001.")


def test_put_writes_edge_cases(redis_url: str) -> None:
    """Test put_writes method with various edge cases.

    This covers more of lines 419-493 in base.py
    """
    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": "edge-case-test",
                "checkpoint_ns": "test-ns",  # Test with namespace
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Test 1: Writes with WRITES_IDX_MAP channels (special channels)
        special_writes = [
            ("__start__", {"started": True}),
            ("__end__", {"completed": True}),
            ("__error__", {"error": "test error"}),
            ("__interrupt__", {"interrupted": True}),
        ]

        saver.put_writes(saved_config, special_writes, "special_task")

        # Verify writes were stored
        result = saver.get_tuple(saved_config)
        assert result is not None
        pending_writes = result.pending_writes

        # Find our special writes
        special_channels = {w[1] for w in pending_writes if w[0] == "special_task"}
        assert "__start__" in special_channels
        assert "__end__" in special_channels
        assert "__error__" in special_channels
        assert "__interrupt__" in special_channels

        # Test 2: Update existing writes (UPSERT case)
        updated_writes = [
            ("__start__", {"started": True, "timestamp": "2024-01-01"}),
        ]

        saver.put_writes(saved_config, updated_writes, "special_task")

        # Verify the write was updated
        result2 = saver.get_tuple(saved_config)
        assert result2 is not None

        # Test 3: Mixed special and regular channels
        mixed_writes = [
            ("regular_channel", "regular_value"),
            ("__error__", {"error": "another error"}),
            ("another_channel", {"data": "test"}),
        ]

        saver.put_writes(saved_config, mixed_writes, "mixed_task")

        # Test 4: Empty task_path (default parameter)
        path_writes = [("path_channel", "path_value")]
        saver.put_writes(
            saved_config, path_writes, "path_task", task_path="custom/path"
        )

        # Verify all writes
        final_result = saver.get_tuple(saved_config)
        assert final_result is not None
        assert len(final_result.pending_writes) > 0


def test_put_writes_with_ttl(redis_url: str) -> None:
    """Test put_writes with TTL configuration.

    This tests TTL application in put_writes method.
    """
    # Create saver with TTL config
    saver = RedisSaver(redis_url, ttl={"default_ttl": 0.1})  # 6 seconds TTL
    saver.setup()

    try:
        thread_id = str(uuid4())

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={},
            step=1,
        )

        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Add writes - should have TTL applied
        writes = [("ttl_channel", "ttl_value")]
        saver.put_writes(saved_config, writes, "ttl_task")

        # Get the actual checkpoint ID from saved_config
        actual_checkpoint_id = saved_config["configurable"]["checkpoint_id"]

        # Check that TTL was applied to write keys using the actual checkpoint ID
        write_keys = list(
            saver._redis.scan_iter(
                match=f"checkpoint_write:{thread_id}:*:{actual_checkpoint_id}:ttl_task:*"
            )
        )

        assert len(write_keys) > 0

        # Verify TTL is set
        for key in write_keys:
            ttl = saver._redis.ttl(key)
            assert ttl > 0 and ttl <= 6

    finally:
        # Don't flush - let tests be isolated by unique thread IDs
        del saver


def test_dump_writes_method(redis_url: str) -> None:
    """Test _dump_writes method directly.

    This covers line 314 and the method implementation.
    """
    with _saver(redis_url) as saver:
        thread_id = "test-thread"
        checkpoint_ns = "test-ns"
        checkpoint_id = "test-checkpoint"
        task_id = "test-task"

        # Test various write scenarios
        writes = [
            ("channel1", "simple_string"),
            ("channel2", {"complex": "object", "nested": {"data": 123}}),
            ("channel3", b"binary_data"),
            ("__error__", {"error": "test_error"}),  # Special channel
        ]

        # Call _dump_writes
        dumped = saver._dump_writes(
            thread_id, checkpoint_ns, checkpoint_id, task_id, writes
        )

        assert len(dumped) == 4

        # Verify structure of dumped writes
        for i, dumped_write in enumerate(dumped):
            assert "thread_id" in dumped_write
            assert "checkpoint_ns" in dumped_write
            assert "checkpoint_id" in dumped_write
            assert "task_id" in dumped_write
            assert "idx" in dumped_write
            assert "channel" in dumped_write
            assert "type" in dumped_write
            assert "blob" in dumped_write

            # Check special channel gets special index
            if writes[i][0] == "__error__":
                assert dumped_write["idx"] == WRITES_IDX_MAP["__error__"]
            else:
                assert dumped_write["idx"] == i


def test_get_next_version_edge_cases(redis_url: str) -> None:
    """Test get_next_version with edge cases.

    This covers line 360 and related logic.
    """
    with _saver(redis_url) as saver:
        # Mock channel
        class MockChannel:
            pass

        channel = MockChannel()

        # Test with integer current version
        version = saver.get_next_version(10, channel)  # type: ignore
        assert version.startswith("00000000000000000000000000000011.")

        # Test with very large integer
        large_version = saver.get_next_version(999999, channel)  # type: ignore
        assert large_version.startswith("00000000000000000000000001000000.")

        # Test version parsing from string with decimal
        existing_version = "00000000000000000000000000000005.1234567890123456"
        next_version = saver.get_next_version(existing_version, channel)
        assert next_version.startswith("00000000000000000000000000000006.")


def test_langchain_message_serialization(redis_url: str) -> None:
    """Test that LangChain messages are properly serialized and deserialized.

    This reproduces the issue where messages stored in LangChain format
    are not properly deserialized back to message objects.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    serializer = JsonPlusRedisSerializer()

    # Test 1: Serialize and deserialize a HumanMessage
    human_msg = HumanMessage(content="Hello, AI!")

    # Serialize
    type_, data = serializer.dumps_typed(human_msg)
    assert type_ == "json"
    assert isinstance(data, str)

    # Deserialize
    deserialized = serializer.loads_typed((type_, data))

    # Should be a HumanMessage object, not a dict
    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "Hello, AI!"
    assert hasattr(deserialized, "content")  # Should have message methods

    # Test 2: Serialize and deserialize an AIMessage
    ai_msg = AIMessage(content="Hello, human!")

    type_, data = serializer.dumps_typed(ai_msg)
    deserialized_ai = serializer.loads_typed((type_, data))

    assert isinstance(deserialized_ai, AIMessage)
    assert deserialized_ai.content == "Hello, human!"

    # Test 3: Serialize and deserialize a list of messages
    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(content="I can help with that."),
    ]

    type_, data = serializer.dumps_typed(messages)
    deserialized_list = serializer.loads_typed((type_, data))

    assert isinstance(deserialized_list, list)
    assert len(deserialized_list) == 2
    assert isinstance(deserialized_list[0], HumanMessage)
    assert isinstance(deserialized_list[1], AIMessage)


def test_checkpoint_with_messages(redis_url: str) -> None:
    """Test that checkpoints containing messages are properly handled.

    This tests the full cycle of saving and loading checkpoints with messages.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

    with _saver(redis_url) as saver:
        thread_id = str(uuid4())

        # Create messages
        messages = [
            HumanMessage(content="What is the weather in SF?"),
            AIMessage(content="Let me check that for you."),
        ]

        # Create checkpoint with messages in channel_values
        checkpoint = create_checkpoint(
            checkpoint=empty_checkpoint(),
            channels={"messages": messages},
            step=1,
        )

        # Add messages to checkpoint's channel_values
        checkpoint["channel_values"]["messages"] = messages

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }

        # Save checkpoint
        saved_config = saver.put(
            config, checkpoint, {"source": "test", "step": 1, "writes": {}}, {}
        )

        # Load checkpoint back
        loaded_checkpoint = saver.get(saved_config)

        assert loaded_checkpoint is not None
        assert "channel_values" in loaded_checkpoint
        assert "messages" in loaded_checkpoint["channel_values"]

        loaded_messages = loaded_checkpoint["channel_values"]["messages"]

        # Verify messages are properly deserialized as objects
        assert len(loaded_messages) == 2
        assert isinstance(loaded_messages[0], HumanMessage)
        assert isinstance(loaded_messages[1], AIMessage)
        assert loaded_messages[0].content == "What is the weather in SF?"
        assert loaded_messages[1].content == "Let me check that for you."


def test_subgraph_state_history_pending_sends(redis_url: str) -> None:
    """Test that get_state_history with subgraphs properly handles pending_sends.

    This reproduces the issue where accessing doc.blob fails because the
    Document from Redis search has the field as '$.blob' not 'blob'.
    """
    from typing import TypedDict

    from langgraph.checkpoint.base import Checkpoint
    from langgraph.graph import START, StateGraph

    # Define subgraph
    class SubgraphState(TypedDict):
        foo: str
        bar: str

    def subgraph_node_1(state: SubgraphState):
        return {"bar": "bar"}

    def subgraph_node_2(state: SubgraphState):
        return {"foo": state["foo"] + state["bar"]}

    subgraph_builder = StateGraph(SubgraphState)
    subgraph_builder.add_node(subgraph_node_1)
    subgraph_builder.add_node(subgraph_node_2)
    subgraph_builder.add_edge(START, "subgraph_node_1")
    subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = subgraph_builder.compile()

    # Define parent graph
    class State(TypedDict):
        foo: str

    def node_1(state: State):
        return {"foo": "hi! " + state["foo"]}

    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", subgraph)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")

    # Compile with Redis checkpointer
    with _saver(redis_url) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test_subgraph"}}

        # Run the graph with subgraphs
        for _, chunk in graph.stream({"foo": "foo"}, config, subgraphs=True):
            pass  # Just run through

        # Now try to get state history - this should trigger the bug
        # where doc.blob fails because Redis returns $.blob
        state_history = list(graph.get_state_history(config))

        # Should be able to find state with subgraph
        state_with_subgraph = [s for s in state_history if s.next == ("node_2",)]

        assert len(state_with_subgraph) > 0, "Should have state before node_2"

        # Get the subgraph config
        subgraph_state = state_with_subgraph[0]
        assert subgraph_state.tasks, "Should have tasks"

        subgraph_config = subgraph_state.tasks[0].state
        assert "checkpoint_ns" in subgraph_config["configurable"]

        # Should be able to get subgraph state
        subgraph_values = graph.get_state(subgraph_config).values
        assert "foo" in subgraph_values
        assert "bar" in subgraph_values
        assert subgraph_values["foo"] == "hi! foobar"
        assert subgraph_values["bar"] == "bar"


def test_message_dict_format_handling(redis_url: str) -> None:
    """Test handling of messages stored in LangChain serialized format.

    This specifically tests the dict format that causes the error:
    {'lc': 1, 'type': 'constructor', 'id': [...], 'kwargs': {...}}
    """
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    serializer = JsonPlusRedisSerializer()

    # This is the format that causes the error in notebooks
    message_dict = {
        "lc": 1,  # LangChain messages use lc: 1
        "type": "constructor",
        "id": ["langchain", "schema", "messages", "HumanMessage"],
        "kwargs": {
            "content": "what is the weather in SF, CA?",
            "type": "human",
            "id": "19fb5cce-473a-408c-8b2b-fcb3587b1661",
        },
    }

    # Convert to JSON string then back to test the full cycle
    import json

    json_str = json.dumps(message_dict)

    # This should properly deserialize to a HumanMessage
    deserialized = serializer.loads(json_str.encode())

    # Should be a HumanMessage object, not a dict
    from langchain_core.messages import HumanMessage

    assert isinstance(deserialized, HumanMessage)
    assert deserialized.content == "what is the weather in SF, CA?"
    assert deserialized.id == "19fb5cce-473a-408c-8b2b-fcb3587b1661"
