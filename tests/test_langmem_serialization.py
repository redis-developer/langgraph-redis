"""Integration tests for langmem object serialization with JsonPlusRedisSerializer."""

import sys

import pytest

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

# langmem requires Python 3.11+ (uses typing.NotRequired)
# Try to import, skip all tests if it fails
try:
    from langmem.short_term.summarization import RunningSummary
except (ImportError, AttributeError):
    # Skip entire module if langmem can't be imported (Python < 3.11)
    pytestmark = pytest.mark.skip(
        reason="langmem requires Python 3.11+ (uses typing.NotRequired)"
    )
    RunningSummary = None  # type: ignore


class TestLangmemSerialization:
    """Test that langmem objects are properly serialized and deserialized."""

    def test_running_summary_serialization(self) -> None:
        """Test that RunningSummary objects roundtrip correctly through serialization."""
        serializer = JsonPlusRedisSerializer()

        # Create a RunningSummary object similar to what langmem creates
        original = RunningSummary(
            summary="This is a summary of the conversation",
            summarized_message_ids={"msg1", "msg2", "msg3"},
            last_summarized_message_id="msg3",
        )

        # Serialize it
        type_str, data_bytes = serializer.dumps_typed(original)

        # Deserialize it
        deserialized = serializer.loads_typed((type_str, data_bytes))

        # Check that it's still a RunningSummary object, not a dict
        assert isinstance(
            deserialized, RunningSummary
        ), f"Expected RunningSummary, got {type(deserialized)}"
        assert deserialized.summary == original.summary
        assert deserialized.summarized_message_ids == original.summarized_message_ids
        assert (
            deserialized.last_summarized_message_id
            == original.last_summarized_message_id
        )

    def test_state_with_context_containing_running_summary(self) -> None:
        """Test that state dicts with RunningSummary in context are properly handled.

        This tests the real-world scenario where create_react_agent stores
        RunningSummary objects in the state's context field.
        """
        serializer = JsonPlusRedisSerializer()

        # Create a state similar to what create_react_agent uses
        state = {
            "messages": [],
            "context": {
                "running_summary": RunningSummary(
                    summary="Previous conversation summary",
                    summarized_message_ids={"msg1", "msg2"},
                    last_summarized_message_id="msg2",
                )
            },
        }

        # Serialize it
        type_str, data_bytes = serializer.dumps_typed(state)

        # Deserialize it
        deserialized = serializer.loads_typed((type_str, data_bytes))

        # Check that the running_summary is still a RunningSummary object
        assert "context" in deserialized
        assert "running_summary" in deserialized["context"]
        running_summary = deserialized["context"]["running_summary"]

        assert isinstance(
            running_summary, RunningSummary
        ), f"Expected RunningSummary, got {type(running_summary)}"
        assert hasattr(
            running_summary, "summarized_message_ids"
        ), "Missing summarized_message_ids attribute"
        assert running_summary.summarized_message_ids == {"msg1", "msg2"}

    def test_nested_running_summary_in_list(self) -> None:
        """Test that RunningSummary objects nested in lists are properly handled."""
        serializer = JsonPlusRedisSerializer()

        state = {
            "summaries": [
                RunningSummary(
                    summary="First summary",
                    summarized_message_ids={"msg1"},
                    last_summarized_message_id="msg1",
                ),
                RunningSummary(
                    summary="Second summary",
                    summarized_message_ids={"msg2", "msg3"},
                    last_summarized_message_id="msg3",
                ),
            ]
        }

        # Serialize and deserialize
        type_str, data_bytes = serializer.dumps_typed(state)
        deserialized = serializer.loads_typed((type_str, data_bytes))

        # Check both summaries are properly reconstructed
        assert len(deserialized["summaries"]) == 2
        for summary in deserialized["summaries"]:
            assert isinstance(summary, RunningSummary)
            assert hasattr(summary, "summarized_message_ids")
