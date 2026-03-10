"""Integration tests for provider metadata stripping in semantic cache deserialization.

Issue: OpenAI Responses API fails with "Duplicate item found with id rs_..."
when cached AIMessages preserve provider-specific IDs in additional_kwargs
and response_metadata. When both the original and cached message exist in
conversation state, the next API call sees duplicate provider IDs.

Fix: _deserialize_response strips all provider metadata on read, keeping only
content and the cached=True marker.

These tests cover EVERY deserialization path in _deserialize_response:
1. LangChain AIMessage with OpenAI Responses API metadata (rs_ IDs)
2. LangChain AIMessage with OpenAI Chat Completions metadata
3. LangChain AIMessage with Anthropic metadata
4. LangChain AIMessage with no provider metadata (baseline)
5. LangChain constructor format that revives to non-AIMessage
6. Plain dict with "content" key
7. Non-dict JSON values (array, number, string, null)
8. Invalid JSON / raw text
9. Round-trip: serialize with provider metadata, deserialize clean
10. Multi-turn conversation scenario (the actual reported bug)
"""

import json
import uuid

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    SemanticCacheConfig,
    SemanticCacheMiddleware,
)
from langgraph.middleware.redis.semantic_cache import (
    _deserialize_response,
    _serialize_response,
)

pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for the test module."""
    container = RedisContainer("redis/redis-stack-server:latest")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def redis_url(redis_container):
    """Get Redis URL from container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


@pytest.fixture
def vectorizer():
    """Shared vectorizer for all tests."""
    return HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")


def _assert_clean_cached_message(msg: AIMessage) -> None:
    """Assert that a cached AIMessage has no provider metadata leakage.

    This is the core assertion used by all tests. A properly deserialized
    cached message must have:
    - additional_kwargs == {"cached": True} (no provider fields)
    - response_metadata == {} (no provider metadata)
    - A valid UUID as the message ID
    - Content preserved from the original
    """
    # Must be marked as cached with NO other keys
    assert msg.additional_kwargs == {"cached": True}, (
        f"additional_kwargs must be exactly {{'cached': True}}, "
        f"got {msg.additional_kwargs}"
    )

    # Must have empty response_metadata
    assert (
        msg.response_metadata == {}
    ), f"response_metadata must be empty, got {msg.response_metadata}"

    # Must have a valid UUID
    assert msg.id is not None, "Message must have an ID"
    uuid.UUID(msg.id)  # raises ValueError if not valid UUID


# ---------------------------------------------------------------------------
# Path 1: LangChain AIMessage with OpenAI Responses API metadata (rs_ IDs)
# This is the EXACT bug scenario from the issue report.
# ---------------------------------------------------------------------------


class TestOpenAIResponsesAPIMetadata:
    """Test stripping of OpenAI Responses API metadata (rs_ IDs)."""

    def test_strips_openai_responses_api_ids(self):
        """OpenAI Responses API embeds rs_-prefixed IDs that must be stripped."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Paris is the capital of France.",
                    "type": "ai",
                    "id": "msg-original-id",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "rs_0158441393b57e570069af698851bc81949fc275d0e5ba22ef": True,
                        "response_id": "rs_resp_abc123",
                    },
                    "response_metadata": {
                        "id": "rs_resp_abc123",
                        "model": "gpt-4o-2024-05-13",
                        "output": [
                            {
                                "id": "rs_0158441393b57e570069af698851bc81949fc275d0e5ba22ef",
                                "type": "message",
                                "content": [{"type": "text", "text": "Paris..."}],
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)

        assert isinstance(result, ModelResponse)
        msg = result.result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "Paris is the capital of France."
        _assert_clean_cached_message(msg)

    @pytest.mark.asyncio
    async def test_openai_responses_api_ids_via_middleware(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """End-to-end: cached OpenAI Responses API message through middleware."""
        cache_name = f"openai_resp_{uuid.uuid4().hex[:8]}"

        # Pre-populate cache with a response containing rs_ IDs
        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "The answer is 42.",
                    "type": "ai",
                    "id": "msg-original",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "rs_deadbeef": True,
                        "response_id": "rs_resp_xyz",
                    },
                    "response_metadata": {
                        "id": "rs_resp_xyz",
                        "model": "gpt-4o",
                        "output": [{"id": "rs_deadbeef", "type": "message"}],
                    },
                },
            }
        )
        cache.store(prompt="What is the answer?", response=cached_response)

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called on cache hit")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="What is the answer?")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            msg = result.result[0]
            assert msg.content == "The answer is 42."
            _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 2: LangChain AIMessage with OpenAI Chat Completions metadata
# ---------------------------------------------------------------------------


class TestOpenAIChatCompletionsMetadata:
    """Test stripping of OpenAI Chat Completions API metadata."""

    def test_strips_chat_completions_metadata(self):
        """Chat Completions API uses different field names but same risk."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Hello! How can I help?",
                    "type": "ai",
                    "id": "chatcmpl-abc123",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "refusal": None,
                    },
                    "response_metadata": {
                        "token_usage": {
                            "completion_tokens": 15,
                            "prompt_tokens": 10,
                            "total_tokens": 25,
                        },
                        "model_name": "gpt-4o-mini",
                        "system_fingerprint": "fp_abc123xyz",
                        "finish_reason": "stop",
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "Hello! How can I help?"
        _assert_clean_cached_message(msg)

    @pytest.mark.asyncio
    async def test_chat_completions_metadata_via_middleware(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """End-to-end: cached Chat Completions message through middleware."""
        cache_name = f"openai_chat_{uuid.uuid4().hex[:8]}"

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "I can help with that.",
                    "type": "ai",
                    "id": "chatcmpl-xyz789",
                    "tool_calls": [],
                    "additional_kwargs": {"refusal": None},
                    "response_metadata": {
                        "model_name": "gpt-4o",
                        "system_fingerprint": "fp_xyz789",
                        "finish_reason": "stop",
                    },
                },
            }
        )
        cache.store(prompt="Can you help me?", response=cached_response)

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="Can you help me?")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            msg = result.result[0]
            assert msg.content == "I can help with that."
            _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 3: LangChain AIMessage with Anthropic metadata
# ---------------------------------------------------------------------------


class TestAnthropicMetadata:
    """Test stripping of Anthropic API metadata."""

    def test_strips_anthropic_metadata(self):
        """Anthropic responses include model, stop_reason, usage in metadata."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "I'm Claude, an AI assistant.",
                    "type": "ai",
                    "id": "msg_01XYZ",
                    "tool_calls": [],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "id": "msg_01XYZ",
                        "model": "claude-sonnet-4-5-20250929",
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 25,
                            "output_tokens": 50,
                        },
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "I'm Claude, an AI assistant."
        _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 4: LangChain AIMessage with no provider metadata (baseline)
# ---------------------------------------------------------------------------


class TestCleanAIMessage:
    """Test baseline: AIMessage with no provider metadata still works."""

    def test_clean_aimessage_no_metadata(self):
        """AIMessage with no additional_kwargs or response_metadata."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Simple response.",
                    "type": "ai",
                    "tool_calls": [],
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "Simple response."
        _assert_clean_cached_message(msg)

    def test_aimessage_with_empty_metadata(self):
        """AIMessage with explicitly empty metadata fields."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Another response.",
                    "type": "ai",
                    "tool_calls": [],
                    "additional_kwargs": {},
                    "response_metadata": {},
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "Another response."
        _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 5: Plain dict with "content" key (non-LangChain format)
# ---------------------------------------------------------------------------


class TestPlainDictDeserialization:
    """Test deserialization of plain dict format {"content": "..."}."""

    def test_plain_dict_content(self):
        """Simple dict with content key."""
        cached_str = json.dumps({"content": "A plain cached response."})

        result = _deserialize_response(cached_str)
        assert isinstance(result, ModelResponse)
        msg = result.result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "A plain cached response."
        _assert_clean_cached_message(msg)

    def test_plain_dict_empty_content(self):
        """Dict with empty content."""
        cached_str = json.dumps({"content": ""})

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == ""
        _assert_clean_cached_message(msg)

    def test_plain_dict_missing_content_key(self):
        """Dict without a content key falls back to empty string."""
        cached_str = json.dumps({"other_field": "value"})

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == ""
        _assert_clean_cached_message(msg)

    @pytest.mark.asyncio
    async def test_plain_dict_via_middleware(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """End-to-end: plain dict cached response through middleware."""
        cache_name = f"plain_dict_{uuid.uuid4().hex[:8]}"

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )
        cache.store(
            prompt="Simple question",
            response=json.dumps({"content": "Simple answer"}),
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def should_not_be_called(request):
            raise AssertionError("Handler should not be called")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="Simple question")]}
            result = await middleware.awrap_model_call(request, should_not_be_called)

            msg = result.result[0]
            assert msg.content == "Simple answer"
            _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 6: Non-dict JSON values (array, number, string, boolean, null)
# ---------------------------------------------------------------------------


class TestNonDictJsonDeserialization:
    """Test deserialization of non-dict JSON values."""

    def test_json_array(self):
        """JSON array is converted to string content."""
        cached_str = json.dumps([1, 2, 3])

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "[1, 2, 3]"
        _assert_clean_cached_message(msg)

    def test_json_number(self):
        """JSON number is converted to string content."""
        cached_str = json.dumps(42)

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "42"
        _assert_clean_cached_message(msg)

    def test_json_string(self):
        """JSON string value."""
        cached_str = json.dumps("just a string")

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "just a string"
        _assert_clean_cached_message(msg)

    def test_json_boolean(self):
        """JSON boolean is converted to string content."""
        cached_str = json.dumps(True)

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "True"
        _assert_clean_cached_message(msg)

    def test_json_null(self):
        """JSON null is converted to string content."""
        cached_str = json.dumps(None)

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "None"
        _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 7: Invalid JSON / raw text
# ---------------------------------------------------------------------------


class TestInvalidJsonDeserialization:
    """Test deserialization of invalid JSON (raw text fallback)."""

    def test_plain_text(self):
        """Raw text that isn't valid JSON."""
        cached_str = "This is just plain text, not JSON at all."

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == cached_str
        _assert_clean_cached_message(msg)

    def test_malformed_json(self):
        """Malformed JSON string."""
        cached_str = '{"content": "missing closing brace"'

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == cached_str
        _assert_clean_cached_message(msg)

    def test_empty_string(self):
        """Empty string input."""
        result = _deserialize_response("")

        msg = result.result[0]
        assert msg.content == ""
        _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 8: Round-trip serialization/deserialization
# ---------------------------------------------------------------------------


class TestRoundTripSerialization:
    """Test serialize -> deserialize round-trip strips provider metadata."""

    def test_round_trip_modelresponse_with_openai_metadata(self):
        """Serialize a ModelResponse with OpenAI metadata, deserialize clean."""
        original = AIMessage(
            content="Round-trip test response.",
            id="chatcmpl-original",
            additional_kwargs={
                "rs_provider_id_123": True,
                "response_id": "rs_resp_456",
            },
            response_metadata={
                "model_name": "gpt-4o",
                "system_fingerprint": "fp_abc",
                "finish_reason": "stop",
                "token_usage": {"total_tokens": 100},
            },
        )
        response = ModelResponse(result=[original], structured_response=None)

        # Serialize (preserves everything)
        serialized = _serialize_response(response)

        # Deserialize (must strip provider metadata)
        deserialized = _deserialize_response(serialized)
        msg = deserialized.result[0]

        assert msg.content == "Round-trip test response."
        _assert_clean_cached_message(msg)

        # Verify the original provider data is NOT present
        assert "rs_provider_id_123" not in msg.additional_kwargs
        assert "response_id" not in msg.additional_kwargs
        assert "model_name" not in msg.response_metadata
        assert "system_fingerprint" not in msg.response_metadata

    def test_round_trip_bare_aimessage_with_anthropic_metadata(self):
        """Serialize a bare AIMessage with Anthropic metadata, deserialize clean."""
        original = AIMessage(
            content="Anthropic round-trip.",
            id="msg_01abc",
            response_metadata={
                "id": "msg_01abc",
                "model": "claude-sonnet-4-5-20250929",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

        serialized = _serialize_response(original)
        deserialized = _deserialize_response(serialized)
        msg = deserialized.result[0]

        assert msg.content == "Anthropic round-trip."
        _assert_clean_cached_message(msg)

    @pytest.mark.asyncio
    async def test_round_trip_via_middleware(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Full round-trip through middleware: store with metadata, retrieve clean."""
        cache_name = f"roundtrip_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def handler_with_provider_metadata(request):
            call_count[0] += 1
            return ModelResponse(
                result=[
                    AIMessage(
                        content="Response with provider data.",
                        id="chatcmpl-handler",
                        additional_kwargs={
                            "rs_item_id": True,
                            "response_id": "rs_resp_handler",
                        },
                        response_metadata={
                            "model_name": "gpt-4o",
                            "system_fingerprint": "fp_handler",
                            "finish_reason": "stop",
                        },
                    )
                ],
                structured_response=None,
            )

        async with SemanticCacheMiddleware(config) as middleware:
            request = {"messages": [HumanMessage(content="Round trip question")]}

            # First call: cache miss, stores the response (with provider metadata)
            result1 = await middleware.awrap_model_call(
                request, handler_with_provider_metadata
            )
            assert call_count[0] == 1

            # Second call: cache hit, must return clean message
            result2 = await middleware.awrap_model_call(
                request, handler_with_provider_metadata
            )
            assert call_count[0] == 1, "Handler should not be called on cache hit"

            msg = result2.result[0]
            assert msg.content == "Response with provider data."
            _assert_clean_cached_message(msg)


# ---------------------------------------------------------------------------
# Path 9: Multi-turn conversation scenario (the actual reported bug)
# ---------------------------------------------------------------------------


class TestMultiTurnDuplicateIdPrevention:
    """Test the exact scenario that causes the OpenAI Responses API error.

    Bug scenario:
    1. User asks "What is X?" -> LLM responds with AIMessage containing rs_ IDs
    2. That response is cached (serialized with all metadata)
    3. User asks "What is X?" again -> cache hit returns AIMessage
    4. Both messages exist in conversation state
    5. Next API call sees duplicate rs_ IDs -> "Duplicate item found" error

    The fix ensures step 3 returns a clean message with no rs_ IDs.
    """

    @pytest.mark.asyncio
    async def test_multi_turn_no_duplicate_provider_ids(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Simulate multi-turn conversation with cached responses.

        Verify that no two messages in the conversation share provider IDs.
        """
        cache_name = f"multiturn_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def handler_returning_openai_response(request):
            call_count[0] += 1
            return ModelResponse(
                result=[
                    AIMessage(
                        content="The capital of France is Paris.",
                        id="chatcmpl-turn1",
                        additional_kwargs={
                            "rs_0158441393b57e570069af698851bc81": True,
                            "response_id": "rs_resp_turn1",
                        },
                        response_metadata={
                            "id": "rs_resp_turn1",
                            "model": "gpt-4o",
                            "output": [
                                {
                                    "id": "rs_0158441393b57e570069af698851bc81",
                                    "type": "message",
                                }
                            ],
                        },
                    )
                ],
                structured_response=None,
            )

        conversation = []

        async with SemanticCacheMiddleware(config) as middleware:
            # Turn 1: User asks, cache miss -> handler called
            user_msg1 = HumanMessage(content="What is the capital of France?")
            conversation.append(user_msg1)

            result1 = await middleware.awrap_model_call(
                {"messages": conversation.copy()},
                handler_returning_openai_response,
            )
            assert call_count[0] == 1
            ai_msg1 = result1.result[0]
            conversation.append(ai_msg1)

            # Turn 2: Same question again -> cache hit
            user_msg2 = HumanMessage(content="What is the capital of France?")
            conversation.append(user_msg2)

            result2 = await middleware.awrap_model_call(
                {"messages": conversation.copy()},
                handler_returning_openai_response,
            )
            assert call_count[0] == 1, "Handler should not be called on cache hit"
            ai_msg2 = result2.result[0]
            conversation.append(ai_msg2)

            # The cached message must not carry rs_ IDs
            _assert_clean_cached_message(ai_msg2)

            # CRITICAL: Verify no duplicate provider IDs across the conversation
            # Collect all rs_ IDs from all AI messages in conversation
            all_rs_ids = set()
            for msg in conversation:
                if isinstance(msg, AIMessage):
                    for key in msg.additional_kwargs:
                        if key.startswith("rs_"):
                            all_rs_ids.add(key)
                    for key in msg.response_metadata:
                        if isinstance(
                            msg.response_metadata.get(key), str
                        ) and msg.response_metadata[key].startswith("rs_"):
                            all_rs_ids.add(msg.response_metadata[key])

            # The cached message (ai_msg2) should contribute ZERO rs_ IDs
            cached_rs_ids = set()
            for key in ai_msg2.additional_kwargs:
                if key.startswith("rs_"):
                    cached_rs_ids.add(key)
            assert len(cached_rs_ids) == 0, (
                f"Cached message must have no rs_ IDs in additional_kwargs, "
                f"found: {cached_rs_ids}"
            )

            # Message IDs must be different
            assert (
                ai_msg1.id != ai_msg2.id
            ), f"Messages must have different IDs: {ai_msg1.id} == {ai_msg2.id}"

    @pytest.mark.asyncio
    async def test_three_turn_conversation_all_clean(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Three identical questions -> first is fresh, next two are cached.

        All cached messages must be clean, all IDs must be unique.
        """
        cache_name = f"threeturn_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        call_count = [0]

        async def handler(request):
            call_count[0] += 1
            return ModelResponse(
                result=[
                    AIMessage(
                        content="Python is a programming language.",
                        id="chatcmpl-first",
                        additional_kwargs={
                            "rs_item_abc": True,
                        },
                        response_metadata={
                            "id": "rs_resp_first",
                            "model": "gpt-4o",
                        },
                    )
                ],
                structured_response=None,
            )

        conversation = []
        ai_messages = []

        async with SemanticCacheMiddleware(config) as middleware:
            for turn in range(3):
                user_msg = HumanMessage(content="What is Python?")
                conversation.append(user_msg)

                result = await middleware.awrap_model_call(
                    {"messages": conversation.copy()}, handler
                )
                ai_msg = result.result[0]
                conversation.append(ai_msg)
                ai_messages.append(ai_msg)

            # Only 1 handler call (turn 1)
            assert call_count[0] == 1

            # All 3 messages must have unique IDs
            ids = [msg.id for msg in ai_messages]
            assert len(set(ids)) == 3, f"All IDs must be unique: {ids}"

            # Turns 2 and 3 (cached) must be clean
            for i, msg in enumerate(ai_messages[1:], start=2):
                _assert_clean_cached_message(msg)

            # All messages must have the same content
            for msg in ai_messages:
                assert msg.content == "Python is a programming language."


# ---------------------------------------------------------------------------
# Path 10: AIMessage with tool_calls preserved in additional_kwargs
# ---------------------------------------------------------------------------


class TestToolCallMetadataStripping:
    """Test that tool-call-related provider metadata is also stripped.

    Some providers embed tool call IDs or function call metadata in
    additional_kwargs. These must also be stripped on cache hit.
    """

    def test_strips_openai_function_call_metadata(self):
        """OpenAI function_call fields in additional_kwargs are stripped."""
        cached_str = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "Based on the search results...",
                    "type": "ai",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "function_call": None,
                        "refusal": None,
                    },
                    "response_metadata": {
                        "token_usage": {"total_tokens": 50},
                        "model_name": "gpt-4o",
                        "system_fingerprint": "fp_xyz",
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert msg.content == "Based on the search results..."
        _assert_clean_cached_message(msg)

        # Verify function_call is not preserved
        assert "function_call" not in msg.additional_kwargs
        assert "refusal" not in msg.additional_kwargs


# ---------------------------------------------------------------------------
# Path 11: LangChain lc=2 format (newer serialization version)
# ---------------------------------------------------------------------------


class TestLangChainV2Format:
    """Test deserialization of LangChain lc=2 constructor format.

    The lc=2 format may not be revivable to AIMessage by the current
    serializer. When revival fails, the code falls through to the
    non-AIMessage path which creates a fresh AIMessage from content.
    Either way, provider metadata must be stripped.
    """

    def test_lc_version_2_with_metadata(self):
        """LangChain v2 serialization format with provider metadata.

        When lc=2 can't be revived to AIMessage, it falls through to
        the non-AIMessage path. The key assertion is that the result
        is still a clean AIMessage with no provider metadata leakage.
        """
        cached_str = json.dumps(
            {
                "lc": 2,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": "V2 format response.",
                    "type": "ai",
                    "tool_calls": [],
                    "additional_kwargs": {
                        "provider_field": "should_be_stripped",
                    },
                    "response_metadata": {
                        "model": "some-model",
                    },
                },
            }
        )

        result = _deserialize_response(cached_str)
        msg = result.result[0]
        assert isinstance(msg, AIMessage)
        # Whether revived or fallen through, provider metadata must be clean
        _assert_clean_cached_message(msg)
        # Provider fields must NOT appear
        assert "provider_field" not in msg.additional_kwargs
        assert "model" not in msg.response_metadata
