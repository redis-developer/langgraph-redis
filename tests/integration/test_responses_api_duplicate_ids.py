"""Integration tests for OpenAI Responses API duplicate ID prevention.

Reproduces the exact customer scenario:
- create_agent with use_responses_api=True (Azure OpenAI)
- Redis middleware stack (SemanticCache + ToolCache + ConversationMemory)
- Multi-turn conversations where cached + original messages coexist in state
- AIMessage.content is a LIST of blocks with embedded rs_ IDs

The bug: _construct_responses_api_input() in langchain_openai extracts 'id'
fields from content blocks and passes them as Responses API item IDs. When
both original and cached messages share the same rs_ IDs, the API fails with:
    "Duplicate item found with id rs_..."

These tests verify:
1. Cache deserialization strips content block IDs
2. MiddlewareStack._sanitize_request strips IDs from checkpoint state messages
3. Multi-turn conversations produce no duplicate IDs across all messages
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
    MiddlewareStack,
    SemanticCacheConfig,
    SemanticCacheMiddleware,
)
from langgraph.middleware.redis.composition import _sanitize_request

pytest.importorskip("sentence_transformers")


# -- Helpers ------------------------------------------------------------------


def _make_responses_api_aimessage(
    text: str,
    block_id: str = "rs_abc123",
    msg_id: str = "msg-original",
    with_reasoning: bool = False,
) -> AIMessage:
    """Create an AIMessage in OpenAI Responses API format.

    This is the exact format produced by langchain_openai when
    use_responses_api=True. Content is a list of blocks with embedded IDs.
    """
    content = []
    if with_reasoning:
        content.append(
            {
                "type": "reasoning",
                "id": f"rs_reasoning_{uuid.uuid4().hex[:8]}",
                "summary": [{"type": "summary_text", "text": "Thinking about it..."}],
            }
        )
    content.append(
        {
            "type": "text",
            "text": text,
            "id": block_id,
            "annotations": [],
        }
    )
    return AIMessage(
        content=content,
        id=msg_id,
        additional_kwargs={},
        response_metadata={
            "id": f"rs_resp_{uuid.uuid4().hex[:8]}",
            "model": "gpt-4o-2024-05-13",
        },
    )


def _collect_content_block_ids(msg: AIMessage) -> set:
    """Collect all 'id' fields from content blocks."""
    ids = set()
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, dict) and "id" in block:
                ids.add(block["id"])
    return ids


# -- Fixtures -----------------------------------------------------------------


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


# -- Test: Cache deserialization strips content block IDs ---------------------


class TestCacheDeserializationStripsBlockIds:
    """Verify _deserialize_response strips 'id' from content blocks."""

    @pytest.mark.asyncio
    async def test_cache_hit_strips_text_block_id(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Cache hit with Responses API format must strip content block IDs."""
        cache_name = f"resp_deser_{uuid.uuid4().hex[:8]}"

        cache = SemanticCache(
            name=cache_name,
            redis_url=redis_url,
            vectorizer=vectorizer,
            distance_threshold=0.1,
        )

        # Store a Responses API formatted message in cache
        cached_response = json.dumps(
            {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Paris is the capital of France.",
                            "id": "rs_0ae41c16891c342b0069b03ac28e508197af",
                            "annotations": [],
                        }
                    ],
                    "type": "ai",
                    "id": "msg-original",
                    "tool_calls": [],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "id": "rs_resp_abc",
                        "model": "gpt-4o",
                    },
                },
            }
        )
        cache.store(
            prompt="What is the capital of France?",
            response=cached_response,
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def fail_handler(request):
            raise AssertionError("Handler should not be called on cache hit")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {
                "messages": [HumanMessage(content="What is the capital of France?")]
            }
            result = await middleware.awrap_model_call(request, fail_handler)

            msg = result.result[0]
            assert isinstance(msg.content, list)
            assert msg.content[0]["text"] == "Paris is the capital of France."

            # The critical assertion: no IDs in content blocks
            block_ids = _collect_content_block_ids(msg)
            assert (
                len(block_ids) == 0
            ), f"Content block IDs must be stripped, found: {block_ids}"

            # Metadata also clean
            assert msg.additional_kwargs == {"cached": True}
            assert msg.response_metadata == {}

    @pytest.mark.asyncio
    async def test_cache_hit_strips_reasoning_and_text_block_ids(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Both reasoning and text blocks have IDs stripped."""
        cache_name = f"resp_reason_{uuid.uuid4().hex[:8]}"

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
                    "content": [
                        {
                            "type": "reasoning",
                            "id": "rs_reasoning_abc",
                            "summary": [
                                {"type": "summary_text", "text": "Let me think..."}
                            ],
                        },
                        {
                            "type": "text",
                            "text": "42 is the answer.",
                            "id": "rs_text_def",
                            "annotations": [],
                        },
                    ],
                    "type": "ai",
                    "id": "msg-with-reasoning",
                    "tool_calls": [],
                    "additional_kwargs": {},
                    "response_metadata": {},
                },
            }
        )
        cache.store(
            prompt="What is the meaning of life?",
            response=cached_response,
        )

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        async def fail_handler(request):
            raise AssertionError("Handler should not be called")

        async with SemanticCacheMiddleware(config) as middleware:
            request = {
                "messages": [HumanMessage(content="What is the meaning of life?")]
            }
            result = await middleware.awrap_model_call(request, fail_handler)

            msg = result.result[0]
            assert isinstance(msg.content, list)
            assert len(msg.content) == 2

            # All IDs stripped
            for block in msg.content:
                assert "id" not in block, f"Block still has 'id': {block}"

            # Content preserved
            assert msg.content[0]["summary"][0]["text"] == "Let me think..."
            assert msg.content[1]["text"] == "42 is the answer."


# -- Test: _sanitize_request strips IDs from checkpoint state -----------------


class TestSanitizeRequestStripsProviderIds:
    """Verify _sanitize_request cleans AIMessages from checkpoint state."""

    def test_strips_content_block_ids_from_dict_request(self):
        """Dict-style request with Responses API AIMessage gets cleaned."""
        ai_msg = _make_responses_api_aimessage(
            "Hello from cache",
            block_id="rs_stale_checkpoint_001",
        )
        request = {
            "messages": [
                HumanMessage(content="Hi"),
                ai_msg,
                HumanMessage(content="Follow-up"),
            ]
        }

        cleaned = _sanitize_request(request)
        cleaned_ai = cleaned["messages"][1]

        assert isinstance(cleaned_ai, AIMessage)
        block_ids = _collect_content_block_ids(cleaned_ai)
        assert len(block_ids) == 0, f"Stale block IDs not stripped: {block_ids}"
        assert cleaned_ai.response_metadata == {}

        # Non-AI messages unchanged
        assert cleaned["messages"][0].content == "Hi"
        assert cleaned["messages"][2].content == "Follow-up"

    def test_strips_content_block_ids_with_reasoning(self):
        """Reasoning + text blocks both get IDs stripped."""
        ai_msg = _make_responses_api_aimessage(
            "Answer",
            block_id="rs_text_block",
            with_reasoning=True,
        )
        request = {"messages": [ai_msg]}

        cleaned = _sanitize_request(request)
        cleaned_ai = cleaned["messages"][0]

        for block in cleaned_ai.content:
            assert "id" not in block, f"Block ID not stripped: {block}"

    def test_preserves_string_content_aimessage(self):
        """AIMessage with string content (Chat Completions) passes through."""
        ai_msg = AIMessage(
            content="Plain string content",
            id="chatcmpl-abc",
            additional_kwargs={},
            response_metadata={},
        )
        request = {"messages": [HumanMessage(content="Q"), ai_msg]}

        cleaned = _sanitize_request(request)
        # No change needed - string content has no block IDs
        assert cleaned["messages"][1].content == "Plain string content"

    def test_no_change_returns_original_request(self):
        """When no cleaning is needed, returns the original request object."""
        request = {"messages": [HumanMessage(content="Hello")]}
        cleaned = _sanitize_request(request)
        assert cleaned is request  # Same object, no copy

    def test_preserves_cached_marker(self):
        """If message already has cached=True, it's preserved."""
        ai_msg = AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "Cached!",
                    "id": "rs_old_id",
                    "annotations": [],
                }
            ],
            id="cached-msg",
            additional_kwargs={"cached": True},
            response_metadata={"model": "gpt-4o"},
        )
        request = {"messages": [ai_msg]}

        cleaned = _sanitize_request(request)
        cleaned_ai = cleaned["messages"][0]
        assert cleaned_ai.additional_kwargs == {"cached": True}
        assert cleaned_ai.response_metadata == {}
        assert "id" not in cleaned_ai.content[0]


# -- Test: MiddlewareStack sanitizes before LLM call --------------------------


class TestMiddlewareStackSanitizesRequests:
    """Verify MiddlewareStack wraps handler with _sanitize_request."""

    @pytest.mark.asyncio
    async def test_stack_strips_stale_ids_before_handler(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Messages in request are cleaned before reaching the LLM handler."""
        captured_requests = []

        async def capture_handler(request):
            captured_requests.append(request)
            return ModelResponse(
                result=[
                    AIMessage(
                        content="Fresh response",
                        id="new-msg",
                    )
                ],
                structured_response=None,
            )

        # Use a unique cache name to avoid collisions
        cache_name = f"stack_sanitize_{uuid.uuid4().hex[:8]}"
        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
            cache_final_only=False,
        )

        middleware = SemanticCacheMiddleware(config)
        stack = MiddlewareStack([middleware])

        # Build request with stale checkpoint AIMessages containing block IDs
        stale_ai_msg = _make_responses_api_aimessage(
            "Stale cached response",
            block_id="rs_stale_from_checkpoint",
        )
        request = {
            "messages": [
                HumanMessage(content="First question"),
                stale_ai_msg,
                HumanMessage(content="New unique question " + uuid.uuid4().hex),
            ]
        }

        try:
            result = await stack.awrap_model_call(request, capture_handler)

            # Handler should have been called (cache miss on unique question)
            assert len(captured_requests) == 1

            # Verify the request that reached the handler has clean messages
            handler_request = captured_requests[0]
            if isinstance(handler_request, dict):
                messages = handler_request["messages"]
            else:
                messages = handler_request.messages

            for msg in messages:
                if isinstance(msg, AIMessage):
                    block_ids = _collect_content_block_ids(msg)
                    assert (
                        len(block_ids) == 0
                    ), f"Stale block IDs reached handler: {block_ids}"
                    assert (
                        msg.response_metadata == {}
                    ), f"Stale metadata reached handler: {msg.response_metadata}"
        finally:
            await stack.aclose()


# -- Test: Multi-turn conversation with Responses API format ------------------


class TestMultiTurnResponsesAPIConversation:
    """End-to-end multi-turn test mirroring the customer's actual setup."""

    @pytest.mark.asyncio
    async def test_multi_turn_no_duplicate_content_block_ids(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Multi-turn conversation: no duplicate content block IDs across messages.

        Scenario:
        1. User asks Q -> LLM returns Responses API AIMessage with rs_ block IDs
        2. Same Q again -> cache hit returns clean message (no block IDs)
        3. Both messages in conversation state
        4. Verify: no duplicate rs_ IDs that would cause API error
        """
        cache_name = f"multiturn_resp_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        call_count = [0]
        original_block_id = "rs_0ae41c16891c342b0069b03ac28e508197af7e9ecae0be58cb"

        async def responses_api_handler(request):
            call_count[0] += 1
            return ModelResponse(
                result=[
                    AIMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Paris is the capital of France.",
                                "id": original_block_id,
                                "annotations": [],
                            }
                        ],
                        id=f"msg-turn-{call_count[0]}",
                        additional_kwargs={},
                        response_metadata={
                            "id": "rs_resp_original",
                            "model": "gpt-4o",
                        },
                    )
                ],
                structured_response=None,
            )

        conversation = []

        async with SemanticCacheMiddleware(config) as middleware:
            # Turn 1: Cache miss -> handler called
            user_msg1 = HumanMessage(content="What is the capital of France?")
            conversation.append(user_msg1)

            result1 = await middleware.awrap_model_call(
                {"messages": conversation.copy()},
                responses_api_handler,
            )
            assert call_count[0] == 1
            ai_msg1 = result1.result[0]
            conversation.append(ai_msg1)

            # Turn 1 message has original block IDs (from handler, not cache)
            turn1_block_ids = _collect_content_block_ids(ai_msg1)

            # Turn 2: Same question -> cache hit
            user_msg2 = HumanMessage(content="What is the capital of France?")
            conversation.append(user_msg2)

            result2 = await middleware.awrap_model_call(
                {"messages": conversation.copy()},
                responses_api_handler,
            )
            assert call_count[0] == 1, "Handler must not be called on cache hit"
            ai_msg2 = result2.result[0]
            conversation.append(ai_msg2)

            # THE CRITICAL ASSERTION: cached message has NO content block IDs
            cached_block_ids = _collect_content_block_ids(ai_msg2)
            assert len(cached_block_ids) == 0, (
                f"Cached message must have no content block IDs, "
                f"found: {cached_block_ids}"
            )

            # Messages have different top-level IDs
            assert ai_msg1.id != ai_msg2.id

            # Content text preserved
            assert ai_msg2.content[0]["text"] == "Paris is the capital of France."

            # No overlap in block IDs between the two messages
            all_block_ids = turn1_block_ids | cached_block_ids
            # Even if turn1 has IDs, turn2 must have none
            assert not turn1_block_ids.intersection(cached_block_ids)

    @pytest.mark.asyncio
    async def test_three_turn_conversation_all_cached_clean(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Three identical questions: first fresh, next two cached.

        All cached messages must have zero content block IDs.
        """
        cache_name = f"three_turn_resp_{uuid.uuid4().hex[:8]}"

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
                        content=[
                            {
                                "type": "reasoning",
                                "id": "rs_reasoning_first",
                                "summary": [
                                    {"type": "summary_text", "text": "Thinking..."}
                                ],
                            },
                            {
                                "type": "text",
                                "text": "Python is a programming language.",
                                "id": "rs_text_first",
                                "annotations": [],
                            },
                        ],
                        id="msg-first",
                        additional_kwargs={},
                        response_metadata={"model": "gpt-4o"},
                    )
                ],
                structured_response=None,
            )

        conversation = []
        ai_messages = []

        async with SemanticCacheMiddleware(config) as middleware:
            for turn in range(3):
                conversation.append(HumanMessage(content="What is Python?"))
                result = await middleware.awrap_model_call(
                    {"messages": conversation.copy()}, handler
                )
                ai_msg = result.result[0]
                conversation.append(ai_msg)
                ai_messages.append(ai_msg)

            # Only 1 handler call
            assert call_count[0] == 1

            # All 3 messages must have unique top-level IDs
            ids = [msg.id for msg in ai_messages]
            assert len(set(ids)) == 3, f"All IDs must be unique: {ids}"

            # Cached messages (turns 2 and 3) must have zero content block IDs
            for i, msg in enumerate(ai_messages[1:], start=2):
                block_ids = _collect_content_block_ids(msg)
                assert (
                    len(block_ids) == 0
                ), f"Turn {i} cached message has content block IDs: {block_ids}"

            # All messages preserve content text
            for msg in ai_messages:
                texts = [b["text"] for b in msg.content if b.get("type") == "text"]
                assert texts == ["Python is a programming language."]

    @pytest.mark.asyncio
    async def test_middleware_stack_multi_turn_responses_api(
        self, redis_url: str, vectorizer: HFTextVectorizer
    ):
        """Full stack (SemanticCache in MiddlewareStack) multi-turn test.

        Verifies the safety-net sanitization in MiddlewareStack works alongside
        the cache deserialization fix.
        """
        cache_name = f"stack_multiturn_{uuid.uuid4().hex[:8]}"

        config = SemanticCacheConfig(
            redis_url=redis_url,
            name=cache_name,
            distance_threshold=0.1,
            vectorizer=vectorizer,
        )

        call_count = [0]
        captured_requests = []
        block_id = "rs_0ae41c16891c342b0069b03ac28e508197af7e9ecae0be58cb"

        async def handler(request):
            call_count[0] += 1
            captured_requests.append(request)
            return ModelResponse(
                result=[
                    AIMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "The answer.",
                                "id": block_id,
                                "annotations": [],
                            }
                        ],
                        id=f"msg-{call_count[0]}",
                        additional_kwargs={},
                        response_metadata={"model": "gpt-4o"},
                    )
                ],
                structured_response=None,
            )

        middleware = SemanticCacheMiddleware(config)
        stack = MiddlewareStack([middleware])

        try:
            conversation = []

            # Turn 1: cache miss
            conversation.append(HumanMessage(content="Question one"))
            result1 = await stack.awrap_model_call(
                {"messages": conversation.copy()}, handler
            )
            ai_msg1 = result1.result[0]
            conversation.append(ai_msg1)

            # Turn 2: different question (cache miss) but AI msg from turn 1
            # is in the conversation with block IDs
            unique_q = f"Different question {uuid.uuid4().hex[:8]}"
            conversation.append(HumanMessage(content=unique_q))
            result2 = await stack.awrap_model_call(
                {"messages": conversation.copy()}, handler
            )

            assert call_count[0] == 2

            # The request that reached the handler on turn 2 should have
            # ai_msg1's content block IDs stripped (safety net sanitization)
            handler_req = captured_requests[1]
            handler_messages = (
                handler_req["messages"]
                if isinstance(handler_req, dict)
                else handler_req.messages
            )

            for msg in handler_messages:
                if isinstance(msg, AIMessage):
                    ids = _collect_content_block_ids(msg)
                    assert len(ids) == 0, f"Stale block IDs reached LLM handler: {ids}"
        finally:
            await stack.aclose()
