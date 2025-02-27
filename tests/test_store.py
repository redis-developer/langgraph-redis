from typing import Any, Dict, Sequence, cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
)
from redis import Redis
from ulid import ULID

from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from tests.conftest import VECTOR_TYPES
from tests.embed_test_utils import CharacterEmbeddings


@pytest.fixture(scope="function", autouse=True)
def clear_test_redis(redis_url: str) -> None:
    client = Redis.from_url(redis_url)
    try:
        client.flushall()
    finally:
        client.close()


@pytest.fixture
def store(redis_url: str) -> RedisStore:
    with RedisStore.from_conn_string(redis_url) as store:
        store.setup()
        return store


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    """Provide a simple embeddings implementation for testing."""
    return CharacterEmbeddings(dims=4)


def test_batch_order(store: RedisStore) -> None:
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    ops: list[Op] = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=(), max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = store.batch(ops)
    assert len(results) == 5

    item = cast(Item, results[0])
    assert isinstance(item, Item)
    assert item.value == {"data": "value1"}
    assert item.key == "key1"

    assert results[1] is None

    search_results = cast(Sequence[SearchItem], results[2])
    assert len(search_results) == 1
    assert search_results[0].value == {"data": "value1"}

    namespaces = cast(list[tuple[str, ...]], results[3])
    assert len(namespaces) >= 2
    assert ("test", "foo") in namespaces
    assert ("test", "bar") in namespaces

    assert results[4] is None


def test_batch_put_ops(store: RedisStore) -> None:
    ops: list[Op] = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),
    ]

    results = store.batch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)

    search_results = store.search(("test",), limit=10)
    assert len(search_results) == 2


def test_batch_search_ops(store: RedisStore) -> None:
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops: list[Op] = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    if isinstance(results[0], list):
        assert len(results[0]) >= 2  # Should find all test documents
    else:
        raise AssertionError(
            "Expected results[0] to be a list, got None or incompatible type"
        )

    if isinstance(results[1], list):
        assert len(results[1]) == 2  # Limited to 2 results
    else:
        raise AssertionError(
            "Expected results[1] to be a list, got None or incompatible type"
        )

    if isinstance(results[2], list):
        assert len(results[2]) == 1  # Only foo namespace
    else:
        raise AssertionError(
            "Expected results[2] to be a list, got None or incompatible type"
        )


def test_batch_list_namespaces_ops(store: RedisStore) -> None:
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops: list[Op] = [
        ListNamespacesOp(match_conditions=(), max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=(), max_depth=2, limit=10, offset=0),
        ListNamespacesOp(
            match_conditions=(MatchCondition("suffix", ("public",)),),
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = store.batch(ops)

    namespaces = cast(list[tuple[str, ...]], results[0])
    assert len(namespaces) == len(test_data)

    namespaces_depth = cast(list[tuple[str, ...]], results[1])
    assert all(len(ns) <= 2 for ns in namespaces_depth)

    namespaces_public = cast(list[tuple[str, ...]], results[2])
    assert all(ns[-1] == "public" for ns in namespaces_public)


def test_list_namespaces(store: RedisStore) -> None:
    # Create test data with various namespaces
    test_namespaces = [
        ("test", "documents", "public"),
        ("test", "documents", "private"),
        ("test", "images", "public"),
        ("test", "images", "private"),
        ("prod", "documents", "public"),
        ("prod", "documents", "private"),
    ]

    # Insert test data
    for namespace in test_namespaces:
        store.put(namespace, "dummy", {"content": "dummy"})

    # Test listing with various filters
    all_namespaces = store.list_namespaces()
    assert len(all_namespaces) == len(test_namespaces)

    # Test prefix filtering
    test_prefix_namespaces = store.list_namespaces(prefix=tuple(["test"]))
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)

    # Test suffix filtering
    public_namespaces = store.list_namespaces(suffix=tuple(["public"]))
    assert len(public_namespaces) == 3
    assert all(ns[-1] == "public" for ns in public_namespaces)

    # Test max depth
    depth_2_namespaces = store.list_namespaces(max_depth=2)
    assert all(len(ns) <= 2 for ns in depth_2_namespaces)

    # Test pagination
    paginated_namespaces = store.list_namespaces(limit=3)
    assert len(paginated_namespaces) == 3

    # Cleanup
    for namespace in test_namespaces:
        store.delete(namespace, "dummy")


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [*[(vt, dt) for vt in VECTOR_TYPES for dt in ["cosine", "inner_product", "l2"]]],
)
def test_vector_search(
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
    redis_url: str,
) -> None:
    index_config: IndexConfig = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "text_fields": ["text"],
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
    }

    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()

        docs = [
            ("doc1", {"text": "short text"}),
            ("doc2", {"text": "longer text document"}),
            ("doc3", {"text": "longest text document here"}),
        ]

        for key, value in docs:
            store.put(("test",), key, value)

        results = store.search(("test",), query="long text")
        assert len(results) > 0

        doc_order = [r.key for r in results]
        assert "doc2" in doc_order
        assert "doc3" in doc_order

        results = store.search(("test",), query="short text")
        assert len(results) > 0
        assert results[0].key == "doc1"


def test_search(store: RedisStore) -> None:
    # Create test data
    test_data = [
        (
            ("test", "docs"),
            "doc1",
            {"title": "First Doc", "author": "Alice", "tags": ["important"]},
        ),
        (
            ("test", "docs"),
            "doc2",
            {"title": "Second Doc", "author": "Bob", "tags": ["draft"]},
        ),
        (
            ("test", "images"),
            "img1",
            {"title": "Image 1", "author": "Alice", "tags": ["final"]},
        ),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    # Test basic search
    all_items = store.search(tuple(["test"]))
    assert len(all_items) == 3

    # Test namespace filtering
    docs_items = store.search(tuple(["test", "docs"]))
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)

    # Test value filtering
    alice_items = store.search(tuple(["test"]), filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(item.value["author"] == "Alice" for item in alice_items)

    # Test pagination
    paginated_items = store.search(tuple(["test"]), limit=2)
    assert len(paginated_items) == 2

    offset_items = store.search(tuple(["test"]), offset=2)
    assert len(offset_items) == 1

    # Cleanup
    for namespace, key, _ in test_data:
        store.delete(namespace, key)


def test_basic_ops(store: RedisStore) -> None:
    store.put(("test",), "key1", {"data": "value1"})
    item = store.get(("test",), "key1")
    assert item is not None
    assert item.value["data"] == "value1"

    store.put(("test",), "key1", {"data": "updated"})
    updated = store.get(("test",), "key1")
    assert updated is not None
    assert updated.value["data"] == "updated"
    assert updated.updated_at > item.updated_at

    store.delete(("test",), "key1")
    deleted = store.get(("test",), "key1")
    assert deleted is None

    # Namespace isolation
    store.put(("test", "ns1"), "key1", {"data": "ns1"})
    store.put(("test", "ns2"), "key1", {"data": "ns2"})

    ns1_item = store.get(("test", "ns1"), "key1")
    ns2_item = store.get(("test", "ns2"), "key1")
    assert ns1_item is not None
    assert ns2_item is not None
    assert ns1_item.value["data"] == "ns1"
    assert ns2_item.value["data"] == "ns2"


def test_large_batches(store: RedisStore) -> None:
    N = 100  # less important that we are performant here
    M = 10

    for m in range(M):
        for i in range(N):
            store.put(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
                value={"foo": "bar" + str(i)},
            )
            store.get(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
            )
            store.list_namespaces(
                prefix=None,
                max_depth=m + 1,
            )
            store.search(
                ("test",),
            )
            store.put(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
                value={"foo": "bar" + str(i)},
            )
            store.delete(
                ("test", "foo", "bar", "baz", str(m % 2)),
                f"key{i}",
            )


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [*[(vt, dt) for vt in VECTOR_TYPES for dt in ["cosine", "inner_product", "l2"]]],
)
def test_vector_update_with_score_verification(
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
    redis_url: str,
) -> None:
    """Test that updating items properly updates their embeddings and scores."""
    index_config: IndexConfig = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "text_fields": ["text"],
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
    }

    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()

        store.put(("test",), "doc1", {"text": "zany zebra Xerxes"})
        store.put(("test",), "doc2", {"text": "something about dogs"})
        store.put(("test",), "doc3", {"text": "text about birds"})

        results_initial = store.search(("test",), query="Zany Xerxes")
        assert len(results_initial) > 0
        assert results_initial[0].key == "doc1"
        assert results_initial[0].score is not None
        initial_score = results_initial[0].score

        store.put(("test",), "doc1", {"text": "new text about dogs"})

        # After updating content to be about dogs instead of zebras,
        # searching for the original zebra content should give a much lower score
        results_after = store.search(("test",), query="Zany Xerxes")
        # The doc may not even be in top results anymore since content changed
        after_doc = next((r for r in results_after if r.key == "doc1"), None)
        assert after_doc is None or (
            after_doc.score is not None and after_doc.score < initial_score
        )

        # When searching for dog content, doc1 should now score highly
        results_new = store.search(("test",), query="new text about dogs")
        doc1_new = next((r for r in results_new if r.key == "doc1"), None)
        assert doc1_new is not None and doc1_new.score is not None
        if after_doc is not None and after_doc.score is not None:
            assert doc1_new.score > after_doc.score

        # Don't index this one
        store.put(("test",), "doc4", {"text": "new text about dogs"}, index=False)
        results_new = store.search(("test",), query="new text about dogs", limit=3)
        assert not any(r.key == "doc4" for r in results_new)


@pytest.mark.requires_api_keys
def test_store_with_memory_persistence(redis_url: str) -> None:
    """Test store functionality with memory persistence.

    Tests the complete flow of:
    1. Storing a memory when asked
    2. Retrieving that memory in a subsequent interaction
    3. Verifying responses reflect the stored information
    """
    index_config: IndexConfig = {
        "dims": 1536,
        "text_fields": ["data"],
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "ann_index_config": {
            "vector_type": "vector",
        },
        "distance_type": "cosine",
    }

    with RedisStore.from_conn_string(redis_url, index=index_config) as store:
        store.setup()
        model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

        def call_model(
            state: MessagesState, config: RunnableConfig, *, store: BaseStore
        ) -> Dict[str, Any]:
            user_id = config["configurable"]["user_id"]
            namespace = ("memories", user_id)
            last_message = cast(BaseMessage, state["messages"][-1])
            memories = store.search(namespace, query=str(last_message.content))
            info = "\n".join([d.value["data"] for d in memories])
            system_msg = (
                f"You are a helpful assistant talking to the user. User info: {info}"
            )

            # Store new memories if the user asks the model to remember
            if "remember" in last_message.content.lower():  # type:ignore[union-attr]
                memory = "User name is Bob"
                store.put(namespace, str(ULID()), {"data": memory})

            messages = [{"role": "system", "content": system_msg}]
            messages.extend([msg.model_dump() for msg in state["messages"]])
            response = model.invoke(messages)
            return {"messages": response}

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)  # type:ignore[arg-type]
        builder.add_edge(START, "call_model")

        checkpointer = None
        with RedisSaver.from_conn_string(redis_url) as cp:
            cp.setup()
            checkpointer = cp

        # Compile graph with store and checkpointer
        graph = builder.compile(checkpointer=checkpointer, store=store)

        # Test 1: Initial message asking to remember name
        config: RunnableConfig = {
            "configurable": {"thread_id": "sync1", "user_id": "1"}
        }
        input_message = HumanMessage(content="Hi! Remember: my name is Bob")
        response = graph.invoke({"messages": [input_message]}, config)

        assert "Hi Bob" in response["messages"][1].content

        # Test 2: inspect the Redis store and verify that we have in fact saved the memories for the user
        for memory in store.search(("memories", "1")):
            assert memory.value["data"] == "User name is Bob"

        # run the graph for another user to verify that the memories about the first user are self-contained
        input_message = HumanMessage(content="what's my name?")
        response = graph.invoke({"messages": [input_message]}, config)

        assert "Bob" in response["messages"][1].content

        # Test 3: New conversation (different thread) shouldn't know the name
        new_config: RunnableConfig = {
            "configurable": {"thread_id": "sync3", "user_id": "2"}
        }
        input_message = HumanMessage(content="what's my name?")
        response = graph.invoke({"messages": [input_message]}, new_config)

        assert "Bob" not in response["messages"][1].content
