"""An interrupt's response_schema survives the checkpoint round trip.

``interrupt(value, response_schema=...)`` stores the JSON Schema of the expected
resume value on the ``Interrupt`` that is written to the checkpoint. Clients read
it back from the thread state to know what to send when they resume.
"""

import sys
from typing import Any, TypedDict
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from langgraph.checkpoint.redis import AsyncRedisSaver, RedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

DECISION_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}},
    "required": ["approved"],
}


class State(TypedDict):
    answer: Any


def _graph(checkpointer: Any, schema: dict[str, Any] = DECISION_SCHEMA) -> Any:
    def ask(state: State) -> State:
        return {"answer": interrupt("Approve?", response_schema=schema)}

    builder = StateGraph(State)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=checkpointer)


def test_serializer_keeps_response_schema() -> None:
    serde = JsonPlusRedisSerializer()
    original = Interrupt(
        value="Approve?", id="interrupt-1", response_schema=DECISION_SCHEMA
    )

    restored = serde.loads_typed(serde.dumps_typed(original))

    assert isinstance(restored, Interrupt)
    assert restored.value == "Approve?"
    assert restored.id == "interrupt-1"
    assert restored.response_schema == DECISION_SCHEMA


@pytest.mark.parametrize("saver_cls", [RedisSaver, ShallowRedisSaver])
def test_pending_interrupt_keeps_response_schema(
    redis_url: str, saver_cls: Any
) -> None:
    with saver_cls.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _graph(saver)
        config = {"configurable": {"thread_id": str(uuid4())}}

        graph.invoke({"answer": None}, config)

        [task] = graph.get_state(config).tasks
        [pending] = task.interrupts
        assert pending.response_schema == DECISION_SCHEMA

        resumed = graph.invoke(Command(resume={"approved": True}), config)
        assert resumed["answer"] == {"approved": True}


def test_pending_interrupt_keeps_a_response_schema_that_holds_a_set(
    redis_url: str,
) -> None:
    schema = {"type": "string", "enum": {"yes", "no"}}
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        graph = _graph(saver, schema)
        config = {"configurable": {"thread_id": str(uuid4())}}

        graph.invoke({"answer": None}, config)

        [task] = graph.get_state(config).tasks
        [pending] = task.interrupts
        assert pending.response_schema == schema

        resumed = graph.invoke(Command(resume="yes"), config)
        assert resumed["answer"] == "yes"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="interrupt() in async context requires Python 3.11+",
)
@pytest.mark.parametrize("saver_cls", [AsyncRedisSaver, AsyncShallowRedisSaver])
async def test_async_pending_interrupt_keeps_response_schema(
    redis_url: str, saver_cls: Any
) -> None:
    async with saver_cls.from_conn_string(redis_url) as saver:
        graph = _graph(saver)
        config = {"configurable": {"thread_id": str(uuid4())}}

        await graph.ainvoke({"answer": None}, config)

        [task] = (await graph.aget_state(config)).tasks
        [pending] = task.interrupts
        assert pending.response_schema == DECISION_SCHEMA

        resumed = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert resumed["answer"] == {"approved": True}
