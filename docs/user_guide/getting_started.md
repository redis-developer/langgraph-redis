---
title: Getting Started
---

# Getting Started

This tutorial walks through creating a LangGraph workflow with Redis-backed
checkpoint persistence. By the end, you will have a graph that saves its state
to Redis and can resume from any checkpoint.

## Prerequisites

- Python 3.10+
- A running Redis 8.0+ instance (see {doc}`installation`)
- `langgraph-checkpoint-redis` installed

## Step 1: Define a Simple Graph

Create a minimal LangGraph `StateGraph` that tracks a list of messages:

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END


# Define the state schema
class State(TypedDict):
    messages: Annotated[list[str], lambda a, b: a + b]


# Define graph nodes
def greet(state: State) -> dict:
    return {"messages": ["Hello! How can I help you?"]}


def farewell(state: State) -> dict:
    return {"messages": ["Goodbye!"]}


# Build the graph
builder = StateGraph(State)
builder.add_node("greet", greet)
builder.add_node("farewell", farewell)
builder.add_edge(START, "greet")
builder.add_edge("greet", "farewell")
builder.add_edge("farewell", END)
```

## Step 2: Add Redis Persistence

Use `RedisSaver` as the checkpointer when compiling the graph:

```python
from langgraph.checkpoint.redis import RedisSaver

REDIS_URL = "redis://localhost:6379"

with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
    checkpointer.setup()  # Create search indices in Redis

    # Compile the graph with the Redis checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    # Run the graph with a thread ID
    config = {"configurable": {"thread_id": "my-first-thread"}}
    result = graph.invoke({"messages": ["Hi there"]}, config)
    print(result["messages"])
    # Output: ['Hi there', 'Hello! How can I help you?', 'Goodbye!']
```

## Step 3: Retrieve State from a Thread

After the graph has run, you can retrieve the saved state at any time:

```python
with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
    checkpointer.setup()

    config = {"configurable": {"thread_id": "my-first-thread"}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if checkpoint_tuple:
        print("Thread ID:", config["configurable"]["thread_id"])
        print("Checkpoint ID:", checkpoint_tuple.config["configurable"]["checkpoint_id"])
        print("Metadata:", checkpoint_tuple.metadata)
```

## Step 4: List Checkpoint History

View all checkpoints stored for a thread:

```python
with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
    checkpointer.setup()

    config = {"configurable": {"thread_id": "my-first-thread"}}
    for checkpoint_tuple in checkpointer.list(config):
        print(
            f"Step {checkpoint_tuple.metadata.get('step')}: "
            f"checkpoint {checkpoint_tuple.config['configurable']['checkpoint_id']}"
        )
```

## Using the Async Variant

For async applications, use `AsyncRedisSaver` with the same API surface:

```python
import asyncio
from langgraph.checkpoint.redis import AsyncRedisSaver


async def main():
    async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        await checkpointer.asetup()

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "async-thread"}}
        result = await graph.ainvoke({"messages": ["Hi"]}, config)
        print(result["messages"])


asyncio.run(main())
```

## Using a Direct Redis Client

If you already have a Redis client instance, pass it directly instead of a
connection string:

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

redis_client = Redis.from_url("redis://localhost:6379")
saver = RedisSaver(redis_client=redis_client)
saver.setup()

graph = builder.compile(checkpointer=saver)
result = graph.invoke(
    {"messages": ["Hi"]},
    {"configurable": {"thread_id": "direct-client-thread"}},
)
print(result["messages"])

# Remember to close the client when done
redis_client.close()
```

## What Gets Stored in Redis

When a graph runs with a Redis checkpointer, the following data is stored:

- **Checkpoint documents** -- the full graph state at each step, keyed as
  `checkpoint:{thread_id}:{namespace}:{checkpoint_id}`
- **Checkpoint writes** -- intermediate write operations, keyed as
  `checkpoint_write:{thread_id}:{namespace}:{checkpoint_id}:{task_id}`
- **Search indices** -- RediSearch indices for efficient querying by thread,
  namespace, or checkpoint ID

## Next Steps

- {doc}`checkpointers` -- detailed guide to checkpoint configuration options
- {doc}`stores` -- use `RedisStore` for cross-thread key-value persistence
- {doc}`ttl` -- configure automatic expiration of checkpoints
