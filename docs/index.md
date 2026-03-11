# langgraph-checkpoint-redis

Redis implementations for [LangGraph](https://langchain-ai.github.io/langgraph/) — checkpoint savers, stores with vector search, and agent middleware.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Concepts
:link: concepts/index
:link-type: doc

Understand the architecture, key patterns, and design decisions behind Redis-backed checkpointing, stores, and middleware.
:::

:::{grid-item-card} User Guides
:link: user_guide/index
:link-type: doc

Step-by-step guides for installation, configuration, TTL management, middleware setup, and enterprise deployment.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

23 Jupyter notebooks covering checkpoints, human-in-the-loop, memory, middleware, and ReAct agents.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Complete API documentation for all public classes — savers, stores, middleware, and utilities.
:::

::::

## Quick Start

Install the package:

```bash
pip install langgraph-checkpoint-redis
```

Use Redis as your LangGraph checkpoint saver:

```python
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    checkpointer.setup()

    # Use with any LangGraph graph
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "my-thread"}}
    result = graph.invoke({"messages": [("human", "Hello!")]}, config)
```

## Features

- **Checkpoint Savers** — Full and shallow variants, sync and async, with automatic TTL
- **Stores** — Key-value storage with optional vector search for semantic retrieval
- **Middleware** — Semantic caching, tool result caching, conversation memory, and semantic routing
- **Cluster Support** — Automatic detection and handling of Redis Cluster, Azure Managed Redis, and Redis Enterprise
- **TTL Management** — Native Redis TTL with refresh-on-read and per-thread pinning

```{toctree}
:maxdepth: 2
:hidden:

concepts/index
user_guide/index
examples/index
api/index
```

```{toctree}
:caption: Links
:maxdepth: 1
:hidden:

Changelog <https://github.com/redis-developer/langgraph-redis/releases>
PyPI <https://pypi.org/project/langgraph-checkpoint-redis/>
GitHub <https://github.com/redis-developer/langgraph-redis>
```
