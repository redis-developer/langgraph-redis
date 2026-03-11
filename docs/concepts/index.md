# Concepts

Core ideas and design decisions behind `langgraph-checkpoint-redis`.

::::{grid} 2 2 3 3
:gutter: 3

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc

Dual sync/async implementation, Redis module dependencies, schema-driven indexing, and package structure.
:::

:::{grid-item-card} Checkpointing
:link: checkpointing
:link-type: doc

Checkpoint lifecycle, threads and namespaces, pending writes, and full vs shallow savers.
:::

:::{grid-item-card} Stores
:link: stores
:link-type: doc

Persistent key-value storage with namespaces, vector search integration, and cross-thread memory.
:::

:::{grid-item-card} TTL Management
:link: ttl
:link-type: doc

Native Redis TTL, refresh-on-read, TTL propagation, and pinning with PERSIST.
:::

:::{grid-item-card} Cluster Support
:link: cluster
:link-type: doc

Automatic cluster detection, cross-slot handling, Azure Managed Redis, and Redis 8.0+ compatibility.
:::

:::{grid-item-card} Middleware
:link: middleware
:link-type: doc

Semantic caching, tool result caching, conversation memory, semantic routing, and middleware composition.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

architecture
checkpointing
stores
ttl
cluster
middleware
```
