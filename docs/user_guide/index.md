---
title: User Guide
---

# User Guide

Learn how to use `langgraph-checkpoint-redis` for persistent LangGraph workflows
backed by Redis. These guides cover everything from installation to advanced
deployment patterns.

::::{grid} 3
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install the package, set up Redis, and verify your environment.
:::

:::{grid-item-card} Getting Started
:link: getting_started
:link-type: doc

Build your first LangGraph workflow with Redis persistence in minutes.
:::

:::{grid-item-card} Checkpointers
:link: checkpointers
:link-type: doc

Deep dive into `RedisSaver` and `AsyncRedisSaver` for full checkpoint history.
:::

:::{grid-item-card} Shallow Checkpointers
:link: shallow_checkpointers
:link-type: doc

Store only the latest checkpoint per thread for reduced storage overhead.
:::

:::{grid-item-card} Stores
:link: stores
:link-type: doc

Key-value storage with optional vector search using `RedisStore`.
:::

:::{grid-item-card} TTL Configuration
:link: ttl
:link-type: doc

Configure automatic expiration and refresh-on-read for checkpoints and stores.
:::

:::{grid-item-card} Middleware
:link: middleware
:link-type: doc

Add semantic caching, tool caching, routing, and conversation memory to agents.
:::

:::{grid-item-card} Azure and Enterprise
:link: azure_enterprise
:link-type: doc

Deploy with Azure Managed Redis, Azure Cache for Redis, or Redis Enterprise.
:::

:::{grid-item-card} Message Exporter
:link: message_exporter
:link-type: doc

Extract and export conversation messages from checkpoints for analytics and auditing.
:::

:::{grid-item-card} Migration Guide
:link: migration
:link-type: doc

Upgrade between versions with step-by-step migration instructions.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

installation
getting_started
checkpointers
shallow_checkpointers
stores
ttl
middleware
message_exporter
azure_enterprise
migration
```
