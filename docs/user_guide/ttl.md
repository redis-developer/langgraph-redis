---
title: TTL Configuration
---

# TTL Configuration

Time-to-live (TTL) controls automatic expiration of checkpoints and store items
in Redis. TTL keeps your Redis instance from growing unboundedly by removing
data that is no longer needed after a configured period.

## TTL Config Dictionary

TTL is configured through a dictionary with two optional keys:

```python
ttl_config = {
    "default_ttl": 60,          # TTL in minutes
    "refresh_on_read": True,    # Reset TTL when data is read
}
```

| Key | Type | Description |
|-----|------|-------------|
| `default_ttl` | `float` | Time-to-live in **minutes**. After this period, keys are automatically removed by Redis. |
| `refresh_on_read` | `bool` | When `True`, reading a checkpoint or store item resets its TTL to the full `default_ttl`. Keeps active data alive. |

:::{important}
The `default_ttl` value is specified in **minutes**, not seconds. A value of
`60` means one hour.
:::

## Checkpointer TTL

### With from_conn_string

Pass the TTL config when creating the checkpointer:

```python
from langgraph.checkpoint.redis import RedisSaver

ttl_config = {"default_ttl": 120, "refresh_on_read": True}  # 2 hours

with RedisSaver.from_conn_string("redis://localhost:6379", ttl=ttl_config) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)
    # Checkpoints will expire after 2 hours of inactivity
```

### With Direct Constructor

```python
saver = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl={"default_ttl": 60, "refresh_on_read": True},
)
saver.setup()
```

### What Gets TTL

When TTL is configured, it applies to all keys related to a checkpoint:

- The checkpoint document itself
- All checkpoint write documents
- The latest-checkpoint pointer key
- Key registry sorted sets

All related keys receive the same TTL so they expire together.

## Store TTL

TTL works for `RedisStore` items as well:

```python
from langgraph.store.redis import RedisStore

with RedisStore.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 30},  # 30 minutes
) as store:
    store.setup()

    # This item will expire after 30 minutes
    store.put(("sessions",), "session-1", {"user": "alice"})
```

### Per-Item TTL

Individual items can specify their own TTL, overriding the global default:

```python
# This item expires in 5 minutes regardless of the global TTL
store.put(
    ("cache",), "temp-data",
    {"result": "ephemeral"},
    ttl=5,
)
```

## Refresh-on-Read Behavior

When `refresh_on_read` is `True`, any read operation resets the TTL for
the accessed data:

```python
ttl_config = {"default_ttl": 60, "refresh_on_read": True}

with RedisSaver.from_conn_string("redis://localhost:6379", ttl=ttl_config) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "active-thread"}}

    # First invocation: checkpoint created with 60-minute TTL
    graph.invoke(inputs, config)

    # 30 minutes later: get_tuple refreshes TTL back to 60 minutes
    checkpoint = saver.get_tuple(config)
    # The checkpoint now has a fresh 60-minute TTL
```

This pattern is useful for keeping active conversations alive while allowing
idle ones to expire naturally.

## Pinning Threads (Removing TTL)

To make a thread persistent (remove its TTL), use `_apply_ttl_to_keys` with
a value of `-1`:

```python
with RedisSaver.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 60, "refresh_on_read": True},
) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "important-thread"}}
    graph.invoke(inputs, config)

    # Pin this thread: remove TTL so it never expires
    checkpoint_tuple = saver.get_tuple(config)
    if checkpoint_tuple:
        checkpoint_key = saver._make_redis_checkpoint_key_cached(
            config["configurable"]["thread_id"],
            "",  # checkpoint_ns
            checkpoint_tuple.config["configurable"]["checkpoint_id"],
        )
        saver._apply_ttl_to_keys(checkpoint_key, ttl_minutes=-1)
```

## Example: Ephemeral vs. Persistent Threads

A common pattern is to have most threads expire automatically while pinning
important ones:

```python
from langgraph.checkpoint.redis import RedisSaver

# Most threads expire after 2 hours
ttl_config = {"default_ttl": 120, "refresh_on_read": True}

with RedisSaver.from_conn_string("redis://localhost:6379", ttl=ttl_config) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    # Regular user sessions expire naturally
    for user_message in user_messages:
        config = {"configurable": {"thread_id": f"session:{user_message.user_id}"}}
        graph.invoke({"messages": [user_message.text]}, config)

    # Admin sessions are pinned and never expire
    admin_config = {"configurable": {"thread_id": "admin:dashboard"}}
    graph.invoke({"messages": ["Show system status"]}, admin_config)

    # Pin the admin thread
    ct = saver.get_tuple(admin_config)
    if ct:
        key = saver._make_redis_checkpoint_key_cached(
            "admin:dashboard", "",
            ct.config["configurable"]["checkpoint_id"],
        )
        saver._apply_ttl_to_keys(key, ttl_minutes=-1)
```

## Async TTL

TTL works identically with async checkpointers and stores:

```python
from langgraph.checkpoint.redis import AsyncRedisSaver

async with AsyncRedisSaver.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 60, "refresh_on_read": True},
) as saver:
    await saver.asetup()
    graph = builder.compile(checkpointer=saver)
    result = await graph.ainvoke(inputs, config)
```

## Next Steps

- {doc}`checkpointers` -- checkpointer reference
- {doc}`stores` -- store reference with per-item TTL
- {doc}`azure_enterprise` -- TTL considerations for enterprise deployments
