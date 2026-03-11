# TTL Management

Time-to-live (TTL) controls how long data persists in Redis before automatic expiration. For production deployments, TTL prevents unbounded storage growth and ensures stale data is cleaned up without manual intervention.

## Why TTL Matters

Without TTL, every checkpoint, store item, and associated blob remains in Redis indefinitely. In a production system with thousands of concurrent conversations, this leads to:

- **Storage costs** -- Redis keeps all data in memory. Unlimited retention means memory usage grows without bound.
- **Data hygiene** -- Abandoned conversations and stale session data accumulate over time.
- **Compliance** -- Some applications require data to be automatically deleted after a retention period.

TTL is optional. When no TTL is configured, all data is persistent by default, which is appropriate for development and for use cases where data must be retained indefinitely.

## Configuration

TTL is configured via a dictionary passed to the checkpoint saver or store constructor:

```python
from langgraph.checkpoint.redis import RedisSaver

saver = RedisSaver.from_conn_string(
    "redis://localhost:6379",
    ttl={"default_ttl": 60, "refresh_on_read": True},
)
```

The `ttl` dictionary accepts two keys:

### `default_ttl`

The default time-to-live in **minutes** for all keys managed by this saver or store. When set, every key created by the component will have a Redis `EXPIRE` applied.

```python
# All checkpoints expire after 24 hours
ttl={"default_ttl": 1440}
```

### `refresh_on_read`

When `True`, reading a checkpoint refreshes its TTL back to the full `default_ttl` value. This keeps actively used data alive while allowing idle data to expire.

```python
# Checkpoints expire after 1 hour of inactivity
ttl={"default_ttl": 60, "refresh_on_read": True}
```

Without `refresh_on_read`, a checkpoint created at time T will expire at T + `default_ttl` regardless of how many times it is read.

## Native Redis TTL

The library uses Redis's native `EXPIRE` command to set TTLs. This has several advantages over application-level TTL sweepers:

- **Atomic** -- Redis guarantees the key is removed at expiration time.
- **No background threads** -- No need for a sweeper thread in the application.
- **Efficient** -- Redis handles expiration internally with negligible overhead.

The `sweep_ttl()`, `start_ttl_sweeper()`, and `stop_ttl_sweeper()` methods exist on stores for API compatibility but are no-ops. Redis handles expiration automatically.

## TTL Propagation

A single logical checkpoint consists of multiple Redis keys: the checkpoint document, channel blobs, and pending writes. When TTL is applied, it must propagate to **all related keys** to avoid orphaned data.

For example, when a checkpoint is written:

```
checkpoint:thread-1::01JEXAMPLE          -> EXPIRE 3600
checkpoint_blob:thread-1::messages:v3    -> EXPIRE 3600
checkpoint_write:thread-1::01JEXAMPLE:t1 -> EXPIRE 3600
```

The `_apply_ttl_to_keys` method handles this propagation. In standalone Redis, it uses a transactional pipeline to apply `EXPIRE` atomically to all related keys. In cluster mode, it applies `EXPIRE` to each key individually (since cluster pipelines cannot guarantee atomicity across slots).

## Pinning with PERSIST

Sometimes a specific checkpoint or item needs to be kept indefinitely even when a `default_ttl` is configured. Setting `ttl_minutes=-1` applies the Redis `PERSIST` command, which removes any existing TTL:

```python
# Remove TTL from a specific checkpoint (make it permanent)
saver._apply_ttl_to_keys(
    checkpoint_key,
    related_keys=[blob_key, write_key],
    ttl_minutes=-1,
)
```

This is useful for pinning important checkpoints (e.g., a human-approved state) while letting routine checkpoints expire normally.

## Store TTL

Stores support TTL at both the store level and the per-item level:

```python
from langgraph.store.redis import RedisStore

store = RedisStore(
    conn=redis_client,
    ttl={"default_ttl": 1440},  # 24-hour default
)
```

When a store item has an associated vector document (`store_vectors:{uuid}`), the TTL is applied to both the item document and the vector document to keep them synchronized.

## Default Behavior

| Configuration | Behavior |
|---|---|
| No `ttl` dict | All data is persistent (no expiration) |
| `default_ttl` only | Keys expire after the specified minutes |
| `default_ttl` + `refresh_on_read` | TTL resets on every read; idle data expires |
| `ttl_minutes=-1` on a key | Removes TTL; key becomes persistent |
