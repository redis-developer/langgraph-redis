---
title: Migration Guide
---

# Migration Guide

This guide summarizes the key changes between versions of
`langgraph-checkpoint-redis` and provides guidance for upgrading. For full
details, refer to the migration documents in the repository root:

- [`MIGRATION_0.1.0.md`](https://github.com/redis-developer/langgraph-redis/blob/main/MIGRATION_0.1.0.md) -- initial migration guide
- [`MIGRATION_0.2.0.md`](https://github.com/redis-developer/langgraph-redis/blob/main/MIGRATION_0.2.0.md) -- migration from 0.1.x to 0.2.0

## Migrating to 0.2.0 from 0.1.x

Version 0.2.0 is a major upgrade that brings compatibility with LangGraph 1.0,
LangGraph Checkpoint 3.0, and LangChain Core 1.0. Several breaking changes
require code modifications.

### Python Version Requirement

The minimum Python version increased from **3.9 to 3.10**. Verify your
environment before upgrading:

```bash
python --version
# Must be 3.10 or higher
```

### Dependency Updates

Update your dependencies:

```bash
pip install --upgrade langgraph-checkpoint-redis>=0.2.0
```

The key version changes:

| Package | 0.1.x | 0.2.0+ |
|---------|-------|--------|
| `langgraph` | `>=0.3.0` | `>=1.0.0` |
| `langgraph-checkpoint` | `>=2.0.21,<3.0.0` | `>=3.0.0,<4.0.0` |
| `langchain-core` | `0.3.x` | `>=1.0.0` |
| `redisvl` | `>=0.4.0` | `>=0.11.0` |
| `python` | `>=3.9` | `>=3.10` |

### Interrupt API Changes

The `Interrupt` class was simplified from four fields to two:

**Removed fields:** `resumable`, `ns`, `when`

**Retained fields:** `value`, `id`

Before (0.1.x):

```python
from langgraph.types import Interrupt

interrupt = Interrupt(
    value={"user_input": "data"},
    resumable=True,
    ns="my_namespace",
    when="before",
)
```

After (0.2.0):

```python
from langgraph.types import Interrupt

interrupt = Interrupt(
    value={"user_input": "data"},
    id="unique-interrupt-id",
)
```

If you relied on the removed fields, move that information into the `value`
dictionary.

### Serialization Changes

The checkpoint serializer now uses `bytes` instead of `str` for the data
component:

- `dumps_typed` returns `tuple[str, bytes]` instead of `tuple[str, str]`
- `loads_typed` accepts `tuple[str, bytes]` instead of `tuple[str, str]`

This change is handled internally by the library's `JsonPlusRedisSerializer`.
**No action is required** unless you implemented a custom serializer.

### Blob Storage Format

Checkpoint blobs are now stored as base64-encoded strings. This is transparent
to users but means that **checkpoints created with 0.1.x are not directly
compatible with 0.2.0**.

### Data Migration Options

Existing checkpoints from 0.1.x cannot be loaded by 0.2.0 without migration.
Choose one of these approaches:

**Fresh start** (recommended for development):

```python
import redis

client = redis.from_url("redis://localhost:6379")
# Clear old checkpoint data
for key in client.scan_iter("checkpoint:*"):
    client.delete(key)
for key in client.scan_iter("checkpoint_write:*"):
    client.delete(key)

# Recreate indices with the new version
from langgraph.checkpoint.redis import RedisSaver
with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
```

**Parallel deployment** (recommended for production):

Run both versions side by side using different Redis databases:

```python
REDIS_URL_OLD = "redis://localhost:6379/0"  # 0.1.x data
REDIS_URL_NEW = "redis://localhost:6379/1"  # 0.2.0 data
```

Gradually migrate traffic to the new version.

:::{note}
`RedisStore` data is not affected by the checkpoint serialization changes
and should remain compatible across versions.
:::

## Migrating to 0.1.0

The 0.1.0 release established the foundational patterns. Key areas that may
differ from pre-release versions:

### Redis Key Patterns

Standardized key formats were introduced:

```text
checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}
checkpoint_blob:{thread_id}:{checkpoint_ns}:{channel}:{version}
checkpoint_write:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}
```

### Shallow vs. Full Checkpointing

Two checkpointing modes are available:

- **`RedisSaver`** -- stores full checkpoint history per thread
- **`ShallowRedisSaver`** -- stores only the latest checkpoint per
  thread/namespace, using constant storage

### TTL Configuration

TTL was introduced with values specified in **minutes**:

```python
saver = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl={
        "default_ttl": 60,       # 60 minutes
        "refresh_on_read": True,
    },
)
```

### Cache Configuration for Shallow Savers

`ShallowRedisSaver` includes configurable LRU cache sizes:

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

saver = ShallowRedisSaver(
    redis_url="redis://localhost:6379",
    key_cache_max_size=2000,
    channel_cache_max_size=200,
)
```

## General Upgrade Steps

When upgrading between any versions:

1. **Check the release notes** on the
   [GitHub releases page](https://github.com/redis-developer/langgraph-redis/releases).

2. **Back up your Redis data** before upgrading:
   ```bash
   redis-cli -h <host> -p <port> SAVE
   ```

3. **Test in a development environment** before deploying to production.

4. **Update your dependencies** and run your test suite:
   ```bash
   pip install --upgrade langgraph-checkpoint-redis
   pytest
   ```

5. **Monitor for errors** after deployment, particularly around checkpoint
   loading and serialization.

## Troubleshooting

### ImportError for Interrupt

```text
ImportError: cannot import name 'Interrupt' from 'langgraph.types'
```

Ensure `langgraph>=1.0.0` is installed:

```bash
pip install --upgrade langgraph>=1.0.0
```

### AttributeError on Interrupt Fields

```text
AttributeError: 'Interrupt' object has no attribute 'resumable'
```

Update code to use only the `value` and `id` fields.

### JSONDecodeError Loading Checkpoints

```text
orjson.JSONDecodeError: unexpected character
```

This indicates 0.1.x checkpoint data being read by 0.2.0 code. See the
data migration options above.

### Checking Installed Versions

```bash
pip show langgraph-checkpoint-redis langgraph langgraph-checkpoint langchain-core
```

## Next Steps

- {doc}`installation` -- verify your setup after upgrading
- {doc}`getting_started` -- test basic operations after migration
- {doc}`checkpointers` -- review current API
