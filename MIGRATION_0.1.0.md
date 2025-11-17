# Migration Guide

This document provides guidance for migrating between different versions of langgraph-checkpoint-redis.

## Version Compatibility

This library is currently at version 0.1.0. As the library evolves, the following areas may change between versions:

- Redis key naming patterns
- JSON data structure within Redis
- Index schemas for RediSearch
- API interfaces

### Data Migration

**Important**: When upgrading between major versions, checkpoint data may not be directly compatible. If you need to preserve existing checkpoint data:

1. **Export existing data** before upgrading using the old version
2. **Upgrade the library** to the new version
3. **Re-import or recreate** your checkpoints using the new version

### Key Structure

The library uses structured Redis key patterns:

**Standard RedisSaver (full history):**

```txt
checkpoint:{thread_id}:{checkpoint_ns}:{checkpoint_id}
checkpoint_blob:{thread_id}:{checkpoint_ns}:{channel}:{version}
checkpoint_write:{thread_id}:{checkpoint_ns}:{checkpoint_id}:{task_id}:{idx}
```

**ShallowRedisSaver (latest only):**

```txt
checkpoint:{thread_id}:{checkpoint_ns}  # Single checkpoint per thread/namespace
```

### Shallow vs Full Checkpointing

The library now supports two checkpoint storage modes:

- **ShallowRedisSaver**: Stores only the most recent checkpoint per thread/namespace
- **RedisSaver**: Stores full checkpoint history

When migrating, consider which mode best fits your use case.

## Configuration Changes

### Cache Configuration (New in latest version)

The ShallowRedisSaver now supports configurable cache sizes:

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

# Configure cache sizes
saver = ShallowRedisSaver(
    redis_url="redis://localhost:6379",
    key_cache_max_size=2000,  # Default: 1000
    channel_cache_max_size=200  # Default: 100
)
```

### TTL Configuration

TTL (Time To Live) configuration:

```python
from langgraph.checkpoint.redis import RedisSaver

saver = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl={
        "default_ttl": 60,  # Time in MINUTES (60 minutes = 1 hour)
        "refresh_on_read": True  # Refresh TTL when checkpoint is read
    }
)
```

**Important:** The `default_ttl` value is specified in **minutes**, not seconds.

## Environment Variables

### New Environment Variables

- `LANGGRAPH_REDIS_PYPROJECT_SEARCH_DEPTH`: Controls how many directory levels to search for pyproject.toml when determining version in development mode (default: 5)

## Redis Module Requirements

### Redis 8.0+

- Includes RedisJSON and RediSearch modules by default
- Recommended for production use

### Redis < 8.0

- Requires Redis Stack or manual installation of:
  - RedisJSON module
  - RediSearch module

## Best Practices for Migration

1. **Test in Development**: Always test the migration process in a development environment first
2. **Backup Data**: Create backups of your Redis data before migration
3. **Gradual Migration**: If possible, run old and new versions in parallel during transition
4. **Monitor Performance**: The new LRU cache implementation may have different performance characteristics

## Getting Help

If you encounter issues during migration:

1. Check the [GitHub Issues](https://github.com/redis-developer/langgraph-redis/issues) for known problems
2. Review the [Release Notes](https://github.com/redis-developer/langgraph-redis/releases) for version changes
3. Open a new issue with details about your migration scenario
