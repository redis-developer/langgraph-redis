# Migration Guide: 0.1.x to 0.2.0

This guide helps you migrate from langgraph-checkpoint-redis 0.1.x to 0.2.0, which includes breaking changes due to upgrades to LangGraph 1.0 and LangGraph Checkpoint 3.0.

## Table of Contents

- [Overview](#overview)
- [Breaking Changes](#breaking-changes)
- [Dependency Updates](#dependency-updates)
- [Migration Steps](#migration-steps)
- [Code Changes Required](#code-changes-required)
- [Data Migration](#data-migration)
- [Troubleshooting](#troubleshooting)

## Overview

Version 0.2.0 represents a major upgrade that brings compatibility with:

- **LangGraph 1.0.3** (from 0.4.9)
- **LangGraph Checkpoint 3.0** (from 2.x)
- **LangChain Core 1.0.5** (from 0.3.74)

These upgrades include breaking API changes that require code modifications.

## Breaking Changes

### 1. Python Version Requirement

**Changed:** Minimum Python version increased from **3.9 to 3.10**

**Reason:** Python 3.9 reached end-of-life in October 2025

**Action Required:**

```bash
# Ensure you're using Python 3.10 or higher
python --version  # Should show 3.10.x or higher
```

### 2. Interrupt API Changes (LangGraph 1.0)

**Changed:** The `Interrupt` class structure has been simplified from 4 fields to 2 fields.

**Removed fields:**

- `resumable` (bool) - Removed in LangGraph v0.6.0
- `ns` (str) - Removed in LangGraph v0.6.0
- `when` (str) - Removed in LangGraph v0.6.0
- `interrupt_id` (property) - Deprecated in favor of `id`

**Retained fields:**

- `value` (Any) - The value associated with the interrupt
- `id` (str) - Unique identifier for the interrupt

**Before (0.1.x):**

```python
from langgraph.types import Interrupt

# Creating an interrupt
interrupt = Interrupt(
    value={"user_input": "data"},
    resumable=True,
    ns="my_namespace",
    when="before"
)

# Accessing fields
if interrupt.resumable:
    process(interrupt.value)
```

**After (0.2.0):**

```python
from langgraph.types import Interrupt

# Creating an interrupt
interrupt = Interrupt(
    value={"user_input": "data"},
    id="unique-interrupt-id"
)

# Accessing fields
# The 'id' field identifies the interrupt
process(interrupt.value, interrupt.id)
```

**Migration Note:** If you were using `resumable`, `ns`, or `when` fields to control interrupt behavior, you'll need to include this information in the `value` dict instead.

### 3. Checkpoint Serialization API (Checkpoint 3.0)

**Changed:** Serializer signatures now use `bytes` instead of `str`

The `dumps_typed` and `loads_typed` methods now use `tuple[str, bytes]` instead of `tuple[str, str]`.

**Before (Checkpoint 2.x):**

```python
from langgraph.checkpoint.serde.base import SerializerProtocol

class CustomSerializer(SerializerProtocol):
    def dumps_typed(self, obj: Any) -> tuple[str, str]:
        # Returns (type_string, data_string)
        return ("json", json.dumps(obj))

    def loads_typed(self, data: tuple[str, str]) -> Any:
        type_str, data_str = data
        return json.loads(data_str)
```

**After (Checkpoint 3.0):**

```python
from langgraph.checkpoint.serde.base import SerializerProtocol

class CustomSerializer(SerializerProtocol):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        # Returns (type_string, data_bytes)
        return ("json", json.dumps(obj).encode())

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_str, data_bytes = data
        return json.loads(data_bytes.decode())
```

**Impact:** This change is handled internally by `JsonPlusRedisSerializer`. **No action required** unless you've implemented a custom serializer.

### 4. Blob Storage Format

**Changed:** All checkpoint blobs are now stored as base64-encoded strings in Redis JSON documents.

**Impact:** This is transparent to users - the library handles encoding/decoding automatically.

**Internal Change:**

```python
# 0.1.x: Blobs stored as JSON strings
{"channel": "messages", "data": "{...}"}

# 0.2.0: Blobs stored as base64-encoded bytes
{"channel": "messages", "data": "eyJ0eXBlIjoi..."}  # base64 string
```

**Note:** Existing checkpoints from 0.1.x will **not** be automatically migrated. See [Data Migration](#data-migration) section.

## Dependency Updates

```toml
# pyproject.toml changes

[tool.poetry.dependencies]
# Before (0.1.x)
python = ">=3.9,<3.14"
langgraph = ">=0.3.0"
langgraph-checkpoint = ">=2.0.21,<3.0.0"

# After (0.2.0)
python = ">=3.10,<3.14"
langgraph = ">=1.0.0"
langgraph-checkpoint = ">=3.0.0,<4.0.0"
redisvl = ">=0.11.0,<1.0.0"  # New: Security fix for CVE-2025-64439
```

### Updated Transitive Dependencies

- `langchain-core`: 0.3.74 → 1.0.5
- `langgraph-prebuilt`: 0.2.2 → 1.0.4
- `langgraph-sdk`: (via langgraph) updated to 0.2.x

## Migration Steps

### Step 1: Check Python Version

```bash
python --version
# Must be 3.10 or higher
```

If you're on Python 3.9, upgrade to Python 3.10+ before proceeding.

### Step 2: Backup Your Data

**Critical:** Create a backup of your Redis data before upgrading.

```bash
# Option 1: Redis SAVE command
redis-cli -h <host> -p <port> SAVE

# Option 2: Export specific keys (if you know the patterns)
redis-cli -h <host> -p <port> --scan --pattern "checkpoint:*" > checkpoint_keys.txt
```

### Step 3: Update Dependencies

```bash
# Update langgraph-checkpoint-redis
pip install --upgrade langgraph-checkpoint-redis==0.2.0

# Or with poetry
poetry add langgraph-checkpoint-redis@^0.2.0
poetry update
```

### Step 4: Update Your Code

Review and update any code that uses:

1. `Interrupt` objects (see [Interrupt API Changes](#2-interrupt-api-changes-langgraph-10))
2. Custom serializers (see [Checkpoint Serialization API](#3-checkpoint-serialization-api-checkpoint-30))

### Step 5: Test Migration

```python
# Test basic checkpoint operations
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()

    # Test saving a checkpoint
    from langgraph.checkpoint.base import Checkpoint
    config = {"configurable": {"thread_id": "test-migration"}}

    checkpoint = Checkpoint(
        v=1,
        id="test-checkpoint",
        ts="2025-01-01T00:00:00Z",
        channel_values={},
        channel_versions={},
        versions_seen={},
    )

    # This should work without errors
    metadata = {"source": "migration_test"}
    saver.put(config, checkpoint, metadata, {})

    # Verify retrieval
    retrieved = saver.get_tuple(config)
    assert retrieved is not None
    print("✓ Migration test passed!")
```

## Code Changes Required

### If You Use Interrupts

**Search for:** `Interrupt(` in your codebase

**Update pattern:**

```python
# Before
from langgraph.types import Interrupt
interrupt = Interrupt(
    value=data,
    resumable=True,
    ns="my_ns",
    when="before"
)

# After
from langgraph.types import Interrupt
import uuid

# Move metadata into the value if needed
interrupt = Interrupt(
    value={
        "data": data,
        # Optional: Include old metadata in value if needed
        "_metadata": {
            "resumable": True,
            "ns": "my_ns",
            "when": "before"
        }
    },
    id=str(uuid.uuid4())  # or your own ID generation
)
```

### If You Access Interrupt Fields

**Search for:** `.resumable`, `.ns`, `.when`, `.interrupt_id` in your codebase

**Update pattern:**

```python
# Before
if interrupt.resumable:
    handle_resumable(interrupt.value)

namespace = interrupt.ns
timing = interrupt.when

# After
# Check metadata in value instead
metadata = interrupt.value.get("_metadata", {})
if metadata.get("resumable"):
    handle_resumable(interrupt.value["data"])

namespace = metadata.get("ns")
timing = metadata.get("when")
```

### If You Have Custom Serializers

**Search for:** Classes that inherit from `SerializerProtocol` or override `dumps_typed`/`loads_typed`

**Update pattern:**

```python
from langgraph.checkpoint.serde.base import SerializerProtocol
from typing import Any

class MySerializer(SerializerProtocol):
    # Before: Returns tuple[str, str]
    # After: Returns tuple[str, bytes]

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        import json
        # Return bytes instead of str
        return ("json", json.dumps(obj).encode("utf-8"))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        import json
        type_str, data_bytes = data
        # Expect bytes instead of str
        return json.loads(data_bytes.decode("utf-8"))
```

## Data Migration

### Checkpoint Data Compatibility

**Important:** Checkpoints created with 0.1.x are **not directly compatible** with 0.2.0 due to:

1. Blob encoding changes (base64 format)
2. Serialization format changes (bytes vs strings)
3. Interrupt object structure changes

### Migration Options

#### Option 1: Fresh Start (Recommended for Development)

If you don't need to preserve checkpoint history:

```python
from langgraph.checkpoint.redis import RedisSaver
import redis

# Clear old checkpoints
redis_client = redis.from_url("redis://localhost:6379")
keys = redis_client.keys("checkpoint:*")
if keys:
    redis_client.delete(*keys)

# Recreate indices with new version
with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
```

#### Option 2: Parallel Deployment (Recommended for Production)

Run both versions in parallel during transition:

```python
# Deploy 0.2.0 with a different key prefix or database
REDIS_URL_NEW = "redis://localhost:6379/1"  # Different database
REDIS_URL_OLD = "redis://localhost:6379/0"  # Old database

# New code uses 0.2.0
from langgraph.checkpoint.redis import RedisSaver
saver_new = RedisSaver.from_conn_string(REDIS_URL_NEW)

# Gradually migrate traffic to new version
# Old traffic continues on 0.1.x until fully migrated
```

#### Option 3: Custom Migration Script

If you must migrate existing checkpoints:

```python
from langgraph.checkpoint.redis import RedisSaver
import redis
import json
import base64

def migrate_checkpoints():
    """
    WARNING: This is a template. Test thoroughly before using in production.
    """
    old_saver = RedisSaver.from_conn_string("redis://localhost:6379")

    # Get all checkpoint keys
    redis_client = redis.from_url("redis://localhost:6379")
    checkpoint_keys = redis_client.keys("checkpoint:*")

    for key in checkpoint_keys:
        try:
            # Read old checkpoint
            old_data = redis_client.json().get(key)

            # Transform blobs to base64 if needed
            if "channel_values" in old_data:
                for channel, value in old_data["channel_values"].items():
                    if isinstance(value, str):
                        # Encode to base64
                        old_data["channel_values"][channel] = base64.b64encode(
                            value.encode()
                        ).decode()

            # Update Interrupt objects if present
            # This requires inspecting your specific checkpoint structure

            # Write back
            redis_client.json().set(key, "$", old_data)
            print(f"✓ Migrated {key}")

        except Exception as e:
            print(f"✗ Failed to migrate {key}: {e}")
            raise

# IMPORTANT: Test on a copy of your data first!
# migrate_checkpoints()
```

**Warning:** Custom migration scripts should be thoroughly tested on copies of your data. Consider the parallel deployment approach instead.

### RedisStore Data

`RedisStore` data (cross-thread persistence) should remain compatible. The store uses a different indexing scheme and doesn't rely on the checkpoint serialization format.

## Troubleshooting

### Issue: ImportError for Interrupt

```python
ImportError: cannot import name 'Interrupt' from 'langgraph.types'
```

**Solution:** Ensure langgraph is upgraded to 1.0+

```bash
pip install --upgrade langgraph>=1.0.0
```

### Issue: AttributeError: 'Interrupt' object has no attribute 'resumable'

```python
AttributeError: 'Interrupt' object has no attribute 'resumable'
```

**Solution:** Update code to use only `value` and `id` fields (see [Interrupt API Changes](#2-interrupt-api-changes-langgraph-10))

### Issue: orjson.JSONDecodeError when loading checkpoints

```python
orjson.JSONDecodeError: unexpected character: ...
```

**Solution:** This indicates you're trying to load a checkpoint created with 0.1.x. See [Data Migration](#data-migration) for options.

### Issue: Tests failing after upgrade

**Common causes:**

1. Test mocks assume old Interrupt structure
2. Fixtures use old checkpoint format
3. Test assertions check removed fields

**Solution:** Update tests to match new APIs:

```python
# Before
def test_interrupt():
    interrupt = Interrupt(value={"test": "data"}, resumable=True)
    assert interrupt.resumable is True

# After
def test_interrupt():
    interrupt = Interrupt(value={"test": "data"}, id="test-id")
    assert interrupt.id == "test-id"
```

### Issue: Existing checkpoints not loading

**Symptoms:** `get_tuple()` returns None for previously saved checkpoints

**Cause:** Blob encoding format changed from JSON strings to base64 bytes

**Solutions:**

1. Fresh start: Clear old data and recreate (see Option 1 in [Data Migration](#data-migration))
2. Parallel deployment: Run 0.2.0 in parallel with 0.1.x (see Option 2)
3. Custom migration: Write a migration script (see Option 3)

### Issue: SecurityError about redisvl version

```python
WARNING: redisvl version <0.11.0 has known vulnerability CVE-2025-64439
```

**Solution:** Ensure redisvl>=0.11.0 is installed

```bash
pip install --upgrade redisvl>=0.11.0
```

## Configuration Changes

### TTL Configuration (Unchanged)

TTL configuration remains the same:

```python
from langgraph.checkpoint.redis import RedisSaver

saver = RedisSaver(
    redis_url="redis://localhost:6379",
    ttl={
        "default_ttl": 60,  # Still in MINUTES
        "refresh_on_read": True
    }
)
```

### Shallow Checkpointing (Unchanged)

ShallowRedisSaver usage remains unchanged:

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

saver = ShallowRedisSaver(
    redis_url="redis://localhost:6379",
    key_cache_max_size=2000,
    channel_cache_max_size=200
)
```

## Testing Your Migration

Create a comprehensive test suite to verify the migration:

```python
import pytest
from langgraph.checkpoint.redis import RedisSaver
from langgraph.types import Interrupt
import uuid

def test_checkpoint_roundtrip():
    """Test basic checkpoint save/load"""
    with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
        saver.setup()

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Create and save checkpoint
        from langgraph.checkpoint.base import Checkpoint
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid.uuid4()),
            ts="2025-01-01T00:00:00Z",
            channel_values={"messages": ["test"]},
            channel_versions={"messages": 1},
            versions_seen={},
        )

        saver.put(config, checkpoint, {"test": True}, {})

        # Retrieve and verify
        retrieved = saver.get_tuple(config)
        assert retrieved is not None
        assert retrieved.checkpoint.channel_values == checkpoint.channel_values

def test_interrupt_serialization():
    """Test Interrupt object handling"""
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    serializer = JsonPlusRedisSerializer()

    # Create interrupt with new API
    interrupt = Interrupt(
        value={"test": "data"},
        id="test-interrupt-id"
    )

    # Serialize
    type_str, data_bytes = serializer.dumps_typed(interrupt)
    assert type_str == "json"
    assert isinstance(data_bytes, bytes)

    # Deserialize
    result = serializer.loads_typed((type_str, data_bytes))
    assert isinstance(result, Interrupt)
    assert result.value == {"test": "data"}
    assert result.id == "test-interrupt-id"

def test_interrupt_in_checkpoint():
    """Test interrupts within checkpoint workflow"""
    from langgraph.types import interrupt

    # The interrupt() function should work as expected
    # This would be tested in an actual graph context
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Getting Help

If you encounter issues during migration:

1. **Check the release notes:** Review the [CHANGELOG](CHANGELOG.md) for detailed changes
2. **GitHub Issues:** Search [langgraph-redis issues](https://github.com/redis-developer/langgraph-redis/issues) for similar problems
3. **LangGraph Documentation:** Review [LangGraph 1.0 docs](https://langchain-ai.github.io/langgraph/) for upstream changes
4. **Open an issue:** If you find a bug, [open a new issue](https://github.com/redis-developer/langgraph-redis/issues/new) with:
   - Your Python version
   - Full error traceback
   - Minimal code to reproduce the issue
   - Version information: `pip list | grep langgraph`

## Reference

### Version Information

```bash
# Check installed versions
pip show langgraph-checkpoint-redis langgraph langgraph-checkpoint langchain-core

# Expected versions for 0.2.0:
# langgraph-checkpoint-redis: 0.2.0
# langgraph: >=1.0.0
# langgraph-checkpoint: >=3.0.0
# langchain-core: >=1.0.0
```

### Related Documentation

- [LangGraph 1.0 Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Core 1.0 Release](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [Checkpoint 3.0 Changes](https://github.com/langchain-ai/langgraph/releases)
- [Migration from 0.1.0](MIGRATION_0.1.0.md) (previous migration guide)

---

**Last Updated:** January 2025
**Applies to:** langgraph-checkpoint-redis 0.2.0
