# LangGraph Redis

This repository contains Redis implementations for LangGraph, providing both Checkpoint Savers and Stores functionality.

## Overview

The project consists of two main components:
1. **Redis Checkpoint Savers**: Implementations for storing and managing checkpoints using Redis
2. **Redis Stores**: Redis-backed key-value stores with optional vector search capabilities

## Dependencies

The project requires the following main dependencies:
- `redis>=5.2.1`
- `redisvl>=0.3.7`
- `langgraph-checkpoint>=2.0.10`

## Installation

Install the library using pip:

```bash
pip install langgraph-checkpoint-redis
```

## Redis Checkpoint Savers

### Important Notes

> [!IMPORTANT]
> When using Redis checkpointers for the first time, make sure to call `.setup()` method on them to create required indices. See examples below.

### Standard Implementation

```python
from langgraph.checkpoint.redis import RedisSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

with RedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    # Call setup to initialize indices
    checkpointer.setup()
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
                "__start__": 1
            },
            "node": {
                "start:node": 2
            }
        },
        "pending_sends": [],
    }

    # Store checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # Retrieve checkpoint
    loaded_checkpoint = checkpointer.get(read_config)

    # List all checkpoints
    checkpoints = list(checkpointer.list(read_config))
```

### Async Implementation

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async def main():
    write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    read_config = {"configurable": {"thread_id": "1"}}

    async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        # Call setup to initialize indices
        await checkpointer.asetup()
        checkpoint = {
            "v": 1,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "channel_values": {
                "my_key": "meow",
                "node": "node"
            },
            "channel_versions": {
                "__start__": 2,
                "my_key": 3,
                "start:node": 3,
                "node": 3
            },
            "versions_seen": {
                "__input__": {},
                "__start__": {
                    "__start__": 1
                },
                "node": {
                    "start:node": 2
                }
            },
            "pending_sends": [],
        }

        # Store checkpoint
        await checkpointer.aput(write_config, checkpoint, {}, {})

        # Retrieve checkpoint
        loaded_checkpoint = await checkpointer.aget(read_config)

        # List all checkpoints
        checkpoints = [c async for c in checkpointer.alist(read_config)]

# Run the async main function
import asyncio
asyncio.run(main())
```

### Shallow Implementations

Shallow Redis checkpoint savers store only the latest checkpoint in Redis. These implementations are useful when retaining a complete checkpoint history is unnecessary.

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
# For async version: from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

with ShallowRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    checkpointer.setup()
    # ... rest of the implementation follows similar pattern
```

## Redis Stores

Redis Stores provide a persistent key-value store with optional vector search capabilities.

### Synchronous Implementation

```python
from langgraph.store.redis import RedisStore

# Basic usage
with RedisStore.from_conn_string("redis://localhost:6379") as store:
    store.setup()
    # Use the store...

# With vector search configuration
index_config = {
    "dims": 1536,  # Vector dimensions
    "distance_type": "cosine",  # Distance metric
    "fields": ["text"],  # Fields to index
}

with RedisStore.from_conn_string("redis://localhost:6379", index=index_config) as store:
    store.setup()
    # Use the store with vector search capabilities...
```

### Async Implementation

```python
from langgraph.store.redis.aio import AsyncRedisStore

async def main():
    async with AsyncRedisStore.from_conn_string("redis://localhost:6379") as store:
        await store.setup()
        # Use the store asynchronously...

asyncio.run(main())
```

## Implementation Details

### Indexing

The Redis implementation creates these main indices:

1. **Checkpoints Index**: Stores checkpoint metadata and versioning
2. **Channel Values Index**: Stores channel-specific data
3. **Writes Index**: Tracks pending writes and intermediate states

For Redis Stores with vector search:
1. **Store Index**: Main key-value store
2. **Vector Index**: Optional vector embeddings for similarity search

## Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/langchain-ai/langgraph
   cd langgraph
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

### Available Commands

The project includes several make commands for development:

- **Testing**:
  ```bash
  make test           # Run all tests
  make test_watch    # Run tests in watch mode
  ```

- **Linting and Formatting**:
  ```bash
  make lint          # Run all linters
  make lint_diff     # Lint only changed files
  make lint_package  # Lint only the package
  make lint_tests    # Lint only tests
  make format        # Format all files
  make format_diff   # Format only changed files
  ```

### Contribution Guidelines

1. Create a new branch for your changes
2. Write tests for new functionality
3. Ensure all tests pass: `make test`
4. Format your code: `make format`
5. Run linting checks: `make lint`
6. Submit a pull request with a clear description of your changes
7. Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages

## License

This project is licensed under the MIT License.