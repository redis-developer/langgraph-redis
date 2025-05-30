# LangGraph Redis

This repository contains Redis implementations for LangGraph, providing both Checkpoint Savers and Stores functionality.

## Overview

The project consists of two main components:

1. **Redis Checkpoint Savers**: Implementations for storing and managing checkpoints using Redis
2. **Redis Stores**: Redis-backed key-value stores with optional vector search capabilities

## Dependencies

### Python Dependencies

The project requires the following main Python dependencies:

- `redis>=5.2.1`
- `redisvl>=0.5.1`
- `langgraph-checkpoint>=2.0.24`

### Redis Modules Requirements

**IMPORTANT:** This library requires Redis with the following modules:

- **RedisJSON** - For storing and manipulating JSON data
- **RediSearch** - For search and indexing capabilities

#### Redis 8.0+

If you're using Redis 8.0 or higher, both RedisJSON and RediSearch modules are included by default as part of the core Redis distribution. No additional installation is required.

#### Redis < 8.0

If you're using a Redis version lower than 8.0, you'll need to ensure these modules are installed:

- Use [Redis Stack](https://redis.io/docs/stack/), which bundles Redis with these modules
- Or install the modules separately in your Redis instance

Failure to have these modules available will result in errors during index creation and checkpoint operations.

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

## Redis Checkpoint TTL Support

Both Redis checkpoint savers and stores support Time-To-Live (TTL) functionality for automatic key expiration:

```python
# Configure TTL for checkpoint savers
ttl_config = {
    "default_ttl": 60,     # Default TTL in minutes
    "refresh_on_read": True,  # Refresh TTL when checkpoint is read
}

# Use with any checkpoint saver implementation
with RedisSaver.from_conn_string("redis://localhost:6379", ttl=ttl_config) as checkpointer:
    checkpointer.setup()
    # Use the checkpointer...
```

This makes it easy to manage storage and ensure ephemeral data is automatically cleaned up.

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

# With TTL configuration
ttl_config = {
    "default_ttl": 60,     # Default TTL in minutes
    "refresh_on_read": True,  # Refresh TTL when store entries are read
}

with RedisStore.from_conn_string(
    "redis://localhost:6379", 
    index=index_config,
    ttl=ttl_config
) as store:
    store.setup()
    # Use the store with vector search and TTL capabilities...
```

### Async Implementation

```python
from langgraph.store.redis.aio import AsyncRedisStore

async def main():
    # TTL also works with async implementations
    ttl_config = {
        "default_ttl": 60,     # Default TTL in minutes
        "refresh_on_read": True,  # Refresh TTL when store entries are read
    }
    
    async with AsyncRedisStore.from_conn_string(
        "redis://localhost:6379", 
        ttl=ttl_config
    ) as store:
        await store.setup()
        # Use the store asynchronously...

asyncio.run(main())
```

## Examples

The `examples` directory contains Jupyter notebooks demonstrating the usage of Redis with LangGraph:

- `persistence_redis.ipynb`: Demonstrates the usage of Redis checkpoint savers with LangGraph
- `create-react-agent-memory.ipynb`: Shows how to create an agent with persistent memory using Redis
- `cross-thread-persistence.ipynb`: Demonstrates cross-thread persistence capabilities
- `persistence-functional.ipynb`: Shows functional persistence patterns with Redis

### Running Example Notebooks

To run the example notebooks with Docker:

1. Navigate to the examples directory:

   ```bash
   cd examples
   ```

2. Start the Docker containers:

   ```bash
   docker compose up
   ```

3. Open the URL shown in the console (typically <http://127.0.0.1:8888/tree>) in your browser to access Jupyter.

4. When finished, stop the containers:

   ```bash
   docker compose down
   ```

## Implementation Details

### Redis Module Usage

This implementation relies on specific Redis modules:

- **RedisJSON**: Used for storing structured JSON data as native Redis objects
- **RediSearch**: Used for creating and querying indices on JSON data

### Indexing

The Redis implementation creates these main indices using RediSearch:

1. **Checkpoints Index**: Stores checkpoint metadata and versioning
2. **Channel Values Index**: Stores channel-specific data
3. **Writes Index**: Tracks pending writes and intermediate states

For Redis Stores with vector search:

1. **Store Index**: Main key-value store
2. **Vector Index**: Optional vector embeddings for similarity search

### TTL Implementation

Both Redis checkpoint savers and stores leverage Redis's native key expiration:

- **Native Redis TTL**: Uses Redis's built-in `EXPIRE` command
- **Automatic Cleanup**: Redis automatically removes expired keys
- **Configurable Default TTL**: Set a default TTL for all keys in minutes
- **TTL Refresh on Read**: Optionally refresh TTL when keys are accessed
- **Applied to All Related Keys**: TTL is applied to all related keys (checkpoint, blobs, writes)

## Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/redis-developer/langgraph-redis
   cd langgraph-redis
   ```

2. Install dependencies:

   ```bash
   `poetry install --all-extras`
   ```

### Available Commands

The project includes several make commands for development:

- **Testing**:

  ```bash
  make test           # Run all tests
  make test-all       # Run all tests including API tests
  ```

- **Linting and Formatting**:

  ```bash
  make format        # Format all files with Black and isort
  make lint          # Run formatting, type checking, and other linters
  make check-types   # Run mypy type checking
  ```

- **Redis for Development/Testing**:

  ```bash
  make redis-start   # Start Redis Stack in Docker (includes RedisJSON and RediSearch modules)
  make redis-stop    # Stop Redis container
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
