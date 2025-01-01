# LangGraph Checkpoint Redis

This README provides documentation for the LangGraph CheckpointSaver Redis implementation. This integration leverages Redis as a persistence layer for LangGraph, enabling fast and reliable storage and retrieval of checkpoint data.

## Overview

LangGraph Checkpointers are a vital component in managing graph states, ensuring resiliency and offering advanced features like human-in-the-loop processing and session-based memory. The Redis implementation provides both synchronous and asynchronous interfaces for checkpoint saving, utilizing Redis JSON and RediSearch for efficient storage and querying.

## Features

- **High Performance**: Optimized for speed and scalability using Redis.
- **Flexible Storage**: Supports JSON storage for checkpoint data.
- **Advanced Querying**: Uses RediSearch to enable complex queries over checkpoint metadata.
- **Asynchronous Support**: Async and sync implementations available to suit diverse use cases.
- **Easy Integration**: Simple API design that conforms to the LangGraph Checkpointer interface.

## Installation

Install the library using pip:

```bash
pip install langgraph-checkpoint-redis
```

This will automatically install all required dependencies as defined in the `pyproject.toml` file.

## Usage

### Synchronous Example

```python
from langgraph.checkpoint.redis import RedisSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

# Initialize RedisSaver
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

### Asynchronous Example

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async def main():
    write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    read_config = {"configurable": {"thread_id": "1"}}

    # Initialize AsyncRedisSaver
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

#### Synchronous Example

```python
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

# Initialize ShallowRedisSaver
with ShallowRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
    # Call setup to initialize indices
    checkpointer.setup()
    checkpoint = {
        "v": 1,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "metadata": {"source": "input", "step": 1}
    }

    # Store the latest checkpoint
    checkpointer.put(write_config, checkpoint, {}, {})

    # Retrieve the latest checkpoint
    loaded_checkpoint = checkpointer.get_tuple(read_config)
```

#### Asynchronous Example

```python
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver

async def main():
    write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    read_config = {"configurable": {"thread_id": "1"}}

    # Initialize AsyncShallowRedisSaver
    async with AsyncShallowRedisSaver.from_conn_string("redis://localhost:6379") as checkpointer:
        # Call setup to initialize indices
        await checkpointer.asetup()
        checkpoint = {
            "v": 1,
            "ts": "2024-07-31T20:14:19.804150+00:00",
            "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            "metadata": {"source": "input", "step": 1}
        }

        # Store the latest checkpoint
        await checkpointer.aput(write_config, checkpoint, {}, {})

        # Retrieve the latest checkpoint
        loaded_checkpoint = await checkpointer.aget_tuple(read_config)

# Run the async main function
import asyncio
asyncio.run(main())
```

## Implementation Details

### Indexing

This implementation creates three main indices in Redis using RediSearch:

1. **Checkpoints Index**: Stores checkpoint metadata and versioning.
2. **Channel Values Index**: Stores the channel-specific data for each checkpoint.
3. **Writes Index**: Tracks pending writes and intermediate states.

These indices enable efficient querying and retrieval operations for various use cases.

### Schema

Below is the schema used for the checkpoint indices:

- **Checkpoints Index**:
  - `thread_id`: Thread identifier (Tag)
  - `checkpoint_ns`: Namespace (Tag)
  - `checkpoint_id`: Checkpoint ID (Tag)
  - `source`: Metadata source (Tag)
  - `step`: Step number (Numeric)
  - `score`: Checkpoint score (Numeric)

- **Channel Values Index**:
  - `thread_id`: Thread identifier (Tag)
  - `checkpoint_ns`: Namespace (Tag)
  - `channel`: Channel name (Tag)
  - `version`: Version (Tag)
  - `type`: Data type (Tag)
  - `blob`: Data content (Text)

- **Writes Index**:
  - `thread_id`: Thread identifier (Tag)
  - `checkpoint_ns`: Namespace (Tag)
  - `checkpoint_id`: Checkpoint ID (Tag)
  - `task_id`: Task ID (Tag)
  - `channel`: Channel name (Tag)

## Testing

### Unit Tests

The implementation includes an extensive suite of unit tests covering both synchronous and asynchronous functionality. See the `tests` directory for detailed test cases.

### Running Tests

Use `pytest` to run all tests:

```bash
pytest tests/
```

For asynchronous tests, ensure `pytest-asyncio` is installed.

## Contributing

We welcome contributions to improve this library. Here’s how you can contribute:

1. **Report Issues**: Found a bug or have a suggestion? Please open an issue on our [GitHub repository](https://www.github.com/langchain-ai/langgraph).
2. **Submit Pull Requests**: If you want to contribute code, fork the repository, create a new branch, make your changes, and submit a pull request.
3. **Improve Documentation**: If you find any discrepancies or omissions in the documentation, feel free to contribute fixes or enhancements.

### Development Setup

To set up the development environment:

1. Clone the repository:

   ```bash
   git clone https://github.com/langchain-ai/langgraph
   cd langgraph
   ```

2. Install dependencies:

   ```bash
   poetry install
   ```

3. Format and lint the code before submitting your changes:

   ```bash
   make format
   make lint
   ```

4. Craft commit messages following [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) to maintain a clear and structured history.

5. Run tests to ensure everything is set up correctly:

   ```bash
   poetry run pytest
   ```

## License

This project is licensed under the MIT License.
