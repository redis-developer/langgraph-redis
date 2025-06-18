# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Dependencies

```bash
poetry install --all-extras  # Install all dependencies with poetry (from README)
make redis-start             # Start Redis Stack container (includes RedisJSON and RediSearch)
make redis-stop              # Stop Redis container
```

### Testing

```bash
make test                    # Run tests with verbose output
make test-all               # Run all tests including API tests
pytest tests/test_specific.py  # Run specific test file
pytest tests/test_specific.py::test_function  # Run specific test
pytest --run-api-tests      # Include API integration tests
```

### Code Quality

```bash
make format          # Format code with black and isort
make lint            # Run formatting, type checking, and other linters
make check-types     # Run mypy type checking
make check           # Run both linting and tests
```

### Development

```bash
make clean           # Remove cache and build artifacts
```

## Architecture Overview

### Core Components

**Checkpoint Savers** (`langgraph/checkpoint/redis/`):

- `base.py`: `BaseRedisSaver` - Abstract base class with shared Redis operations, schemas, and TTL management
- `__init__.py`: `RedisSaver` - Standard sync implementation
- `aio.py`: `AsyncRedisSaver` - Async implementation
- `shallow.py` / `ashallow.py`: Shallow variants that store only latest checkpoint

**Stores** (`langgraph/store/redis/`):

- `base.py`: `BaseRedisStore` - Abstract base with Redis operations, vector search, and TTL support
- `__init__.py`: `RedisStore` - Sync store with key-value and vector search
- `aio.py`: `AsyncRedisStore` - Async store implementation

### Key Architecture Patterns

**Dual Implementation Strategy**: Each major component has both sync and async variants that share common base classes. The base classes (`BaseRedisSaver`, `BaseRedisStore`) contain the bulk of the business logic, while concrete implementations handle Redis client management and specific I/O patterns.

**Redis Module Dependencies**: The library requires RedisJSON and RediSearch modules. Redis 8.0+ includes these by default; earlier versions need Redis Stack. All operations use structured JSON storage with search indices for efficient querying.

**Schema-Driven Indexing**: Both checkpoints and stores use predefined schemas (`SCHEMAS` constants) that define Redis Search indices. Checkpoint indices track thread/namespace/version hierarchies; store indices support both key-value lookup and optional vector similarity search.

**TTL Integration**: Native Redis TTL support is integrated throughout, with configurable defaults and refresh-on-read capabilities. TTL applies to all related keys (main document, vectors, writes) atomically.

**Cluster Support**: Full Redis Cluster support with automatic detection and cluster-aware operations (individual key operations vs. pipelined operations).

**Type System**: Heavy use of generics (`BaseRedisSaver[RedisClientType, IndexType]`) to maintain type safety across sync/async variants while sharing implementation code.

### Redis Key Patterns

- Checkpoints: `checkpoint:{thread_id}:{namespace}:{checkpoint_id}`
- Checkpoint blobs: `checkpoint_blob:{thread_id}:{namespace}:{channel}:{version}`
- Checkpoint writes: `checkpoint_write:{thread_id}:{namespace}:{checkpoint_id}:{task_id}`
- Store items: `store:{uuid}`
- Store vectors: `store_vectors:{uuid}`

### Testing Strategy

Tests are organized by functionality:

- `test_sync.py` / `test_async.py`: Core checkpoint functionality
- `test_store.py` / `test_async_store.py`: Store operations
- `test_cluster_mode.py`: Redis Cluster specific tests
- `test_*_ttl.py`: TTL functionality
- `test_key_parsing.py`: Key generation and parsing logic
- `test_semantic_search_*.py`: Vector search capabilities

### Important Dependencies

- Requires Redis with RedisJSON and RediSearch modules
- Uses `redisvl` for vector operations and search index management
- Uses `python-ulid` for unique document IDs
- Integrates with LangGraph's checkpoint and store base classes
