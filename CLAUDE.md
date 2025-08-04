# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to Write Tests for Coverage

When improving test coverage, follow these principles:

1. **Focus on Integration Tests**: Write tests that use real Redis instances and test actual usage scenarios. Unit tests
   are secondary to integration tests.

2. **Test What Code SHOULD Do**: Don't write tests that mirror what the code currently does. Test against the expected
   behavior and requirements.

3. **Use Meaningful Test Names**: Test names should describe the behavior being tested, not generic names like "
   test_function_x".

4. **Research Before Writing**: Find and understand existing tests for the feature/area before adding new tests.

5. **Test Error Paths and Edge Cases**: Focus on uncovered error handling, boundary conditions, and edge cases.

6. **Run Tests Incrementally**: Run `make test-all` after every 5 tests to ensure no regressions.

7. **Avoid "Ugly Mirror" Testing**: Don't create tests that simply verify the current implementation. Test the contract
   and expected behavior.

Example of a good integration test for error handling:

```python
def test_malformed_base64_blob_handling(redis_url: str) -> None:
    """Test handling of malformed base64 data in blob decoding."""
    with _saver(redis_url) as saver:
        # Set up real scenario
        # Test error condition
        # Verify graceful handling
```

## CRITICAL: Always Use TestContainers for Redis

**NEVER use Docker directly or manually start Redis containers!** All tests, benchmarks, and profiling scripts MUST use
TestContainers. The library handles container lifecycle automatically.

```python
from testcontainers.redis import RedisContainer

# Use redis:8 (has all required modules) or redis/redis-stack-server:latest
redis_container = RedisContainer("redis:8")
redis_container.start()
try:
    redis_url = f"redis://{redis_container.get_container_host_ip()}:{redis_container.get_exposed_port(6379)}"
    # Use redis_url...
finally:
    redis_container.stop()
```

## Development Commands

### Setup and Dependencies

```bash
poetry install --all-extras  # Install all dependencies with poetry (from README)
```

### Testing

```bash
make test-all               # PREFERRED: Run all tests including API tests when evaluating changes
make test                   # Run tests with verbose output
make test-coverage          # Run tests with coverage
make coverage-report        # Show coverage report in terminal
make coverage-html          # Generate HTML coverage report
pytest tests/test_specific.py  # Run specific test file
pytest tests/test_specific.py::test_function  # Run specific test
pytest --run-api-tests      # Include API integration tests
```

**Important**: Always use `make test-all` when evaluating changes to ensure all tests pass, including API integration
tests.

Note: Tests automatically use TestContainers for Redis - do not manually start Redis containers.

### Code Quality

```bash
make format          # Format code with black and isort
make lint            # Run formatting, type checking, and other linters
make check-types     # Run mypy type checking
make check           # Run both linting and tests
make find-dead-code  # Find unused code with vulture
poetry run check-format      # Check formatting without modifying
poetry run check-sort-imports # Check import sorting
poetry run check-lint        # Run all linting checks
```

### Development

```bash
make clean           # Remove cache and build artifacts
```

## Code Style Guidelines

- Use Black for formatting with target versions py39-py313
- Sort imports with isort (black profile)
- Strict typing required (disallow_untyped_defs=True)
- Follow PEP 8 naming conventions (snake_case for functions/variables)
- Type annotations required for all function parameters and return values
- Explicit error handling with descriptive error messages
- Test all functionality with both sync and async variants
- Maintain test coverage with pytest
- Use contextlib for resource management
- Document public APIs with docstrings

## Architecture Overview

### Core Components

**Checkpoint Savers** (`langgraph/checkpoint/redis/`):

- `base.py`: `BaseRedisSaver` - Abstract base class with shared Redis operations, schemas, and TTL management
- `__init__.py`: `RedisSaver` - Standard sync implementation
- `aio.py`: `AsyncRedisSaver` - Async implementation
- `shallow.py` / `ashallow.py`: Shallow variants that store only latest checkpoint
- `key_registry.py`: Checkpoint key registry using sorted sets for efficient write tracking
- `scan_utils.py`: Utilities for efficient key scanning and pattern matching

**Stores** (`langgraph/store/redis/`):

- `base.py`: `BaseRedisStore` - Abstract base with Redis operations, vector search, and TTL support
- `__init__.py`: `RedisStore` - Sync store with key-value and vector search
- `aio.py`: `AsyncRedisStore` - Async store implementation

### Key Architecture Patterns

**Dual Implementation Strategy**: Each major component has both sync and async variants that share common base classes.
The base classes (`BaseRedisSaver`, `BaseRedisStore`) contain the bulk of the business logic, while concrete
implementations handle Redis client management and specific I/O patterns.

**Redis Module Dependencies**: The library requires RedisJSON and RediSearch modules. Redis 8.0+ includes these by
default; earlier versions need Redis Stack. All operations use structured JSON storage with search indices for efficient
querying.

**Schema-Driven Indexing**: Both checkpoints and stores use predefined schemas (`SCHEMAS` constants) that define Redis
Search indices. Checkpoint indices track thread/namespace/version hierarchies; store indices support both key-value
lookup and optional vector similarity search.

**TTL Integration**: Native Redis TTL support is integrated throughout, with configurable defaults and refresh-on-read
capabilities. TTL applies to all related keys (main document, vectors, writes) atomically.

**Cluster Support**: Full Redis Cluster support with automatic detection and cluster-aware operations (individual key
operations vs. pipelined operations).

**Type System**: Heavy use of generics (`BaseRedisSaver[RedisClientType, IndexType]`) to maintain type safety across
sync/async variants while sharing implementation code.

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
- `test_cluster_mode.py` / `test_async_cluster_mode.py`: Redis Cluster specific tests
- `test_checkpoint_ttl.py`: TTL functionality for checkpoints
- `test_key_parsing.py` / `test_subgraph_key_parsing.py`: Key generation and parsing logic
- `test_semantic_search_*.py`: Vector search capabilities
- `test_interruption.py` / `test_streaming*.py`: Advanced workflow tests
- `test_shallow_*.py`: Shallow checkpoint implementation tests
- `test_decode_responses.py`: Redis response decoding tests
- `test_crossslot_integration.py`: Cross-slot operation tests

## Notebooks and Examples

The `examples/` directory contains Jupyter notebooks demonstrating Redis integration with LangGraph:

- All notebooks MUST use Redis implementations (RedisSaver, RedisStore), not in-memory equivalents
- Notebooks can be run via Docker Compose: `cd examples && docker compose up`
- Each notebook includes installation of required dependencies within the notebook cells
- TestContainers should be used for any new notebook examples requiring Redis

### Running Notebooks

1. With Docker (recommended):
   ```bash
   cd examples
   docker compose up
   ```

2. Locally:
    - Ensure Redis is running with required modules (RedisJSON, RediSearch)
    - Install dependencies: `pip install langgraph-checkpoint-redis jupyter`
    - Run: `jupyter notebook`

### Important Dependencies

- Requires Redis with RedisJSON and RediSearch modules
- Uses `redisvl` for vector operations and search index management
- Uses `python-ulid` for unique document IDs
- Integrates with LangGraph's checkpoint and store base classes
