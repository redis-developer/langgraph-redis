# LangGraph Redis Checkpointer Benchmarks

Performance benchmarks comparing Redis checkpointer implementations against all other official LangGraph checkpointers.

## What It Tests

The `checkpointer_benchmark.py` benchmarks:

- **Selected LangGraph checkpointers**: Memory, SQLite, MongoDB, MySQL (regular and shallow), Redis (regular and shallow)
- **Workload patterns**: Fanout-to-subgraph pattern with parallel executions
- **Scaling behavior**: Tests from 100 to 900 parallel executions
- **Individual operations**: Put, get, list checkpoints, and writes operations

## Quick Start

```bash
# Install dependencies
make install

# Run benchmark and generate plots
make all
```

## Prerequisites

- **Docker** (required for TestContainers)

The benchmark uses TestContainers to automatically spin up database instances as needed. No manual database setup required.

## Commands

```bash
make benchmark    # Run performance tests
make plots        # Generate charts from results
make clean        # Remove reports and plots
```

## Results

Results are saved to `reports/` directory:

- `run-TIMESTAMP/` - Individual JSON reports per checkpointer
- `summary_TIMESTAMP.json` - Comparative summary
- Performance plots (PNG files) showing scaling behavior

## Checkpointer Coverage

**Regular implementations tested:**
- Memory (reference)
- SQLite
- MongoDB  
- MySQL
- Redis

**Shallow implementations tested:**
- MySQL (shallow)
- Redis (shallow)

Note: SQLite and MongoDB do not have shallow variants available and Postgres shallow impl. is broken

