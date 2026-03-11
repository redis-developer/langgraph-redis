---
title: Installation
---

# Installation

This guide covers installing `langgraph-checkpoint-redis` and setting up the
required Redis server.

## Install the Package

Install from PyPI using pip:

```bash
pip install langgraph-checkpoint-redis
```

Or using Poetry:

```bash
poetry add langgraph-checkpoint-redis
```

The package brings in the following key dependencies automatically:

- `langgraph` and `langgraph-checkpoint` -- core LangGraph framework
- `redis` -- Python Redis client
- `redisvl` -- Redis vector library for search index management
- `python-ulid` -- ULID generation for document IDs

## Redis Requirements

The library requires a Redis server with the **RedisJSON** and **RediSearch**
modules. These modules provide structured JSON document storage and full-text
plus vector search indexing.

### Redis 8.0+ (Recommended)

Redis 8.0 and later include RedisJSON and RediSearch as built-in modules. No
additional configuration is needed.

Start Redis 8 with Docker:

```bash
docker run -d --name redis -p 6379:6379 redis:8
```

### Redis Stack (Older Versions)

For Redis versions prior to 8.0, use Redis Stack which bundles the required
modules:

```bash
docker run -d --name redis -p 6379:6379 redis/redis-stack-server:latest
```

### Redis Cloud

For a managed Redis service with all required modules, use
[Redis Cloud](https://redis.io/cloud). Redis Cloud provides fully managed Redis
instances with RedisJSON and RediSearch enabled by default. No module
installation or server management is required.

## Verify Your Setup

After starting Redis, verify that the required modules are loaded:

```bash
redis-cli INFO modules
```

Look for entries containing `ReJSON` (RedisJSON) and `search` (RediSearch) in
the output. Both must be present:

```text
# Modules
module:name=ReJSON,ver=...
module:name=search,ver=...
```

You can also verify from Python:

```python
import redis

r = redis.from_url("redis://localhost:6379")
modules = r.module_list()
module_names = [m[b"name"].decode() for m in modules]
print("Modules:", module_names)
assert b"ReJSON" in [m[b"name"] for m in modules], "RedisJSON not found"
assert b"search" in [m[b"name"] for m in modules], "RediSearch not found"
print("All required modules are available.")
```

## Python Version

The library requires **Python 3.10 or higher**. Verify your Python version:

```bash
python --version
# Should show 3.10.x or higher
```

## Next Steps

With the package installed and Redis running, proceed to
{doc}`getting_started` to build your first LangGraph workflow with Redis
persistence.
