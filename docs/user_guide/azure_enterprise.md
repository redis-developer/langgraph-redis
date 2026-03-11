---
title: Azure and Enterprise Deployment
---

# Azure and Enterprise Deployment

This guide covers deploying `langgraph-checkpoint-redis` with managed Redis
services, including Azure Managed Redis, Azure Cache for Redis Enterprise, and
Redis Enterprise.

## Key Principle: Use Standard Redis Client

Managed Redis services such as Azure Managed Redis and Redis Enterprise use a
**proxy layer** that routes commands to the appropriate shard. Because the proxy
handles cluster topology, you must use the standard `Redis` client -- **not**
`RedisCluster`:

```python
from redis import Redis

# Correct: standard Redis client with managed services
client = Redis(
    host="my-redis.eastus.redis.azure.net",
    port=10000,
    ssl=True,
    password="your-access-key",
)
```

Using `RedisCluster` with these proxy-based services will cause connection
errors because the proxy does not support the `CLUSTER` command set.

## Azure Managed Redis

Azure Managed Redis (preview) is Microsoft's latest managed Redis offering with
built-in support for Redis modules.

### Prerequisites

- Select the **Enterprise** or **Enterprise Flash** tier when creating the
  resource. The Basic/Standard/Premium tiers do not support RedisJSON or
  RediSearch.
- Enable the **RedisJSON** and **RediSearch** modules at creation time. Modules
  cannot be added after the resource is created.

### Connection Configuration

```python
from langgraph.checkpoint.redis import RedisSaver

# Azure Managed Redis uses port 10000 for TLS connections
REDIS_URL = "rediss://:your-access-key@my-redis.eastus.redis.azure.net:10000"

with RedisSaver.from_conn_string(REDIS_URL) as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)
```

Or with explicit client configuration:

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

client = Redis(
    host="my-redis.eastus.redis.azure.net",
    port=10000,
    password="your-access-key",
    ssl=True,
    ssl_cert_reqs="required",
    decode_responses=False,
)

saver = RedisSaver(redis_client=client)
saver.setup()
```

### Port Configuration

| Port | Protocol | Usage |
|------|----------|-------|
| 10000 | TLS (rediss://) | Default for Enterprise tier with SSL |
| 6379 | Non-TLS (redis://) | Available if non-TLS access is enabled |

## Azure Cache for Redis Enterprise

Azure Cache for Redis Enterprise is the established enterprise-grade Redis
service on Azure.

### Setup

1. Create an Azure Cache for Redis resource with the **Enterprise** tier.
2. Under **Modules**, enable **RedisJSON** and **RediSearch**.
3. Note the hostname and access key from the **Access keys** blade.

### Connection Example

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

# Standard Redis client -- do NOT use RedisCluster
client = Redis(
    host="my-cache.region.redisenterprise.cache.azure.net",
    port=10000,
    password="your-access-key",
    ssl=True,
)

# Checkpointer
saver = RedisSaver(redis_client=client)
saver.setup()

# Store (with same or separate client)
store = RedisStore(client)
store.setup()

graph = builder.compile(checkpointer=saver, store=store)
```

## Redis Enterprise (Self-Managed)

For self-managed Redis Enterprise deployments:

### Module Requirements

Ensure the following modules are enabled on your database:

- **RedisJSON** -- structured document storage
- **RediSearch** -- full-text and vector search indexing

Modules are configured at the database level in the Redis Enterprise admin
console.

### Connection Configuration

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

# Redis Enterprise proxy endpoint
client = Redis(
    host="redis-enterprise.internal.example.com",
    port=12000,  # Database-specific port
    password="database-password",
    ssl=True,
    ssl_ca_certs="/path/to/ca.pem",
)

saver = RedisSaver(redis_client=client)
saver.setup()
```

## SSL/TLS Configuration

For production deployments, always use SSL/TLS:

```python
from redis import Redis

client = Redis(
    host="redis.example.com",
    port=6380,
    ssl=True,
    ssl_certfile="/path/to/client-cert.pem",
    ssl_keyfile="/path/to/client-key.pem",
    ssl_ca_certs="/path/to/ca-cert.pem",
    ssl_cert_reqs="required",
)
```

Using a connection URL with TLS:

```python
# The "rediss://" scheme enables TLS
REDIS_URL = "rediss://:password@redis.example.com:6380"

with RedisSaver.from_conn_string(REDIS_URL) as saver:
    saver.setup()
```

## TTL Considerations for Enterprise

TTL operations use Redis `EXPIRE` commands. On some enterprise proxy
configurations, `EXPIRE` commands in pipelines may fail independently of
`JSON.SET` commands. The library handles this gracefully:

- Write operations (`put`, `put_writes`) apply TTL as a best-effort step
  **after** the critical data write succeeds.
- A TTL failure does not cause data loss -- the checkpoint is still stored,
  just without expiration.
- Warning messages are logged when TTL application fails.

## Complete Azure Example

A full example combining checkpointer, store, and middleware on Azure:

```python
from redis import Redis
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

# Single client for Azure Managed Redis
client = Redis(
    host="my-redis.eastus.redis.azure.net",
    port=10000,
    password="your-access-key",
    ssl=True,
)

# TTL configured for 2-hour expiration
ttl_config = {"default_ttl": 120, "refresh_on_read": True}

saver = RedisSaver(redis_client=client, ttl=ttl_config)
saver.setup()

store = RedisStore(client, ttl=ttl_config)
store.setup()

graph = builder.compile(checkpointer=saver, store=store)

# Run with thread management
config = {"configurable": {"thread_id": "azure-session-1"}}
result = graph.invoke({"messages": ["Hello from Azure"]}, config)
```

## Next Steps

- {doc}`ttl` -- TTL configuration details
- {doc}`checkpointers` -- checkpointer reference
- {doc}`middleware` -- add middleware to your deployment
