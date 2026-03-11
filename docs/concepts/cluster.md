# Cluster Support

`langgraph-checkpoint-redis` works with standalone Redis, Redis Cluster, and managed Redis services. Cluster mode is detected automatically, and the library adapts its operations to work within cluster constraints.

## Automatic Cluster Detection

When you call `setup()` on a checkpoint saver, the library inspects the Redis client to determine whether it is a `RedisCluster` instance:

```python
def _detect_cluster_mode(self) -> None:
    if isinstance(self._redis, RedisCluster):
        self.cluster_mode = True
    else:
        self.cluster_mode = False
```

You can also set `cluster_mode` explicitly in the constructor to skip auto-detection:

```python
saver = RedisSaver(redis_url="redis://localhost:6379", cluster_mode=True)
```

The store (`RedisStore`) similarly accepts a `cluster_mode` parameter.

## How Cluster Mode Affects Operations

Redis Cluster distributes keys across slots based on a hash of the key name. This means keys with different names may live on different nodes. Two operations are affected:

### Pipelines

In standalone Redis, pipelines are transactional: multiple commands execute atomically on a single server. In cluster mode, pipeline commands may target different nodes, so atomicity is not guaranteed. The library handles this by:

- **Standalone** -- Using `pipeline(transaction=True)` for batched operations like TTL propagation.
- **Cluster** -- Executing commands individually per key, since the keys may reside on different nodes.

```python
if self.cluster_mode:
    for key in keys_to_delete:
        self._redis.delete(key)
else:
    pipeline = self._redis.pipeline(transaction=True)
    for key in keys_to_delete:
        pipeline.delete(key)
    pipeline.execute()
```

### Key Deletion

Bulk deletion of checkpoint data (e.g., clearing a thread's history) follows the same pattern: individual `DELETE` calls in cluster mode, pipelined `DELETE` in standalone mode.

## Cross-Slot Considerations

In Redis Cluster, a multi-key command fails if the keys hash to different slots. The library avoids multi-key commands entirely, performing all operations on individual keys. This design ensures compatibility with both standalone and cluster deployments without requiring hash tags or key co-location.

## Azure Managed Redis and Redis Enterprise

Azure Managed Redis and Redis Enterprise use a **proxy architecture**: the client connects to a single endpoint that routes commands to the appropriate shard internally. From the client's perspective, this looks like a standalone Redis server.

For these deployments, use the standard `Redis` client, not `RedisCluster`:

```python
from langgraph.checkpoint.redis import RedisSaver

# Azure Managed Redis or Redis Enterprise -- use standard client
saver = RedisSaver.from_conn_string("rediss://my-instance.redis.cache.windows.net:6380")
saver.setup()
```

The proxy handles sharding transparently, and `cluster_mode` will correctly auto-detect as `False` since the client is not a `RedisCluster` instance. Transactional pipelines work normally through the proxy.

## Redis 8.0+ vs Redis Stack

The library requires the RedisJSON and RediSearch modules. There are two ways to get them:

| Option | When to Use |
|---|---|
| **Redis 8.0+** | Production deployments. Both modules are built into the core server. No additional configuration required. |
| **Redis Stack** | Older Redis versions (6.x, 7.x). Bundles the modules as a separate distribution. Available as `redis/redis-stack-server` on Docker Hub. |

In both cases, the library uses the same API. The only difference is server-side packaging.

## Sentinel Support

Redis Sentinel provides high availability for standalone Redis. The checkpoint saver supports Sentinel URLs:

```python
saver = RedisSaver.from_conn_string(
    "redis+sentinel://sentinel-host:26379/myservice/0"
)
```

Sentinel is not applicable to Redis Cluster deployments, which have their own built-in failover mechanism.

## Choosing a Deployment

| Deployment | Client | `cluster_mode` | Notes |
|---|---|---|---|
| Standalone Redis | `Redis` | `False` (auto) | Simplest setup; suitable for development and small workloads |
| Redis Cluster (OSS) | `RedisCluster` | `True` (auto) | Horizontal scaling with client-side routing |
| Azure Managed Redis | `Redis` | `False` (auto) | Proxy-based; use standard client with TLS |
| Redis Enterprise | `Redis` | `False` (auto) | Proxy-based; use standard client |
| Redis with Sentinel | `Redis` | `False` (auto) | High availability for standalone Redis |
