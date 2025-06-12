from typing import Any, Optional, TypeVar, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redisvl.index import AsyncSearchIndex, SearchIndex

RedisClientType = TypeVar(
    "RedisClientType", bound=Union[Redis, AsyncRedis, RedisCluster, AsyncRedisCluster]
)
IndexType = TypeVar("IndexType", bound=Union[SearchIndex, AsyncSearchIndex])
MetadataInput = Optional[dict[str, Any]]
