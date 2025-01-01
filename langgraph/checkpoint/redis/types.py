from typing import Any, Optional, TypeVar, Union

from redisvl.index import AsyncSearchIndex, SearchIndex

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

RedisClientType = TypeVar("RedisClientType", bound=Union[Redis, AsyncRedis])
IndexType = TypeVar("IndexType", bound=Union[SearchIndex, AsyncSearchIndex])
MetadataInput = Optional[dict[str, Any]]
