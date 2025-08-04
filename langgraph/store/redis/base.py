"""Base implementation for Redis-backed store with optional vector search capabilities."""

from __future__ import annotations

import copy
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

from langgraph.store.base import (
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError
from redisvl.index import SearchIndex
from redisvl.query.filter import Tag, Text
from redisvl.utils.token_escaper import TokenEscaper

from .token_unescaper import TokenUnescaper
from .types import IndexType, RedisClientType

_token_escaper = TokenEscaper()
_token_unescaper = TokenUnescaper()

logger = logging.getLogger(__name__)

REDIS_KEY_SEPARATOR = ":"
STORE_PREFIX = "store"
STORE_VECTOR_PREFIX = "store_vectors"

# Schemas for Redis Search indices
SCHEMAS = [
    {
        "index": {
            "name": "store",
            "prefix": STORE_PREFIX + REDIS_KEY_SEPARATOR,
            "storage_type": "json",
        },
        "fields": [
            {"name": "prefix", "type": "text"},
            {"name": "key", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "ttl_minutes", "type": "numeric"},
            {"name": "expires_at", "type": "numeric"},
        ],
    },
    {
        "index": {
            "name": "store_vectors",
            "prefix": STORE_VECTOR_PREFIX + REDIS_KEY_SEPARATOR,
            "storage_type": "json",
        },
        "fields": [
            {"name": "prefix", "type": "text"},
            {"name": "key", "type": "tag"},
            {"name": "field_name", "type": "tag"},
            {"name": "embedding", "type": "vector"},
            {"name": "created_at", "type": "numeric"},
            {"name": "updated_at", "type": "numeric"},
            {"name": "ttl_minutes", "type": "numeric"},
            {"name": "expires_at", "type": "numeric"},
        ],
    },
]


def _ensure_string_or_literal(value: Any) -> str:
    """Convert value to string safely."""
    if hasattr(value, "lower"):
        return value.lower()
    return str(value)


C = TypeVar("C", bound=Union[Redis, AsyncRedis])


class RedisDocument(TypedDict, total=False):
    prefix: str
    key: str
    value: Optional[str]
    created_at: int
    updated_at: int
    ttl_minutes: Optional[float]
    expires_at: Optional[int]


class BaseRedisStore(Generic[RedisClientType, IndexType]):
    """Base Redis implementation for persistent key-value store with optional vector search."""

    _redis: RedisClientType
    store_index: IndexType
    vector_index: IndexType
    _ttl_sweeper_thread: Optional[threading.Thread] = None
    _ttl_stop_event: threading.Event | None = None
    # Whether to operate in Redis cluster mode; None triggers auto-detection
    cluster_mode: Optional[bool] = None
    SCHEMAS = SCHEMAS

    supports_ttl: bool = True
    ttl_config: Optional[TTLConfig] = None

    def _apply_ttl_to_keys(
        self,
        main_key: str,
        related_keys: Optional[list[str]] = None,
        ttl_minutes: Optional[float] = None,
    ) -> Any:
        """Apply Redis native TTL to keys.

        Args:
            main_key: The primary Redis key
            related_keys: Additional Redis keys that should expire at the same time
            ttl_minutes: Time-to-live in minutes
        """
        if ttl_minutes is None:
            # Check if there's a default TTL in config
            if self.ttl_config and "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

        if ttl_minutes is not None:
            ttl_seconds = int(ttl_minutes * 60)

            # Use the cluster_mode attribute to determine the approach
            if self.cluster_mode:
                # Cluster path: direct expire calls
                self._redis.expire(main_key, ttl_seconds)
                if related_keys:
                    for key in related_keys:
                        self._redis.expire(key, ttl_seconds)
            else:
                # Non-cluster path: transactional pipeline
                pipeline = self._redis.pipeline(transaction=True)
                pipeline.expire(main_key, ttl_seconds)
                if related_keys:
                    for key in related_keys:
                        pipeline.expire(key, ttl_seconds)
                pipeline.execute()

    def sweep_ttl(self) -> int:
        """Clean up any remaining expired items.

        This is not needed with Redis native TTL, but kept for API compatibility.
        Redis automatically removes expired keys.

        Returns:
            int: Always returns 0 as Redis handles expiration automatically
        """
        return 0

    def start_ttl_sweeper(self, _sweep_interval_minutes: Optional[int] = None) -> None:
        """Start TTL sweeper.

        This is a no-op with Redis native TTL, but kept for API compatibility.
        Redis automatically removes expired keys.

        Args:
            _sweep_interval_minutes: Ignored parameter, kept for API compatibility
        """
        # No-op: Redis handles TTL expiration automatically
        pass

    def stop_ttl_sweeper(self, _timeout: Optional[float] = None) -> bool:
        """Stop TTL sweeper.

        This is a no-op with Redis native TTL, but kept for API compatibility.

        Args:
            _timeout: Ignored parameter, kept for API compatibility

        Returns:
            bool: Always True as there's no sweeper to stop
        """
        # No-op: Redis handles TTL expiration automatically
        return True

    def __init__(
        self,
        conn: RedisClientType,
        *,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,  # Corrected type hint for ttl
        cluster_mode: Optional[bool] = None,
    ) -> None:
        """Initialize store with Redis connection and optional index config."""
        self.index_config = index
        self.ttl_config = ttl
        self._redis = conn
        # Store cluster_mode; None means auto-detect in RedisStore or AsyncRedisStore
        self.cluster_mode = cluster_mode

        if self.index_config:
            self.index_config = self.index_config.copy()
            self.embeddings = ensure_embeddings(
                self.index_config.get("embed"),
            )
            fields = self.index_config.get("fields", ["$"]) or []
            if isinstance(fields, str):
                fields = [fields]
            self.index_config["__tokenized_fields"] = [
                (p, tokenize_path(p)) if p != "$" else (p, p) for p in fields
            ]

        # Initialize search indices
        self.store_index = SearchIndex.from_dict(
            self.SCHEMAS[0], redis_client=self._redis
        )

        # Configure vector index if needed
        if self.index_config:
            # Get storage type from index config, default to "json"
            # Cast to dict to safely access potential extra fields
            index_dict = dict(self.index_config)
            vector_storage_type = index_dict.get("vector_storage_type", "json")

            vector_schema: Dict[str, Any] = copy.deepcopy(self.SCHEMAS[1])
            # Update storage type in schema
            vector_schema["index"]["storage_type"] = vector_storage_type

            vector_fields = vector_schema.get("fields", [])
            vector_field = None
            for f in vector_fields:
                if isinstance(f, dict) and f.get("name") == "embedding":
                    vector_field = f
                    break

            if vector_field:
                # Configure vector field with index config values
                vector_field["attrs"] = {
                    "algorithm": "flat",  # Default to flat
                    "datatype": "float32",
                    "dims": self.index_config["dims"],
                    # Map distance metrics to Redis-accepted literals
                    "distance_metric": {
                        "cosine": "COSINE",
                        "inner_product": "IP",
                        "l2": "L2",
                    }[
                        _ensure_string_or_literal(
                            index_dict.get("distance_type", "cosine")
                        )
                    ],
                }

                # Apply any additional vector type config
                if "ann_index_config" in index_dict:
                    vector_field["attrs"].update(index_dict["ann_index_config"])

            self.vector_index = SearchIndex.from_dict(
                vector_schema, redis_client=self._redis
            )

        # Set client information in Redis
        self.set_client_info()

    def set_client_info(self) -> None:
        """Set client info for Redis monitoring."""

        from langgraph.checkpoint.redis.version import __redisvl_version__

        # Create the client info string with only the redisvl version
        client_info = f"redis-py(redisvl_v{__redisvl_version__})"

        try:
            # Try to use client_setinfo command if available
            self._redis.client_setinfo("LIB-NAME", client_info)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                self._redis.echo(client_info)
            except Exception:
                # Silently fail if even echo doesn't work
                pass

    async def aset_client_info(self) -> None:
        """Set client info for Redis monitoring asynchronously."""

        from langgraph.checkpoint.redis.version import __redisvl_version__

        # Create the client info string with only the redisvl version
        client_info = f"redis-py(redisvl_v{__redisvl_version__})"

        try:
            # Try to use client_setinfo command if available
            await self._redis.client_setinfo("LIB-NAME", client_info)
        except (ResponseError, AttributeError):
            # Fall back to a simple echo if client_setinfo is not available
            try:
                # Call with await to ensure it's an async call
                echo_result = self._redis.echo(client_info)
                if hasattr(echo_result, "__await__"):
                    await echo_result
            except Exception:
                # Silently fail if even echo doesn't work
                pass

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, Sequence, tuple[str, ...], list]]:
        """Convert GET operations into Redis queries."""
        namespace_groups = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))

        results: list[tuple[str, Sequence, tuple[str, ...], list]] = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items)
            # Use Tag helper to properly escape all special characters
            prefix_filter = Text("prefix") == _namespace_to_text(namespace)
            filter_str = f"({prefix_filter} "
            if keys:
                key_filter = Tag("key") == list(keys)
                filter_str += f"{key_filter})"
            else:
                filter_str += ")"
            results.append((filter_str, [], namespace, items))
        return results

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[
        list[RedisDocument], Optional[tuple[str, list[tuple[str, str, str, str]]]]
    ]:
        # Last-write wins
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        operations: list[RedisDocument] = []
        embedding_request = None
        to_embed: list[tuple[str, str, str, str]] = []

        if deletes:
            # Delete matching documents
            for op in deletes:
                prefix = _namespace_to_text(op.namespace)
                query = f"(@prefix:{prefix} @key:{{{op.key}}})"
                results = self.store_index.search(query)
                for doc in results.docs:
                    self._redis.delete(doc.id)

        # Handle inserts
        if inserts:
            for op in inserts:
                now = int(datetime.now(timezone.utc).timestamp() * 1_000_000)

                # With native Redis TTL, we don't need to store TTL in document
                ttl_minutes = None
                expires_at = None
                if hasattr(op, "ttl") and op.ttl is not None:
                    ttl_minutes = op.ttl
                    # Calculate expiration but don't rely on it for actual expiration
                    # as we'll use Redis native TTL
                    expires_at = int(
                        (
                            datetime.now(timezone.utc) + timedelta(minutes=op.ttl)
                        ).timestamp()
                    )

                doc = RedisDocument(
                    prefix=_namespace_to_text(op.namespace),
                    key=op.key,
                    value=op.value,
                    created_at=now,
                    updated_at=now,
                    ttl_minutes=ttl_minutes,
                    expires_at=expires_at,
                )
                operations.append(doc)

                if self.index_config and op.index is not False:
                    paths = (
                        self.index_config["__tokenized_fields"]
                        if op.index is None
                        else [(ix, tokenize_path(ix)) for ix in op.index]
                    )

                    for path, tokenized_path in paths:
                        texts = get_text_at_path(op.value, tokenized_path)
                        for text in texts:
                            to_embed.append(
                                (_namespace_to_text(op.namespace), op.key, path, text)
                            )

            if to_embed:
                embedding_request = ("", to_embed)

        return operations, embedding_request

    def _get_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[list[tuple[str, list, int, int]], list[tuple[int, str]]]:
        """Convert search operations into Redis queries."""
        queries = []
        embedding_requests = []

        for idx, op in search_ops:
            filter_conditions = []
            if op.namespace_prefix:
                prefix = _namespace_to_text(op.namespace_prefix)
                filter_conditions.append(f"@prefix:{prefix}*")

            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

            query = " ".join(filter_conditions) if filter_conditions else "*"
            limit = op.limit if op.limit is not None else 10
            offset = op.offset if op.offset is not None else 0
            params = [limit, offset]
            queries.append((query, params, limit, offset))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, list]]:
        """Convert list namespaces operations into Redis queries."""
        queries = []
        for _, op in list_ops:
            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        path = _namespace_to_text(condition.path, handle_wildcards=True)
                        conditions.append(f"@prefix:{path}*")
                    elif condition.match_type == "suffix":
                        path = _namespace_to_text(condition.path, handle_wildcards=True)
                        conditions.append(f"@prefix:*{path}")

            query = " ".join(conditions) if conditions else "*"
            params = [op.limit, op.offset] if op.limit or op.offset else []
            queries.append((query, params))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> str:
        """Get Redis search filter condition for an operator."""
        if op == "$eq":
            return f'@{key}:"{value}"'
        elif op == "$gt":
            return f"@{key}:[({value} inf]"
        elif op == "$gte":
            return f"@{key}:[{value} inf]"
        elif op == "$lt":
            return f"@{key}:[-inf ({value}]"
        elif op == "$lte":
            return f"@{key}:[-inf {value}]"
        elif op == "$ne":
            return f'-@{key}:"{value}"'
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def _cosine_similarity(
        self, vec1: list[float], vecs: list[list[float]]
    ) -> list[float]:
        """Compute cosine similarity between vectors."""
        # Note: For production use, consider importing numpy for better performance
        similarities = []
        for vec2 in vecs:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = (sum(x * x for x in vec1)) ** 0.5
            norm2 = (sum(x * x for x in vec2)) ** 0.5
            if norm1 == 0 or norm2 == 0:
                similarities.append(0)
            else:
                similarities.append(dot_product / (norm1 * norm2))
        return similarities


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to text string with proper escaping.

    Args:
        namespace: Tuple of strings representing namespace components
        handle_wildcards: Whether to handle wildcard characters specially

    Returns:
        Properly escaped string representation of namespace
    """
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)

    # First join with dots
    ns_text = _token_escaper.escape(".".join(namespace))

    return ns_text


def _decode_ns(ns: str) -> tuple[str, ...]:
    """Convert a dotted namespace string back into a tuple."""
    return tuple(_token_unescaper.unescape(ns).split("."))


def _row_to_item(namespace: tuple[str, ...], row: dict[str, Any]) -> Item:
    """Convert a row from Redis to an Item."""
    return Item(
        value=row["value"],
        key=row["key"],
        namespace=namespace,
        created_at=datetime.fromtimestamp(row["created_at"] / 1_000_000, timezone.utc),
        updated_at=datetime.fromtimestamp(row["updated_at"] / 1_000_000, timezone.utc),
    )


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    score: Optional[float] = None,
) -> SearchItem:
    """Convert a row from Redis to a SearchItem."""
    return SearchItem(
        value=row["value"],
        key=row["key"],
        namespace=namespace,
        created_at=datetime.fromtimestamp(row["created_at"] / 1_000_000, timezone.utc),
        updated_at=datetime.fromtimestamp(row["updated_at"] / 1_000_000, timezone.utc),
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    """Group operations by type for batch processing."""
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot
