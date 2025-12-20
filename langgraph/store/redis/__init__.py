"""Synchronous Redis store implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional, Sequence, cast

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)
from redis import Redis
from redis.cluster import RedisCluster as SyncRedisCluster
from redis.commands.search.query import Query
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.token_escaper import TokenEscaper
from ulid import ULID

from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
    BaseRedisStore,
    RedisDocument,
    _decode_ns,
    _group_ops,
    _namespace_to_text,
    _row_to_item,
    _row_to_search_item,
)

from .token_unescaper import TokenUnescaper

_token_escaper = TokenEscaper()
_token_unescaper = TokenUnescaper()

logger = logging.getLogger(__name__)


def _convert_redis_score_to_similarity(score: float, distance_type: str) -> float:
    """Convert Redis vector distance to similarity score."""
    if distance_type == "cosine":
        # Redis returns cosine distance (1 - cosine_similarity)
        # Convert back to similarity
        return 1.0 - score
    elif distance_type == "l2":
        # For L2, smaller distance means more similar
        # Use a simple exponential decay
        return math.exp(-score)
    elif distance_type == "inner_product":
        # For inner product, Redis already returns what we want
        return score
    return score


class RedisStore(BaseStore, BaseRedisStore[Redis, SearchIndex]):
    """Redis-backed store with optional vector search.

    Provides synchronous operations for storing and retrieving data with optional
    vector similarity search support.
    """

    # Enable TTL support
    supports_ttl = True
    ttl_config: Optional[TTLConfig] = None

    def __init__(
        self,
        conn: Redis,
        *,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
        cluster_mode: Optional[bool] = None,
        store_prefix: str = STORE_PREFIX,
        vector_prefix: str = STORE_VECTOR_PREFIX,
    ) -> None:
        BaseStore.__init__(self)
        BaseRedisStore.__init__(
            self,
            conn,
            index=index,
            ttl=ttl,
            cluster_mode=cluster_mode,
            store_prefix=store_prefix,
            vector_prefix=vector_prefix,
        )
        # Set client info for monitoring (sync store can call this safely)
        self.set_client_info()
        # Detection will happen in setup()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
        store_prefix: str = STORE_PREFIX,
        vector_prefix: str = STORE_VECTOR_PREFIX,
    ) -> Iterator[RedisStore]:
        """Create store from Redis connection string."""
        client = None
        try:
            client = RedisConnectionFactory.get_redis_connection(conn_string)
            store = cls(
                client,
                index=index,
                ttl=ttl,
                store_prefix=store_prefix,
                vector_prefix=vector_prefix,
            )
            # Client info is set in __init__, but set it again here to ensure
            # it's available even if called before setup()
            store.set_client_info()
            yield store
        finally:
            if client:
                client.close()
                client.connection_pool.disconnect()

    def setup(self) -> None:
        """Initialize store indices."""
        # Detect if we're connected to a Redis cluster
        self._detect_cluster_mode()

        self.store_index.create(overwrite=False)
        if self.index_config:
            self.vector_index.create(overwrite=False)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch of operations."""
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped_ops:
            self._batch_get_ops(
                cast(list[tuple[int, GetOp]], grouped_ops[GetOp]), results
            )

        if PutOp in grouped_ops:
            self._batch_put_ops(cast(list[tuple[int, PutOp]], grouped_ops[PutOp]))

        if SearchOp in grouped_ops:
            self._batch_search_ops(
                cast(list[tuple[int, SearchOp]], grouped_ops[SearchOp]), results
            )

        if ListNamespacesOp in grouped_ops:
            self._batch_list_namespaces_ops(
                cast(
                    Sequence[tuple[int, ListNamespacesOp]],
                    grouped_ops[ListNamespacesOp],
                ),
                results,
            )

        return results

    def _detect_cluster_mode(self) -> None:
        """Detect if the Redis client is a cluster client by inspecting its class."""
        # If we passed in_cluster_mode explicitly, respect it
        if self.cluster_mode is not None:
            logger.info(
                f"Redis cluster_mode explicitly set to {self.cluster_mode}, skipping detection."
            )
            return

        if isinstance(self._redis, SyncRedisCluster):
            self.cluster_mode = True
            logger.info("Redis cluster client detected for RedisStore.")
        else:
            self.cluster_mode = False
            logger.info("Redis standalone client detected for RedisStore.")

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        """Execute list namespaces operations in batch."""
        for idx, op in list_ops:
            # Construct base query for namespace search
            base_query = "*"  # Start with all documents
            if op.match_conditions:
                conditions = []
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        prefix = _namespace_to_text(
                            condition.path, handle_wildcards=True
                        )
                        conditions.append(f"@prefix:{prefix}*")
                    elif condition.match_type == "suffix":
                        suffix = _namespace_to_text(
                            condition.path, handle_wildcards=True
                        )
                        conditions.append(f"@prefix:*{suffix}")
                if conditions:
                    base_query = " ".join(conditions)

            # Execute search with return_fields=["prefix"] to get just namespaces
            query = FilterQuery(filter_expression=base_query, return_fields=["prefix"])
            res = self.store_index.search(query)

            # Extract unique namespaces
            namespaces = set()
            for doc in res.docs:
                if hasattr(doc, "prefix"):
                    ns = tuple(_token_unescaper.unescape(doc.prefix).split("."))
                    # Apply max_depth if specified
                    if op.max_depth is not None:
                        ns = ns[: op.max_depth]
                    namespaces.add(ns)

            # Sort and apply pagination
            sorted_namespaces = sorted(namespaces)
            if op.limit or op.offset:
                offset = op.offset or 0
                limit = op.limit or 10
                sorted_namespaces = sorted_namespaces[offset : offset + limit]

            results[idx] = sorted_namespaces

    def _batch_get_ops(
        self,
        get_ops: list[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        """Execute GET operations in batch."""
        refresh_keys_by_idx: dict[int, list[str]] = (
            {}
        )  # Track keys that need TTL refreshed by op index

        for query, _, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            res = self.store_index.search(Query(query))
            # Parse JSON from each document
            key_to_row = {
                json.loads(doc.json)["key"]: (json.loads(doc.json), doc.id)
                for doc in res.docs
            }

            for idx, key in items:
                if key in key_to_row:
                    data, doc_id = key_to_row[key]
                    results[idx] = _row_to_item(
                        namespace, data, deserialize_fn=self._deserialize_value
                    )

                    # Find the corresponding operation by looking it up in the operation list
                    # This is needed because idx is the index in the overall operation list
                    op_idx = None
                    for i, (local_idx, op) in enumerate(get_ops):
                        if local_idx == idx:
                            op_idx = i
                            break

                    if op_idx is not None:
                        op = get_ops[op_idx][1]
                        if hasattr(op, "refresh_ttl") and op.refresh_ttl:
                            if idx not in refresh_keys_by_idx:
                                refresh_keys_by_idx[idx] = []
                            refresh_keys_by_idx[idx].append(doc_id)

                            # Also add vector keys for the same document
                            doc_uuid = doc_id.split(":")[-1]
                            vector_key = (
                                f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_uuid}"
                            )
                            refresh_keys_by_idx[idx].append(vector_key)

        # Now refresh TTLs for any keys that need it
        if refresh_keys_by_idx and self.ttl_config:
            # Get default TTL from config
            ttl_minutes = None
            if "default_ttl" in self.ttl_config:
                ttl_minutes = self.ttl_config.get("default_ttl")

            if ttl_minutes is not None:
                ttl_seconds = int(ttl_minutes * 60)
                if self.cluster_mode:
                    for keys_to_refresh in refresh_keys_by_idx.values():
                        for key in keys_to_refresh:
                            ttl = self._redis.ttl(key)
                            if ttl > 0:
                                self._redis.expire(key, ttl_seconds)
                else:
                    pipeline = self._redis.pipeline(transaction=True)
                    for keys in refresh_keys_by_idx.values():
                        for key in keys:
                            # Only refresh TTL if the key exists and has a TTL
                            ttl = self._redis.ttl(key)
                            if ttl > 0:  # Only refresh if key exists and has TTL
                                pipeline.expire(key, ttl_seconds)
                    if pipeline.command_stack:
                        pipeline.execute()

    def _batch_put_ops(
        self,
        put_ops: list[tuple[int, PutOp]],
    ) -> None:
        """Execute PUT operations in batch."""
        operations, embedding_request = self._prepare_batch_PUT_queries(put_ops)

        # First delete any existing documents that are being updated/deleted
        for _, op in put_ops:
            namespace = _namespace_to_text(op.namespace)
            query = f"@prefix:{namespace} @key:{{{_token_escaper.escape(op.key)}}}"
            results = self.store_index.search(query)

            if self.cluster_mode:
                for doc in results.docs:
                    self._redis.delete(doc.id)
                if self.index_config:
                    vector_results = self.vector_index.search(query)
                    for doc_vec in vector_results.docs:
                        self._redis.delete(doc_vec.id)
            else:
                pipeline = self._redis.pipeline(transaction=True)
                for doc in results.docs:
                    pipeline.delete(doc.id)

                if self.index_config:
                    vector_results = self.vector_index.search(query)
                    for doc_vec in vector_results.docs:
                        pipeline.delete(doc_vec.id)

                if pipeline.command_stack:
                    pipeline.execute()

        # Now handle new document creation
        doc_ids: dict[tuple[str, str], str] = {}
        store_docs: list[RedisDocument] = []
        store_keys: list[str] = []
        ttl_tracking: dict[str, tuple[list[str], Optional[float]]] = (
            {}
        )  # Tracks keys that need TTL + their TTL values

        # Generate IDs for PUT operations
        for _, op in put_ops:
            if op.value is not None:
                generated_doc_id = str(ULID())
                namespace = _namespace_to_text(op.namespace)
                doc_ids[(namespace, op.key)] = generated_doc_id
                # Track TTL for this document if specified
                if hasattr(op, "ttl") and op.ttl is not None:
                    main_key = (
                        f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{generated_doc_id}"
                    )
                    ttl_tracking[main_key] = ([], op.ttl)

        # Load store docs with explicit keys
        for doc in operations:
            store_key = (doc["prefix"], doc["key"])
            doc_id = doc_ids[store_key]
            # Remove TTL fields - they're not needed with Redis native TTL
            if "ttl_minutes" in doc:
                doc.pop("ttl_minutes", None)
            if "expires_at" in doc:
                doc.pop("expires_at", None)

            store_docs.append(doc)
            redis_key = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
            store_keys.append(redis_key)

        if store_docs:
            if self.cluster_mode:
                # Load individually if cluster
                for i, store_doc_item in enumerate(store_docs):
                    self.store_index.load([store_doc_item], keys=[store_keys[i]])
            else:
                self.store_index.load(store_docs, keys=store_keys)

        # Handle vector embeddings with same IDs
        if embedding_request and self.embeddings:
            _, text_params = embedding_request
            vectors = self.embeddings.embed_documents(
                [text for _, _, _, text in text_params]
            )

            vector_docs: list[dict[str, Any]] = []
            vector_keys: list[str] = []

            # Check if we're using hash storage for vectors
            vector_storage_type = "json"  # default
            if self.index_config:
                index_dict = dict(self.index_config)
                vector_storage_type = index_dict.get("vector_storage_type", "json")

            for (ns, key, path, _), vector in zip(text_params, vectors):
                vector_key: tuple[str, str] = (ns, key)
                doc_id = doc_ids[vector_key]

                # Prepare vector based on storage type
                if vector_storage_type == "hash":
                    # For hash storage, convert vector to byte string
                    from redisvl.redis.utils import array_to_buffer

                    vector_list = (
                        vector.tolist() if hasattr(vector, "tolist") else vector
                    )
                    embedding_value = array_to_buffer(vector_list, "float32")
                else:
                    # For JSON storage, keep as list
                    embedding_value = (
                        vector.tolist() if hasattr(vector, "tolist") else vector
                    )

                vector_docs.append(
                    {
                        "prefix": ns,
                        "key": key,
                        "field_name": path,
                        "embedding": embedding_value,
                        "created_at": datetime.now(timezone.utc).timestamp(),
                        "updated_at": datetime.now(timezone.utc).timestamp(),
                    }
                )
                redis_vector_key = f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                vector_keys.append(redis_vector_key)

                # Add this vector key to the related keys list for TTL
                main_key = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                if main_key in ttl_tracking:
                    ttl_tracking[main_key][0].append(redis_vector_key)

            if vector_docs:
                if self.cluster_mode:
                    # Load individually if cluster
                    for i, vector_doc_item in enumerate(vector_docs):
                        self.vector_index.load([vector_doc_item], keys=[vector_keys[i]])
                else:
                    self.vector_index.load(vector_docs, keys=vector_keys)

        # Now apply TTLs after all documents are loaded
        for main_key, (related_keys, ttl_minutes) in ttl_tracking.items():
            self._apply_ttl_to_keys(main_key, related_keys, ttl_minutes)

    def _batch_search_ops(
        self,
        search_ops: list[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        """Execute search operations in batch."""
        queries, embedding_requests = self._get_batch_search_queries(search_ops)

        # Handle vector search
        query_vectors = {}
        if embedding_requests and self.embeddings:
            vectors = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )
            query_vectors = dict(zip([idx for idx, _ in embedding_requests], vectors))

        # Process each search operation
        for (idx, op), (query_str, params, limit, offset) in zip(search_ops, queries):
            if op.query and idx in query_vectors:
                # Vector similarity search
                vector = query_vectors[idx]
                vector_query = VectorQuery(
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    vector_field_name="embedding",
                    filter_expression=f"@prefix:{_namespace_to_text(op.namespace_prefix)}*",
                    return_fields=["prefix", "key", "vector_distance"],
                    num_results=limit,  # Use the user-specified limit
                )
                vector_results = self.vector_index.query(vector_query)

                # Get matching store docs
                result_map = {}  # Map store key to vector result with distances

                if self.cluster_mode:
                    store_docs = []
                    # Direct JSON GET for cluster mode
                    for doc in vector_results:
                        doc_id = (
                            doc.get("id")
                            if isinstance(doc, dict)
                            else getattr(doc, "id", None)
                        )
                        if doc_id:
                            doc_uuid = doc_id.split(":")[1]
                            store_key = (
                                f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_uuid}"
                            )
                            result_map[store_key] = doc
                            # Fetch individually in cluster mode
                            store_doc_item = self._redis.json().get(store_key)
                            store_docs.append(store_doc_item)
                    store_docs_raw = store_docs
                else:
                    pipe = self._redis.pipeline(transaction=True)
                    for doc in vector_results:
                        doc_id = (
                            doc.get("id")
                            if isinstance(doc, dict)
                            else getattr(doc, "id", None)
                        )
                        if not doc_id:
                            continue
                        doc_uuid = doc_id.split(":")[1]
                        store_key = (
                            f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_uuid}"
                        )
                        result_map[store_key] = doc
                        pipe.json().get(store_key)
                    # Execute all lookups in one batch
                    store_docs_raw = pipe.execute()

                # Process results maintaining order and applying filters
                items = []
                refresh_keys = []  # Track keys that need TTL refreshed
                store_docs_iter = iter(store_docs_raw)

                for store_key in result_map.keys():
                    store_doc = next(store_docs_iter, None)
                    if store_doc:
                        vector_result = result_map[store_key]
                        # Get vector_distance from original search result
                        dist = (
                            vector_result.get("vector_distance")
                            if isinstance(vector_result, dict)
                            else getattr(vector_result, "vector_distance", 0)
                        )
                        # Convert to similarity score
                        score = (1.0 - float(dist)) if dist is not None else 0.0
                        if not isinstance(store_doc, dict):
                            try:
                                # Cast needed: redis-py types json().get() incorrectly
                                store_doc = json.loads(
                                    cast(str, store_doc)
                                )  # Attempt to parse if it's a JSON string
                            except (json.JSONDecodeError, TypeError):
                                logger.error(f"Failed to parse store_doc: {store_doc}")
                                continue  # Skip this problematic document

                        if isinstance(
                            store_doc, dict
                        ):  # Check again after potential parsing
                            store_doc["vector_distance"] = dist
                        else:
                            # if still not a dict, this means it's a problematic entry
                            logger.error(
                                f"store_doc is not a dict after parsing attempt: {store_doc}"
                            )
                            continue

                        # Apply value filters if needed
                        if op.filter:
                            matches = True
                            value = store_doc.get("value", {})
                            for key, expected in op.filter.items():
                                actual = value.get(key)
                                if isinstance(expected, list):
                                    if actual not in expected:
                                        matches = False
                                        break
                                elif actual != expected:
                                    matches = False
                                    break
                            if not matches:
                                continue

                        # If refresh_ttl is true, add to list for refreshing
                        if op.refresh_ttl:
                            refresh_keys.append(store_key)
                            # Also find associated vector keys with same ID
                            doc_id = store_key.split(":")[-1]
                            vector_key = (
                                f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                            )
                            refresh_keys.append(vector_key)

                        items.append(
                            _row_to_search_item(
                                _decode_ns(store_doc["prefix"]),
                                store_doc,
                                score=score,
                                deserialize_fn=self._deserialize_value,
                            )
                        )

                # Refresh TTL if requested
                if op.refresh_ttl and refresh_keys and self.ttl_config:
                    # Get default TTL from config
                    ttl_minutes = None
                    if "default_ttl" in self.ttl_config:
                        ttl_minutes = self.ttl_config.get("default_ttl")

                    if ttl_minutes is not None:
                        ttl_seconds = int(ttl_minutes * 60)
                        if self.cluster_mode:
                            for key in refresh_keys:
                                ttl = self._redis.ttl(key)
                                if ttl > 0:
                                    self._redis.expire(key, ttl_seconds)
                        else:
                            pipeline = self._redis.pipeline(transaction=True)
                            for key in refresh_keys:
                                # Only refresh TTL if the key exists and has a TTL
                                ttl = self._redis.ttl(key)
                                if ttl > 0:  # Only refresh if key exists and has TTL
                                    pipeline.expire(key, ttl_seconds)
                            if pipeline.command_stack:
                                pipeline.execute()

                results[idx] = items
            else:
                # Regular search
                # Create a query with LIMIT and OFFSET parameters
                query = Query(query_str).paging(offset, limit)

                # Execute search with limit and offset applied by Redis
                res = self.store_index.search(query)
                items = []
                refresh_keys = []  # Track keys that need TTL refreshed

                for doc in res.docs:
                    data = json.loads(doc.json)
                    # Apply value filters
                    if op.filter:
                        matches = True
                        value = data.get("value", {})
                        for key, expected in op.filter.items():
                            actual = value.get(key)
                            if isinstance(expected, list):
                                if actual not in expected:
                                    matches = False
                                    break
                            elif actual != expected:
                                matches = False
                                break
                        if not matches:
                            continue

                    # If refresh_ttl is true, add the key to refresh list
                    if op.refresh_ttl:
                        refresh_keys.append(doc.id)
                        # Also find associated vector keys with same ID
                        doc_id = doc.id.split(":")[-1]
                        vector_key = (
                            f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                        )
                        refresh_keys.append(vector_key)

                    items.append(
                        _row_to_search_item(
                            _decode_ns(data["prefix"]),
                            data,
                            deserialize_fn=self._deserialize_value,
                        )
                    )

                # Refresh TTL if requested
                if op.refresh_ttl and refresh_keys and self.ttl_config:
                    # Get default TTL from config
                    ttl_minutes = None
                    if "default_ttl" in self.ttl_config:
                        ttl_minutes = self.ttl_config.get("default_ttl")

                    if ttl_minutes is not None:
                        ttl_seconds = int(ttl_minutes * 60)
                        if self.cluster_mode:
                            for key in refresh_keys:
                                ttl = self._redis.ttl(key)
                                if ttl > 0:
                                    self._redis.expire(key, ttl_seconds)
                        else:
                            pipeline = self._redis.pipeline(transaction=True)
                            for key in refresh_keys:
                                # Only refresh TTL if the key exists and has a TTL
                                ttl = self._redis.ttl(key)
                                if ttl > 0:  # Only refresh if key exists and has TTL
                                    pipeline.expire(key, ttl_seconds)
                            if pipeline.command_stack:
                                pipeline.execute()

                results[idx] = items

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch of operations asynchronously."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)


__all__ = ["AsyncRedisStore", "RedisStore"]
