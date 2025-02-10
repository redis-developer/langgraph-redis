"""Synchronous Redis store implementation."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
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
)
from redis import Redis
from redis.commands.search.query import Query
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.utils.token_escaper import TokenEscaper

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

    def __init__(
        self,
        conn: Redis,
        *,
        index: Optional[IndexConfig] = None,
    ) -> None:
        BaseStore.__init__(self)
        BaseRedisStore.__init__(self, conn, index=index)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[IndexConfig] = None,
    ) -> Iterator[RedisStore]:
        """Create store from Redis connection string."""
        client = None
        try:
            client = RedisConnectionFactory.get_redis_connection(conn_string)
            yield cls(client, index=index)
        finally:
            if client:
                client.close()
                client.connection_pool.disconnect()

    def setup(self) -> None:
        """Initialize store indices."""
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
        for query, _, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            res = self.store_index.search(Query(query))
            # Parse JSON from each document
            key_to_row = {
                json.loads(doc.json)["key"]: json.loads(doc.json) for doc in res.docs
            }
            for idx, key in items:
                if key in key_to_row:
                    results[idx] = _row_to_item(namespace, key_to_row[key])

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
            for doc in results.docs:
                self._redis.delete(doc.id)
            if self.index_config:
                results = self.vector_index.search(query)
                for doc in results.docs:
                    self._redis.delete(doc.id)

        # Now handle new document creation
        doc_ids: dict[tuple[str, str], str] = {}
        store_docs: list[RedisDocument] = []
        store_keys: list[str] = []

        # Generate IDs for PUT operations
        for _, op in put_ops:
            if op.value is not None:
                generated_doc_id = uuid.uuid4().hex
                namespace = _namespace_to_text(op.namespace)
                doc_ids[(namespace, op.key)] = generated_doc_id

        # Load store docs with explicit keys
        for doc in operations:
            store_key = (doc["prefix"], doc["key"])
            doc_id = doc_ids[store_key]
            store_docs.append(doc)
            store_keys.append(f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id}")
        if store_docs:
            self.store_index.load(store_docs, keys=store_keys)

        # Handle vector embeddings with same IDs
        if embedding_request and self.embeddings:
            _, text_params = embedding_request
            vectors = self.embeddings.embed_documents(
                [text for _, _, _, text in text_params]
            )

            vector_docs: list[dict[str, Any]] = []
            vector_keys: list[str] = []
            for (ns, key, path, _), vector in zip(text_params, vectors):
                vector_key: tuple[str, str] = (ns, key)
                doc_id = doc_ids[vector_key]
                vector_docs.append(
                    {
                        "prefix": ns,
                        "key": key,
                        "field_name": path,
                        "embedding": (
                            vector.tolist() if hasattr(vector, "tolist") else vector
                        ),
                        "created_at": datetime.now(timezone.utc).timestamp(),
                        "updated_at": datetime.now(timezone.utc).timestamp(),
                    }
                )
                vector_keys.append(
                    f"{STORE_VECTOR_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id}"
                )
            if vector_docs:
                self.vector_index.load(vector_docs, keys=vector_keys)

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
        for (idx, op), (query_str, params) in zip(search_ops, queries):
            if op.query and idx in query_vectors:
                # Vector similarity search
                vector = query_vectors[idx]
                vector_query = VectorQuery(
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    vector_field_name="embedding",
                    filter_expression=f"@prefix:{_namespace_to_text(op.namespace_prefix)}*",
                    return_fields=["prefix", "key", "vector_distance"],
                    num_results=op.limit,
                )
                vector_results = self.vector_index.query(vector_query)

                # Get matching store docs in pipeline
                pipe = self._redis.pipeline()
                result_map = {}  # Map store key to vector result with distances

                for doc in vector_results:
                    doc_id = (
                        doc.get("id")
                        if isinstance(doc, dict)
                        else getattr(doc, "id", None)
                    )
                    if doc_id:
                        store_key = f"{STORE_PREFIX}{REDIS_KEY_SEPARATOR}{doc_id.split(':')[1]}"  # Convert vector:ID to store:ID
                        result_map[store_key] = doc
                        pipe.json().get(store_key)

                # Execute all lookups in one batch
                store_docs = pipe.execute()

                # Process results maintaining order and applying filters
                items = []
                for store_key, store_doc in zip(result_map.keys(), store_docs):
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
                        store_doc["vector_distance"] = dist

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

                        items.append(
                            _row_to_search_item(
                                _decode_ns(store_doc["prefix"]),
                                store_doc,
                                score=score,
                            )
                        )

                results[idx] = items
            else:
                # Regular search
                query = Query(query_str)
                # Get all potential matches for filtering
                res = self.store_index.search(query)
                items = []

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
                    items.append(_row_to_search_item(_decode_ns(data["prefix"]), data))

                # Apply pagination after filtering
                if params:
                    limit, offset = params
                    items = items[offset : offset + limit]

                results[idx] = items

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch of operations asynchronously."""
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)


__all__ = ["AsyncRedisStore", "RedisStore"]
