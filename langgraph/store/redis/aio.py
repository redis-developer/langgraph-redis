from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, AsyncIterator, Iterable, Optional, Sequence, cast

from langgraph.store.base import (
    GetOp,
    IndexConfig,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.query import Query
from redisvl.index import AsyncSearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.utils.token_escaper import TokenEscaper
from ulid import ULID

from langgraph.store.redis.base import (
    REDIS_KEY_SEPARATOR,
    STORE_PREFIX,
    STORE_VECTOR_PREFIX,
    BaseRedisStore,
    RedisDocument,
    _decode_ns,
    _ensure_string_or_literal,
    _group_ops,
    _namespace_to_text,
    _row_to_item,
    _row_to_search_item,
    logger,
)

from .token_unescaper import TokenUnescaper

_token_escaper = TokenEscaper()
_token_unescaper = TokenUnescaper()
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster


class AsyncRedisStore(
    BaseRedisStore[AsyncRedis, AsyncSearchIndex], AsyncBatchedBaseStore
):
    """Async Redis store with optional vector search."""

    store_index: AsyncSearchIndex
    vector_index: AsyncSearchIndex
    _owns_its_client: bool
    supports_ttl: bool = True
    # Use a different name to avoid conflicting with the base class attribute
    _async_ttl_stop_event: asyncio.Event | None = None
    _ttl_sweeper_task: asyncio.Task | None = None
    ttl_config: Optional[TTLConfig] = None
    # Whether to assume the Redis server is a cluster; None triggers auto-detection
    cluster_mode: Optional[bool] = None

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[AsyncRedis] = None,
        index: Optional[IndexConfig] = None,
        connection_args: Optional[dict[str, Any]] = None,
        ttl: Optional[TTLConfig] = None,
        cluster_mode: Optional[bool] = None,
        store_prefix: str = STORE_PREFIX,
        vector_prefix: str = STORE_VECTOR_PREFIX,
    ) -> None:
        """Initialize store with Redis connection and optional index config."""
        if redis_url is None and redis_client is None:
            raise ValueError("Either redis_url or redis_client must be provided")

        # Initialize base classes
        AsyncBatchedBaseStore.__init__(self)

        # Configure client first
        self.configure_client(
            redis_url=redis_url,
            redis_client=redis_client,
            connection_args=connection_args or {},
        )

        # Validate and store cluster_mode; None means auto-detect later
        if cluster_mode is not None and not isinstance(cluster_mode, bool):
            raise TypeError("cluster_mode must be a boolean or None")

        # Initialize BaseRedisStore with prefix parameters
        BaseRedisStore.__init__(
            self,
            conn=self._redis,
            index=index,
            ttl=ttl,
            cluster_mode=cluster_mode,
            store_prefix=store_prefix,
            vector_prefix=vector_prefix,
        )

        # Update store_index to async version
        from copy import deepcopy

        store_schema = {
            "index": {
                "name": self.store_prefix,
                "prefix": self.store_prefix + REDIS_KEY_SEPARATOR,
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
        }
        self.store_index = AsyncSearchIndex.from_dict(
            store_schema, redis_client=self._redis
        )

        # Configure vector index if needed
        if self.index_config:
            # Create custom vector schema with instance prefix
            vector_schema = {
                "index": {
                    "name": self.vector_prefix,
                    "prefix": self.vector_prefix + REDIS_KEY_SEPARATOR,
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
            }
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
                    "distance_metric": {
                        "cosine": "COSINE",
                        "inner_product": "IP",
                        "l2": "L2",
                    }[
                        _ensure_string_or_literal(
                            self.index_config.get("distance_type", "cosine")
                        )
                    ],
                }

                # Apply any additional vector type config
                if "ann_index_config" in self.index_config:
                    vector_field["attrs"].update(self.index_config["ann_index_config"])

            try:
                self.vector_index = AsyncSearchIndex.from_dict(
                    vector_schema, redis_client=self._redis
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create vector index with schema: {vector_schema}. Error: {str(e)}"
                ) from e

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None

        # Use direct AsyncRedis.from_url to avoid the deprecated get_async_redis_connection
        if redis_client is None:
            if not redis_url:
                redis_url = os.environ.get("REDIS_URL")
                if not redis_url:
                    raise ValueError("REDIS_URL env var not set")
            self._redis = AsyncRedis.from_url(redis_url, **(connection_args or {}))
        else:
            self._redis = redis_client

    async def setup(self) -> None:
        """Initialize store indices."""
        # Handle embeddings in same way as sync store
        if self.index_config:
            self.embeddings = ensure_embeddings(
                self.index_config.get("embed"),
            )

        # Auto-detect cluster mode if not explicitly set
        if self.cluster_mode is None:
            await self._detect_cluster_mode()
        else:
            logger.info(
                f"Redis cluster_mode explicitly set to {self.cluster_mode}, skipping detection."
            )

        # Create indices in Redis
        await self.store_index.create(overwrite=False)
        if self.index_config:
            await self.vector_index.create(overwrite=False)

    async def _detect_cluster_mode(self) -> None:
        """Detect if the Redis client is a cluster client by inspecting its class."""
        # Determine cluster mode based on client class
        if isinstance(self._redis, AsyncRedisCluster):
            self.cluster_mode = True
            logger.info("Redis cluster client detected for AsyncRedisStore.")
        else:
            self.cluster_mode = False
            logger.info("Redis standalone client detected for AsyncRedisStore.")

    # This can't be properly typed due to covariance issues with async methods
    async def _apply_ttl_to_keys(
        self,
        main_key: str,
        related_keys: Optional[list[str]] = None,
        ttl_minutes: Optional[float] = None,
    ) -> Any:
        """Apply Redis native TTL to keys asynchronously.

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
            if self.cluster_mode:
                await self._redis.expire(main_key, ttl_seconds)
                if related_keys:
                    for key in related_keys:
                        await self._redis.expire(key, ttl_seconds)
            else:
                pipeline = self._redis.pipeline(transaction=True)

                # Set TTL for main key
                pipeline.expire(main_key, ttl_seconds)

                # Set TTL for related keys
                if related_keys:  # Check if related_keys is not None
                    for key in related_keys:
                        pipeline.expire(key, ttl_seconds)

                await pipeline.execute()

    # This can't be properly typed due to covariance issues with async methods
    async def sweep_ttl(self) -> int:  # type: ignore[override]
        """Clean up any remaining expired items.

        This is not needed with Redis native TTL, but kept for API compatibility.
        Redis automatically removes expired keys.

        Returns:
            int: Always returns 0 as Redis handles expiration automatically
        """
        return 0

    # This can't be properly typed due to covariance issues with async methods
    async def start_ttl_sweeper(  # type: ignore[override]
        self, _sweep_interval_minutes: Optional[int] = None
    ) -> None:
        """Start TTL sweeper.

        This is a no-op with Redis native TTL, but kept for API compatibility.
        Redis automatically removes expired keys.

        Args:
            _sweep_interval_minutes: Ignored parameter, kept for API compatibility
        """
        # No-op: Redis handles TTL expiration automatically
        pass

    # This can't be properly typed due to covariance issues with async methods
    async def stop_ttl_sweeper(self, _timeout: Optional[float] = None) -> bool:  # type: ignore[override]
        """Stop TTL sweeper.

        This is a no-op with Redis native TTL, but kept for API compatibility.

        Args:
            _timeout: Ignored parameter, kept for API compatibility

        Returns:
            bool: Always True as there's no sweeper to stop
        """
        # No-op: Redis handles TTL expiration automatically
        return True

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
        store_prefix: str = STORE_PREFIX,
        vector_prefix: str = STORE_VECTOR_PREFIX,
    ) -> AsyncIterator[AsyncRedisStore]:
        """Create store from Redis connection string."""
        async with cls(
            redis_url=conn_string,
            index=index,
            ttl=ttl,
            store_prefix=store_prefix,
            vector_prefix=vector_prefix,
        ) as store:
            await store.setup()
            # Set client information after setup
            await store.aset_client_info()
            yield store

    def create_indexes(self) -> None:
        """Create async indices."""
        self.store_index = AsyncSearchIndex.from_dict(
            self.SCHEMAS[0], redis_client=self._redis
        )
        if self.index_config:
            self.vector_index = AsyncSearchIndex.from_dict(
                self.SCHEMAS[1], redis_client=self._redis
            )

    async def __aenter__(self) -> AsyncRedisStore:
        """Async context manager enter."""
        # Client info was already set in __init__,
        # but we'll set it again here to be consistent with checkpoint code
        await self.aset_client_info()
        return self

    async def __aexit__(
        self,
        _exc_type: Optional[type[BaseException]] = None,
        _exc_value: Optional[BaseException] = None,
        _traceback: Optional[TracebackType] = None,
    ) -> None:
        """Async context manager exit."""
        # Cancel the background task created by AsyncBatchedBaseStore
        if hasattr(self, "_task") and self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Close Redis connections if we own them
        if self._owns_its_client:
            await self._redis.aclose()  # type: ignore[attr-defined]
            await self._redis.connection_pool.disconnect()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch of operations asynchronously."""
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        tasks = []

        if GetOp in grouped_ops:
            tasks.append(
                self._batch_get_ops(
                    list(cast(list[tuple[int, GetOp]], grouped_ops[GetOp])), results
                )
            )

        if PutOp in grouped_ops:
            tasks.append(
                self._batch_put_ops(
                    list(cast(list[tuple[int, PutOp]], grouped_ops[PutOp]))
                )
            )

        if SearchOp in grouped_ops:
            tasks.append(
                self._batch_search_ops(
                    list(cast(list[tuple[int, SearchOp]], grouped_ops[SearchOp])),
                    results,
                )
            )

        if ListNamespacesOp in grouped_ops:
            tasks.append(
                self._batch_list_namespaces_ops(
                    list(
                        cast(
                            list[tuple[int, ListNamespacesOp]],
                            grouped_ops[ListNamespacesOp],
                        )
                    ),
                    results,
                )
            )

        await asyncio.gather(*tasks)

        return results

    def batch(self: AsyncRedisStore, ops: Iterable[Op]) -> list[Result]:
        """Execute batch of operations synchronously.

        Args:
            ops: Operations to execute in batch

        Returns:
            Results from batch execution

        Raises:
            asyncio.InvalidStateError: If called from the main event loop
        """
        try:
            if asyncio.get_running_loop():
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncRedisStore are only allowed from a "
                    "different thread. From the main thread, use the async interface."
                    "For example, use `await store.abatch(...)` or `await "
                    "store.aget(...)`"
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.abatch(ops), asyncio.get_event_loop()
        ).result()

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        """Execute GET operations in batch asynchronously."""
        refresh_keys_by_idx: dict[int, list[str]] = (
            {}
        )  # Track keys that need TTL refreshed by op index

        for query, _, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            res = await self.store_index.search(Query(query))
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
                            ttl = await self._redis.ttl(key)
                            if ttl > 0:
                                await self._redis.expire(key, ttl_seconds)
                else:
                    # In cluster mode, we must use transaction=False # Comment no longer relevant
                    pipeline = self._redis.pipeline(
                        transaction=True
                    )  # Assuming non-cluster or single node for now

                    for keys in refresh_keys_by_idx.values():
                        for key in keys:
                            # Only refresh TTL if the key exists and has a TTL
                            ttl = await self._redis.ttl(key)
                            if ttl > 0:  # Only refresh if key exists and has TTL
                                pipeline.expire(key, ttl_seconds)

                    await pipeline.execute()

    async def _aprepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[
        list[RedisDocument], Optional[tuple[str, list[tuple[str, str, str, str]]]]
    ]:
        """Prepare queries - no Redis operations in async version."""
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
                results = await self.store_index.search(query)
                for doc in results.docs:
                    await self._redis.delete(doc.id)

        # Handle inserts
        if inserts:
            for op in inserts:
                now = int(datetime.now(timezone.utc).timestamp() * 1_000_000)

                # Handle TTL
                ttl_minutes = None
                expires_at = None
                if op.ttl is not None:
                    ttl_minutes = op.ttl
                    expires_at = int(
                        (
                            datetime.now(timezone.utc) + timedelta(minutes=op.ttl)
                        ).timestamp()
                    )

                doc = RedisDocument(
                    prefix=_namespace_to_text(op.namespace),
                    key=op.key,
                    value=self._serialize_value(op.value),
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

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> None:
        """Execute PUT operations in batch asynchronously."""
        operations, embedding_request = await self._aprepare_batch_PUT_queries(put_ops)

        # First delete any existing documents that are being updated/deleted
        for _, op in put_ops:
            namespace = _namespace_to_text(op.namespace)
            query = f"@prefix:{namespace} @key:{{{_token_escaper.escape(op.key)}}}"
            results = await self.store_index.search(query)

            if self.cluster_mode:
                for doc in results.docs:
                    await self._redis.delete(doc.id)
                if self.index_config:
                    vector_results = await self.vector_index.search(query)
                    for doc_vec in vector_results.docs:
                        await self._redis.delete(doc_vec.id)
            else:
                pipeline = self._redis.pipeline(transaction=True)
                for doc in results.docs:
                    pipeline.delete(doc.id)

                if self.index_config:
                    vector_results = await self.vector_index.search(query)
                    for doc_vec in vector_results.docs:
                        pipeline.delete(doc_vec.id)

                if (
                    pipeline.command_stack
                ):  # Check if pipeline has commands before executing
                    await pipeline.execute()

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
                # For cluster mode, load documents individually if SearchIndex.load isn't cluster-safe for batching.
                # This is a conservative approach. If redisvl's load is cluster-safe, this can be optimized.
                for i, store_doc_item in enumerate(store_docs):
                    await self.store_index.load([store_doc_item], keys=[store_keys[i]])
            else:
                await self.store_index.load(store_docs, keys=store_keys)

        # Handle vector embeddings with same IDs
        if embedding_request and self.embeddings:
            _, text_params = embedding_request
            vectors = await self.embeddings.aembed_documents(
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
                redis_vector_key = f"{self.vector_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                vector_keys.append(redis_vector_key)

                # Add this vector key to the related keys list for TTL
                main_key = f"{self.store_prefix}{REDIS_KEY_SEPARATOR}{doc_id}"
                if main_key in ttl_tracking:
                    ttl_tracking[main_key][0].append(redis_vector_key)

            if vector_docs:
                if self.cluster_mode:
                    # Similar to store_docs, load vector docs individually in cluster mode as a precaution.
                    for i, vector_doc_item in enumerate(vector_docs):
                        await self.vector_index.load(
                            [vector_doc_item], keys=[vector_keys[i]]
                        )
                else:
                    await self.vector_index.load(vector_docs, keys=vector_keys)

        # Now apply TTLs after all documents are loaded
        for main_key, (related_keys, ttl_minutes) in ttl_tracking.items():
            await self._apply_ttl_to_keys(main_key, related_keys, ttl_minutes)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        """Execute search operations in batch asynchronously."""
        queries, embedding_requests = self._get_batch_search_queries(search_ops)

        # Handle vector search
        query_vectors = {}
        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents(
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
                vector_query.paging(offset, limit)
                vector_results_docs = await self.vector_index.query(vector_query)

                # Get matching store docs
                result_map = {}

                if self.cluster_mode:
                    store_docs = []
                    for doc in vector_results_docs:
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
                            store_doc_item = await self._redis.json().get(store_key)  # type: ignore[misc]
                            store_docs.append(store_doc_item)
                    store_docs_raw = store_docs
                else:
                    pipeline = self._redis.pipeline(transaction=False)
                    for (
                        doc
                    ) in (
                        vector_results_docs
                    ):  # doc_vr is now an individual doc from the list
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
                            pipeline.json().get(store_key)
                    store_docs_raw = await pipeline.execute()

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
                        # Ensure store_doc is a dictionary before trying to assign to it
                        if not isinstance(store_doc, dict):
                            try:
                                store_doc = json.loads(
                                    store_doc
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
                                ttl = await self._redis.ttl(key)
                                if ttl > 0:
                                    await self._redis.expire(key, ttl_seconds)
                        else:
                            pipeline = self._redis.pipeline(transaction=True)
                            for key in refresh_keys:
                                # Only refresh TTL if the key exists and has a TTL
                                ttl = await self._redis.ttl(key)
                                if ttl > 0:  # Only refresh if key exists and has TTL
                                    pipeline.expire(key, ttl_seconds)
                            if pipeline.command_stack:
                                await pipeline.execute()

                results[idx] = items

            else:
                # Regular search
                # Create a query with LIMIT and OFFSET parameters
                query = Query(query_str).paging(offset, limit)

                # Execute search with limit and offset applied by Redis
                res = await self.store_index.search(query)
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
                                ttl = await self._redis.ttl(key)
                                if ttl > 0:
                                    await self._redis.expire(key, ttl_seconds)
                        else:
                            pipeline = self._redis.pipeline(transaction=True)
                            for key in refresh_keys:
                                # Only refresh TTL if the key exists and has a TTL
                                ttl = await self._redis.ttl(key)
                                if ttl > 0:  # Only refresh if key exists and has TTL
                                    pipeline.expire(key, ttl_seconds)
                            if pipeline.command_stack:
                                await pipeline.execute()

                results[idx] = items

    async def _batch_list_namespaces_ops(
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
            res = await self.store_index.search(query)

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

    # We don't need _run_background_tasks anymore as AsyncBatchedBaseStore provides this
