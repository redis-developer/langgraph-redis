from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Num, Tag
from redisvl.redis.connection import RedisConnectionFactory

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.ashallow import AsyncShallowRedisSaver
from langgraph.checkpoint.redis.base import BaseRedisSaver
from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
from redis import Redis


class RedisSaver(BaseRedisSaver[Redis, SearchIndex]):
    """Standard Redis implementation for checkpoint saving."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            redis_client=redis_client,
            index_prefix=index_prefix,
            connection_args=connection_args,
        )

    def configure_client(
        self,
        redis_url: Optional[str] = None,
        redis_client: Optional[Redis] = None,
        connection_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the Redis client."""
        self._owns_its_client = redis_client is None

        self._redis = redis_client or RedisConnectionFactory.get_redis_connection(
            redis_url, **connection_args
        )

    def create_indexes(self) -> None:
        self.checkpoint_index = SearchIndex.from_dict(self.SCHEMAS[0])
        self.channel_index = SearchIndex.from_dict(self.SCHEMAS[1])
        self.writes_index = SearchIndex.from_dict(self.SCHEMAS[2])

        # Connect Redis client to indices
        self.checkpoint_index.set_client(self._redis)
        self.channel_index.set_client(self._redis)
        self.writes_index.set_client(self._redis)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Redis."""
        # Construct the filter expression
        filter_expression = []
        if config:
            filter_expression.append(
                Tag("thread_id") == config["configurable"]["thread_id"]
            )
            if checkpoint_ns := config["configurable"].get("checkpoint_ns"):
                filter_expression.append(Tag("checkpoint_ns") == checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                filter_expression.append(Tag("checkpoint_id") == checkpoint_id)

        if filter:
            for k, v in filter.items():
                if k == "source":
                    filter_expression.append(Tag("source") == v)
                elif k == "step":
                    filter_expression.append(Num("step") == v)
                else:
                    raise ValueError(f"Unsupported filter key: {k}")

        if before:
            filter_expression.append(Tag("checkpoint_id") < get_checkpoint_id(before))

        # Combine all filter expressions
        combined_filter = filter_expression[0] if filter_expression else "*"
        for expr in filter_expression[1:]:
            combined_filter &= expr

        # Construct the Redis query
        query = FilterQuery(
            filter_expression=combined_filter,
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "checkpoint",
                "metadata",
                "pending_sends",
                "pending_writes",
            ],
            num_results=limit or 10,
        )

        # Execute the query
        results = self.checkpoint_index.search(query)

        # Process the results
        for doc in results.docs:
            # Access fields directly using attributes
            checkpoint_ns = getattr(doc, "checkpoint_ns", "")
            checkpoint_id = getattr(doc, "checkpoint_id", "")
            thread_id = getattr(doc, "thread_id", "")
            metadata = cast(
                CheckpointMetadata, getattr(doc, "metadata", {})
            )  # Enforce type

            # Extract channel values
            channel_query = FilterQuery(
                filter_expression=(Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
            )
            channel_values = {}
            channel_results = self.channel_index.search(channel_query)
            for val in channel_results.docs:
                channel_values[getattr(val, "channel", "")] = {
                    "type": getattr(val, "type", ""),
                    "blob": getattr(val, "blob", ""),
                }

            # Load checkpoint data
            checkpoint = self._load_checkpoint(
                getattr(doc, "checkpoint", {}),
                channel_values,
                getattr(doc, "pending_sends", []),
            )

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,  # Metadata is now properly typed
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": getattr(doc, "parent_checkpoint_id", ""),
                        }
                    }
                    if hasattr(doc, "parent_checkpoint_id")
                    else None
                ),
                pending_writes=getattr(doc, "pending_writes", []),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint to Redis."""
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = configurable.pop(
            "checkpoint_id", configurable.pop("thread_ts", None)
        )

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # Store checkpoint data
        checkpoint_data = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "checkpoint": self._dump_checkpoint(copy),
            "metadata": self._dump_metadata(metadata),
            "pending_sends": [],
            "pending_writes": [],
        }

        self.checkpoint_index.load([checkpoint_data])

        # Store channel values
        blobs = self._dump_blobs(
            thread_id,
            checkpoint_ns,
            copy.get("channel_values", {}),
            new_versions,
        )

        if blobs:
            self.channel_index.load(blobs)

        return next_config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Construct the query
        query = FilterQuery(
            filter_expression=(
                (Tag("thread_id") == thread_id)
                & (Tag("checkpoint_ns") == checkpoint_ns)
                & (Tag("checkpoint_id") == checkpoint_id)
            ),
            return_fields=[
                "thread_id",
                "checkpoint_ns",
                "checkpoint_id",
                "parent_checkpoint_id",
                "checkpoint",
                "$.metadata",  # Fetch metadata using JSON path
            ],
            num_results=1,
        )

        # Execute the query
        results = self.checkpoint_index.search(query)
        if not results.docs:
            return None

        doc = results.docs[0]
        print(f"Full document retrieved: {doc.__dict__}")  # Debugging

        # Fetch and parse metadata
        raw_metadata = getattr(doc, "$.metadata", "{}")
        metadata_dict = (
            json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        )

        # Ensure metadata matches CheckpointMetadata type
        sanitized_metadata = {
            k.replace("\u0000", ""): v.replace("\u0000", "")
            if isinstance(v, str)
            else v
            for k, v in metadata_dict.items()
        }
        metadata = cast(CheckpointMetadata, sanitized_metadata)

        # Get writes for this checkpoint
        write_key = (
            f"{self.index_prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        )
        writes = self._load_writes_from_redis(write_key)

        return CheckpointTuple(
            config=config,
            checkpoint=self._load_checkpoint(getattr(doc, "checkpoint", {}), {}, []),
            metadata=metadata,
            parent_config=None,
            pending_writes=writes,
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        redis_url: Optional[str] = None,
        *,
        redis_client: Optional[Redis] = None,
        index_prefix: str = "checkpoint",
        connection_args: Optional[dict[str, Any]] = None,
    ) -> Iterator[RedisSaver]:
        """Create a new RedisSaver instance."""
        saver: Optional[RedisSaver] = None
        try:
            saver = cls(
                redis_url=redis_url,
                redis_client=redis_client,
                index_prefix=index_prefix,
                connection_args=connection_args,
            )

            yield saver
        finally:
            if saver and saver._owns_its_client:  # Ensure saver is not None
                saver._redis.close()
                saver._redis.connection_pool.disconnect()


__all__ = [
    "RedisSaver",
    "AsyncRedisSaver",
    "BaseRedisSaver",
    "ShallowRedisSaver",
    "AsyncShallowRedisSaver",
]
