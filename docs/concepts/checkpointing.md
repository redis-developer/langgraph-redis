# Checkpointing

Checkpointing is LangGraph's mechanism for persisting graph state. After every superstep of graph execution, the checkpoint saver writes the current state to durable storage. This enables time-travel debugging, human-in-the-loop workflows, error recovery, and multi-turn conversations.

`langgraph-checkpoint-redis` implements the LangGraph `BaseCheckpointSaver` interface using Redis as the storage backend.

## Why Checkpointing Matters

Without checkpointing, graph state exists only in memory for the duration of a single invocation. Checkpointing solves several problems:

- **Conversation continuity** -- A user can return to a conversation hours later and resume from where they left off.
- **Human-in-the-loop** -- The graph can pause at an interrupt point, wait for human approval, and resume from the saved state.
- **Fault tolerance** -- If a process crashes mid-execution, the graph can restart from the last committed checkpoint.
- **Debugging** -- You can inspect the full history of state transitions for a given thread.

## Threads and Namespaces

Every checkpoint is associated with a **thread** and a **namespace**:

- **Thread ID** -- Identifies a single conversation or execution session. All checkpoints for one conversation share the same thread ID.
- **Checkpoint namespace** -- Distinguishes between parent graphs and subgraphs within the same thread. The root graph uses an empty namespace; subgraphs use a hierarchical namespace like `"subgraph:node_name"`.

Together, `thread_id` and `checkpoint_ns` form the scope for checkpoint queries. When you call `graph.invoke(input, config)`, the `thread_id` in the config determines which checkpoint history to use.

```python
config = {"configurable": {"thread_id": "user-123-session-1"}}
result = graph.invoke({"messages": [("human", "Hello")]}, config)
```

## Checkpoint Lifecycle

### Writing

After each superstep, LangGraph calls `put()` on the checkpoint saver with:

- **Checkpoint** -- The serialized graph state, including a unique `checkpoint_id` (ULID-based for chronological ordering).
- **Metadata** -- Source (`"input"` or `"loop"`), step number, and write details.
- **Channel values** -- The current values of all graph channels, stored as separate blobs for efficient partial reads.
- **Pending writes** -- Any writes from the current step that have not yet been committed to channel values.

The saver stores these as JSON documents in Redis with RediSearch indexes for efficient querying.

### Reading

When a graph resumes, it calls `get_tuple()` to retrieve the latest (or a specific) checkpoint. The saver:

1. Queries the checkpoint index for the matching `thread_id` and `checkpoint_ns`.
2. Fetches the checkpoint document, channel blobs, and pending writes.
3. Deserializes and returns a `CheckpointTuple` containing the full state.

### Listing

`list()` returns an iterator over all checkpoints for a given thread, ordered by recency. This supports time-travel: you can retrieve any prior state by specifying a `checkpoint_id` in the config.

## Channel Values and Blobs

Channel values are the per-channel state of the graph (e.g., the `messages` list in a chat agent). Rather than storing all channel values inside the checkpoint document, the saver writes each channel as a separate **blob** keyed by channel name and version:

```
checkpoint_blob:{thread_id}:{namespace}:{channel}:{version}
```

This design has two benefits:

- **Deduplication** -- If a channel value does not change between supersteps, the blob is not rewritten. The checkpoint simply references the same version.
- **Partial reads** -- When loading a checkpoint, only the blobs for channels that have changed need to be fetched.

## Pending Writes

Pending writes capture the outputs of individual tasks within a superstep before they are committed to channel values. They are stored as separate documents:

```
checkpoint_write:{thread_id}:{namespace}:{checkpoint_id}:{task_id}
```

Pending writes enable human-in-the-loop workflows. When a graph hits an interrupt, the writes from the interrupted step are saved but not committed. After human approval, the graph can resume and apply the pending writes.

## Full vs Shallow Checkpointers

The library provides two checkpointer variants:

### Full Checkpointer (`RedisSaver` / `AsyncRedisSaver`)

Stores the complete checkpoint history for every thread. You can list all prior checkpoints, retrieve any historical state, and traverse the parent chain. This is the default and recommended choice for most applications.

### Shallow Checkpointer (`ShallowRedisSaver` / `AsyncShallowRedisSaver`)

Stores only the most recent checkpoint per thread. Previous checkpoints are overwritten. This reduces storage usage and is appropriate when you only need the latest state (e.g., a stateless API that resumes the most recent turn).

The shallow variant includes in-memory caching of key mappings and channel versions to reduce Redis round-trips during writes.

## How Redis Stores Checkpoints

Each checkpoint is a JSON document stored via RedisJSON:

```json
{
    "thread_id": "user-123",
    "checkpoint_ns": "",
    "checkpoint_id": "01JEXAMPLE",
    "parent_checkpoint_id": "01JPARENT",
    "checkpoint": "{...serialized state...}",
    "metadata": "{...}",
    "source": "loop",
    "step": 3
}
```

A RediSearch index over these documents enables queries like "find the latest checkpoint for thread X" without scanning keys. The `redisvl` library builds and executes these queries using the schema defined in `BaseRedisSaver`.

## Usage

```python
from langgraph.checkpoint.redis import RedisSaver

with RedisSaver.from_conn_string("redis://localhost:6379") as saver:
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "thread-1"}}
    result = graph.invoke({"messages": [("human", "Hi")]}, config)

    # Resume the same thread later
    result = graph.invoke({"messages": [("human", "What did I say?")]}, config)
```
