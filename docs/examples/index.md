# Examples

23 Jupyter notebooks demonstrating how to use `langgraph-checkpoint-redis` for
persistent LangGraph workflows. Each notebook is self-contained and uses Redis
as the backing store for checkpoints, memory, and middleware.

## Checkpoints

Persist graph state across invocations using Redis-backed checkpoint savers,
including cross-thread sharing, functional APIs, and subgraph patterns.

| Notebook | Description |
|----------|-------------|
| [Cross-Thread Persistence](checkpoints/cross-thread-persistence) | Cross-thread persistence with Redis stores |
| [Cross-Thread Persistence (Functional)](checkpoints/cross-thread-persistence-functional) | Cross-thread persistence using the functional API |
| [Functional Persistence](checkpoints/persistence-functional) | Functional persistence patterns |
| [Subgraph Persistence](checkpoints/subgraph-persistence) | Subgraph checkpoint persistence |
| [Managing State in Subgraphs](checkpoints/subgraphs-manage-state) | Managing state in subgraphs |

## Human-in-the-Loop

Interrupt graph execution for human review, edit state mid-run, and navigate
checkpoint history with time travel.

| Notebook | Description |
|----------|-------------|
| [Breakpoints](human_in_the_loop/breakpoints) | Adding breakpoints for human review |
| [Dynamic Breakpoints](human_in_the_loop/dynamic_breakpoints) | Dynamic breakpoints based on conditions |
| [Edit Graph State](human_in_the_loop/edit-graph-state) | Editing graph state during execution |
| [Review Tool Calls](human_in_the_loop/review-tool-calls) | Reviewing tool calls before execution |
| [Review Tool Calls (OpenAI)](human_in_the_loop/review-tool-calls-openai) | Reviewing tool calls with the OpenAI Responses API |
| [Time Travel](human_in_the_loop/time-travel) | Time travel to previous checkpoints |
| [Wait for User Input](human_in_the_loop/wait-user-input) | Waiting for user input during execution |

## Memory

Manage conversation history, summarize past messages, and perform semantic
search over stored memories using `RedisStore`.

| Notebook | Description |
|----------|-------------|
| [Summarize Conversation History](memory/add-summary-conversation-history) | Summarizing conversation history |
| [Delete Messages](memory/delete-messages) | Deleting messages from state |
| [Manage Conversation History](memory/manage-conversation-history) | Managing conversation history length |
| [Semantic Search](memory/semantic-search) | Semantic search over stored memories |

## Middleware

Add caching, conversation memory, and semantic routing to LangGraph agents
using the Redis middleware layer.

| Notebook | Description |
|----------|-------------|
| [Semantic Cache](middleware/middleware_semantic_cache) | LLM response caching with semantic matching |
| [Tool Result Caching](middleware/middleware_tool_caching) | Tool result caching with metadata control |
| [Conversation Memory](middleware/middleware_conversation_memory) | Semantic conversation history retrieval |
| [Middleware Composition](middleware/middleware_composition) | Combining middleware with shared connections |

## ReAct Agents

Build ReAct agents with Redis-backed persistence, human-in-the-loop review,
and long-term memory.

| Notebook | Description |
|----------|-------------|
| [ReAct Agent with HITL](react_agent/create-react-agent-hitl) | ReAct agent with human-in-the-loop |
| [ReAct Agent with Message History](react_agent/create-react-agent-manage-message-history) | ReAct agent with message history management |
| [ReAct Agent with Memory](react_agent/create-react-agent-memory) | ReAct agent with persistent memory |

```{toctree}
:hidden:
:maxdepth: 1
:glob:

checkpoints/*
human_in_the_loop/*
memory/*
middleware/*
react_agent/*
```
