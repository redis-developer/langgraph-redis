#!/usr/bin/env python3
"""Debug script: reproduce the notebook conversation memory scenario
and extract all diagnostic information.

Uses real Redis via TestContainers. No mocks, no guessing.
"""

import asyncio
import time
from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from testcontainers.redis import RedisContainer

from langgraph.middleware.redis import (
    ConversationMemoryConfig,
    ConversationMemoryMiddleware,
)


async def run_debug() -> None:
    # ── Spin up Redis ──────────────────────────────────────────────────
    print("Starting Redis container...")
    redis_container = RedisContainer("redis/redis-stack-server:latest")
    redis_container.start()
    redis_url = (
        f"redis://{redis_container.get_container_host_ip()}"
        f":{redis_container.get_exposed_port(6379)}"
    )
    print(f"Redis URL: {redis_url}\n")

    try:
        config = ConversationMemoryConfig(
            redis_url=redis_url,
            name="debug_conversation_memory",
            session_tag="user_123",
            top_k=3,
            distance_threshold=0.7,
            graceful_degradation=False,  # Surface all errors
        )

        # ── The exact notebook conversation ────────────────────────────
        turns = [
            {
                "user": "Hi! My name is Alice and I'm a software engineer.",
                "response": "Hi Alice! It's nice to meet you. How can I assist you today?",
            },
            {
                "user": "I'm really interested in machine learning and I work with Python.",
                "response": (
                    "That's great to hear! Machine learning is a fascinating field, "
                    "and Python is one of the most popular languages for working with it."
                ),
            },
            {
                "user": "What Python libraries would be most useful for me?",
                "response": "For machine learning with Python, I'd recommend scikit-learn, TensorFlow, and PyTorch.",
            },
            {
                "user": "What's my name and what do I do for work?",
                "response": "Your name is Alice, and you are a software engineer.",
            },
        ]

        async with ConversationMemoryMiddleware(config) as middleware:
            for i, turn in enumerate(turns):
                turn_num = i + 1
                query = turn["user"]
                mock_response = turn["response"]

                print(f"{'='*70}")
                print(f"TURN {turn_num}")
                print(f"{'='*70}")
                print(f"User: {query!r}\n")

                # Capture what the LLM actually sees
                seen_messages = []

                async def mock_llm(request: Any) -> ModelResponse:
                    if isinstance(request, dict):
                        msgs = request.get("messages", [])
                    else:
                        msgs = getattr(request, "messages", [])
                    seen_messages.extend(msgs)
                    return ModelResponse(
                        result=[AIMessage(content=mock_response)]
                    )

                request = {"messages": [HumanMessage(content=query)]}
                await middleware.awrap_model_call(request, mock_llm)

                # ── Report what the LLM saw ────────────────────────────
                print(f"Messages sent to LLM ({len(seen_messages)} total):")
                for j, msg in enumerate(seen_messages):
                    msg_type = type(msg).__name__
                    content = getattr(msg, "content", str(msg))
                    if isinstance(msg, SystemMessage):
                        print(f"  [{j}] {msg_type}:")
                        # Print each line of the system message indented
                        for line in content.split("\n"):
                            print(f"       {line}")
                    else:
                        preview = content[:120] + "..." if len(content) > 120 else content
                        print(f"  [{j}] {msg_type}: {preview}")

                has_context = any(isinstance(m, SystemMessage) for m in seen_messages)
                print(f"\n  Context injected: {'YES' if has_context else 'NO'}")
                print()

                time.sleep(0.5)  # Let index update

        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print("Turn 1: No context expected (first message)")
        print("Turn 2: Should have context from Turn 1 (introduction)")
        print("Turn 3: Should have ML/Python context from Turn 2")
        print("Turn 4: Should have name/work context from Turn 1")

    finally:
        redis_container.stop()
        print("\nRedis container stopped.")


if __name__ == "__main__":
    asyncio.run(run_debug())
