"""Message exporter for extracting conversation messages from checkpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

import orjson


class MessageRecipe(Protocol):
    """Protocol for message extraction recipes.

    Implement this interface to support custom message formats.
    """

    def extract(self, message: Any) -> Optional[Dict[str, Any]]:
        """Extract structured data from a message.

        Args:
            message: The message to extract data from.

        Returns:
            Dict with at least 'role' and 'content' keys, or None if message cannot be extracted.
        """
        ...


class LangChainRecipe:
    """Default recipe for extracting LangChain messages."""

    def extract(self, message: Any) -> Optional[Dict[str, Any]]:
        """Extract data from LangChain message objects."""
        try:
            from langchain_core.messages import BaseMessage

            if isinstance(message, BaseMessage):
                # Handle actual message objects
                return {
                    "role": message.__class__.__name__.replace("Message", "").lower(),
                    "content": message.content,
                    "type": message.__class__.__name__,
                    "id": getattr(message, "id", None),
                    "metadata": {
                        "name": getattr(message, "name", None),
                        "tool_calls": getattr(message, "tool_calls", None),
                        "additional_kwargs": getattr(message, "additional_kwargs", {}),
                    },
                }
        except ImportError:
            # langchain_core not available, handle as dict
            pass

        if isinstance(message, dict):
            # Handle serialized LangChain format
            if message.get("lc") and message.get("type") == "constructor":
                kwargs = message.get("kwargs", {})
                message_type = (
                    message.get("id", ["unknown"])[-1]
                    if isinstance(message.get("id"), list)
                    else "unknown"
                )
                return {
                    "role": message_type.replace("Message", "").lower(),
                    "content": kwargs.get("content", ""),
                    "type": message_type,
                    "id": kwargs.get("id"),
                    "metadata": kwargs,
                }
            # Handle simple dict format
            elif "role" in message and "content" in message:
                return message
        elif isinstance(message, str):
            # Plain string message
            return {"role": "unknown", "content": message, "type": "string"}

        return None


class MessageExporter:
    """Export messages from Redis checkpoints."""

    def __init__(
        self, redis_saver: Any, recipe: Optional[MessageRecipe] = None
    ) -> None:
        self.saver = redis_saver
        self.recipe = recipe or LangChainRecipe()

    def export(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Export messages from checkpoint data.

        Args:
            thread_id: The conversation thread ID
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            List of extracted message dictionaries
        """
        # Get checkpoint
        if checkpoint_id:
            config = {
                "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
            }
            checkpoint = self.saver.get(config)
        else:
            # Get latest checkpoint
            checkpoint_tuple = self.saver.get_tuple(
                {"configurable": {"thread_id": thread_id}}
            )
            checkpoint = checkpoint_tuple.checkpoint if checkpoint_tuple else None

        if not checkpoint:
            return []

        # Extract messages from channel_values
        messages = checkpoint.get("channel_values", {}).get("messages", [])

        extracted = []
        for msg in messages:
            extracted_msg = self.recipe.extract(msg)
            if extracted_msg:
                extracted.append(extracted_msg)

        return extracted

    def export_thread(self, thread_id: str) -> Dict[str, Any]:
        """Export all messages from all checkpoints in a thread.

        Args:
            thread_id: The conversation thread ID

        Returns:
            Dict with thread_id, messages, and export timestamp
        """
        messages = []
        seen_ids = set()

        # Get all checkpoints for thread
        for checkpoint_tuple in self.saver.list(
            {"configurable": {"thread_id": thread_id}}
        ):
            checkpoint_messages = checkpoint_tuple.checkpoint.get(
                "channel_values", {}
            ).get("messages", [])

            for msg in checkpoint_messages:
                extracted = self.recipe.extract(msg)
                if extracted:
                    # Add checkpoint metadata
                    extracted["checkpoint_id"] = checkpoint_tuple.checkpoint.get("id")
                    extracted["checkpoint_ts"] = checkpoint_tuple.checkpoint.get("ts")

                    # Deduplicate by message ID if available
                    msg_id = extracted.get("id")
                    if msg_id:
                        if msg_id in seen_ids:
                            continue
                        seen_ids.add(msg_id)

                    messages.append(extracted)

        return {
            "thread_id": thread_id,
            "messages": messages,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }
