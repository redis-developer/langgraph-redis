import base64
import logging
from typing import Any, Union

import orjson
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class JsonPlusRedisSerializer(JsonPlusSerializer):
    """Redis-optimized serializer using orjson for faster JSON processing.

    This serializer handles the conversion of LangChain objects (including messages)
    to and from their serialized format. It specifically addresses the MESSAGE_COERCION_FAILURE
    issue by ensuring that LangChain message objects stored in their serialized format
    (with 'lc', 'type', 'constructor' fields) are properly reconstructed as message objects
    rather than being left as raw dictionaries.

    The serialized format for LangChain objects looks like:
    {
        'lc': 1,  # LangChain version marker
        'type': 'constructor',
        'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
        'kwargs': {'content': '...', 'type': 'human', 'id': '...'}
    }

    This serializer ensures such objects are properly deserialized back to their
    original message object form (e.g., HumanMessage, AIMessage) to prevent
    downstream errors when the application expects message objects with specific
    attributes and methods.
    """

    SENTINEL_FIELDS = [
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "parent_checkpoint_id",
    ]

    def dumps(self, obj: Any) -> bytes:
        """Use orjson for serialization with LangChain object support via default handler."""
        # Use orjson with default handler for LangChain objects
        # The _default method from parent class handles LangChain serialization
        return orjson.dumps(obj, default=self._default)

    def loads(self, data: bytes) -> Any:
        """Use orjson for JSON parsing with reviver support, fallback to parent for msgpack data."""
        try:
            # Fast path: Use orjson for JSON data
            parsed = orjson.loads(data)
            # Apply reviver for LangChain objects (lc format)
            return self._revive_if_needed(parsed)
        except (orjson.JSONDecodeError, TypeError):
            # Fallback: Parent handles msgpack and other formats via loads_typed
            # Attempt to detect type and use loads_typed
            try:
                # Try loading as msgpack via parent's loads_typed
                return super().loads_typed(("msgpack", data))
            except Exception:
                # If that fails, try loading as json string
                return super().loads_typed(("json", data))

    def _revive_if_needed(self, obj: Any) -> Any:
        """Recursively apply reviver to handle LangChain serialized objects.

        This method is crucial for preventing MESSAGE_COERCION_FAILURE by ensuring
        that LangChain message objects stored in their serialized format are properly
        reconstructed. Without this, messages would remain as dictionaries with
        'lc', 'type', and 'constructor' fields, causing errors when the application
        expects actual message objects with 'role' and 'content' attributes.

        This also handles Interrupt objects that may be stored as plain dictionaries
        with 'value' and 'id' keys, reconstructing them as proper Interrupt instances
        to prevent AttributeError when accessing the 'id' attribute.

        Args:
            obj: The object to potentially revive, which may be a dict, list, or primitive.

        Returns:
            The revived object with LangChain objects properly reconstructed.
        """
        if isinstance(obj, dict):
            # Check if this is a LangChain serialized object
            if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                # Use parent's reviver method to reconstruct the object
                # This converts {'lc': 1, 'type': 'constructor', ...} back to
                # the actual LangChain object (e.g., HumanMessage, AIMessage)
                return self._reviver(obj)

            # Check if this looks like an Interrupt object stored as a plain dict
            # Interrupt objects have 'value' and 'id' keys, and possibly nothing else
            # We need to be careful not to accidentally convert other dicts
            if (
                "value" in obj
                and "id" in obj
                and len(obj) == 2
                and isinstance(obj.get("id"), str)
            ):
                # Try to reconstruct as an Interrupt object
                try:
                    from langgraph.types import Interrupt

                    return Interrupt(value=obj["value"], id=obj["id"])  # type: ignore[call-arg]
                except (ImportError, TypeError, ValueError):
                    # If we can't import or construct Interrupt, fall through
                    pass

            # Recursively process nested dicts
            return {k: self._revive_if_needed(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process lists (e.g., lists of messages)
            return [self._revive_if_needed(item) for item in obj]
        else:
            # Return primitives as-is
            return obj

    def dumps_typed(self, obj: Any) -> tuple[str, str]:  # type: ignore[override]
        if isinstance(obj, (bytes, bytearray)):
            return "base64", base64.b64encode(obj).decode("utf-8")
        else:
            # All objects should be JSON-serializable (LangChain objects are pre-serialized)
            return "json", self.dumps(obj).decode("utf-8")

    def loads_typed(self, data: tuple[str, Union[str, bytes]]) -> Any:
        type_, data_ = data
        if type_ == "base64":
            decoded = base64.b64decode(
                data_ if isinstance(data_, bytes) else data_.encode()
            )
            return decoded
        elif type_ == "json":
            data_bytes = data_ if isinstance(data_, bytes) else data_.encode()
            return self.loads(data_bytes)
