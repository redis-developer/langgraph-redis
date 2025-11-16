import logging
from typing import Any

import orjson
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class JsonPlusRedisSerializer(JsonPlusSerializer):
    """Redis-optimized serializer using orjson for JSON processing.

    Redis requires JSON-serializable data (not msgpack), so this serializer:
    1. Uses orjson for fast JSON serialization
    2. Handles LangChain objects by encoding them in the LC constructor format
    3. Handles Interrupt objects with custom serialization/deserialization
    4. Applies parent's _reviver for security-checked object reconstruction

    In checkpoint 3.0, the serializer API uses only dumps_typed/loads_typed
    with tuple[str, bytes] signatures (changed from tuple[str, str] in 2.x).
    """

    SENTINEL_FIELDS = [
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "parent_checkpoint_id",
    ]

    def _default_handler(self, obj: Any) -> Any:
        """Custom JSON encoder for objects that orjson can't serialize.

        This handles LangChain objects by delegating to the parent's
        _encode_constructor_args method which creates the LC format.
        """
        # Try to encode using parent's constructor args encoder
        # This creates the {"lc": 2, "type": "constructor", ...} format
        try:
            # _encode_constructor_args needs the CLASS, not the instance
            # For LangChain objects with to_json(), use that data for kwargs
            if hasattr(obj, "to_json"):
                json_dict = obj.to_json()
                if isinstance(json_dict, dict) and "lc" in json_dict:
                    # Already in LC format, return as-is
                    return json_dict

            # For other objects, encode with constructor args
            # Pass the class and the instance's __dict__ as kwargs
            return self._encode_constructor_args(
                type(obj),
                kwargs=obj.__dict__ if hasattr(obj, "__dict__") else {}
            )
        except Exception:
            # For types we can't handle, raise TypeError
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize using orjson for JSON.

        Returns:
            tuple[str, bytes]: Type identifier and serialized bytes
        """
        if isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", bytes(obj)
        elif obj is None:
            return "null", b""
        else:
            # Use orjson for JSON serialization with custom default handler
            json_bytes = orjson.dumps(obj, default=self._default_handler)
            return "json", json_bytes

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize with custom revival for LangChain/LangGraph objects.

        Args:
            data: Tuple of (type_str, data_bytes)

        Returns:
            Deserialized object with proper revival of LangChain/LangGraph types
        """
        type_, data_bytes = data

        if type_ == "null":
            return None
        elif type_ == "bytes":
            return data_bytes
        elif type_ == "bytearray":
            return bytearray(data_bytes)
        elif type_ == "json":
            # Use orjson for parsing, then apply our custom revival
            parsed = orjson.loads(data_bytes)
            return self._revive_if_needed(parsed)
        elif type_ == "msgpack":
            # Handle backward compatibility with old checkpoints that used msgpack
            return super().loads_typed(data)
        else:
            # Unknown type, try parent
            return super().loads_typed(data)

    def _revive_if_needed(self, obj: Any) -> Any:
        """Recursively apply reviver to handle LangChain and LangGraph serialized objects.

        This method is crucial for preventing MESSAGE_COERCION_FAILURE by ensuring
        that LangChain message objects stored in their serialized format are properly
        reconstructed. Without this, messages would remain as dictionaries with
        'lc', 'type', and 'constructor' fields, causing errors when the application
        expects actual message objects with 'role' and 'content' attributes.

        It also handles LangGraph Interrupt objects which serialize to {"value": ..., "resumable": ..., "ns": ..., "when": ...}
        and must be reconstructed to prevent AttributeError when accessing Interrupt attributes.

        Args:
            obj: The object to potentially revive, which may be a dict, list, or primitive.

        Returns:
            The revived object with LangChain/LangGraph objects properly reconstructed.
        """
        if isinstance(obj, dict):
            # Check if this is a LangChain serialized object
            if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                # Use parent's reviver method to reconstruct the object
                # This converts {'lc': 1, 'type': 'constructor', ...} back to
                # the actual LangChain object (e.g., HumanMessage, AIMessage)
                return self._reviver(obj)

            # Check if this is a serialized Interrupt object
            # Interrupt objects serialize to {"value": ..., "resumable": ..., "ns": ..., "when": ...}
            # This must be done before recursively processing to avoid losing the structure
            if (
                "value" in obj
                and "resumable" in obj
                and "when" in obj
                and len(obj) == 4
                and isinstance(obj.get("resumable"), bool)
            ):
                # Try to reconstruct as an Interrupt object
                try:
                    from langgraph.types import Interrupt

                    return Interrupt(
                        value=self._revive_if_needed(obj["value"]),
                        resumable=obj["resumable"],
                        ns=obj["ns"],
                        when=obj["when"],
                    )
                except (ImportError, TypeError, ValueError) as e:
                    # If we can't import or construct Interrupt, log and fall through
                    logger.debug(
                        "Failed to deserialize Interrupt object: %s", e, exc_info=True
                    )

            # Recursively process nested dicts
            return {k: self._revive_if_needed(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process lists (e.g., lists of messages)
            return [self._revive_if_needed(item) for item in obj]
        else:
            # Return primitives as-is
            return obj
