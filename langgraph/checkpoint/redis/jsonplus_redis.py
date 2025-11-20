import dataclasses
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
        # Bytes/bytearray in nested structures require msgpack - signal to fallback
        if isinstance(obj, (bytes, bytearray)):
            raise TypeError("bytes/bytearray in nested structure - use msgpack")

        # Handle Interrupt objects with a type marker to avoid false positives
        from langgraph.types import Interrupt

        if isinstance(obj, Interrupt):
            return {
                "__interrupt__": True,
                "value": obj.value,
                "id": obj.id,
            }

        # Handle Send objects with a type marker (issue #94)
        from langgraph.types import Send

        if isinstance(obj, Send):
            return {
                "__send__": True,
                "node": obj.node,
                "arg": obj.arg,
            }

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
                type(obj), kwargs=obj.__dict__ if hasattr(obj, "__dict__") else {}
            )
        except (AttributeError, KeyError, ValueError, TypeError):
            # For types we can't handle, raise TypeError
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _preprocess_interrupts(self, obj: Any) -> Any:
        """Recursively add type markers to Interrupt and Send objects before serialization.

        This prevents false positives where user data with {value, id} fields
        could be incorrectly deserialized as Interrupt objects.

        Also handles dataclass instances to preserve type information during serialization.
        """
        from langgraph.types import Interrupt, Send

        if isinstance(obj, Interrupt):
            # Add type marker to distinguish from plain dicts
            return {
                "__interrupt__": True,
                "value": self._preprocess_interrupts(obj.value),
                "id": obj.id,
            }
        elif isinstance(obj, Send):
            # Add type marker to distinguish from plain dicts (issue #94)
            return {
                "__send__": True,
                "node": obj.node,
                "arg": self._preprocess_interrupts(obj.arg),
            }
        elif isinstance(obj, set):
            # Handle sets by converting to list for JSON serialization
            # Will be reconstructed back to set on deserialization
            return {
                "lc": 2,
                "type": "constructor",
                "id": ["builtins", "set"],
                "kwargs": {
                    "__set_items__": [self._preprocess_interrupts(item) for item in obj]
                },
            }
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            # Handle dataclass instances (like langmem's RunningSummary)
            # Convert to LangChain constructor format to preserve type information
            # Recursively process the dataclass fields
            processed_dict = {
                k: self._preprocess_interrupts(v)
                for k, v in dataclasses.asdict(obj).items()
            }
            return self._encode_constructor_args(type(obj), kwargs=processed_dict)
        elif isinstance(obj, dict):
            return {k: self._preprocess_interrupts(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            processed = [self._preprocess_interrupts(item) for item in obj]
            # Preserve tuple type
            return tuple(processed) if isinstance(obj, tuple) else processed
        else:
            return obj

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize using orjson for JSON.

        Falls back to msgpack for structures containing bytes/bytearray.

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
            try:
                # Preprocess to add type markers to Interrupt objects
                processed_obj = self._preprocess_interrupts(obj)
                # Try orjson first with custom default handler
                json_bytes = orjson.dumps(processed_obj, default=self._default_handler)
                return "json", json_bytes
            except (TypeError, orjson.JSONEncodeError):
                # Fall back to parent's msgpack serialization for bytes in nested structures
                return super().dumps_typed(obj)

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

    def _reconstruct_from_constructor(self, obj: dict[str, Any]) -> Any:
        """Reconstruct an object from LangChain constructor format.

        This handles objects that were serialized using _encode_constructor_args
        but are not LangChain objects (e.g., dataclasses, regular classes, sets).

        Args:
            obj: Dict with 'lc', 'type', 'id', and 'kwargs' keys

        Returns:
            Reconstructed object instance

        Raises:
            Exception: If object cannot be reconstructed
        """
        # Get the class from the id field
        id_parts = obj.get("id", [])
        if not id_parts or len(id_parts) < 2:
            raise ValueError(f"Invalid constructor format: {obj}")

        # Import the module and get the class
        module_path = ".".join(id_parts[:-1])
        class_name = id_parts[-1]

        try:
            import importlib

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Cannot import {class_name} from {module_path}: {e}"
            ) from e

        # Get the kwargs and recursively revive nested objects
        kwargs = obj.get("kwargs", {})
        revived_kwargs = {k: self._revive_if_needed(v) for k, v in kwargs.items()}

        # Special handling for sets
        if cls is set and "__set_items__" in revived_kwargs:
            return set(revived_kwargs["__set_items__"])

        # Reconstruct the object
        return cls(**revived_kwargs)

    def _revive_if_needed(self, obj: Any) -> Any:
        """Recursively apply reviver to handle LangChain and LangGraph serialized objects.

        This method is crucial for preventing MESSAGE_COERCION_FAILURE by ensuring
        that LangChain message objects stored in their serialized format are properly
        reconstructed. Without this, messages would remain as dictionaries with
        'lc', 'type', and 'constructor' fields, causing errors when the application
        expects actual message objects with 'role' and 'content' attributes.

        It also handles LangGraph Interrupt objects which serialize to {"value": ..., "resumable": ..., "ns": ..., "when": ...}
        and must be reconstructed to prevent AttributeError when accessing Interrupt attributes.

        Additionally, it handles dataclass objects (like langmem's RunningSummary) that are serialized
        using the LangChain constructor format but need special reconstruction logic.

        Args:
            obj: The object to potentially revive, which may be a dict, list, or primitive.

        Returns:
            The revived object with LangChain/LangGraph objects properly reconstructed.
        """
        if isinstance(obj, dict):
            # Check if this is a LangChain serialized object
            if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                # First try to use parent's reviver method to reconstruct LangChain objects
                # This converts {'lc': 1, 'type': 'constructor', ...} back to
                # the actual LangChain object (e.g., HumanMessage, AIMessage)
                revived = self._reviver(obj)

                # If reviver returns a dict unchanged, it means it couldn't reconstruct it
                # This happens with dataclasses or other non-LangChain objects
                if isinstance(revived, dict) and revived.get("lc") in (1, 2):
                    # Try to reconstruct it manually
                    try:
                        return self._reconstruct_from_constructor(obj)
                    except Exception:
                        # If reconstruction fails, fall through to recursive dict processing
                        pass
                else:
                    # Reviver successfully reconstructed the object
                    return revived

            # Check if this is a serialized Interrupt object with type marker
            # LangGraph 1.0+: Interrupt objects serialize to {"__interrupt__": True, "value": ..., "id": ...}
            # This must be done before recursively processing to avoid losing the structure
            if (
                obj.get("__interrupt__") is True
                and "value" in obj
                and "id" in obj
                and len(obj) == 3
            ):
                # Try to reconstruct as an Interrupt object
                try:
                    from langgraph.types import Interrupt

                    return Interrupt(
                        value=self._revive_if_needed(obj["value"]),
                        id=obj["id"],
                    )
                except (ImportError, TypeError, ValueError) as e:
                    # If we can't import or construct Interrupt, log and fall through
                    logger.debug(
                        "Failed to deserialize Interrupt object: %s", e, exc_info=True
                    )

            # Check if this is a serialized Send object with type marker (issue #94)
            # Send objects serialize to {"__send__": True, "node": ..., "arg": ...}
            if (
                obj.get("__send__") is True
                and "node" in obj
                and "arg" in obj
                and len(obj) == 3
            ):
                # Try to reconstruct as a Send object
                try:
                    from langgraph.types import Send

                    return Send(
                        node=obj["node"],
                        arg=self._revive_if_needed(obj["arg"]),
                    )
                except (ImportError, TypeError, ValueError) as e:
                    # If we can't import or construct Send, log and fall through
                    logger.debug(
                        "Failed to deserialize Send object: %s", e, exc_info=True
                    )

            # Recursively process nested dicts
            return {k: self._revive_if_needed(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process lists (e.g., lists of messages)
            return [self._revive_if_needed(item) for item in obj]
        else:
            # Return primitives as-is
            return obj
