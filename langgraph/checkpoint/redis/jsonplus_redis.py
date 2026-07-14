import dataclasses
import logging
from collections.abc import Callable, Sequence
from typing import Any

import orjson
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


def _interrupt_fields() -> set[str]:
    """Return the set of field names on the installed Interrupt dataclass.

    langgraph <=1.0.x has Interrupt(value, id).
    langgraph >=1.1.x has Interrupt(value, resumable, ns, when).
    """
    from langgraph.types import Interrupt

    return {f.name for f in dataclasses.fields(Interrupt)}


def _serialize_interrupt(obj: Any, *, preprocess: Any = None) -> dict[str, Any]:
    """Serialize an Interrupt object using whichever fields exist."""
    fields = _interrupt_fields()
    result: dict[str, Any] = {"__interrupt__": True}
    if preprocess is not None:
        result["value"] = preprocess(obj.value)
    else:
        result["value"] = obj.value
    # Persist every field the installed Interrupt has (except value, handled above)
    for field in ("id", "resumable", "ns", "when"):
        if field in fields:
            result[field] = getattr(obj, field)
    return result


def _deserialize_interrupt(obj: dict[str, Any], *, revive: Any = None) -> Any:
    """Reconstruct an Interrupt from a serialized dict, tolerating both formats."""
    from langgraph.types import Interrupt

    fields = _interrupt_fields()
    value = revive(obj["value"]) if revive is not None else obj["value"]
    kwargs: dict[str, Any] = {"value": value}
    # Pass through whichever optional fields the installed class accepts
    for field in ("id", "resumable", "ns", "when"):
        if field in fields and field in obj:
            kwargs[field] = obj[field]
    return Interrupt(**kwargs)


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

    def _encode_constructor_envelope(
        self,
        constructor: Callable[..., Any] | type[Any],
        *,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an lc:2 default-constructor envelope.

        This preserves type information for JSON values that should be revived
        by calling the target class constructor.
        """
        out: dict[str, Any] = {
            "lc": 2,
            "type": "constructor",
            "id": [*constructor.__module__.split("."), constructor.__name__],
        }
        if args is not None:
            out["args"] = args
        if kwargs is not None:
            out["kwargs"] = kwargs
        return out

    def _default_handler(self, obj: Any) -> Any:
        """Custom JSON encoder for objects that orjson can't serialize.

        This handles LangChain objects by encoding them in the LC constructor
        format that the loader can revive.
        """
        # Bytes/bytearray in nested structures require msgpack - signal to fallback
        if isinstance(obj, (bytes, bytearray)):
            raise TypeError("bytes/bytearray in nested structure - use msgpack")

        # Handle Interrupt objects with a type marker to avoid false positives
        from langgraph.types import Interrupt

        if isinstance(obj, Interrupt):
            return _serialize_interrupt(obj)

        # Handle Send objects with a type marker (issue #94)
        from langgraph.types import Send

        if isinstance(obj, Send):
            return {
                "__send__": True,
                "node": obj.node,
                "arg": obj.arg,
            }

        # Try to encode using the constructor envelope format.
        # This creates the {"lc": 2, "type": "constructor", ...} format
        try:
            # The envelope needs the CLASS, not the instance.
            # For LangChain objects with to_json(), use that data for kwargs
            if hasattr(obj, "to_json"):
                json_dict = obj.to_json()
                if isinstance(json_dict, dict) and "lc" in json_dict:
                    # Already in LC format, return as-is
                    return json_dict

            # For other objects, encode with constructor args
            # Pass the class and the instance's __dict__ as kwargs
            return self._encode_constructor_envelope(
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
            return _serialize_interrupt(obj, preprocess=self._preprocess_interrupts)
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
            return self._encode_constructor_envelope(
                set,
                args=[[self._preprocess_interrupts(item) for item in obj]],
            )
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            # Handle dataclass instances (like langmem's RunningSummary)
            # Convert to LangChain constructor format to preserve type information
            # Recursively process the dataclass fields without dataclasses.asdict(),
            # which would erase nested dataclass type information.
            processed_dict = {
                field.name: self._preprocess_interrupts(getattr(obj, field.name))
                for field in dataclasses.fields(obj)
            }
            return self._encode_constructor_envelope(type(obj), kwargs=processed_dict)
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

    def _constructor_class(self, obj: dict[str, Any]) -> type[Any]:
        """Return the class targeted by an lc constructor envelope."""
        id_parts = obj.get("id", [])
        if (
            not isinstance(id_parts, list)
            or len(id_parts) < 2
            or not all(isinstance(part, str) for part in id_parts)
        ):
            raise ValueError(f"Invalid constructor format: {obj}")

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

        return cls

    def _reconstruct_set_constructor(self, obj: dict[str, Any]) -> set[Any]:
        """Reconstruct current and legacy Redis JSON set envelopes."""
        if "args" in obj:
            args = obj["args"]
            if not isinstance(args, list) or len(args) != 1:
                raise ValueError(f"Invalid set constructor args: {obj}")
            return set(args[0])

        # Compatibility with checkpoints written by older JsonPlusRedisSerializer
        # versions that stored set items under a non-constructor kwarg.
        kwargs = obj.get("kwargs", {})
        if isinstance(kwargs, dict) and "__set_items__" in kwargs:
            return set(kwargs["__set_items__"])

        raise ValueError(f"Invalid set constructor envelope: {obj}")

    def _reconstruct_dataclass_constructor(
        self, obj: dict[str, Any], cls: type[Any]
    ) -> Any:
        """Reconstruct a dataclass from its already-revived constructor fields."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"Constructor does not target a dataclass: {obj}")

        args = obj.get("args", [])
        kwargs = obj.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError(f"Invalid dataclass constructor payload: {obj}")

        return cls(*args, **kwargs)

    def _revive_if_needed(self, obj: Any) -> Any:
        """Recursively apply reviver to handle LangChain and LangGraph serialized objects.

        This method is crucial for preventing MESSAGE_COERCION_FAILURE by ensuring
        that LangChain message objects stored in their serialized format are properly
        reconstructed. Without this, messages would remain as dictionaries with
        'lc', 'type', and 'constructor' fields, causing errors when the application
        expects actual message objects with 'role' and 'content' attributes.

        It also handles LangGraph Interrupt objects (both old and new serialization formats)
        and must be reconstructed to prevent AttributeError when accessing Interrupt attributes.

        Additionally, it handles dataclass objects (like langmem's RunningSummary) that are serialized
        using the LangChain constructor format but need special reconstruction logic.

        Args:
            obj: The object to potentially revive, which may be a dict, list, or primitive.

        Returns:
            The revived object with LangChain/LangGraph objects properly reconstructed.
        """
        if isinstance(obj, list):
            return [self._revive_if_needed(item) for item in obj]
        if isinstance(obj, dict):
            # Match json.loads(object_hook=...) semantics by reviving children first.
            revived = {k: self._revive_if_needed(v) for k, v in obj.items()}

            if (
                revived.get("lc") in (1, 2)
                and revived.get("type") == "constructor"
            ):
                # If the dictionary is an lc constructor envelope
                if revived.get("id") == ["builtins", "set"]:
                    return self._reconstruct_set_constructor(revived)

                try:
                    cls = self._constructor_class(revived)
                except ValueError:
                    cls = None

                if cls is not None and dataclasses.is_dataclass(cls):
                    try:
                        return self._reconstruct_dataclass_constructor(revived, cls)
                    except Exception:
                        # If reconstruction fails, fall back to the parent reviver.
                        pass

                # First try to use parent's reviver method to reconstruct LangChain objects
                # This converts {'lc': 1, 'type': 'constructor', ...} back to
                # the actual LangChain object (e.g., HumanMessage, AIMessage)
                # Do not manually import and call arbitrary constructor envelopes here.
                # Unknown lc envelopes should follow upstream JsonPlusSerializer behavior.
                return self._reviver(revived)

            # Check if this is a serialized Interrupt object with type marker.
            # Handles both formats:
            #   langgraph <=1.0.x: {"__interrupt__": True, "value": ..., "id": ...}
            #   langgraph >=1.1.x: {"__interrupt__": True, "value": ..., "resumable": ..., "ns": ..., "when": ...}
            if revived.get("__interrupt__") is True and "value" in revived:
                try:
                    return _deserialize_interrupt(revived)
                except (ImportError, TypeError, ValueError) as e:
                    logger.debug(
                        "Failed to deserialize Interrupt object: %s", e, exc_info=True
                    )

            # Check if this is a serialized Send object with type marker (issue #94)
            # Send objects serialize to {"__send__": True, "node": ..., "arg": ...}
            if (
                revived.get("__send__") is True
                and "node" in revived
                and "arg" in revived
                and len(revived) == 3
            ):
                # Try to reconstruct as a Send object
                try:
                    from langgraph.types import Send

                    return Send(
                        node=revived["node"],
                        arg=revived["arg"],
                    )
                except (ImportError, TypeError, ValueError) as e:
                    # If we can't import or construct Send, log and fall through
                    logger.debug(
                        "Failed to deserialize Send object: %s", e, exc_info=True
                    )

            return revived
        return obj
