import base64
import binascii
import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any

import orjson
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

_TYPED_VALUE_MARKER = "__redis_jsonplus_typed__"
_TYPED_VALUE_KEYS = {_TYPED_VALUE_MARKER, "type", "data"}


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

    @classmethod
    def is_native_json_value(cls, obj: Any) -> bool:
        """Return whether JSON can store a value without changing its meaning.

        Exact type checks are intentional: ``Enum`` and scalar subclasses can be
        accepted by JSON encoders while losing their Python type.
        """
        if obj is None or type(obj) in (str, bool, int):
            return True
        if type(obj) is float:
            return math.isfinite(obj)
        if type(obj) in (list, tuple):
            return all(cls.is_native_json_value(item) for item in obj)
        if type(obj) is dict:
            return all(
                type(key) is str and cls.is_native_json_value(value)
                for key, value in obj.items()
            )
        return False

    def _encode_typed_value(self, obj: Any) -> dict[str, Any]:
        """Wrap an upstream-serialized value in a RedisJSON-safe envelope."""
        type_, data = super().dumps_typed(obj)
        return {
            _TYPED_VALUE_MARKER: True,
            "type": type_,
            "data": base64.b64encode(bytes(data)).decode("ascii"),
        }

    def _decode_typed_value(self, obj: dict[str, Any]) -> Any:
        """Decode a value produced by :meth:`_encode_typed_value`."""
        type_ = obj["type"]
        data = obj["data"]
        if not isinstance(type_, str) or not isinstance(data, str):
            return obj
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            return obj
        return super().loads_typed((type_, decoded))

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

        from langgraph.checkpoint.serde.types import _DeltaSnapshot

        if isinstance(obj, _DeltaSnapshot):
            return {"__delta_snapshot__": True, "value": obj.value}

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
                    # Already in LC format; preserve typed values in its fields.
                    return self._preprocess_redis_json(json_dict)

            # For other objects with instance state, encode with constructor kwargs.
            if hasattr(obj, "__dict__"):
                return self._encode_constructor_envelope(
                    type(obj),
                    kwargs={
                        key: self._preprocess_redis_json(value)
                        for key, value in obj.__dict__.items()
                    },
                )

            # Built-in and extension types without __dict__ must not be encoded
            # as an empty constructor. Delegate to the upstream msgpack codec,
            # which preserves types such as Decimal, timedelta and frozenset.
            return self._encode_typed_value(obj)
        except (AttributeError, KeyError, ValueError, TypeError):
            # For types we can't handle, raise TypeError
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _preprocess_redis_json(
        self,
        obj: Any,
        *,
        encode_bytes: Callable[[bytes], Any] | None = None,
    ) -> Any:
        """Recursively normalize values before Redis JSON encoding.

        The order here is intentional:

        1. Escape scalar values that Redis JSON cannot store directly, such as
           bytes and bytearray, when an encode_bytes callback is provided.
        2. Preserve allowlisted semantic/type-carrying values before treating
           them as containers. This includes explicit Redis markers for
           Interrupt, _DeltaSnapshot, and Send, plus constructor envelopes for
           set and dataclass instances. This is especially important for
           _DeltaSnapshot because it is a NamedTuple and would otherwise be
           flattened by the generic tuple branch.
        3. Recurse through ordinary containers only after those special cases
           have had a chance to preserve their type information.

        When encode_bytes is provided, nested bytes/bytearray values are escaped
        as Redis JSON markers. Without it, bytes are left in place so orjson can
        raise and dumps_typed can fall back to msgpack.
        """
        from langgraph.checkpoint.serde.types import _DeltaSnapshot
        from langgraph.types import Interrupt, Send

        if isinstance(obj, (bytes, bytearray)):
            return (
                {"__bytes__": encode_bytes(bytes(obj))}
                if encode_bytes is not None
                else obj
            )
        if isinstance(obj, Interrupt):
            return _serialize_interrupt(
                obj,
                preprocess=lambda value: self._preprocess_redis_json(
                    value, encode_bytes=encode_bytes
                ),
            )
        elif isinstance(obj, _DeltaSnapshot):
            return {
                "__delta_snapshot__": True,
                "value": self._preprocess_redis_json(
                    obj.value, encode_bytes=encode_bytes
                ),
            }
        elif isinstance(obj, Send):
            # Add type marker to distinguish from plain dicts (issue #94)
            return {
                "__send__": True,
                "node": obj.node,
                "arg": self._preprocess_redis_json(obj.arg, encode_bytes=encode_bytes),
            }
        elif isinstance(obj, set):
            # Handle sets by converting to list for JSON serialization
            # Will be reconstructed back to set on deserialization
            return self._encode_constructor_envelope(
                set,
                args=[
                    [
                        self._preprocess_redis_json(item, encode_bytes=encode_bytes)
                        for item in obj
                    ]
                ],
            )
        elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            # Handle dataclass instances (like langmem's RunningSummary)
            # Convert to LangChain constructor format to preserve type information
            # Recursively process the dataclass fields without dataclasses.asdict(),
            # which would erase nested dataclass type information.
            processed_dict = {
                field.name: self._preprocess_redis_json(
                    getattr(obj, field.name), encode_bytes=encode_bytes
                )
                for field in dataclasses.fields(obj)
            }
            return self._encode_constructor_envelope(type(obj), kwargs=processed_dict)
        elif isinstance(obj, dict):
            # RedisJSON requires string object keys. During msgpack-to-RedisJSON
            # normalization, preserve non-string keys in a typed envelope instead
            # of allowing orjson.OPT_NON_STR_KEYS to coerce or merge them.
            if encode_bytes is not None and any(type(key) is not str for key in obj):
                return self._encode_typed_value(obj)
            return {
                k: self._preprocess_redis_json(v, encode_bytes=encode_bytes)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            processed = [
                self._preprocess_redis_json(item, encode_bytes=encode_bytes)
                for item in obj
            ]
            # Preserve tuple type
            return tuple(processed) if isinstance(obj, tuple) else processed
        elif isinstance(obj, Enum):
            # orjson otherwise serializes Enum members as their raw value.
            return self._encode_typed_value(obj)
        elif self.is_native_json_value(obj):
            return obj
        elif hasattr(obj, "to_json") or hasattr(obj, "__dict__"):
            # Keep LangChain and ordinary constructor objects on the existing
            # default-handler path.
            return obj
        else:
            # orjson has native encoders for datetime, date, time and UUID, but
            # those encodings discard the Python type. The upstream codec also
            # covers Decimal, timedelta, frozenset, deque and non-finite floats.
            return self._encode_typed_value(obj)

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
                # Normalize marker types before JSON encoding.
                processed_obj = self._preprocess_redis_json(obj)
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
                set(revived) == _TYPED_VALUE_KEYS
                and revived.get(_TYPED_VALUE_MARKER) is True
            ):
                return self._decode_typed_value(revived)

            if revived.get("lc") in (1, 2) and revived.get("type") == "constructor":
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

            # _DeltaSnapshot marker (issue #201). Redis JSON cannot store msgpack
            # ext code 7, so wrap the seed with a local constructor.
            if (
                revived.get("__delta_snapshot__") is True
                and "value" in revived
                and len(revived) == 2
            ):
                try:
                    from langgraph.checkpoint.serde.types import _DeltaSnapshot

                    return _DeltaSnapshot(revived["value"])
                except (ImportError, TypeError, ValueError) as e:
                    logger.debug(
                        "Failed to deserialize _DeltaSnapshot: %s", e, exc_info=True
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
