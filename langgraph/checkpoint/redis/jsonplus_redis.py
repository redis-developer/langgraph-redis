import base64
import logging
from typing import Any, Union

import orjson
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


class JsonPlusRedisSerializer(JsonPlusSerializer):
    """Redis-optimized serializer using orjson for faster JSON processing."""

    SENTINEL_FIELDS = [
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "parent_checkpoint_id",
    ]

    def dumps(self, obj: Any) -> bytes:
        """Use orjson for simple objects, fallback to parent for complex objects."""
        try:
            # Fast path: Use orjson for JSON-serializable objects
            return orjson.dumps(obj)
        except TypeError:
            # Complex objects (Send, etc.) need parent's msgpack serialization
            return super().dumps(obj)

    def loads(self, data: bytes) -> Any:
        """Use orjson for JSON parsing with reviver support, fallback to parent for msgpack data."""
        try:
            # Fast path: Use orjson for JSON data
            parsed = orjson.loads(data)
            # Apply reviver for LangChain objects (lc format)
            return self._revive_if_needed(parsed)
        except orjson.JSONDecodeError:
            # Fallback: Parent handles msgpack and other formats
            return super().loads(data)

    def _revive_if_needed(self, obj: Any) -> Any:
        """Recursively apply reviver to handle LangChain serialized objects."""
        if isinstance(obj, dict):
            # Check if this is a LangChain serialized object
            if obj.get("lc") in (1, 2) and obj.get("type") == "constructor":
                # Use parent's reviver method to reconstruct the object
                return self._reviver(obj)
            # Recursively process nested dicts
            return {k: self._revive_if_needed(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process lists
            return [self._revive_if_needed(item) for item in obj]
        else:
            # Return primitives as-is
            return obj

    def dumps_typed(self, obj: Any) -> tuple[str, str]:  # type: ignore[override]
        if isinstance(obj, (bytes, bytearray)):
            return "base64", base64.b64encode(obj).decode("utf-8")
        else:
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
