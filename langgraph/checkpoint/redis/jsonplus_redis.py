import base64
import logging
from typing import Any, Union

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


logger = logging.getLogger(__name__)


# RediSearch versions below 2.10 don't support indexing and querying
# empty strings, so we use a sentinel value to represent empty strings.
EMPTY_STRING_SENTINEL = "__empty__"


class JsonPlusRedisSerializer(JsonPlusSerializer):
    """Redis-optimized serializer that stores strings directly."""

    SENTINEL_FIELDS = [
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "parent_checkpoint_id",
    ]

    def dumps_typed(self, obj: Any) -> tuple[str, str]:  # type: ignore[override]
        if isinstance(obj, (bytes, bytearray)):
            return "base64", base64.b64encode(obj).decode("utf-8")
        else:
            for field in self.SENTINEL_FIELDS:
                try:
                    if field in obj and not obj[field]:
                        obj[field] = EMPTY_STRING_SENTINEL
                except (KeyError, AttributeError):
                    try:
                        if hasattr(obj, field) and not getattr(obj, field, None):
                            setattr(obj, field, EMPTY_STRING_SENTINEL)
                    except Exception as e:
                        logger.debug(
                            f"Error setting {field} from empty string to sentinel: {e}"
                        )
            results = self.dumps(obj).decode("utf-8")
            return "json", results

    def loads_typed(self, data: tuple[str, Union[str, bytes]]) -> Any:
        type_, data_ = data
        if type_ == "base64":
            decoded = base64.b64decode(
                data_ if isinstance(data_, bytes) else data_.encode()
            )
            return decoded
        elif type_ == "json":
            data_bytes = data_ if isinstance(data_, bytes) else data_.encode()
            results = self.loads(data_bytes)
            for field in self.SENTINEL_FIELDS:
                try:
                    if field in results and results[field] == EMPTY_STRING_SENTINEL:
                        results[field] = ""
                except (KeyError, AttributeError):
                    try:
                        if (
                            hasattr(results, field)
                            and getattr(results, field) == EMPTY_STRING_SENTINEL
                        ):
                            setattr(results, field, "")
                    except Exception as e:
                        logger.debug(
                            f"Error setting {field} from sentinel to empty string: {e}"
                        )
                        pass
            return results
