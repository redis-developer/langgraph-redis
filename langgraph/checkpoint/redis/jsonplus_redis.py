import base64
import logging
from typing import Any, Union

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)


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
