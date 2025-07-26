import concurrent.futures

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from langgraph.checkpoint.redis import RedisSaver


def _increment(
    checkpointer: RedisSaver, config: RunnableConfig, iterations: int
) -> None:
    for _ in range(iterations):
        with checkpointer.thread_lock(config["configurable"]["thread_id"]):
            tup = checkpointer.get_tuple(config)
            cp = tup.checkpoint if tup else empty_checkpoint()
            cp.setdefault("channel_values", {}).setdefault("count", 0)
            cp.setdefault("channel_versions", {}).setdefault("count", "0")
            cp["channel_values"]["count"] += 1
            cp["channel_versions"]["count"] = str(cp["channel_values"]["count"])
            checkpointer.put(config, cp, {}, {"count": cp["channel_versions"]["count"]})


@pytest.mark.skip("Requires running Redis container")
def test_thread_lock_serialization(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        config: RunnableConfig = {
            "configurable": {"thread_id": "t", "checkpoint_ns": ""}
        }
        saver.put(config, empty_checkpoint(), {}, {})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_increment, saver, config, 5) for _ in range(2)]
            for f in futures:
                f.result()

        final = saver.get_tuple(config)
        assert final is not None
        assert final.checkpoint["channel_values"]["count"] == 10
