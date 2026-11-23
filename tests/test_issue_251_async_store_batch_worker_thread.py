"""Regression tests for issue #251.

``AsyncRedisStore.batch`` is meant to be called from a worker thread while the
store's event loop keeps running elsewhere. It used to look up the calling
thread's loop with ``asyncio.get_event_loop()``, which raises in a worker thread.
These tests replace ``abatch`` with a fake so no Redis server is needed; they
check which loop the work is submitted to.
"""

import asyncio
import threading
from typing import Any, Iterable

import pytest
from langgraph.store.base import GetOp, Op, PutOp, Result
from redis.asyncio import Redis as AsyncRedis

from langgraph.store.redis import AsyncRedisStore


def _make_store() -> AsyncRedisStore:
    # The client is created lazily and never connects in these tests.
    return AsyncRedisStore(redis_client=AsyncRedis())


async def _fake_setup(store: AsyncRedisStore) -> None:
    async def _no_op(*args: Any, **kwargs: Any) -> None:
        return None

    store.store_index.create = _no_op  # type: ignore[method-assign]
    store.cluster_mode = False
    await store.setup()


def _run_in_thread(fn: Any) -> Any:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread, outcome


async def test_batch_from_worker_thread_runs_on_store_loop() -> None:
    store = _make_store()
    await _fake_setup(store)
    seen: dict[str, Any] = {}

    async def fake_abatch(ops: Iterable[Op]) -> list[Result]:
        seen["loop"] = asyncio.get_running_loop()
        return [None for _ in ops]

    store.abatch = fake_abatch  # type: ignore[method-assign]
    ops = [PutOp(("ns",), "k", {"v": 1}), GetOp(("ns",), "k")]

    thread, outcome = _run_in_thread(lambda: store.batch(ops))
    while thread.is_alive():
        await asyncio.sleep(0.01)
    thread.join()

    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"] == [None, None]
    assert seen["loop"] is asyncio.get_running_loop()


async def test_batch_from_store_loop_raises_invalid_state() -> None:
    store = _make_store()
    await _fake_setup(store)

    with pytest.raises(asyncio.InvalidStateError):
        store.batch([GetOp(("ns",), "k")])


def test_batch_before_setup_raises_runtime_error() -> None:
    async def build() -> AsyncRedisStore:
        return _make_store()

    store = asyncio.run(build())

    with pytest.raises(RuntimeError, match="setup"):
        store.batch([GetOp(("ns",), "k")])


@pytest.mark.asyncio
async def test_batch_from_worker_thread_against_real_redis(redis_url) -> None:
    """End to end: a worker thread writes and reads through ``batch()`` on a live Redis."""
    namespace = ("issue_251", "worker_thread")
    async with AsyncRedisStore.from_conn_string(redis_url) as store:
        await store.setup()

        def put_then_get() -> tuple[list, list]:
            # Two calls: operations inside one batch are not ordered put-then-get.
            put = store.batch([PutOp(namespace=namespace, key="k1", value={"n": 1})])
            got = store.batch([GetOp(namespace=namespace, key="k1")])
            return put, got

        # The store's loop keeps running here while the worker thread blocks on batch().
        put, got = await asyncio.to_thread(put_then_get)

        assert put == [None]
        assert got[0] is not None
        assert got[0].value == {"n": 1}
        stored = await store.aget(namespace, "k1")
        assert stored is not None and stored.value == {"n": 1}
