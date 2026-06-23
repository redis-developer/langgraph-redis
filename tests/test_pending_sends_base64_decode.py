"""Regression test: pending-send blobs must be base64-decoded on load.

Blobs are stored base64-encoded in Redis JSON (see ``BaseRedisSaver._encode_blob``).
The pending-writes load path decodes them via ``_decode_blob`` before
deserialization, but the pending-sends load path returned the raw base64 string.
``_load_checkpoint`` then handed that string to the serializer, and ``orjson``
raised on the first (non-JSON) character::

    orjson.JSONDecodeError: unexpected character: line 1 column 1 (char 0)

This broke resuming any graph that left a queued ``Send`` on the TASKS channel
(e.g. a human-in-the-loop ``interrupt()`` or a map step) when the checkpoint was
read back through the FT.SEARCH load path.

These tests store a real ``Send`` on a parent checkpoint's TASKS channel and
confirm it round-trips back to a ``Send`` object through ``_load_checkpoint``.
"""

import uuid

import pytest
from langgraph.constants import TASKS
from langgraph.types import Send

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver


def _checkpoint(checkpoint_id: str) -> dict:
    return {
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00",
        "v": 1,
        "channel_values": {"messages": []},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_pending_sends_blob_is_base64_decoded_sync(redis_url: str) -> None:
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        thread_id = f"test-pending-sends-{uuid.uuid4()}"
        base = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        parent_id = str(uuid.uuid4())
        parent_cfg = saver.put(
            base, _checkpoint(parent_id), {"source": "loop"}, {"messages": "1"}
        )
        # A queued Send on the parent's TASKS channel, as an interrupt leaves behind.
        saver.put_writes(
            parent_cfg, [(TASKS, Send("tools", {"request": "approve?"}))], "task1"
        )

        # FT.SEARCH load path returns the stored (type, blob) pairs (blob is base64).
        pending_sends = saver._load_pending_sends(thread_id, "", parent_id)
        assert pending_sends, "expected the TASKS write to be loaded as a pending send"

        # _load_checkpoint is the single point that deserializes pending sends.
        # Before the fix this raised orjson.JSONDecodeError on the base64 blob.
        checkpoint = saver._load_checkpoint(_checkpoint(parent_id), {}, pending_sends)

        sends = checkpoint["pending_sends"]
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "tools"
        assert sends[0].arg == {"request": "approve?"}

        saver.delete_thread(thread_id)


@pytest.mark.asyncio
async def test_pending_sends_blob_is_base64_decoded_async(redis_url: str) -> None:
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        thread_id = f"test-pending-sends-{uuid.uuid4()}"
        base = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        parent_id = str(uuid.uuid4())
        parent_cfg = await saver.aput(
            base, _checkpoint(parent_id), {"source": "loop"}, {"messages": "1"}
        )
        await saver.aput_writes(
            parent_cfg, [(TASKS, Send("tools", {"request": "approve?"}))], "task1"
        )

        # _abatch_load_pending_sends uses the FT.SEARCH ($.blob) path resume hits.
        pending_sends_map = await saver._abatch_load_pending_sends(
            [(thread_id, "", parent_id)]
        )
        pending_sends = pending_sends_map.get((thread_id, "", parent_id), [])
        assert pending_sends, "expected the TASKS write to be loaded as a pending send"

        checkpoint = saver._load_checkpoint(_checkpoint(parent_id), {}, pending_sends)

        sends = checkpoint["pending_sends"]
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "tools"
        assert sends[0].arg == {"request": "approve?"}

        await saver.adelete_thread(thread_id)
