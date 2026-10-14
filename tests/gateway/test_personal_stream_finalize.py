"""Explicit finalize receipts distinguish real edits from sends and deferred attempts."""

import asyncio

import pytest

from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _EditableTransport:
    """A regular transport whose send leaves an editable message open until finalize."""

    REQUIRES_EDIT_FINALIZE = True
    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, first_edit="success"):
        self.first_edit = first_edit
        self.sent = []
        self.edits = []
        self.visible = ""
        self.closed = False

    async def send(self, chat_id, content, **kwargs):
        self.sent.append(content)
        self.visible = content
        return SendResult(success=True, message_id="preview")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, **kwargs):
        assert message_id == "preview"
        self.edits.append((content, finalize))
        if len(self.edits) == 1:
            if self.first_edit == "skipped":
                return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})
            if self.first_edit == "flood":
                return SendResult(success=False, error="429 flood control: retry after")
            if self.first_edit == "exception":
                raise RuntimeError("temporary transport failure")
        self.visible = content
        self.closed = finalize
        return SendResult(success=True, message_id=message_id)


@pytest.mark.asyncio
async def test_fresh_regular_send_still_receives_its_first_finalize_edit():
    adapter = _EditableTransport()
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(cursor=""))
    consumer.on_delta("Complete answer")
    consumer.finish("Complete answer")
    await consumer.run()
    assert adapter.sent == ["Complete answer"]
    assert adapter.edits == [("Complete answer", True)]
    assert adapter.closed
    assert consumer.delivered_final_matches("Complete answer") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("first_edit", ["success", "skipped", "flood", "exception"])
@pytest.mark.parametrize("final", ["Preview", "Complete final answer"])
async def test_existing_finalize_only_dedupes_a_successful_actual_edit(first_edit, final):
    adapter = _EditableTransport(first_edit)
    sent = asyncio.Event()
    consumer = GatewayStreamConsumer(
        adapter, "chat", StreamConsumerConfig(cursor="", edit_interval=0, buffer_threshold=1),
        on_new_message=sent.set,
    )
    consumer.on_delta("Preview")
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(sent.wait(), 5)
        consumer.finish(final)
        await asyncio.wait_for(task, 5)
        attempts = 1 if first_edit == "success" else 2
        assert adapter.edits == [(final, True)] * attempts
        assert adapter.sent == ["Preview"]
        assert adapter.closed
        assert adapter.visible == final
        assert consumer.final_content_delivered
        assert consumer.delivered_final_matches(final) is True
    finally:
        if not task.done():
            consumer.finish()
        await asyncio.wait_for(task, 5)
