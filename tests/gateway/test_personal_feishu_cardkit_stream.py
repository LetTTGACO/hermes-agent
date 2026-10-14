"""Exercise upstream streaming and progress through real Feishu CardKit delivery."""

import asyncio
import json
import queue
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext
from plugins.platforms.feishu import adapter as adapter_module
from plugins.platforms.feishu import feishu_cardkit as cardkit
from tests.waiting import async_wait_until


class _SDKBuilder:
    def __init__(self):
        self.fields = {}

    def __getattr__(self, name):
        def set_field(value):
            self.fields[name] = value
            return self
        return set_field

    def build(self):
        return SimpleNamespace(**self.fields)


class _SDKModel:
    @staticmethod
    def builder():
        return _SDKBuilder()


class _Transport:
    """SDK/network double that records requests without making delivery decisions."""

    def __init__(self):
        self.sent_message_ids: list[str] = []
        self.deleted_message_ids: list[str] = []
        self.card_text: dict[str, str] = {}
        self.closed_card_ids: list[str] = []
        self.card_snapshots: dict[str, list[str]] = {}
        self.message_cards: dict[str, str] = {}
        self.im_requests: list = []
        self.failed_update_cards: set[str] = set()

    @staticmethod
    def _response(**data):
        return SimpleNamespace(
            success=lambda: True, code=0, msg="", data=SimpleNamespace(**data),
        )

    async def acreate(self, request):
        card_id = f"card-{len(self.card_text) + 1}"
        payload = json.loads(request.request_body.data)
        text = payload["body"]["elements"][0]["content"]
        self.card_text[card_id] = text
        self.card_snapshots[card_id] = [text]
        return self._response(card_id=card_id)

    async def acontent(self, request):
        if request.card_id in self.failed_update_cards:
            raise RuntimeError("external CardKit update unavailable")
        self.card_text[request.card_id] = request.request_body.content
        self.card_snapshots[request.card_id].append(request.request_body.content)
        return self._response()

    async def asettings(self, request):
        settings = json.loads(request.request_body.settings)
        assert settings["config"]["streaming_mode"] is False
        self.closed_card_ids.append(request.card_id)
        return self._response()

    def send(self, request):
        message_id = f"om-sent-{len(self.sent_message_ids) + 1}"
        self.sent_message_ids.append(message_id)
        self.im_requests.append(request)
        payload = json.loads(request.request_body.content)
        self.message_cards[message_id] = payload["data"]["card_id"]
        return self._response(message_id=message_id)

    def delete(self, request):
        self.deleted_message_ids.append(request.message_id)
        return self._response()


@pytest_asyncio.fixture
async def cardkit_chain(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cardkit, "_import_cardkit_sdk", lambda: (_SDKModel,) * 6)
    for name in (
        "CreateMessageRequest", "CreateMessageRequestBody", "ReplyMessageRequest",
        "ReplyMessageRequestBody", "DeleteMessageRequest",
    ):
        monkeypatch.setattr(adapter_module, name, _SDKModel)
    transport = _Transport()
    sdk = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(acreate=transport.acreate, asettings=transport.asettings),
            card_element=SimpleNamespace(acontent=transport.acontent),
        )),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
            create=transport.send, reply=transport.send, delete=transport.delete,
        ))),
    )
    adapter = adapter_module.FeishuAdapter(PlatformConfig(
        enabled=True, extra={"app_id": "cli_test", "app_secret": "secret_test"},
    ))
    adapter._client = sdk
    client = cardkit.FeishuCardKitClient(get_sdk_client=lambda: adapter._client)
    runtime = cardkit.FeishuCardKitRuntime(
        client=client, send_card_reference=adapter._send_cardkit_reference_once,
    )
    adapter._cardkit_runtime = runtime
    try:
        yield adapter, runtime, transport
    finally:
        await runtime.close_all()
        adapter._shutdown_sdk_executor()


@asynccontextmanager
async def _stream_preview(chain, preview_text):
    adapter, _, transport = chain
    sent = asyncio.Event()
    consumer = GatewayStreamConsumer(
        adapter, "oc-chat",
        StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        metadata={"thread_id": "topic-A", "_cardkit_turn_id": "om-inbound"},
        on_new_message=sent.set,
        initial_reply_to_id="om-inbound",
    )
    consumer.on_delta(preview_text)
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(sent.wait(), 5)
        message_id = consumer.message_id
        assert message_id == transport.sent_message_ids[0]
        card_id = transport.message_cards[message_id]
        yield consumer, task, message_id, card_id
    finally:
        if not task.done():
            consumer.finish()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_identical_final_snapshot_still_closes_card(cardkit_chain):
    _, runtime, transport = cardkit_chain
    final_text = "The complete answer is already visible."
    async with _stream_preview(cardkit_chain, final_text) as (consumer, task, message_id, card_id):
        assert card_id not in transport.closed_card_ids
        consumer.finish(final_text)
        await asyncio.wait_for(task, 5)
        assert transport.closed_card_ids.count(card_id) == 1
        assert len(transport.sent_message_ids) == 1
        assert transport.card_text[card_id] == final_text
        assert message_id not in runtime._sessions
        assert consumer.delivered_final_matches(final_text)
        request = transport.im_requests[0]
        assert request.message_id == "om-inbound"
        assert request.request_body.reply_in_thread is True


@pytest.mark.asyncio
async def test_fence_balancing_snapshots_replace_without_duplication(cardkit_chain):
    _, _, transport = cardkit_chain
    preview_text = "Example:\n```python\nprint(1)"
    final_text = "Example:\n```python\nprint(1)\nprint(2)\n```\nFinished."
    async with _stream_preview(cardkit_chain, preview_text) as (consumer, task, _, card_id):
        first_snapshot = transport.card_text[card_id]
        assert first_snapshot == preview_text + "\n```"
        consumer.on_delta("\nprint(2)")
        await async_wait_until(
            lambda: transport.card_text[card_id] != first_snapshot,
            what="second fence-balanced snapshot",
        )
        second_snapshot = transport.card_text[card_id]
        assert not second_snapshot.startswith(first_snapshot)
        assert second_snapshot == preview_text + "\nprint(2)\n```"
        consumer.finish(final_text)
        await asyncio.wait_for(task, 5)
        assert transport.card_text[card_id] == final_text
        assert len(transport.sent_message_ids) == 1
        assert transport.closed_card_ids.count(card_id) == 1
        assert consumer.delivered_final_matches(final_text)


@pytest.mark.asyncio
async def test_full_fallback_deletes_preview_and_releases_card(cardkit_chain):
    _, runtime, transport = cardkit_chain
    final_text = "Corrected authoritative final answer."
    async with _stream_preview(cardkit_chain, "Original preview answer.") as (consumer, task, message_id, card_id):
        transport.failed_update_cards.add(card_id)
        consumer.finish(final_text)
        await asyncio.wait_for(task, 5)
        assert message_id in transport.deleted_message_ids
        assert message_id not in runtime._sessions
        assert len(transport.sent_message_ids) == 2
        replacement_id = transport.sent_message_ids[-1]
        assert transport.card_text[transport.message_cards[replacement_id]] == final_text
        assert transport.closed_card_ids.count(card_id) == 1
        assert consumer.delivered_final_matches(final_text)


@pytest.mark.asyncio
async def test_tail_fallback_preserves_preview_prefix(cardkit_chain):
    _, runtime, transport = cardkit_chain
    preview_text = "Visible head\n"
    final_text = preview_text + "The rest of the answer."
    async with _stream_preview(cardkit_chain, preview_text) as (consumer, task, message_id, card_id):
        transport.failed_update_cards.add(card_id)
        consumer.finish(final_text)
        await asyncio.wait_for(task, 5)
        assert message_id not in transport.deleted_message_ids
        assert len(transport.sent_message_ids) == 2
        tail_id = transport.sent_message_ids[-1]
        sent_tail_text = transport.card_text[transport.message_cards[tail_id]]
        assert preview_text + sent_tail_text == final_text
        assert transport.card_text[card_id] == preview_text
        assert message_id in runtime._sessions
        assert consumer.delivered_final_matches(final_text)


@pytest.mark.asyncio
@pytest.mark.parametrize("final_delivery", ["same-turn-stream", "unowned-static"])
async def test_progress_edits_stay_open_until_same_turn_final(cardkit_chain, final_delivery):
    adapter, runtime, transport = cardkit_chain
    source = SessionSource(
        platform=Platform.FEISHU, chat_id="oc-chat", chat_type="group", thread_id="topic-A",
    )
    ctx = TurnContext(
        source=source, inbound_message_id="om-inbound", _run_still_current=lambda: True,
        progress_mode="all", tool_progress_enabled=True, progress_queue=queue.Queue(),
        _progress_reply_to="om-inbound",
        _progress_metadata={
            "thread_id": "topic-A", "_cardkit_turn_id": "om-inbound",
            "message_kind": "tool_progress",
        },
    )
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _: adapter), ctx)
    turn.progress_callback("tool.started", "terminal", "pwd", {})
    progress_task = asyncio.create_task(turn.send_progress_messages())
    try:
        await async_wait_until(lambda: runtime._sessions, what="real TurnRunner progress send")
        progress_message_id = transport.sent_message_ids[0]
        progress_card_id = transport.message_cards[progress_message_id]
        initial_text = transport.card_text[progress_card_id]
        turn.progress_callback("tool.started", "browser_navigate", "https://example.com", {})
        # Cancellation drains the real progress loop with finalize=True, without waiting
        # for its edit throttle. No CardKit close should happen for a progress edit.
        progress_task.cancel()
        await asyncio.wait_for(progress_task, 5)
        edited_text = transport.card_text[progress_card_id]
        assert initial_text in edited_text
        assert "example.com" in edited_text
        assert progress_card_id not in transport.closed_card_ids
        assert progress_message_id in runtime._sessions

        if final_delivery == "same-turn-stream":
            consumer = GatewayStreamConsumer(
                adapter, source.chat_id, StreamConsumerConfig(cursor=""),
                metadata={"thread_id": "topic-A", "_cardkit_turn_id": "om-inbound"},
            )
            consumer.on_delta("The assistant final answer.")
            consumer.finish("The assistant final answer.")
            await consumer.run()
            assert consumer.final_response_sent
        else:
            sent = await adapter.send(
                source.chat_id, "The non-streamed final answer.",
                metadata={"thread_id": "topic-A"},
            )
            assert sent.success
            assert progress_card_id not in transport.closed_card_ids
            event = MessageEvent(text="hello", source=source, message_id="om-inbound")
            await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert transport.closed_card_ids.count(progress_card_id) == 1
        assert progress_message_id not in runtime._sessions
    finally:
        if not progress_task.done():
            progress_task.cancel()
        await asyncio.gather(progress_task, return_exceptions=True)
