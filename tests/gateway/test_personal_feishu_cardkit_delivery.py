"""Final-review regressions through real CardKit, stream, and turn delivery paths."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.run import GatewayRunner
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext
from plugins.platforms.feishu import adapter as adapter_module
from plugins.platforms.feishu.feishu_cardkit import FeishuCardKitClient, FeishuCardKitRuntime


class _Transport:
    """External I/O only; real installed SDK builders construct every request."""

    def __init__(self):
        self.cards = {}
        self.messages = {}
        self.closed = []
        self.close_attempts = []
        self.im_updates = []
        self.fail_content = False
        self.fail_close = False
        self.fail_send = False

    @staticmethod
    def ok(**data):
        return SimpleNamespace(success=lambda: True, code=0, msg="", data=SimpleNamespace(**data))

    async def create_card(self, request):
        card_id = f"card-{len(self.cards) + 1}"
        self.cards[card_id] = json.loads(request.request_body.data)["body"]["elements"][0]["content"]
        return self.ok(card_id=card_id)

    async def content(self, request):
        if self.fail_content:
            raise RuntimeError("external content update temporarily unavailable")
        self.cards[request.card_id] = request.request_body.content
        return self.ok()

    async def settings(self, request):
        assert json.loads(request.request_body.settings)["config"]["streaming_mode"] is False
        self.close_attempts.append(request.card_id)
        if self.fail_close:
            raise RuntimeError("external close temporarily unavailable")
        self.closed.append(request.card_id)
        return self.ok()

    def send(self, request):
        if self.fail_send:
            return SimpleNamespace(success=lambda: False, code=230001, msg="delivery unavailable")
        body = request.request_body
        message_id = f"om-{len(self.messages) + 1}"
        self.messages[message_id] = (body.msg_type, json.loads(body.content))
        return self.ok(message_id=message_id)

    def update(self, request):
        self.im_updates.append((request.message_id, request.request_body.msg_type))
        # IM message.update cannot edit interactive messages with text/post payloads.
        assert self.messages[request.message_id][0] == "interactive"
        return SimpleNamespace(success=lambda: False, code=230001,
                               msg="text/post update cannot edit an interactive card")

    def delete(self, request):
        del self.messages[request.message_id]
        return self.ok()


@pytest_asyncio.fixture
async def chain(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert adapter_module._load_lark_oapi()
    transport = _Transport()
    sdk = SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(acreate=transport.create_card, asettings=transport.settings),
            card_element=SimpleNamespace(acontent=transport.content))),
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
            create=transport.send, reply=transport.send, update=transport.update,
            delete=transport.delete))),
    )
    adapter = adapter_module.FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = sdk
    adapter._cardkit_runtime = FeishuCardKitRuntime(
        client=FeishuCardKitClient(get_sdk_client=lambda: adapter._client),
        send_card_reference=adapter._send_cardkit_reference_once,
    )
    try:
        yield adapter, transport
    finally:
        transport.fail_content = transport.fail_close = False
        await adapter._cardkit_runtime.close_all()
        adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_failure", [False, True])
async def test_acknowledged_initial_card_survives_redundant_update_failure(chain, close_failure, caplog):
    adapter, transport = chain
    transport.fail_content = True
    transport.fail_close = close_failure
    result = await adapter.send("oc-chat", "answer ▉", metadata={
        "expect_edits": True, "_cardkit_turn_id": "om-in",
    })
    assert result.success
    assert len(transport.messages) == 1
    assert transport.messages[result.message_id][0] == "interactive"
    session = adapter._cardkit_runtime._sessions[result.message_id]
    assert session.current_text == transport.cards[session.card_id] == "answer"
    assert transport.close_attempts == []
    assert any("initial update failed" in record.message and record.exc_info for record in caplog.records)

    transport.fail_content = False
    assert (await adapter.edit_message("oc-chat", result.message_id, "answer extended")).success
    closed = await adapter.edit_message("oc-chat", result.message_id, "answer extended", finalize=True)
    assert closed.success is not close_failure
    if close_failure:
        assert result.message_id in adapter._cardkit_runtime._sessions
        transport.fail_close = False
        assert (await adapter.edit_message("oc-chat", result.message_id, "answer extended", finalize=True)).success
    assert result.message_id not in adapter._cardkit_runtime._sessions
    assert transport.closed == [session.card_id]
    assert transport.im_updates == []
    assert len(transport.messages) == 1


@asynccontextmanager
async def _preview(chain, text="Original preview"):
    adapter, _ = chain
    sent = asyncio.Event()
    consumer = GatewayStreamConsumer(
        adapter, "oc-chat", StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
        metadata={"_cardkit_turn_id": "om-in"}, on_new_message=sent.set,
    )
    consumer.on_delta(text)
    task = asyncio.create_task(consumer.run())
    try:
        await asyncio.wait_for(sent.wait(), 5)
        yield consumer, task
    finally:
        if not task.done():
            consumer.finish()
        await asyncio.wait_for(task, 5)


def _turn(consumer):
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", chat_type="dm")
    ctx = TurnContext(source=source, inbound_message_id="om-in", stream_consumer_holder=[consumer])
    runner = GatewayTurnMixin()
    return runner, ctx, TurnRunner(runner, ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("final", ["Original preview", "Authoritative transformed final"])
async def test_successful_done_edit_closes_once_without_standard_im_fallback(chain, final):
    adapter, transport = chain
    async with _preview(chain) as (consumer, task):
        consumer.finish(final)
        await asyncio.wait_for(task, 5)
        assert consumer.delivered_final_matches(final) is True
        assert consumer.final_content_delivered
        assert list(transport.cards.values()) == [final]
        assert len(transport.messages) == 1
        assert transport.closed == ["card-1"]
        assert not adapter._cardkit_runtime._sessions
        assert transport.im_updates == []


@pytest.mark.asyncio
async def test_transformed_authoritative_finish_is_not_resent_after_durable_delivery(chain):
    _, transport = chain
    async with _preview(chain) as (consumer, task):
        runner, ctx, turn = _turn(consumer)
        response = {"final_response": "Authoritative transformed final", "response_transformed": True}
        turn._finish_stream_consumer(response, [], consumer)
        await asyncio.wait_for(task, 5)
        assert consumer.delivered_final_matches(response["final_response"]) is True
        before = len(transport.im_updates)
        await runner._run_agent_mark_streamed_delivery(response, ctx)
        assert response.get("already_sent") is True
        assert len(transport.im_updates) == before
        assert len(transport.messages) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["mismatch", "unknown", "split", "failed"])
async def test_transformed_reconciliation_keeps_unconfirmed_final_send(chain, state):
    _, transport = chain
    async with _preview(chain) as (consumer, task):
        runner, ctx, turn = _turn(consumer)
        response = {"final_response": "Delivered final", "response_transformed": True}
        turn._finish_stream_consumer(response, [], consumer)
        await asyncio.wait_for(task, 5)
        if state == "failed":
            response["failed"] = True
        else:
            response["final_response"] = "Changed after finish"
            if state in {"unknown", "split"}:
                # Exercise the real matcher's legacy ambiguous / payload-less split states.
                consumer._delivered_final_text = None
                consumer._delivery_ambiguous = True
                consumer._turn_split_delivery = state == "split"
            assert consumer.delivered_final_matches(response["final_response"]) is (
                None if state == "unknown" else False)
        before = len(transport.im_updates)
        await runner._run_agent_mark_streamed_delivery(response, ctx)
        assert not response.get("already_sent")
        assert len(transport.im_updates) == before + (state != "failed")


@pytest.mark.asyncio
async def test_required_transformed_edit_and_fallback_failure_do_not_claim_delivery(chain):
    """#119323: preview-first; the authoritative final never reaches any transport."""
    _, transport = chain
    async with _preview(chain) as (consumer, task):
        runner, ctx, turn = _turn(consumer)
        transport.fail_content = transport.fail_send = True
        response = {"final_response": "Authoritative transformed final", "response_transformed": True}
        turn._finish_stream_consumer(response, [], consumer)
        await asyncio.wait_for(task, 5)
        assert not consumer.final_content_delivered
        assert not consumer.final_response_sent
        assert consumer.delivered_final_matches(response["final_response"]) is False
        await runner._run_agent_mark_streamed_delivery(response, ctx)
        assert not response.get("already_sent")
        assert len(transport.messages) == 1
        _, payload = next(iter(transport.messages.values()))
        assert transport.cards[payload["data"]["card_id"]] == "Original preview"


class _Ticks:
    """Each next wait acknowledges the previous heartbeat completed."""

    def __init__(self):
        self.entered = asyncio.Queue()
        self.release = asyncio.Queue()

    async def sleep(self, _interval):
        self.entered.put_nowait(None)
        await self.release.get()

    async def ready(self):
        await asyncio.wait_for(self.entered.get(), 5)


@pytest.mark.asyncio
async def test_heartbeat_updates_one_owned_card_until_processing_complete(chain, monkeypatch):
    adapter, transport = chain
    ticks = _Ticks()
    # Replace only the scheduler's clock boundary, never the shared asyncio module.
    monkeypatch.setattr("gateway.run_turn.asyncio", SimpleNamespace(sleep=ticks.sleep))
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "180")
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", chat_type="group", thread_id="topic-A")
    metadata = {"thread_id": "topic-A", "_cardkit_turn_id": "om-in"}
    ctx = TurnContext(source=source, session_key="session", _status_thread_metadata=metadata)
    agent = object()
    ctx.agent_holder[0] = agent
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {"session": agent}
    runner._draining = runner._restart_requested = False
    runner._delivery_adapter_for = lambda _source: adapter
    action = ["first task"]
    runner._agent_activity_summary = lambda _agent: {"current_tool": action[0]}
    disp = SimpleNamespace(
        _display_surface_mode=lambda *a, **kw: "on",
        resolve_display_setting=lambda *a: False, user_config={}, platform_key="feishu",
    )
    task = asyncio.create_task(runner._run_agent_notify_long_running(disp, ctx, [None]))
    try:
        await ticks.ready()
        ticks.release.put_nowait(None)
        await ticks.ready()
        assert len(transport.messages) == 1
        message_id = next(iter(transport.messages))
        session = adapter._cardkit_runtime._sessions.get(message_id)
        assert session is not None
        assert session.scope == ("oc-chat", "topic-A", "om-in")
        first = transport.cards[session.card_id]
        action[0] = "second task"
        ticks.release.put_nowait(None)
        await ticks.ready()
        assert transport.cards[session.card_id] != first
        assert "second task" in transport.cards[session.card_id]
        assert len(transport.messages) == 1
        assert transport.im_updates == []
        assert transport.closed == []
        assert metadata == {"thread_id": "topic-A", "_cardkit_turn_id": "om-in"}
        runner._running_agents.clear()
        ticks.release.put_nowait(None)
        await asyncio.wait_for(task, 5)
        event = MessageEvent(text="work", source=source, message_id="om-in")
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert transport.closed == [session.card_id]
        assert not adapter._cardkit_runtime._sessions
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
