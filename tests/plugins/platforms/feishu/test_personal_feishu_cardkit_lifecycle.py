"""CardKit sessions follow the adapter SDK and belong to one inbound turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource
from plugins.platforms.feishu import adapter as adapter_module
from plugins.platforms.feishu import feishu_cardkit as cardkit
from plugins.platforms.feishu.feishu_cardkit import FeishuCardKitClient, FeishuCardKitRuntime


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


def _make_sdk(card_id):
    response = SimpleNamespace(
        success=lambda: True, code=0, msg="", data=SimpleNamespace(card_id=card_id)
    )
    return SimpleNamespace(
        cardkit=SimpleNamespace(v1=SimpleNamespace(
            card=SimpleNamespace(
                acreate=AsyncMock(return_value=response),
                asettings=AsyncMock(return_value=response),
            ),
            card_element=SimpleNamespace(acontent=AsyncMock(return_value=response)),
        ))
    )


@pytest.fixture
def cardkit_sdk(monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
        lambda: (_SDKModel,) * 6,
    )


@pytest.mark.asyncio
async def test_cardkit_uses_rebuilt_sdk_for_existing_session(monkeypatch, cardkit_sdk):
    sdk_a, sdk_b = _make_sdk("card-A"), _make_sdk("card-B")
    adapter = adapter_module.FeishuAdapter(PlatformConfig(
        enabled=True,
        extra={
            "app_id": "cli_test", "app_secret": "secret_test",
            "connection_mode": "webhook", "verification_token": "test-token",
        },
    ))
    monkeypatch.setattr(adapter_module, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(
        adapter_module, "acquire_scoped_lock", lambda *args, **kwargs: (True, None)
    )
    monkeypatch.setattr(adapter_module, "_sdk_domain", lambda domain: domain)
    monkeypatch.setattr(adapter, "_build_lark_client", lambda domain: sdk_a)
    monkeypatch.setattr(adapter, "_build_event_handler", lambda: object())
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda *_: None)

    async def connect_transport():
        adapter._prepare_client()

    monkeypatch.setattr(adapter, "_connect_with_retry", connect_transport)
    monkeypatch.setattr(
        adapter, "_feishu_send_with_retry", AsyncMock(return_value=SimpleNamespace(
            success=lambda: True, code=0, msg="",
            data=SimpleNamespace(message_id="om-card"),
        ))
    )
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", AsyncMock()
    )
    assert await adapter.connect() is True
    runtime_before = adapter._cardkit_runtime
    sent = await adapter.send("oc-chat", "initial", metadata={"expect_edits": True})
    assert sent.success is True
    assert sent.message_id == "om-card"

    monkeypatch.setattr(adapter, "_build_lark_client", lambda domain: sdk_b)
    adapter._prepare_client()
    edited = await adapter.edit_message("oc-chat", sent.message_id, "updated snapshot")
    finalized = await adapter.edit_message(
        "oc-chat", sent.message_id, "updated snapshot", finalize=True
    )

    assert edited.success is True
    assert finalized.success is True
    assert edited.message_id == finalized.message_id == sent.message_id
    assert adapter._cardkit_runtime is runtime_before
    assert sdk_a.cardkit.v1.card.acreate.await_count == 1
    assert sdk_b.cardkit.v1.card_element.acontent.await_count == 1
    assert sdk_b.cardkit.v1.card_element.acontent.await_args.args[0].card_id == "card-A"
    assert sdk_b.cardkit.v1.card.asettings.await_count == 1
    assert sdk_b.cardkit.v1.card_element.acontent.await_args.args[0].request_body.content == "updated snapshot"
    assert sdk_b.cardkit.v1.card.asettings.await_args.args[0].card_id == "card-A"
    assert sdk_a.cardkit.v1.card_element.acontent.await_count == 1
    sdk_a.cardkit.v1.card.asettings.assert_not_awaited()
    sdk_b.cardkit.v1.card.acreate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "close"])
async def test_cardkit_client_unavailable_fails_without_stale_sdk_use(cardkit_sdk, operation):
    sdk_a = _make_sdk("card-A")
    current_sdk = sdk_a
    client = FeishuCardKitClient(get_sdk_client=lambda: current_sdk)
    assert await client.create_card(content="initial") == "card-A"
    current_sdk = None

    with pytest.raises(RuntimeError, match="SDK client is unavailable"):
        if operation == "create":
            await client.create_card(content="next")
        elif operation == "update":
            await client.update_element_content("card-A", "content", "next", 2)
        else:
            await client.close_card("card-A", "final", 3)

    assert sdk_a.cardkit.v1.card.acreate.await_count == 1
    sdk_a.cardkit.v1.card_element.acontent.assert_not_awaited()
    sdk_a.cardkit.v1.card.asettings.assert_not_awaited()


@pytest.fixture
def runtime_pair(monkeypatch):
    monkeypatch.setattr(cardkit.asyncio, "sleep", AsyncMock())
    client = SimpleNamespace(
        create_card=AsyncMock(side_effect=[f"card-{n}" for n in range(10)]),
        update_element_content=AsyncMock(),
        close_card=AsyncMock(),
    )

    async def send_reference(**kwargs):
        return SendResult(success=True, message_id=f"om-{kwargs['card_id']}")

    return FeishuCardKitRuntime(client=client, send_card_reference=send_reference), client


def _owned_metadata(owner="om-in", thread="topic-A", **extra):
    return {"_cardkit_turn_id": owner, "thread_id": thread, **extra}


async def _send_progress(runtime, owner="om-in", thread="topic-A", reply_to="anchor-A"):
    result = await runtime.try_send(
        "oc-chat", "running", reply_to=reply_to,
        metadata=_owned_metadata(owner, thread, message_kind="tool_progress"),
    )
    assert result.success
    return result.message_id


@pytest.mark.asyncio
async def test_other_topic_does_not_close_progress(runtime_pair):
    runtime, client = runtime_pair
    message_id = await _send_progress(runtime)
    sent = await runtime.try_send(
        "oc-chat", "topic B final", reply_to=None,
        metadata=_owned_metadata(thread="topic-B"),
    )
    assert sent.success
    assert client.close_card.await_count == 0
    assert (await runtime.edit_if_managed("oc-chat", message_id, "A continues")).success


@pytest.mark.asyncio
async def test_same_owner_different_anchor_closes_progress(runtime_pair):
    runtime, client = runtime_pair
    message_id = await _send_progress(runtime)
    sent = await runtime.try_send(
        "oc-chat", "done", reply_to="different-anchor", metadata=_owned_metadata(),
    )
    assert sent.success
    client.close_card.assert_awaited_once()
    assert await runtime.edit_if_managed("oc-chat", message_id, "late") is None


@pytest.mark.asyncio
async def test_same_topic_other_owner_keeps_progress(runtime_pair):
    runtime, client = runtime_pair
    message_id = await _send_progress(runtime)
    sent = await runtime.try_send(
        "oc-chat", "new turn", reply_to=None, metadata=_owned_metadata(owner="om-new"),
    )
    assert sent.success
    client.close_card.assert_not_awaited()
    assert (await runtime.edit_if_managed("oc-chat", message_id, "continues")).success


@pytest.mark.asyncio
async def test_unowned_cron_does_not_close_progress(runtime_pair):
    runtime, client = runtime_pair
    message_id = await _send_progress(runtime)
    sent = await runtime.try_send(
        "oc-chat", "cron report", reply_to=None, metadata={"thread_id": "topic-A"},
    )
    assert sent.success
    client.close_card.assert_not_awaited()
    assert (await runtime.edit_if_managed("oc-chat", message_id, "continues")).success


@pytest.mark.parametrize("thread_metadata", [{}, {"thread_id": None}, {"thread_id": ""}])
def test_scope_normalizes_empty_thread(thread_metadata):
    metadata = {"_cardkit_turn_id": "om-in", **thread_metadata}
    original = metadata.copy()
    assert cardkit.cardkit_scope("oc-chat", metadata) == ("oc-chat", None, "om-in")
    assert metadata == original


@pytest.mark.parametrize("metadata", [None, {}, {"_cardkit_turn_id": ""}, {"_cardkit_turn_id": None}])
def test_scope_requires_owner(metadata):
    assert cardkit.cardkit_scope("oc-chat", metadata) is None


def test_scope_converts_nonempty_values_to_strings():
    assert cardkit.cardkit_scope("oc-chat", {"thread_id": "topic-A", "_cardkit_turn_id": "om-in"}) == (
        "oc-chat", "topic-A", "om-in"
    )
    assert cardkit.cardkit_scope("oc-chat", {"thread_id": 123, "_cardkit_turn_id": 456}) == (
        "oc-chat", "123", "456"
    )


@pytest.mark.asyncio
async def test_scope_is_captured_before_first_await(runtime_pair):
    runtime, client = runtime_pair
    metadata = _owned_metadata(message_kind="tool_progress")

    async def create_card(**kwargs):
        metadata.update(_owned_metadata(owner="om-new", thread="topic-B"))
        return "card-A"

    client.create_card.side_effect = create_card
    sent = await runtime.try_send("oc-chat", "running", reply_to=None, metadata=metadata)
    await runtime.close_scope(("oc-chat", "topic-A", "om-in"))
    client.close_card.assert_awaited_once()
    assert await runtime.edit_if_managed("oc-chat", sent.message_id, "late") is None


def _adapter_with_runtime(runtime):
    adapter = adapter_module.FeishuAdapter(PlatformConfig(enabled=True))
    adapter._cardkit_runtime = runtime
    adapter._reactions_enabled = lambda: True
    adapter._remove_reaction = AsyncMock(return_value=True)
    adapter._add_reaction = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_two_adapter_runtimes_do_not_share_sessions(runtime_pair):
    runtime, client = runtime_pair
    other = FeishuCardKitRuntime(client=client, send_card_reference=runtime._send_card_reference)
    adapter_a, adapter_b = _adapter_with_runtime(runtime), _adapter_with_runtime(other)
    message_id = await _send_progress(adapter_a._cardkit_runtime)
    await adapter_b._cardkit_runtime.try_send(
        "oc-chat", "done", reply_to=None, metadata=_owned_metadata(),
    )
    await adapter_b._cardkit_runtime.close_scope(("oc-chat", "topic-A", "om-in"))
    client.close_card.assert_not_awaited()
    assert (await adapter_a._cardkit_runtime.edit_if_managed("oc-chat", message_id, "continues")).success


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
@pytest.mark.parametrize("reactions_enabled", [True, False])
@pytest.mark.parametrize("thread", [None, "topic-A"])
async def test_old_processing_completion_keeps_new_turn_cards(runtime_pair, outcome, reactions_enabled, thread):
    runtime, client = runtime_pair
    ids = {}
    for owner in ("om-old", "om-new"):
        assistant = await runtime.try_send(
            "oc-chat", "answer", reply_to=None,
            metadata=_owned_metadata(owner, thread, expect_edits=True),
        )
        ids[owner] = [assistant.message_id, await _send_progress(runtime, owner, thread)]
    client.close_card.reset_mock()
    adapter = _adapter_with_runtime(runtime)
    adapter._reactions_enabled = lambda: reactions_enabled
    adapter._pending_processing_reactions["om-old"] = "reaction-old"
    event = MessageEvent(
        text="old", message_id="om-old",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", thread_id=thread),
    )

    await adapter.on_processing_complete(event, outcome)

    assert client.close_card.await_count == 2
    for message_id in ids["om-old"]:
        assert await runtime.edit_if_managed("oc-chat", message_id, "late") is None
    for message_id in ids["om-new"]:
        assert (await runtime.edit_if_managed("oc-chat", message_id, "new continues")).success
    if reactions_enabled:
        adapter._remove_reaction.assert_awaited_once_with("om-old", "reaction-old")
        if outcome is ProcessingOutcome.FAILURE:
            adapter._add_reaction.assert_awaited_once_with("om-old", adapter_module._FEISHU_REACTION_FAILURE)
        else:
            adapter._add_reaction.assert_not_awaited()
    else:
        adapter._remove_reaction.assert_not_awaited()
        adapter._add_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_processing_cleanup_exception_preserves_reaction_path(runtime_pair):
    runtime, _ = runtime_pair
    adapter = _adapter_with_runtime(runtime)
    runtime.close_scope = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    adapter._pending_processing_reactions["om-in"] = "reaction-in"
    event = MessageEvent(
        text="in", message_id="om-in",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", thread_id="topic-A"),
    )
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    runtime.close_scope.assert_awaited_once_with(("oc-chat", "topic-A", "om-in"))
    adapter._remove_reaction.assert_awaited_once_with("om-in", "reaction-in")
    adapter._add_reaction.assert_awaited_once_with("om-in", adapter_module._FEISHU_REACTION_FAILURE)


@pytest.mark.asyncio
async def test_close_scope_retains_failed_session_and_continues(runtime_pair):
    runtime, client = runtime_pair
    assistant = await runtime.try_send(
        "oc-chat", "answer", reply_to=None, metadata=_owned_metadata(expect_edits=True),
    )
    progress_id = await _send_progress(runtime)
    client.close_card.side_effect = [RuntimeError("close unavailable"), None, None]
    scope = ("oc-chat", "topic-A", "om-in")
    await runtime.close_scope(None)
    client.close_card.assert_not_awaited()
    await runtime.close_scope(scope)
    assert client.close_card.await_count == 2
    assert (await runtime.edit_if_managed("oc-chat", assistant.message_id, "retry pending")).success
    assert await runtime.edit_if_managed("oc-chat", progress_id, "late") is None
    await runtime.close_scope(scope)
    assert client.close_card.await_count == 3
    assert await runtime.edit_if_managed("oc-chat", assistant.message_id, "late") is None
