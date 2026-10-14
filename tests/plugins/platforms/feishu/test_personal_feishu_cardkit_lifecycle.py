"""CardKit sessions follow the adapter SDK and belong to one inbound turn."""

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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

    class ObservedLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                attempted.set()
            return await super().acquire()

    attempted = asyncio.Event()
    runtime._registry_lock = ObservedLock()
    async with runtime._registry_lock:
        task = asyncio.create_task(runtime.try_send(
            "oc-chat", "running", reply_to=None, metadata=metadata,
        ))
        await asyncio.wait_for(attempted.wait(), 5)
        assert not task.done()
        client.create_card.assert_not_awaited()
        metadata.update(_owned_metadata(owner="om-new", thread="topic-B"))
    sent = await asyncio.wait_for(task, 5)
    assert runtime._sessions[sent.message_id].scope == ("oc-chat", "topic-A", "om-in")
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


@pytest.fixture
def lifecycle_adapter(monkeypatch, cardkit_sdk):
    """Keep adapter, CardKit client and runtime real; replace SDK/network I/O."""
    monkeypatch.setattr(cardkit.asyncio, "sleep", AsyncMock())
    sdk = _make_sdk("card-A")
    im_response = SimpleNamespace(
        success=lambda: True, code=0, msg="", data=SimpleNamespace(message_id="om-card"),
    )
    sdk.im = SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(
        delete=MagicMock(return_value=im_response),
    )))
    adapter = _adapter_with_runtime(None)
    adapter._client = sdk
    transport = AsyncMock(return_value=im_response)
    monkeypatch.setattr(adapter, "_send_raw_message", transport)
    client = FeishuCardKitClient(get_sdk_client=lambda: adapter._client)
    runtime = FeishuCardKitRuntime(
        client=client, send_card_reference=adapter._send_cardkit_reference_once,
    )
    adapter._cardkit_runtime = runtime
    yield adapter, runtime, sdk, transport
    adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["assistant", "static"])
@pytest.mark.parametrize("close_fails", [False, True], ids=["close-success", "close-failure"])
async def test_cancelled_reference_closes_known_card_without_fallback(lifecycle_adapter, monkeypatch, streaming, close_fails):
    adapter, runtime, sdk, transport = lifecycle_adapter
    reference_started, release_reference = asyncio.Event(), asyncio.Event()

    async def blocked_reference(**kwargs):
        reference_started.set()
        await release_reference.wait()

    transport.side_effect = blocked_reference
    if close_fails:
        sdk.cardkit.v1.card.asettings.side_effect = RuntimeError("close unavailable")
    standard_send = AsyncMock(wraps=adapter._send_standard_message)
    monkeypatch.setattr(adapter, "_send_standard_message", standard_send)
    task = asyncio.create_task(adapter.send("oc-chat", "preview", metadata={"expect_edits": streaming}))
    await asyncio.wait_for(reference_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    sdk.cardkit.v1.card.asettings.assert_awaited_once()
    assert sdk.cardkit.v1.card.asettings.await_args.args[0].card_id == "card-A"
    standard_send.assert_not_awaited()
    assert runtime._sessions == {}
    assert runtime._sends_drained.is_set()


@pytest.mark.asyncio
async def test_cancelled_initial_update_closes_known_card_without_fallback(lifecycle_adapter, monkeypatch):
    adapter, runtime, sdk, _ = lifecycle_adapter
    update_started, release_update = asyncio.Event(), asyncio.Event()

    async def blocked_update(request):
        update_started.set()
        await release_update.wait()

    sdk.cardkit.v1.card_element.acontent.side_effect = blocked_update
    standard_send = AsyncMock(wraps=adapter._send_standard_message)
    monkeypatch.setattr(adapter, "_send_standard_message", standard_send)
    task = asyncio.create_task(adapter.send("oc-chat", "preview", metadata={"expect_edits": True}))
    await asyncio.wait_for(update_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    sdk.cardkit.v1.card.asettings.assert_awaited_once()
    standard_send.assert_not_awaited()
    assert runtime._sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["assistant", "static"])
async def test_disconnect_drains_cancelled_send_cleanup(lifecycle_adapter, streaming):
    adapter, runtime, sdk, transport = lifecycle_adapter
    reference_started, release_reference = asyncio.Event(), asyncio.Event()
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()

    async def blocked_reference(**kwargs):
        reference_started.set()
        await release_reference.wait()

    async def blocked_cleanup(request):
        cleanup_started.set()
        await release_cleanup.wait()
        return SimpleNamespace(success=lambda: True)

    transport.side_effect = blocked_reference
    sdk.cardkit.v1.card.asettings.side_effect = blocked_cleanup
    send_task = asyncio.create_task(adapter.send("oc-chat", "preview", metadata={"expect_edits": streaming}))
    close_task = None
    try:
        await asyncio.wait_for(reference_started.wait(), 2)
        send_task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 2)
        close_task = asyncio.create_task(runtime.close_all())
        checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(checkpoint.set)
        await checkpoint.wait()
        assert runtime._closing
        assert not close_task.done()
        assert not runtime._sends_drained.is_set()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await send_task
        await asyncio.wait_for(close_task, 2)
        sdk.cardkit.v1.card.asettings.assert_awaited_once()
        assert runtime._sessions == {}
    finally:
        release_reference.set()
        release_cleanup.set()
        await asyncio.gather(send_task, *([close_task] if close_task else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_delete_success_releases_managed_card(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})

    assert await adapter.delete_message("oc-chat", sent.message_id)

    sdk.im.v1.message.delete.assert_called_once()
    assert sdk.im.v1.message.delete.call_args.args[0].message_id == sent.message_id
    sdk.cardkit.v1.card.asettings.assert_awaited_once()
    assert runtime._sessions == {}
    assert await runtime.edit_if_managed("oc-chat", sent.message_id, "late") is None


@pytest.mark.asyncio
async def test_delete_failure_keeps_card_editable(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})
    sdk.im.v1.message.delete.return_value = SimpleNamespace(success=lambda: False, code=230001, msg="denied")

    assert not await adapter.delete_message("oc-chat", sent.message_id)

    sdk.cardkit.v1.card.asettings.assert_not_awaited()
    assert (await adapter.edit_message("oc-chat", sent.message_id, "still editable")).success
    assert sdk.cardkit.v1.card_element.acontent.await_args.args[0].request_body.content == "still editable"
    assert sent.message_id in runtime._sessions


@pytest.mark.asyncio
async def test_delete_success_survives_card_close_failure(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})
    sdk.cardkit.v1.card.asettings.side_effect = RuntimeError("close unavailable")

    assert await adapter.delete_message("oc-chat", sent.message_id)

    sdk.cardkit.v1.card.asettings.assert_awaited_once()
    assert runtime._sessions == {}


@pytest.mark.asyncio
async def test_delete_unknown_or_wrong_chat_keeps_other_sessions(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})

    await runtime.forget_deleted_message("oc-chat", "om-unknown")
    await runtime.forget_deleted_message("oc-other", sent.message_id)

    sdk.cardkit.v1.card.asettings.assert_not_awaited()
    assert (await adapter.edit_message("oc-chat", sent.message_id, "still editable")).success


@pytest.mark.asyncio
async def test_delete_waits_for_active_edit_before_closing(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})
    update_started, release_update, delete_succeeded = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked_update(request):
        update_started.set()
        await release_update.wait()
        return SimpleNamespace(success=lambda: True)

    def deletion_success():
        delete_succeeded.set()
        return True

    sdk.cardkit.v1.card_element.acontent.side_effect = blocked_update
    sdk.im.v1.message.delete.return_value = SimpleNamespace(success=deletion_success)
    edit_task = asyncio.create_task(adapter.edit_message("oc-chat", sent.message_id, "latest snapshot"))
    delete_task = None
    try:
        await asyncio.wait_for(update_started.wait(), 2)
        delete_task = asyncio.create_task(adapter.delete_message("oc-chat", sent.message_id))
        await asyncio.wait_for(delete_succeeded.wait(), 2)
        sdk.cardkit.v1.card.asettings.assert_not_awaited()
        assert not delete_task.done()
        release_update.set()
        assert (await edit_task).success
        assert await delete_task
        close = sdk.cardkit.v1.card.asettings.await_args.args[0]
        assert "latest snapshot" in close.request_body.settings
        assert runtime._sessions == {}
    finally:
        release_update.set()
        await asyncio.gather(edit_task, *([delete_task] if delete_task else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_delete_cleanup_preserves_replacement_session(lifecycle_adapter):
    adapter, runtime, sdk, _ = lifecycle_adapter
    sent = await adapter.send("oc-chat", "preview", metadata={"expect_edits": True})
    original = runtime._sessions[sent.message_id]
    close_started, release_close = asyncio.Event(), asyncio.Event()

    async def blocked_close(request):
        close_started.set()
        await release_close.wait()
        return SimpleNamespace(success=lambda: True)

    sdk.cardkit.v1.card.asettings.side_effect = blocked_close
    delete_task = asyncio.create_task(adapter.delete_message("oc-chat", sent.message_id))
    try:
        await asyncio.wait_for(close_started.wait(), 2)
        sdk.cardkit.v1.card.acreate.return_value.data.card_id = "card-B"
        replacement = await adapter.send("oc-chat", "replacement", metadata={"expect_edits": True})
        assert replacement.message_id == sent.message_id
        assert runtime._sessions[sent.message_id] is not original
        release_close.set()
        assert await delete_task
        assert (await adapter.edit_message("oc-chat", replacement.message_id, "still editable")).success
        assert sdk.cardkit.v1.card_element.acontent.await_args.args[0].card_id == "card-B"
    finally:
        release_close.set()
        await asyncio.gather(delete_task, return_exceptions=True)
