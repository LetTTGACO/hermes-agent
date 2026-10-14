"""Feishu adapter coverage for its CardKit runtime boundary."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu.feishu_cardkit import FeishuCardKitRuntime


def test_adapter_import_does_not_eagerly_load_lark_oapi():
    script = (
        "import sys; "
        "import plugins.platforms.feishu.adapter; "
        "raise SystemExit(1 if 'lark_oapi' in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _make_adapter():
    return FeishuAdapter(
        PlatformConfig(
            enabled=True,
            extra={"app_id": "cli_test", "app_secret": "secret_test"},
        )
    )


@pytest.mark.asyncio
async def test_send_returns_runtime_success_without_standard_send(monkeypatch):
    adapter = _make_adapter()
    adapter._client = object()
    runtime_result = SendResult(success=True, message_id="om-card")
    runtime = SimpleNamespace(try_send=AsyncMock(return_value=runtime_result))
    adapter._cardkit_runtime = runtime
    standard = AsyncMock()
    monkeypatch.setattr(adapter, "_send_standard_message", standard, raising=False)

    result = await adapter.send("oc_chat", "hello", metadata={"expect_edits": True})

    assert result is runtime_result
    standard.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_runtime_failure_falls_back_exactly_once(monkeypatch):
    adapter = _make_adapter()
    adapter._client = object()
    runtime = SimpleNamespace(
        try_send=AsyncMock(return_value=SendResult(success=False, error="boom"))
    )
    adapter._cardkit_runtime = runtime
    fallback = SendResult(success=True, message_id="om-standard")
    standard = AsyncMock(return_value=fallback)
    monkeypatch.setattr(adapter, "_send_standard_message", standard, raising=False)

    result = await adapter.send("oc_chat", "hello", metadata=None)

    assert result is fallback
    runtime.try_send.assert_awaited_once()
    standard.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_without_runtime_uses_standard_once(monkeypatch):
    adapter = _make_adapter()
    adapter._client = object()
    adapter._cardkit_runtime = None
    fallback = SendResult(success=True, message_id="om-standard")
    standard = AsyncMock(return_value=fallback)
    monkeypatch.setattr(adapter, "_send_standard_message", standard, raising=False)

    result = await adapter.send("oc_chat", "hello", metadata=None)

    assert result is fallback
    standard.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_cardkit_reference_once_uses_interactive_payload_and_finalizes(
    monkeypatch,
):
    adapter = _make_adapter()
    response = SimpleNamespace(data=SimpleNamespace(message_id="om-card"))
    send_with_retry = AsyncMock(return_value=response)
    finalized = SendResult(
        success=True, message_id="om-card", raw_response=response
    )
    finalize = MagicMock(return_value=finalized)
    monkeypatch.setattr(adapter, "_feishu_send_with_retry", send_with_retry)
    monkeypatch.setattr(adapter, "_finalize_send_result", finalize)
    metadata = {
        "expect_edits": True,
        "thread_id": "topic-1",
        "reply_to_message_id": "om-trigger",
    }

    result = await adapter._send_cardkit_reference_once(
        card_id="card-1",
        chat_id="oc-chat",
        reply_to="om-trigger",
        metadata=metadata,
    )

    assert result is finalized
    send_with_retry.assert_awaited_once_with(
        chat_id="oc-chat",
        msg_type="interactive",
        payload=json.dumps(
            {"type": "card", "data": {"card_id": "card-1"}},
            ensure_ascii=False,
        ),
        reply_to="om-trigger",
        metadata=metadata,
    )
    finalize.assert_called_once_with(response, "cardkit send failed")


@pytest.mark.asyncio
async def test_adapter_runtime_callback_preserves_interactive_thread_routing(
    monkeypatch,
):
    adapter = _make_adapter()
    adapter._client = object()
    cardkit_client = SimpleNamespace(
        create_card=AsyncMock(return_value="card-1"),
        update_element_content=AsyncMock(return_value=None),
        close_card=AsyncMock(return_value=None),
    )
    adapter._cardkit_runtime = FeishuCardKitRuntime(
        client=cardkit_client,
        send_card_reference=adapter._send_cardkit_reference_once,
    )
    response = SimpleNamespace(
        success=lambda: True,
        code=0,
        msg="",
        data=SimpleNamespace(message_id="om-card"),
    )
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(adapter, "_send_raw_message", transport)
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", AsyncMock()
    )
    metadata = {
        "expect_edits": True,
        "thread_id": "topic-1",
        "reply_to_message_id": "om-trigger",
    }

    result = await adapter.send(
        "oc-chat", "hello", reply_to="om-trigger", metadata=metadata
    )

    assert result.success is True
    assert result.message_id == "om-card"
    transport.assert_awaited_once_with(
        chat_id="oc-chat",
        msg_type="interactive",
        payload=json.dumps(
            {"type": "card", "data": {"card_id": "card-1"}},
            ensure_ascii=False,
        ),
        reply_to="om-trigger",
        metadata=metadata,
    )
    cardkit_client.create_card.assert_awaited_once()
    assert cardkit_client.update_element_content.await_args.args[2] == "hello"


@pytest.mark.asyncio
async def test_edit_uses_runtime_only_when_message_is_managed(monkeypatch):
    adapter = _make_adapter()
    adapter._client = object()
    managed = SendResult(success=True, message_id="om-card")
    runtime = SimpleNamespace(edit_if_managed=AsyncMock(return_value=managed))
    adapter._cardkit_runtime = runtime
    build = MagicMock(side_effect=AssertionError("standard edit should not run"))
    monkeypatch.setattr(adapter, "_build_outbound_payload", build)

    result = await adapter.edit_message("oc_chat", "om-card", "final", finalize=True)

    assert result is managed
    runtime.edit_if_managed.assert_awaited_once_with(
        "oc_chat", "om-card", "final", finalize=True
    )


@pytest.mark.asyncio
async def test_edit_unmanaged_message_uses_official_standard_path(monkeypatch):
    adapter = _make_adapter()
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(update=object())))
    )
    runtime = SimpleNamespace(edit_if_managed=AsyncMock(return_value=None))
    adapter._cardkit_runtime = runtime
    standard = SendResult(success=True, message_id="om-standard")
    monkeypatch.setattr(
        adapter, "_build_outbound_payload", lambda content: ("text", "payload")
    )
    monkeypatch.setattr(
        adapter, "_build_update_message_body", lambda **kwargs: "body"
    )
    monkeypatch.setattr(
        adapter, "_build_update_message_request", lambda **kwargs: "request"
    )
    monkeypatch.setattr(adapter, "_run_blocking", AsyncMock(return_value="response"))
    monkeypatch.setattr(
        adapter, "_finalize_send_result", lambda response, default: standard
    )

    result = await adapter.edit_message(
        "oc_chat", "om-standard", "updated", finalize=False
    )

    assert result is standard
    runtime.edit_if_managed.assert_awaited_once_with(
        "oc_chat", "om-standard", "updated", finalize=False
    )


def test_requires_edit_finalize_tracks_runtime_availability():
    adapter = _make_adapter()
    assert adapter.REQUIRES_EDIT_FINALIZE is False
    adapter._cardkit_runtime = object()
    assert adapter.REQUIRES_EDIT_FINALIZE is True


@pytest.mark.asyncio
async def test_disconnect_closes_runtime_before_sdk_teardown(monkeypatch):
    adapter = _make_adapter()
    order = []
    runtime = SimpleNamespace(
        close_all=AsyncMock(side_effect=lambda: order.append("cards"))
    )
    adapter._cardkit_runtime = runtime
    monkeypatch.setattr(
        adapter,
        "_stop_webhook_server",
        AsyncMock(side_effect=lambda: order.append("transport")),
    )
    monkeypatch.setattr(adapter, "_cancel_pending_tasks", AsyncMock())
    monkeypatch.setattr(adapter, "_release_app_lock", AsyncMock())
    monkeypatch.setattr(
        adapter, "_shutdown_sdk_executor", lambda: order.append("sdk")
    )

    await adapter.disconnect()

    assert order.index("cards") < order.index("transport")
    assert order.index("cards") < order.index("sdk")
    assert adapter._cardkit_runtime is None


@pytest.mark.asyncio
async def test_connect_creates_cardkit_runtime_after_transport(monkeypatch):
    import plugins.platforms.feishu.adapter as module
    adapter = _make_adapter()
    order = []
    sdk_client = object()
    async def connect_transport():
        adapter._client = sdk_client
        order.append("transport")
    monkeypatch.setattr(module, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(module, "acquire_scoped_lock", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(adapter, "_connect_with_retry", connect_transport)
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda *_: None)
    # Avoid starting a real websocket thread while retaining the connect lifecycle.
    adapter._connection_mode = "webhook"
    adapter._verification_token = "test-token"
    class Client:
        def __init__(self, client):
            assert client is sdk_client
            order.append("cardkit")
    monkeypatch.setattr(module, "FeishuCardKitClient", Client, raising=False)
    assert await adapter.connect() is True
    assert order == ["transport", "cardkit"]
    assert isinstance(adapter._cardkit_runtime, FeishuCardKitRuntime)
    assert adapter.REQUIRES_EDIT_FINALIZE is True


@pytest.mark.asyncio
async def test_disconnect_cardkit_error_still_completes_upstream_teardown(monkeypatch):
    adapter = _make_adapter()
    order = []
    supervisor = MagicMock()
    supervisor.cancel.side_effect = lambda: order.append("supervisor")
    adapter._ws_supervisor = supervisor
    async def close_cards():
        order.append("cards")
        raise RuntimeError("card close failed")
    adapter._cardkit_runtime = SimpleNamespace(close_all=close_cards)
    monkeypatch.setattr(adapter, "_stop_webhook_server", AsyncMock(side_effect=lambda: order.append("transport")))
    monkeypatch.setattr(adapter, "_cancel_pending_tasks", AsyncMock())
    monkeypatch.setattr(adapter, "_release_app_lock", AsyncMock(side_effect=lambda: order.append("lock")))
    monkeypatch.setattr(adapter, "_shutdown_sdk_executor", lambda: order.append("sdk"))
    await adapter.disconnect()
    assert order == ["supervisor", "cards", "transport", "sdk", "lock"]
    assert adapter._cardkit_runtime is None
    assert adapter._ws_supervisor is None


@pytest.mark.asyncio
async def test_disconnect_cardkit_cancellation_propagates(monkeypatch):
    adapter = _make_adapter()
    adapter._cardkit_runtime = SimpleNamespace(close_all=AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await adapter.disconnect()
