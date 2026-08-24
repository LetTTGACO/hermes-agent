"""Feishu adapter coverage for its CardKit runtime boundary."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.feishu.adapter import FeishuAdapter


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
    monkeypatch.setattr(adapter, "_send_standard_message", standard)

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
    monkeypatch.setattr(adapter, "_send_standard_message", standard)

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
    monkeypatch.setattr(adapter, "_send_standard_message", standard)

    result = await adapter.send("oc_chat", "hello", metadata=None)

    assert result is fallback
    standard.assert_awaited_once()


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
    monkeypatch.setattr(adapter, "_shutdown_sdk_executor", lambda: None)

    await adapter.disconnect()

    assert order.index("cards") < order.index("transport")
    assert adapter._cardkit_runtime is None
