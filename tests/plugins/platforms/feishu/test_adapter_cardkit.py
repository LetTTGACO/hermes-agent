"""Tests for Feishu adapter CardKit integration helpers."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.platforms.base import SendResult
from plugins.platforms.feishu import adapter as feishu_adapter
from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu.feishu_cardkit import (
    CARDKIT_ASSISTANT_PROFILE,
    CARDKIT_STATIC_PROFILE,
    CARDKIT_TOOL_PROGRESS_PROFILE,
    FeishuCardKitClient,
    FeishuStreamingCardSession,
)


class TestCardkitStreamingMode:
    def test_tool_progress_uses_streaming_tool_progress_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode(
            {"message_kind": "tool_progress"}
        ) == (True, CARDKIT_TOOL_PROGRESS_PROFILE)

    def test_expect_edits_uses_streaming_assistant_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode({"expect_edits": True}) == (
            True,
            CARDKIT_ASSISTANT_PROFILE,
        )

    def test_no_metadata_uses_static_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode(None) == (
            False,
            CARDKIT_STATIC_PROFILE,
        )

    def test_job_metadata_uses_static_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode({"job_id": "cron_123"}) == (
            False,
            CARDKIT_STATIC_PROFILE,
        )

    def test_tool_progress_takes_priority_over_expect_edits(self):
        assert FeishuAdapter._cardkit_streaming_mode(
            {"message_kind": "tool_progress", "expect_edits": True}
        ) == (True, CARDKIT_TOOL_PROGRESS_PROFILE)


class TestToolProgressMetadata:
    def test_thread_tool_progress_metadata_uses_tool_progress_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode(
            {"message_kind": "tool_progress", "thread_id": "ot_test"}
        ) == (True, CARDKIT_TOOL_PROGRESS_PROFILE)

    def test_reply_tool_progress_metadata_uses_tool_progress_profile(self):
        assert FeishuAdapter._cardkit_streaming_mode(
            {
                "message_kind": "tool_progress",
                "thread_id": "ot_test",
                "reply_to_message_id": "om_test",
            }
        ) == (True, CARDKIT_TOOL_PROGRESS_PROFILE)


class TestCardkitAvailability:
    def test_should_try_cardkit_requires_complete_state(self, monkeypatch):
        adapter = object.__new__(FeishuAdapter)

        monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
        monkeypatch.setattr(feishu_adapter, "FeishuCardKitClient", object)
        monkeypatch.setattr(feishu_adapter, "FeishuStreamingCardSession", object)

        assert adapter._should_try_cardkit() is False

        adapter._settings = SimpleNamespace(app_id="cli_test", app_secret="secret")
        adapter._client = object()

        assert adapter._should_try_cardkit() is True

    def test_should_try_cardkit_returns_false_when_client_missing(self, monkeypatch):
        adapter = object.__new__(FeishuAdapter)
        monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
        monkeypatch.setattr(feishu_adapter, "FeishuCardKitClient", object)
        monkeypatch.setattr(feishu_adapter, "FeishuStreamingCardSession", object)
        adapter._settings = SimpleNamespace(app_id="cli_test", app_secret="secret")

        assert adapter._should_try_cardkit() is False

    def test_get_cardkit_client_memoizes_and_refreshes_for_sdk_client(self, monkeypatch):
        created_with = []

        class FakeCardKitClient:
            def __init__(self, sdk_client):
                self.sdk_client = sdk_client
                created_with.append(sdk_client)

        adapter = object.__new__(FeishuAdapter)
        adapter._settings = SimpleNamespace(app_id="cli_test", app_secret="secret")
        adapter._client = object()
        adapter._cardkit_client = None
        adapter._cardkit_sdk_source = None

        monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
        monkeypatch.setattr(feishu_adapter, "FeishuCardKitClient", FakeCardKitClient)
        monkeypatch.setattr(feishu_adapter, "FeishuStreamingCardSession", object)

        first = adapter._get_cardkit_client()
        second = adapter._get_cardkit_client()
        adapter._client = object()
        third = adapter._get_cardkit_client()

        assert first is second
        assert third is not first
        assert created_with == [first.sdk_client, third.sdk_client]


class TestCardkitLazyImportBinding:
    def test_lazy_import_path_binds_cardkit_globals(self, monkeypatch):
        class Dummy:
            pass

        fake_modules = {
            "lark_oapi": ModuleType("lark_oapi"),
            "lark_oapi.api": ModuleType("lark_oapi.api"),
            "lark_oapi.api.application": ModuleType("lark_oapi.api.application"),
            "lark_oapi.api.application.v6": ModuleType("lark_oapi.api.application.v6"),
            "lark_oapi.api.im": ModuleType("lark_oapi.api.im"),
            "lark_oapi.api.im.v1": ModuleType("lark_oapi.api.im.v1"),
            "lark_oapi.core": ModuleType("lark_oapi.core"),
            "lark_oapi.core.const": ModuleType("lark_oapi.core.const"),
            "lark_oapi.core.model": ModuleType("lark_oapi.core.model"),
            "lark_oapi.event": ModuleType("lark_oapi.event"),
            "lark_oapi.event.callback": ModuleType("lark_oapi.event.callback"),
            "lark_oapi.event.callback.model": ModuleType("lark_oapi.event.callback.model"),
            "lark_oapi.event.callback.model.p2_card_action_trigger": ModuleType(
                "lark_oapi.event.callback.model.p2_card_action_trigger"
            ),
            "lark_oapi.event.dispatcher_handler": ModuleType("lark_oapi.event.dispatcher_handler"),
            "lark_oapi.ws": ModuleType("lark_oapi.ws"),
        }
        for name, module in fake_modules.items():
            monkeypatch.setitem(sys.modules, name, module)

        fake_modules["lark_oapi.api.application.v6"].GetApplicationRequest = Dummy
        for name in (
            "CreateFileRequest",
            "CreateFileRequestBody",
            "CreateImageRequest",
            "CreateImageRequestBody",
            "CreateMessageRequest",
            "CreateMessageRequestBody",
            "GetChatRequest",
            "GetMessageRequest",
            "GetMessageResourceRequest",
            "P2ImMessageMessageReadV1",
            "ReplyMessageRequest",
            "ReplyMessageRequestBody",
            "UpdateMessageRequest",
            "UpdateMessageRequestBody",
        ):
            setattr(fake_modules["lark_oapi.api.im.v1"], name, Dummy)
        fake_modules["lark_oapi.core"].AccessTokenType = Dummy
        fake_modules["lark_oapi.core"].HttpMethod = Dummy
        fake_modules["lark_oapi.core.const"].FEISHU_DOMAIN = "feishu.test"
        fake_modules["lark_oapi.core.const"].LARK_DOMAIN = "lark.test"
        fake_modules["lark_oapi.core.model"].BaseRequest = Dummy
        fake_modules[
            "lark_oapi.event.callback.model.p2_card_action_trigger"
        ].CallBackCard = Dummy
        fake_modules[
            "lark_oapi.event.callback.model.p2_card_action_trigger"
        ].P2CardActionTriggerResponse = Dummy
        fake_modules["lark_oapi.event.dispatcher_handler"].EventDispatcherHandler = Dummy
        fake_modules["lark_oapi.ws"].Client = Dummy

        captured = {}

        def fake_ensure_and_bind(feature, importer, target_globals, **kwargs):
            assert feature == "platform.feishu"
            assert kwargs == {"prompt": False}
            bindings = importer()
            captured.update(bindings)
            for key, value in bindings.items():
                monkeypatch.setitem(target_globals, key, value)
            return True

        monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", False)
        monkeypatch.setattr("tools.lazy_deps.ensure_and_bind", fake_ensure_and_bind)

        assert feishu_adapter.check_feishu_requirements() is True
        assert captured["FeishuCardKitClient"] is FeishuCardKitClient
        assert captured["FeishuStreamingCardSession"] is FeishuStreamingCardSession
        assert captured["CARDKIT_ASSISTANT_PROFILE"] == CARDKIT_ASSISTANT_PROFILE
        assert captured["CARDKIT_TOOL_PROGRESS_PROFILE"] == CARDKIT_TOOL_PROGRESS_PROFILE
        assert captured["CARDKIT_STATIC_PROFILE"] == CARDKIT_STATIC_PROFILE
        assert feishu_adapter.FeishuCardKitClient is FeishuCardKitClient


class TestSendCardkitFirst:
    @staticmethod
    def _adapter():
        adapter = object.__new__(FeishuAdapter)
        adapter._client = object()
        return adapter

    @pytest.mark.asyncio
    async def test_send_uses_cardkit_when_available_and_returns_success(self, monkeypatch):
        adapter = self._adapter()
        cardkit_result = SendResult(success=True, message_id="cardkit_message")
        cardkit = AsyncMock(return_value=cardkit_result)
        standard = AsyncMock()

        monkeypatch.setattr(adapter, "_should_try_cardkit", lambda: True)
        monkeypatch.setattr(adapter, "_send_cardkit", cardkit)
        monkeypatch.setattr(adapter, "_send_standard_message", standard)

        result = await adapter.send(
            "chat_1",
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )

        assert result is cardkit_result
        cardkit.assert_awaited_once_with(
            "chat_1",
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )
        standard.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_falls_back_to_standard_when_cardkit_fails(self, monkeypatch):
        adapter = self._adapter()
        standard_result = SendResult(success=True, message_id="standard_message")
        cardkit = AsyncMock(return_value=SendResult(success=False, error="cardkit failed"))
        standard = AsyncMock(return_value=standard_result)

        monkeypatch.setattr(adapter, "_should_try_cardkit", lambda: True)
        monkeypatch.setattr(adapter, "_send_cardkit", cardkit)
        monkeypatch.setattr(adapter, "_send_standard_message", standard)

        result = await adapter.send(
            "chat_1",
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )

        assert result is standard_result
        cardkit.assert_awaited_once()
        standard.assert_awaited_once_with(
            "chat_1",
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )

    @pytest.mark.asyncio
    async def test_send_skips_cardkit_when_unavailable(self, monkeypatch):
        adapter = self._adapter()
        standard_result = SendResult(success=True, message_id="standard_message")
        cardkit = AsyncMock()
        standard = AsyncMock(return_value=standard_result)

        monkeypatch.setattr(adapter, "_should_try_cardkit", lambda: False)
        monkeypatch.setattr(adapter, "_send_cardkit", cardkit, raising=False)
        monkeypatch.setattr(adapter, "_send_standard_message", standard)

        result = await adapter.send(
            "chat_1",
            "hello",
            reply_to=None,
            metadata=None,
        )

        assert result is standard_result
        cardkit.assert_not_awaited()
        standard.assert_awaited_once_with(
            "chat_1",
            "hello",
            reply_to=None,
            metadata=None,
        )


class TestCardkitSendHelpers:
    @staticmethod
    def _adapter():
        adapter = object.__new__(FeishuAdapter)
        adapter._cardkit_sessions = {}
        adapter._cardkit_open_by_chat = {}
        return adapter

    @pytest.mark.asyncio
    async def test_close_cardkit_siblings_removes_only_closed_tool_progress_sessions(self):
        adapter = self._adapter()
        tool_session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            closed=False,
            close=AsyncMock(return_value=SendResult(success=True, message_id="tool_msg")),
        )
        assistant_session = SimpleNamespace(
            profile=CARDKIT_ASSISTANT_PROFILE,
            closed=False,
            close=AsyncMock(return_value=SendResult(success=True, message_id="assistant_msg")),
        )
        adapter._cardkit_sessions = {
            "tool_msg": tool_session,
            "assistant_msg": assistant_session,
        }
        adapter._cardkit_open_by_chat = {
            "chat_1": {
                "tool_msg": tool_session,
                "assistant_msg": assistant_session,
            }
        }

        await adapter._close_cardkit_siblings("chat_1")

        tool_session.close.assert_awaited_once_with(None)
        assistant_session.close.assert_not_awaited()
        assert adapter._cardkit_sessions == {"assistant_msg": assistant_session}
        assert adapter._cardkit_open_by_chat == {
            "chat_1": {"assistant_msg": assistant_session}
        }

    @pytest.mark.asyncio
    async def test_close_cardkit_siblings_keeps_failed_tool_progress_session_tracked(self):
        adapter = self._adapter()
        failed_session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            closed=False,
            close=AsyncMock(return_value=SendResult(success=False, error="still open")),
        )
        raised_session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            closed=False,
            close=AsyncMock(side_effect=RuntimeError("network")),
        )
        adapter._cardkit_sessions = {
            "failed_msg": failed_session,
            "raised_msg": raised_session,
        }
        adapter._cardkit_open_by_chat = {
            "chat_1": {
                "failed_msg": failed_session,
                "raised_msg": raised_session,
            }
        }

        await adapter._close_cardkit_siblings("chat_1")

        failed_session.close.assert_awaited_once_with(None)
        raised_session.close.assert_awaited_once_with(None)
        assert adapter._cardkit_sessions == {
            "failed_msg": failed_session,
            "raised_msg": raised_session,
        }
        assert adapter._cardkit_open_by_chat == {
            "chat_1": {
                "failed_msg": failed_session,
                "raised_msg": raised_session,
            }
        }

    @pytest.mark.asyncio
    async def test_close_cardkit_siblings_removes_already_closed_tool_progress_session(self):
        adapter = self._adapter()
        closed_session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            closed=True,
            close=AsyncMock(return_value=SendResult(success=True, message_id="closed_msg")),
        )
        adapter._cardkit_sessions = {"closed_msg": closed_session}
        adapter._cardkit_open_by_chat = {"chat_1": {"closed_msg": closed_session}}

        await adapter._close_cardkit_siblings("chat_1")

        closed_session.close.assert_not_awaited()
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_send_cardkit_reference_sends_interactive_payload_and_finalizes(self, monkeypatch):
        adapter = self._adapter()
        response = SimpleNamespace(data=SimpleNamespace(message_id="msg_1"))
        send_with_retry = AsyncMock(return_value=response)
        finalized = SendResult(success=True, message_id="msg_1", raw_response=response)
        finalize_calls = []

        def finalize(actual_response, default_message):
            finalize_calls.append((actual_response, default_message))
            return finalized

        monkeypatch.setattr(adapter, "_feishu_send_with_retry", send_with_retry)
        monkeypatch.setattr(adapter, "_finalize_send_result", finalize)

        result = await adapter._send_cardkit_reference(
            card_id="card_1",
            chat_id="chat_1",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )

        assert result is finalized
        send_with_retry.assert_awaited_once_with(
            chat_id="chat_1",
            msg_type="interactive",
            payload=json.dumps(
                {"type": "card", "data": {"card_id": "card_1"}},
                ensure_ascii=False,
            ),
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )
        assert finalize_calls == [(response, "cardkit send failed")]

    @pytest.mark.asyncio
    async def test_send_cardkit_reference_retries_transient_cardid_invalid(self, monkeypatch):
        adapter = self._adapter()
        invalid = SimpleNamespace(
            success=lambda: False,
            code=230099,
            msg="Failed to create card content, ext=ErrCode: 11310; ErrMsg: cardid is invalid;",
        )
        ok = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="msg_1"))
        send_with_retry = AsyncMock(side_effect=[invalid, invalid, ok])
        sleeps: list[float] = []
        monkeypatch.setattr(feishu_adapter.asyncio, "sleep", AsyncMock(side_effect=lambda s: sleeps.append(s)))
        finalized = SendResult(success=True, message_id="msg_1", raw_response=ok)
        monkeypatch.setattr(adapter, "_feishu_send_with_retry", send_with_retry)
        monkeypatch.setattr(adapter, "_finalize_send_result", lambda resp, default_message: finalized)

        result = await adapter._send_cardkit_reference(
            card_id="card_1", chat_id="chat_1", reply_to=None, metadata=None
        )

        assert result is finalized
        assert send_with_retry.await_count == 3
        # 0.3s initial wait before the first attempt, then (0.5, 1.0) backoff
        # across two retries before the successful third send
        assert sleeps == [0.3, 0.5, 1.0]

    @pytest.mark.asyncio
    async def test_send_cardkit_reference_does_not_retry_non_cardid_failure(self, monkeypatch):
        adapter = self._adapter()
        other = SimpleNamespace(
            success=lambda: False,
            code=230002,
            msg="some other card content error",
        )
        send_with_retry = AsyncMock(return_value=other)
        monkeypatch.setattr(feishu_adapter.asyncio, "sleep", AsyncMock())
        finalized = SendResult(success=False, error="[230002] boom", raw_response=other)
        monkeypatch.setattr(adapter, "_feishu_send_with_retry", send_with_retry)
        monkeypatch.setattr(adapter, "_finalize_send_result", lambda resp, default_message: finalized)

        result = await adapter._send_cardkit_reference(
            card_id="card_1", chat_id="chat_1", reply_to=None, metadata=None
        )

        assert result is finalized
        send_with_retry.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_streaming_cardkit_registers_successful_session(self, monkeypatch):
        adapter = self._adapter()
        close_siblings = AsyncMock()
        monkeypatch.setattr(adapter, "_close_cardkit_siblings", close_siblings)
        created_sessions = []

        class FakeStreamingSession:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.profile = kwargs["profile"]
                self.start = AsyncMock(
                    return_value=SendResult(success=True, message_id="stream_msg")
                )
                created_sessions.append(self)

        monkeypatch.setattr(
            feishu_adapter,
            "FeishuStreamingCardSession",
            FakeStreamingSession,
        )
        cardkit_client = object()

        result = await adapter._send_streaming_cardkit(
            cardkit_client,
            "chat_1",
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
            profile=CARDKIT_ASSISTANT_PROFILE,
        )

        assert result.success is True
        assert result.message_id == "stream_msg"
        close_siblings.assert_awaited_once_with("chat_1")
        session = created_sessions[0]
        assert session.kwargs == {
            "client": cardkit_client,
            "chat_id": "chat_1",
            "send_card_reference": adapter._send_cardkit_reference,
            "profile": CARDKIT_ASSISTANT_PROFILE,
        }
        session.start.assert_awaited_once_with(
            "hello",
            reply_to="reply_1",
            metadata={"expect_edits": True},
        )
        assert adapter._cardkit_sessions == {"stream_msg": session}
        assert adapter._cardkit_open_by_chat == {
            "chat_1": {"stream_msg": session}
        }

    @pytest.mark.asyncio
    async def test_send_static_cardkit_closes_card_when_reference_fails(self, monkeypatch):
        adapter = self._adapter()
        monkeypatch.setattr(adapter, "_close_cardkit_siblings", AsyncMock())
        cardkit_client = MagicMock()
        cardkit_client.create_card = AsyncMock(return_value="card_static")
        cardkit_client.close_card = AsyncMock(return_value=None)
        failed = SendResult(success=False, error="ref failed", raw_response={"r": 1})
        monkeypatch.setattr(
            adapter,
            "_send_cardkit_reference",
            AsyncMock(return_value=failed),
        )

        result = await adapter._send_static_cardkit(
            cardkit_client,
            "chat_1",
            "hello",
            reply_to=None,
            metadata=None,
        )

        assert result is failed
        cardkit_client.create_card.assert_awaited_once()
        # best-effort close fires when the reference send fails
        cardkit_client.close_card.assert_awaited_once()
        assert cardkit_client.close_card.call_args.args[0] == "card_static"
        assert cardkit_client.close_card.call_args.args[1] == "hello"
        assert cardkit_client.close_card.call_args.args[2] == 1

    @pytest.mark.asyncio
    async def test_send_static_cardkit_closes_card_when_reference_raises(self, monkeypatch):
        adapter = self._adapter()
        monkeypatch.setattr(adapter, "_close_cardkit_siblings", AsyncMock())
        cardkit_client = MagicMock()
        cardkit_client.create_card = AsyncMock(return_value="card_static")
        cardkit_client.close_card = AsyncMock(return_value=None)
        monkeypatch.setattr(
            adapter,
            "_send_cardkit_reference",
            AsyncMock(side_effect=RuntimeError("ref boom")),
        )

        with pytest.raises(RuntimeError, match="ref boom"):
            await adapter._send_static_cardkit(
                cardkit_client,
                "chat_1",
                "hello",
                reply_to=None,
                metadata=None,
            )

        cardkit_client.create_card.assert_awaited_once()
        # best-effort close fires even when the reference send raises, and the
        # original exception propagates unchanged
        cardkit_client.close_card.assert_awaited_once()
        assert cardkit_client.close_card.call_args.args[0] == "card_static"

    @pytest.mark.asyncio
    async def test_send_static_cardkit_close_error_is_swallowed(self, monkeypatch):
        # best-effort close must never mask the original reference-send failure
        adapter = self._adapter()
        monkeypatch.setattr(adapter, "_close_cardkit_siblings", AsyncMock())
        cardkit_client = MagicMock()
        cardkit_client.create_card = AsyncMock(return_value="card_static")
        cardkit_client.close_card = AsyncMock(side_effect=RuntimeError("close boom"))
        monkeypatch.setattr(
            adapter,
            "_send_cardkit_reference",
            AsyncMock(return_value=SendResult(success=False, error="ref failed")),
        )

        result = await adapter._send_static_cardkit(
            cardkit_client,
            "chat_1",
            "hello",
            reply_to=None,
            metadata=None,
        )

        assert result.success is False
        assert result.error == "ref failed"
        cardkit_client.close_card.assert_awaited_once()


class TestEditMessageCardkit:
    @staticmethod
    def _adapter():
        adapter = object.__new__(FeishuAdapter)
        adapter._client = object()
        adapter._cardkit_sessions = {}
        adapter._cardkit_open_by_chat = {}
        return adapter

    @pytest.mark.asyncio
    async def test_tracked_cardkit_session_updates_without_finalizing(self):
        adapter = self._adapter()
        result = SendResult(success=True, message_id="msg_1")
        session = SimpleNamespace(update=AsyncMock(return_value=result))
        adapter._cardkit_sessions = {"msg_1": session}

        actual = await adapter.edit_message(
            "chat_1",
            "msg_1",
            "partial content",
            finalize=False,
        )

        assert actual is result
        session.update.assert_awaited_once_with("partial content")
        assert adapter._cardkit_open_by_chat == {
            "chat_1": {"msg_1": session}
        }

    @pytest.mark.asyncio
    async def test_tracked_cardkit_session_closes_and_removes_tracking_on_finalize(self):
        adapter = self._adapter()
        result = SendResult(success=True, message_id="msg_1")
        session = SimpleNamespace(close=AsyncMock(return_value=result))
        adapter._cardkit_sessions = {"msg_1": session}
        adapter._cardkit_open_by_chat = {"chat_1": {"msg_1": session}}

        actual = await adapter.edit_message(
            "chat_1",
            "msg_1",
            "final content",
            finalize=True,
        )

        assert actual is result
        session.close.assert_awaited_once_with("final content")
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_untracked_message_uses_standard_update_path(self, monkeypatch):
        adapter = self._adapter()
        captured = {}

        class MessageAPI:
            def update(self, request):
                captured["request"] = request
                return SimpleNamespace(success=lambda: True)

        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=MessageAPI()))
        )

        async def direct(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr(feishu_adapter.asyncio, "to_thread", direct)

        result = await adapter.edit_message(
            "chat_1",
            "standard_msg",
            "standard content",
            finalize=True,
        )

        assert result.success is True
        assert result.message_id == "standard_msg"
        assert captured["request"].message_id == "standard_msg"
        assert captured["request"].request_body.msg_type == "text"
        assert captured["request"].request_body.content == json.dumps(
            {"text": "standard content"},
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    async def test_edit_tool_progress_finalize_calls_update_not_close(self):
        # tool_progress cards must stay streaming across finalize=True edits
        # (run.py passes finalize=True on every progress edit). Only the
        # assistant profile should close on finalize.
        adapter = self._adapter()
        result = SendResult(success=True, message_id="msg_tp")
        session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            update=AsyncMock(return_value=result),
            close=AsyncMock(return_value=SendResult(success=True, message_id="msg_tp")),
        )
        adapter._cardkit_sessions = {"msg_tp": session}
        adapter._cardkit_open_by_chat = {"chat_1": {"msg_tp": session}}

        actual = await adapter.edit_message(
            "chat_1",
            "msg_tp",
            "progress content",
            finalize=True,
        )

        assert actual is result
        session.update.assert_awaited_once_with("progress content")
        session.close.assert_not_awaited()
        # session REMAINS tracked so subsequent edits hit the CardKit path
        assert adapter._cardkit_sessions == {"msg_tp": session}
        assert adapter._cardkit_open_by_chat == {"chat_1": {"msg_tp": session}}

    @pytest.mark.asyncio
    async def test_edit_assistant_finalize_calls_close_and_pops(self):
        adapter = self._adapter()
        result = SendResult(success=True, message_id="msg_asst")
        session = SimpleNamespace(
            profile=CARDKIT_ASSISTANT_PROFILE,
            update=AsyncMock(return_value=SendResult(success=True, message_id="msg_asst")),
            close=AsyncMock(return_value=result),
        )
        adapter._cardkit_sessions = {"msg_asst": session}
        adapter._cardkit_open_by_chat = {"chat_1": {"msg_asst": session}}

        actual = await adapter.edit_message(
            "chat_1",
            "msg_asst",
            "final content",
            finalize=True,
        )

        assert actual is result
        session.close.assert_awaited_once_with("final content")
        session.update.assert_not_awaited()
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_edit_tool_progress_non_finalize_calls_update(self):
        adapter = self._adapter()
        result = SendResult(success=True, message_id="msg_tp")
        session = SimpleNamespace(
            profile=CARDKIT_TOOL_PROGRESS_PROFILE,
            update=AsyncMock(return_value=result),
            close=AsyncMock(),
        )
        adapter._cardkit_sessions = {"msg_tp": session}
        adapter._cardkit_open_by_chat = {"chat_1": {"msg_tp": session}}

        actual = await adapter.edit_message(
            "chat_1",
            "msg_tp",
            "more progress",
            finalize=False,
        )

        assert actual is result
        session.update.assert_awaited_once_with("more progress")
        session.close.assert_not_awaited()
        assert adapter._cardkit_sessions == {"msg_tp": session}
        assert adapter._cardkit_open_by_chat == {"chat_1": {"msg_tp": session}}


class TestDisconnectCardkitCleanup:
    @staticmethod
    def _adapter():
        adapter = object.__new__(FeishuAdapter)
        adapter._cardkit_sessions = {}
        adapter._cardkit_open_by_chat = {}
        return adapter

    @pytest.mark.asyncio
    async def test_close_all_cardkit_sessions_closes_each_and_clears_dicts(self):
        adapter = self._adapter()
        session_a = SimpleNamespace(
            current_text="alpha",
            close=AsyncMock(return_value=SendResult(success=True, message_id="msg_a")),
        )
        session_b = SimpleNamespace(
            current_text="beta",
            close=AsyncMock(return_value=SendResult(success=True, message_id="msg_b")),
        )
        adapter._cardkit_sessions = {"msg_a": session_a, "msg_b": session_b}
        adapter._cardkit_open_by_chat = {
            "chat_1": {"msg_a": session_a},
            "chat_2": {"msg_b": session_b},
        }

        await adapter._close_all_cardkit_sessions()

        session_a.close.assert_awaited_once_with("alpha")
        session_b.close.assert_awaited_once_with("beta")
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_close_all_cardkit_sessions_swallows_errors_and_still_clears(self):
        # one session raises during close: the other must still be closed and
        # both dicts must still be cleared (best-effort cleanup).
        adapter = self._adapter()
        raising_session = SimpleNamespace(
            current_text="raising",
            close=AsyncMock(side_effect=RuntimeError("boom")),
        )
        ok_session = SimpleNamespace(
            current_text="ok",
            close=AsyncMock(return_value=SendResult(success=True, message_id="msg_ok")),
        )
        adapter._cardkit_sessions = {"msg_raise": raising_session, "msg_ok": ok_session}
        adapter._cardkit_open_by_chat = {
            "chat_1": {"msg_raise": raising_session, "msg_ok": ok_session},
        }

        await adapter._close_all_cardkit_sessions()

        raising_session.close.assert_awaited_once_with("raising")
        ok_session.close.assert_awaited_once_with("ok")
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_close_all_cardkit_sessions_noop_when_empty(self):
        adapter = self._adapter()
        adapter._cardkit_sessions = {}
        adapter._cardkit_open_by_chat = {}

        await adapter._close_all_cardkit_sessions()

        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}

    @pytest.mark.asyncio
    async def test_disconnect_closes_cardkit_sessions_early(self, monkeypatch):
        # thin disconnect coverage: patch the heavy websocket internals and
        # assert open CardKit sessions are closed before websocket teardown.
        adapter = self._adapter()
        session = SimpleNamespace(
            current_text="content",
            close=AsyncMock(return_value=SendResult(success=True, message_id="msg_1")),
        )
        adapter._cardkit_sessions = {"msg_1": session}
        adapter._cardkit_open_by_chat = {"chat_1": {"msg_1": session}}
        adapter._running = True
        adapter._pending_text_batch_tasks = {}
        adapter._pending_media_batch_tasks = {}
        adapter._ws_thread_loop = None
        adapter._ws_future = None
        adapter._client = object()
        adapter._event_handler = None

        monkeypatch.setattr(feishu_adapter, "FEISHU_AVAILABLE", True)
        monkeypatch.setattr(adapter, "_cancel_pending_tasks", AsyncMock())
        monkeypatch.setattr(adapter, "_reset_batch_buffers", lambda *a, **kw: None)
        monkeypatch.setattr(adapter, "_disable_websocket_auto_reconnect", lambda *a, **kw: None)
        monkeypatch.setattr(adapter, "_stop_webhook_server", AsyncMock())
        monkeypatch.setattr(adapter, "_persist_seen_message_ids", lambda *a, **kw: None)
        monkeypatch.setattr(adapter, "_release_app_lock", AsyncMock())
        monkeypatch.setattr(adapter, "_mark_disconnected", lambda *a, **kw: None)

        await adapter.disconnect()

        session.close.assert_awaited_once_with("content")
        assert adapter._cardkit_sessions == {}
        assert adapter._cardkit_open_by_chat == {}
