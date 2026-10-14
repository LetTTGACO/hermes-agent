"""Personal fork tests for Feishu CardKit helper module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from plugins.platforms.feishu.feishu_cardkit import (
    FeishuCardKitClient,
    FeishuCardKitRuntime,
    FeishuStreamingCardSession,
    build_card,
    strip_streaming_cursor,
    truncate_summary,
    CARD_CONTENT_ELEMENT_ID,
    MAX_CARD_TEXT_LENGTH,
    CARDKIT_ASSISTANT_PROFILE,
    CARDKIT_TOOL_PROGRESS_PROFILE,
    CARDKIT_STATIC_PROFILE,
)
from gateway.platforms.base import SendResult
from gateway.stream_consumer import ensure_closed_code_fences


def _mock_builders():
    """Return a 6-tuple of MagicMock builder classes for _import_cardkit_sdk."""
    return tuple(MagicMock() for _ in range(6))


class TestFeishuCardKitClient:
    def _make_sdk_client(self):
        """Create a mock SDK client with cardkit.v1 chain."""
        client = MagicMock()
        card_resource = MagicMock()
        card_element_resource = MagicMock()
        client.cardkit.v1.card = card_resource
        client.cardkit.v1.card_element = card_element_resource
        return client, card_resource, card_element_resource

    @pytest.mark.asyncio
    async def test_create_card_streaming(self):
        sdk_client, card_resource, _ = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = True
        response.code = 0
        response.data.card_id = "card_123"
        card_resource.acreate = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            card_id = await client.create_card(
                content="hello", streaming_mode=True, profile=CARDKIT_ASSISTANT_PROFILE
            )
        assert card_id == "card_123"
        card_resource.acreate.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_card_static(self):
        sdk_client, card_resource, _ = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = True
        response.code = 0
        response.data.card_id = "card_456"
        card_resource.acreate = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            card_id = await client.create_card(
                content="final text", streaming_mode=False, profile=CARDKIT_STATIC_PROFILE
            )
        assert card_id == "card_456"

    @pytest.mark.asyncio
    async def test_create_card_missing_card_id(self):
        sdk_client, card_resource, _ = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = True
        response.code = 0
        response.data.card_id = None
        card_resource.acreate = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            with pytest.raises(RuntimeError, match="missing card_id"):
                await client.create_card(content="hello", streaming_mode=True)

    @pytest.mark.asyncio
    async def test_create_card_api_failure(self):
        sdk_client, card_resource, _ = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = False
        response.code = 999
        response.msg = "permission denied"
        card_resource.acreate = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            with pytest.raises(RuntimeError, match="permission denied"):
                await client.create_card(content="hello", streaming_mode=True)

    @pytest.mark.asyncio
    async def test_update_element_content(self):
        sdk_client, _, card_element_resource = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = True
        response.code = 0
        card_element_resource.acontent = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            await client.update_element_content("card_123", "content", "hello world", 2)
        card_element_resource.acontent.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_element_content_over_limit(self):
        sdk_client, _, card_element_resource = self._make_sdk_client()
        client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
        long_text = "a" * (MAX_CARD_TEXT_LENGTH + 1)
        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            with pytest.raises(ValueError, match="exceeds"):
                await client.update_element_content("card_123", "content", long_text, 2)
        card_element_resource.acontent.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_card(self):
        sdk_client, card_resource, _ = self._make_sdk_client()
        response = MagicMock()
        response.success.return_value = True
        response.code = 0
        card_resource.asettings = AsyncMock(return_value=response)

        with patch(
            "plugins.platforms.feishu.feishu_cardkit._import_cardkit_sdk",
            return_value=_mock_builders(),
        ):
            client = FeishuCardKitClient(get_sdk_client=lambda: sdk_client)
            await client.close_card("card_123", "final text", 5)
        card_resource.asettings.assert_called_once()


class TestStripStreamingCursor:
    def test_strips_trailing_block_cursor(self):
        assert strip_streaming_cursor("hello ▉") == "hello"

    def test_strips_trailing_cursor_no_space(self):
        assert strip_streaming_cursor("hello▉") == "hello"

    def test_no_cursor_unchanged(self):
        assert strip_streaming_cursor("hello") == "hello"

    def test_empty_string(self):
        assert strip_streaming_cursor("") == ""


class TestTruncateSummary:
    def test_short_text_unchanged(self):
        assert truncate_summary("hello") == "hello"

    def test_long_text_truncated(self):
        text = "a" * 100
        assert len(truncate_summary(text)) == 50

    def test_exact_boundary(self):
        text = "a" * 50
        assert truncate_summary(text) == text


class TestBuildCard:
    def test_streaming_assistant_profile(self):
        card = build_card("", streaming_mode=True, profile=CARDKIT_ASSISTANT_PROFILE)
        assert card["schema"] == "2.0"
        assert card["config"]["streaming_mode"] is True
        assert "streaming_config" in card["config"]
        assert card["config"]["streaming_config"]["print_step"]["default"] == 5
        assert card["config"]["streaming_config"]["print_frequency_ms"]["default"] == 50
        assert card["config"]["streaming_config"]["print_strategy"] == "fast"
        assert card["body"]["elements"][0]["tag"] == "markdown"
        assert card["body"]["elements"][0]["element_id"] == CARD_CONTENT_ELEMENT_ID

    def test_streaming_tool_progress_profile(self):
        card = build_card("", streaming_mode=True, profile=CARDKIT_TOOL_PROGRESS_PROFILE)
        assert card["config"]["streaming_mode"] is True
        assert card["config"]["streaming_config"]["print_step"]["default"] == 80
        assert card["config"]["streaming_config"]["print_frequency_ms"]["default"] == 30

    def test_static_card(self):
        content = "Hello, world!"
        card = build_card(content, streaming_mode=False, profile=CARDKIT_STATIC_PROFILE)
        assert card["config"]["streaming_mode"] is False
        assert "streaming_config" not in card["config"]
        assert card["body"]["elements"][0]["content"] == content

    def test_streaming_card_summary_generating(self):
        card = build_card("", streaming_mode=True, profile=CARDKIT_ASSISTANT_PROFILE)
        assert card["config"]["summary"]["content"] == "[Generating...]"

    def test_static_card_summary_from_content(self):
        content = "This is a long response that should be summarized"
        card = build_card(content, streaming_mode=False, profile=CARDKIT_STATIC_PROFILE)
        assert card["config"]["summary"]["content"] == truncate_summary(content)

    def test_streaming_card_with_content_has_summary(self):
        content = "Some visible text here"
        card = build_card(content, streaming_mode=True, profile=CARDKIT_ASSISTANT_PROFILE)
        assert card["config"]["summary"]["content"] == content


class TestFeishuStreamingCardSession:
    """Tests for FeishuStreamingCardSession lifecycle management."""

    def _make_client_mock(self):
        client = MagicMock()
        client.create_card = AsyncMock(return_value="card_123")
        client.update_element_content = AsyncMock(return_value=None)
        client.close_card = AsyncMock(return_value=None)
        return client

    def _make_send_ref_mock(self, message_id="om_abc"):
        send_ref = AsyncMock(return_value=SendResult(success=True, message_id=message_id))
        return send_ref

    @pytest.mark.asyncio
    async def test_start_creates_card_and_sends_reference(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello", reply_to="om_reply", metadata={"k": "v"})

        assert result.success is True
        assert result.message_id == "om_abc"
        assert session.card_id == "card_123"
        assert session.message_id == "om_abc"
        client.create_card.assert_called_once()
        send_ref.assert_called_once()
        assert send_ref.call_args.kwargs["card_id"] == "card_123"
        assert send_ref.call_args.kwargs["chat_id"] == "oc_test"
        assert send_ref.call_args.kwargs["reply_to"] == "om_reply"
        assert send_ref.call_args.kwargs["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_start_with_initial_text_pushes_update(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello world", reply_to=None, metadata=None)

        assert result.success is True
        assert session.current_text == "hello world"
        client.update_element_content.assert_called_once()
        call_args = client.update_element_content.call_args
        assert call_args.args[0] == "card_123"
        assert call_args.args[1] == CARD_CONTENT_ELEMENT_ID
        assert call_args.args[2] == "hello world"

    @pytest.mark.asyncio
    async def test_start_empty_text_skips_update(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("", reply_to=None, metadata=None)

        assert result.success is True
        client.update_element_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_send_ref_failure(self):
        client = self._make_client_mock()
        send_ref = AsyncMock(
            return_value=SendResult(
                success=False, error="send failed", raw_response={"r": 1}
            )
        )
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello", reply_to=None, metadata=None)

        assert result.success is False
        assert session.card_id == "card_123"
        # raw_response passthrough preserved on the failed result
        assert result.raw_response == {"r": 1}
        # best-effort close fires when the reference send fails
        client.close_card.assert_awaited_once()
        assert client.close_card.call_args.args[0] == "card_123"

    @pytest.mark.asyncio
    async def test_start_send_ref_raises_closes_card_and_propagates(self):
        client = self._make_client_mock()
        send_ref = AsyncMock(side_effect=RuntimeError("send boom"))
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        with pytest.raises(RuntimeError, match="send boom"):
            await session.start("hello", reply_to=None, metadata=None)

        # best-effort close fires even when the reference send raises, and the
        # original exception propagates unchanged
        client.close_card.assert_awaited_once()
        assert client.close_card.call_args.args[0] == "card_123"

    @pytest.mark.asyncio
    async def test_start_send_ref_failure_close_error_is_swallowed(self):
        # best-effort close must never mask the original reference-send failure
        client = self._make_client_mock()
        client.close_card = AsyncMock(side_effect=RuntimeError("close boom"))
        send_ref = AsyncMock(
            return_value=SendResult(success=False, error="send failed")
        )
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello", reply_to=None, metadata=None)

        assert result.success is False
        assert result.error == "send failed"
        client.close_card.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_initial_update_failure_preserves_acknowledged_snapshot(self):
        client = self._make_client_mock()
        client.update_element_content = AsyncMock(side_effect=RuntimeError("update failed"))
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello", reply_to=None, metadata=None)

        assert result.success is True
        assert result.message_id == session.message_id
        assert session.current_text == "hello"
        client.close_card.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_pushes_merged_text(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        await session.start("hello", reply_to=None, metadata=None)

        # Reset mock to isolate the update call
        client.update_element_content.reset_mock()
        result = await session.update("hello world")

        assert result.success is True
        client.update_element_content.assert_called_once()
        call_args = client.update_element_content.call_args
        assert call_args.args[2] == "hello world"
        assert session.current_text == "hello world"

    @pytest.mark.asyncio
    async def test_update_replaces_reformatted_full_snapshot_without_duplication(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        await session.start("", reply_to=None, metadata=None)

        # GatewayStreamConsumer passes full snapshots to edit_message(). While
        # streaming, it may append a synthetic backtick after the cursor to
        # balance incomplete Markdown. The next, longer snapshot then no longer
        # has the prior rendered frame as a prefix.
        partial = "前言\n\n## 模型对比\n\n`GLM"
        partial_frame = ensure_closed_code_fences(partial + " ▉")
        complete = (
            "前言\n\n## 模型对比\n\n"
            "`GLM-5.2` vs `Doubao-Seed-2.1-Turbo`\n\n"
            "| 模型 | 特点 |\n| --- | --- |\n| GLM-5.2 | 长文本 |"
        )
        complete_frame = ensure_closed_code_fences(complete + " ▉")

        await session.update(partial_frame)
        await session.update(complete_frame)

        partial_visible = strip_streaming_cursor(partial_frame)
        expected = strip_streaming_cursor(complete_frame)
        assert not expected.startswith(partial_visible)
        assert session.current_text == expected
        assert client.update_element_content.await_args_list[-1].args[2] == expected

    @pytest.mark.asyncio
    async def test_update_same_text_noop(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        await session.start("hello", reply_to=None, metadata=None)

        client.update_element_content.reset_mock()
        result = await session.update("hello")

        assert result.success is True
        client.update_element_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_pushes_final_and_closes(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        await session.start("hello", reply_to=None, metadata=None)

        client.update_element_content.reset_mock()
        client.close_card.reset_mock()
        result = await session.close("hello world")

        assert result.success is True
        assert session.closed is True
        # Final update pushed because merged text differs from current
        client.update_element_content.assert_called_once()
        client.close_card.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_already_closed(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        await session.start("hello", reply_to=None, metadata=None)

        await session.close("hello world")
        client.close_card.reset_mock()
        result = await session.close("hello world again")

        assert result.success is True
        client.close_card.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_without_active_session(self):
        client = self._make_client_mock()
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.close("hello world")

        assert result.success is False
        client.close_card.assert_not_called()


def _runtime_client():
    client = MagicMock()
    client.create_card = AsyncMock(side_effect=["card-progress", "card-final"])
    client.update_element_content = AsyncMock(return_value=None)
    client.close_card = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_runtime_uses_latest_complete_snapshot_without_duplication():
    client = _runtime_client()
    send_reference = AsyncMock(
        return_value=SendResult(success=True, message_id="om-assistant")
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)

    sent = await runtime.try_send(
        "oc_chat", "# Title\nfirst", reply_to=None, metadata={"expect_edits": True}
    )
    edited = await runtime.edit_if_managed(
        "oc_chat", sent.message_id, "# Title\nfirst\nsecond", finalize=False
    )

    assert edited.success is True
    assert client.update_element_content.await_args_list[-1].args[2] == (
        "# Title\nfirst\nsecond"
    )


@pytest.mark.asyncio
async def test_runtime_tool_progress_finalize_updates_until_next_card_closes_it():
    client = _runtime_client()
    send_reference = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id="om-progress"),
            SendResult(success=True, message_id="om-final"),
        ]
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)

    progress = await runtime.try_send(
        "oc_chat", "running", reply_to=None,
        metadata={"message_kind": "tool_progress", "_cardkit_turn_id": "om-in"},
    )
    update = await runtime.edit_if_managed(
        "oc_chat", progress.message_id, "running tool 2", finalize=True
    )
    assert update.success is True
    client.close_card.assert_not_awaited()

    final = await runtime.try_send(
        "oc_chat", "done", reply_to=None, metadata={"_cardkit_turn_id": "om-in"}
    )
    assert final.success is True
    client.close_card.assert_awaited_once()
    assert await runtime.edit_if_managed(
        "oc_chat", progress.message_id, "late", finalize=False
    ) is None


@pytest.mark.asyncio
async def test_runtime_retries_only_cardid_invalid_reference(monkeypatch):
    client = _runtime_client()
    invalid = MagicMock(code=230099, msg="ErrCode: 11310 cardid is invalid")
    send_reference = AsyncMock(side_effect=[
        SendResult(success=False, error="cardid is invalid", raw_response=invalid),
        SendResult(success=True, message_id="om-static"),
    ])
    sleep = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", sleep
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)

    result = await runtime.try_send("oc_chat", "done", reply_to=None, metadata=None)

    assert result.success is True
    assert send_reference.await_count == 2
    assert sleep.await_args_list == [call(0.3), call(0.5)]


@pytest.mark.asyncio
async def test_runtime_does_not_retry_other_reference_failures(monkeypatch):
    client = _runtime_client()
    denied = MagicMock(code=230001, msg="permission denied")
    send_reference = AsyncMock(return_value=SendResult(
        success=False, error="permission denied", raw_response=denied
    ))
    sleep = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", sleep
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)

    result = await runtime.try_send("oc_chat", "done", reply_to=None, metadata=None)

    assert result.success is False
    assert send_reference.await_count == 1
    assert sleep.await_args_list == [call(0.3)]


@pytest.mark.asyncio
async def test_runtime_closes_orphan_when_reference_fails():
    client = _runtime_client()
    send_reference = AsyncMock(
        return_value=SendResult(success=False, error="send failed")
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)

    result = await runtime.try_send("oc_chat", "done", reply_to=None, metadata=None)

    assert result.success is False
    client.close_card.assert_awaited_once_with("card-progress", "done", 1)


@pytest.mark.asyncio
async def test_runtime_close_all_attempts_every_session_and_forgets_them():
    client = _runtime_client()
    send_reference = AsyncMock(side_effect=[
        SendResult(success=True, message_id="om-one"),
        SendResult(success=True, message_id="om-two"),
    ])
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)
    one = await runtime.try_send(
        "oc_one", "one", reply_to=None, metadata={"expect_edits": True}
    )
    two = await runtime.try_send(
        "oc_two", "two", reply_to=None, metadata={"expect_edits": True}
    )
    client.close_card.side_effect = [RuntimeError("first close failed"), None]

    await runtime.close_all()

    assert client.close_card.await_count == 2
    assert await runtime.edit_if_managed(
        "oc_one", one.message_id, "late", finalize=False
    ) is None
    assert await runtime.edit_if_managed(
        "oc_two", two.message_id, "late", finalize=False
    ) is None


@pytest.mark.asyncio
async def test_runtime_failed_assistant_finalize_stays_managed_for_retry():
    client = _runtime_client()
    send_reference = AsyncMock(
        return_value=SendResult(success=True, message_id="om-assistant")
    )
    runtime = FeishuCardKitRuntime(client=client, send_card_reference=send_reference)
    sent = await runtime.try_send(
        "oc_chat", "preview", reply_to=None, metadata={"expect_edits": True}
    )
    client.close_card.side_effect = RuntimeError("close failed")

    failed = await runtime.edit_if_managed(
        "oc_chat", sent.message_id, "final", finalize=True
    )
    client.close_card.side_effect = None
    retried = await runtime.edit_if_managed(
        "oc_chat", sent.message_id, "final", finalize=True
    )

    assert failed.success is False
    assert retried.success is True
    assert await runtime.edit_if_managed(
        "oc_chat", sent.message_id, "late", finalize=False
    ) is None


@pytest.mark.asyncio
async def test_runtime_serializes_progress_update_before_sibling_close(monkeypatch):
    update_started = asyncio.Event()
    release_update = asyncio.Event()
    sibling_close_attempted = asyncio.Event()
    sibling_close_called = asyncio.Event()
    operations = []

    class ControlledClient:
        def __init__(self):
            self.card_ids = iter(("card-progress", "card-static"))

        async def create_card(self, **kwargs):
            return next(self.card_ids)

        async def update_element_content(
            self, card_id, element_id, content, sequence
        ):
            operations.append(("update-start", card_id, sequence, content))
            if content == "running tool 2":
                update_started.set()
                await release_update.wait()
            operations.append(("update-end", card_id, sequence, content))

        async def close_card(self, card_id, final_text, sequence):
            operations.append(("close", card_id, sequence, final_text))
            if card_id == "card-progress":
                sibling_close_called.set()

    async def send_reference(**kwargs):
        message_id = (
            "om-progress"
            if kwargs["card_id"] == "card-progress"
            else "om-static"
        )
        return SendResult(success=True, message_id=message_id)

    sleep = AsyncMock()
    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", sleep
    )
    original_close = FeishuStreamingCardSession.close

    async def announced_close(session, final_text=None):
        if session.card_id == "card-progress":
            sibling_close_attempted.set()
        return await original_close(session, final_text)

    monkeypatch.setattr(FeishuStreamingCardSession, "close", announced_close)

    runtime = FeishuCardKitRuntime(
        client=ControlledClient(), send_card_reference=send_reference
    )
    progress = await runtime.try_send(
        "oc_chat",
        "running",
        reply_to=None,
        metadata={"message_kind": "tool_progress", "_cardkit_turn_id": "om-in"},
    )
    operations.clear()

    update_task = asyncio.create_task(
        runtime.edit_if_managed(
            "oc_chat", progress.message_id, "running tool 2", finalize=True
        )
    )
    await asyncio.wait_for(update_started.wait(), timeout=5)
    final_task = asyncio.create_task(
        runtime.try_send("oc_chat", "done", reply_to=None, metadata={"_cardkit_turn_id": "om-in"})
    )
    await asyncio.wait_for(sibling_close_attempted.wait(), timeout=5)
    close_overlapped_update = sibling_close_called.is_set()

    release_update.set()
    edited, final = await asyncio.gather(update_task, final_task)

    assert edited.success is True
    assert final.success is True
    assert close_overlapped_update is False
    progress_operations = [
        (kind, sequence)
        for kind, card_id, sequence, _ in operations
        if card_id == "card-progress"
    ]
    assert progress_operations == [
        ("update-start", 3),
        ("update-end", 3),
        ("close", 4),
    ]


@pytest.mark.asyncio
async def test_runtime_close_all_closes_session_from_inflight_send(monkeypatch):
    inflight_reference_started = asyncio.Event()
    release_inflight_reference = asyncio.Event()
    close_all_started = asyncio.Event()
    release_existing_close = asyncio.Event()
    closed_cards = []

    class ControlledClient:
        def __init__(self):
            self.card_ids = iter(("card-existing", "card-inflight"))

        async def create_card(self, **kwargs):
            return next(self.card_ids)

        async def update_element_content(self, *args):
            return None

        async def close_card(self, card_id, final_text, sequence):
            closed_cards.append(card_id)
            if card_id == "card-existing":
                close_all_started.set()
                await release_existing_close.wait()

    reference_count = 0

    async def send_reference(**kwargs):
        nonlocal reference_count
        reference_count += 1
        if reference_count == 2:
            inflight_reference_started.set()
            await release_inflight_reference.wait()
        return SendResult(
            success=True,
            message_id=("om-existing" if reference_count == 1 else "om-inflight"),
        )

    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", AsyncMock()
    )
    runtime = FeishuCardKitRuntime(
        client=ControlledClient(), send_card_reference=send_reference
    )
    await runtime.try_send(
        "oc_existing", "", reply_to=None, metadata={"expect_edits": True}
    )
    inflight_task = asyncio.create_task(
        runtime.try_send(
            "oc_inflight", "", reply_to=None, metadata={"expect_edits": True}
        )
    )
    await asyncio.wait_for(inflight_reference_started.wait(), timeout=5)

    close_task = asyncio.create_task(runtime.close_all())
    await asyncio.wait_for(close_all_started.wait(), timeout=5)
    release_inflight_reference.set()
    inflight = await asyncio.wait_for(inflight_task, timeout=5)
    release_existing_close.set()
    await asyncio.wait_for(close_task, timeout=5)

    assert inflight.success is True
    assert closed_cards == ["card-existing", "card-inflight"]
    assert await runtime.edit_if_managed(
        "oc_inflight", inflight.message_id, "late", finalize=False
    ) is None


@pytest.mark.asyncio
async def test_runtime_close_all_gates_new_sends(monkeypatch):
    close_all_started = asyncio.Event()
    release_close = asyncio.Event()
    created_cards = []

    class ControlledClient:
        async def create_card(self, **kwargs):
            card_id = f"card-{len(created_cards) + 1}"
            created_cards.append(card_id)
            return card_id

        async def update_element_content(self, *args):
            return None

        async def close_card(self, card_id, final_text, sequence):
            close_all_started.set()
            await release_close.wait()

    async def send_reference(**kwargs):
        return SendResult(success=True, message_id=f"om-{kwargs['card_id']}")

    monkeypatch.setattr(
        "plugins.platforms.feishu.feishu_cardkit.asyncio.sleep", AsyncMock()
    )
    runtime = FeishuCardKitRuntime(
        client=ControlledClient(), send_card_reference=send_reference
    )
    await runtime.try_send(
        "oc_existing", "", reply_to=None, metadata={"expect_edits": True}
    )
    close_task = asyncio.create_task(runtime.close_all())
    await asyncio.wait_for(close_all_started.wait(), timeout=5)

    late = await runtime.try_send(
        "oc_late", "", reply_to=None, metadata={"expect_edits": True}
    )
    release_close.set()
    await asyncio.wait_for(close_task, timeout=5)

    assert late is None
    assert created_cards == ["card-1"]
