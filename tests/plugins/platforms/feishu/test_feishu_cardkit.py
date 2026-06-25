"""Tests for Feishu CardKit helper module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from plugins.platforms.feishu.feishu_cardkit import (
    FeishuCardKitClient,
    FeishuStreamingCardSession,
    build_card,
    strip_streaming_cursor,
    merge_streaming_text,
    truncate_summary,
    CARD_CONTENT_ELEMENT_ID,
    MAX_CARD_TEXT_LENGTH,
    CARDKIT_ASSISTANT_PROFILE,
    CARDKIT_TOOL_PROGRESS_PROFILE,
    CARDKIT_STATIC_PROFILE,
)
from gateway.platforms.base import SendResult


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
            client = FeishuCardKitClient(sdk_client)
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
            client = FeishuCardKitClient(sdk_client)
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
            client = FeishuCardKitClient(sdk_client)
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
            client = FeishuCardKitClient(sdk_client)
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
            client = FeishuCardKitClient(sdk_client)
            await client.update_element_content("card_123", "content", "hello world", 2)
        card_element_resource.acontent.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_element_content_over_limit(self):
        sdk_client, _, card_element_resource = self._make_sdk_client()
        client = FeishuCardKitClient(sdk_client)
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
            client = FeishuCardKitClient(sdk_client)
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


class TestMergeStreamingText:
    def test_next_extends_previous(self):
        assert merge_streaming_text("hello", "hello world") == "hello world"

    def test_previous_contains_next(self):
        assert merge_streaming_text("hello world", "hello") == "hello world"

    def test_no_overlap(self):
        assert merge_streaming_text("hello", "world") == "helloworld"

    def test_overlap(self):
        assert merge_streaming_text("hello wor", "world") == "hello world"

    def test_empty_previous(self):
        assert merge_streaming_text("", "hello") == "hello"

    def test_empty_next(self):
        assert merge_streaming_text("hello", "") == "hello"

    def test_none_previous(self):
        assert merge_streaming_text(None, "hello") == "hello"

    def test_none_next(self):
        assert merge_streaming_text("hello", None) == "hello"


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
    async def test_start_initial_update_failure_closes_and_fails(self):
        client = self._make_client_mock()
        client.update_element_content = AsyncMock(side_effect=RuntimeError("update failed"))
        send_ref = self._make_send_ref_mock()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_test",
            send_card_reference=send_ref,
        )
        result = await session.start("hello", reply_to=None, metadata=None)

        assert result.success is False
        client.close_card.assert_called_once()

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
