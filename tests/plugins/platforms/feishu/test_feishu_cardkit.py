"""Tests for Feishu CardKit helper module."""

import pytest
from plugins.platforms.feishu.feishu_cardkit import (
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
