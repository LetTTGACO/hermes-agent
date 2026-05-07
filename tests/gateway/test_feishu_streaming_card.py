import asyncio
from types import SimpleNamespace


def test_resolve_api_base_supports_feishu_lark_and_custom():
    from gateway.platforms.feishu_streaming_card import resolve_api_base

    assert resolve_api_base("feishu") == "https://open.feishu.cn/open-apis"
    assert resolve_api_base("lark") == "https://open.larksuite.com/open-apis"
    assert resolve_api_base("https://open.example.test/") == "https://open.example.test/open-apis"


def test_resolve_receive_id_type_uses_open_id_for_user_dm():
    from gateway.platforms.feishu_streaming_card import resolve_receive_id_type

    assert resolve_receive_id_type("ou_user") == "open_id"
    assert resolve_receive_id_type("oc_chat") == "chat_id"


def test_build_streaming_card_json_2_0_defaults():
    from gateway.platforms.feishu_streaming_card import build_streaming_card

    card = build_streaming_card()

    assert card["schema"] == "2.0"
    assert card["config"]["streaming_mode"] is True
    assert card["config"]["summary"] == {"content": "[Generating...]"}
    assert card["config"]["streaming_config"] == {
        "print_frequency_ms": {"default": 70},
        "print_step": {"default": 1},
        "print_strategy": "fast",
    }
    assert card["body"]["elements"] == [
        {"tag": "markdown", "content": "", "element_id": "content"}
    ]


def test_merge_streaming_text_preserves_prefix_and_overlap():
    from gateway.platforms.feishu_streaming_card import merge_streaming_text

    assert merge_streaming_text("", "hello") == "hello"
    assert merge_streaming_text("hello", "hello world") == "hello world"
    assert merge_streaming_text("hello world", "hello") == "hello world"
    assert merge_streaming_text("hello wor", "world") == "hello world"
    assert merge_streaming_text("abc", "xyz") == "abcxyz"


def test_truncate_summary_leaves_short_text_and_truncates_long_text():
    from gateway.platforms.feishu_streaming_card import truncate_summary

    assert truncate_summary("short", max_chars=10) == "short"
    assert truncate_summary("abcdefghij", max_chars=5) == "abcde"


def test_has_natural_streaming_boundary_supports_ascii_and_chinese_punctuation():
    from gateway.platforms.feishu_streaming_card import has_natural_streaming_boundary

    assert has_natural_streaming_boundary("hello.") is True
    assert has_natural_streaming_boundary("你好！") is True
    assert has_natural_streaming_boundary("继续") is False
    assert has_natural_streaming_boundary("") is False


def test_should_push_streaming_update_uses_boundary_and_delta():
    from gateway.platforms.feishu_streaming_card import should_push_streaming_update

    assert should_push_streaming_update("", "hi", block_streaming=True) is True
    assert should_push_streaming_update("hello", "hello there", block_streaming=True) is False
    assert should_push_streaming_update("hello", "hello there.", block_streaming=True) is True
    assert should_push_streaming_update("a", "a" + "b" * 18, block_streaming=True) is True
    assert should_push_streaming_update("hello", "hello there", block_streaming=False) is True


def test_strip_streaming_cursor_removes_gateway_cursor_suffix():
    from gateway.platforms.feishu_streaming_card import strip_streaming_cursor

    assert strip_streaming_cursor("answer ▉") == "answer"
    assert strip_streaming_cursor("answer▉") == "answer"
    assert strip_streaming_cursor("answer") == "answer"


class FakeCardKitClient:
    def __init__(self):
        self.calls = []

    async def get_token(self):
        self.calls.append(("token",))
        return "tenant_token"

    async def create_card(self):
        self.calls.append(("create_card",))
        return "card_123"

    async def update_element_content(self, card_id, element_id, content, sequence):
        self.calls.append(("update", card_id, element_id, content, sequence))

    async def close_card(self, card_id, final_text, sequence):
        self.calls.append(("close", card_id, final_text, sequence))


async def fake_send_card_reference(*, card_id, chat_id, reply_to, metadata):
    return SimpleNamespace(success=True, message_id=f"msg_for_{card_id}")


def test_session_start_creates_card_and_sends_reference():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
        block_streaming=True,
    )

    result = asyncio.run(session.start("hello", reply_to=None, metadata=None))

    assert result.success is True
    assert result.message_id == "msg_for_card_123"
    assert session.message_id == "msg_for_card_123"
    assert client.calls == [
        ("create_card",),
        ("update", "card_123", "content", "hello", 2),
    ]


def test_session_update_sends_full_snapshot_and_increments_sequence():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
        block_streaming=False,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    asyncio.run(session.update("hello world"))

    assert ("update", "card_123", "content", "hello world", 3) in client.calls


def test_session_update_blocks_small_delta_until_close_when_block_streaming_true():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
        block_streaming=True,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    asyncio.run(session.update("hello there"))
    asyncio.run(session.close("hello there"))

    assert ("update", "card_123", "content", "hello there", 3) in client.calls
    assert client.calls[-1] == ("close", "card_123", "hello there", 4)


def test_session_close_disables_streaming_mode_once():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
        block_streaming=True,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    asyncio.run(session.close("hello final"))
    asyncio.run(session.close("ignored"))

    close_calls = [call for call in client.calls if call[0] == "close"]
    assert close_calls == [("close", "card_123", "hello final", 4)]
