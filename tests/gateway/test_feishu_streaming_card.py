import json


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
    assert merge_streaming_text("hello wor", "world") == "hello world"
    assert merge_streaming_text("abc", "xyz") == "abcxyz"


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
