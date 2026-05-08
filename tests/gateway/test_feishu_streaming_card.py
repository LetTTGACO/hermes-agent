import asyncio
import json
from types import SimpleNamespace

import pytest


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
        "print_frequency_ms": {"default": 50},
        "print_step": {"default": 5},
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


def test_should_push_streaming_update_uses_internal_boundary_and_delta_rules():
    from gateway.platforms.feishu_streaming_card import should_push_streaming_update

    assert should_push_streaming_update("", "hi") is True
    assert should_push_streaming_update("hello", "hello there") is False
    assert should_push_streaming_update("hello", "hello there.") is True
    assert should_push_streaming_update("a", "a" + "b" * 18) is True


def test_strip_streaming_cursor_removes_gateway_cursor_suffix():
    from gateway.platforms.feishu_streaming_card import strip_streaming_cursor

    assert strip_streaming_cursor("answer ▉") == "answer"
    assert strip_streaming_cursor("answer▉") == "answer"
    assert strip_streaming_cursor("answer") == "answer"


class FakeCardKitClient:
    def __init__(self):
        self.calls = []

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
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    session.last_update_time = 0.0
    asyncio.run(session.update("hello world."))

    assert ("update", "card_123", "content", "hello world.", 3) in client.calls


def test_session_update_coalesces_small_delta_until_close():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    asyncio.run(session.update("hello there"))
    asyncio.run(session.close("hello there"))

    assert ("update", "card_123", "content", "hello there", 3) in client.calls
    assert client.calls[-1] == ("close", "card_123", "hello there", 4)


def test_session_update_flushes_pending_small_delta_after_throttle():
    from gateway.platforms.feishu_streaming_card import (
        STREAMING_UPDATE_THROTTLE_MS,
        FeishuStreamingCardSession,
    )

    async def run_test():
        client = FakeCardKitClient()
        session = FeishuStreamingCardSession(
            client=client,
            chat_id="oc_chat",
            send_card_reference=fake_send_card_reference,
        )
        await session.start("hello", reply_to=None, metadata=None)
        await session.update("hello there")

        assert ("update", "card_123", "content", "hello there", 3) not in client.calls

        await asyncio.sleep((STREAMING_UPDATE_THROTTLE_MS + 80) / 1000)

        assert ("update", "card_123", "content", "hello there", 3) in client.calls

    asyncio.run(run_test())


def test_session_close_disables_streaming_mode_once():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    client = FakeCardKitClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))
    asyncio.run(session.close("hello final"))
    asyncio.run(session.close("ignored"))

    close_calls = [call for call in client.calls if call[0] == "close"]
    assert close_calls == [("close", "card_123", "hello final", 4)]


def test_session_close_reports_failure_when_final_update_fails():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    class FailingFinalUpdateClient(FakeCardKitClient):
        async def update_element_content(self, card_id, element_id, content, sequence):
            if content == "hello final":
                self.calls.append(
                    ("update_failed", card_id, element_id, content, sequence)
                )
                raise RuntimeError("update failed")
            await super().update_element_content(card_id, element_id, content, sequence)

    client = FailingFinalUpdateClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))

    result = asyncio.run(session.close("hello final"))

    assert result.success is False
    assert result.error == "update failed"
    assert ("update_failed", "card_123", "content", "hello final", 3) in client.calls
    assert not any(call[0] == "close" for call in client.calls)


def test_session_start_closes_card_reference_when_initial_update_fails():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    class FailingInitialUpdateClient(FakeCardKitClient):
        async def update_element_content(self, card_id, element_id, content, sequence):
            self.calls.append(("update_failed", card_id, element_id, content, sequence))
            raise RuntimeError("initial update failed")

    client = FailingInitialUpdateClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
    )

    result = asyncio.run(session.start("hello", reply_to=None, metadata=None))

    assert result.success is False
    assert "initial update failed" in result.error
    assert result.message_id == "msg_for_card_123"
    assert ("update_failed", "card_123", "content", "hello", 2) in client.calls
    assert ("close", "card_123", "", 3) in client.calls


def test_session_close_reports_failure_when_close_card_fails():
    from gateway.platforms.feishu_streaming_card import FeishuStreamingCardSession

    class FailingCloseClient(FakeCardKitClient):
        async def close_card(self, card_id, final_text, sequence):
            self.calls.append(("close_failed", card_id, final_text, sequence))
            raise RuntimeError("close failed")

    client = FailingCloseClient()
    session = FeishuStreamingCardSession(
        client=client,
        chat_id="oc_chat",
        send_card_reference=fake_send_card_reference,
    )
    asyncio.run(session.start("hello", reply_to=None, metadata=None))

    result = asyncio.run(session.close("hello final"))

    assert result.success is False
    assert "close failed" in result.error
    assert session.closed is False
    assert ("close_failed", "card_123", "hello final", 4) in client.calls


class FakeSdkResponse:
    def __init__(self, *, code=0, msg="", data=None):
        self.code = code
        self.msg = msg
        self.data = data

    def success(self):
        return self.code == 0


class FakeSdkCardId:
    card_id = "card_123"


class FakeSdkCardResource:
    def __init__(self):
        self.calls = []

    async def acreate(self, request):
        self.calls.append(("create", request))
        return FakeSdkResponse(data=FakeSdkCardId())

    async def asettings(self, request):
        self.calls.append(("settings", request))
        return FakeSdkResponse()


class FakeSdkCardElementResource:
    def __init__(self):
        self.calls = []

    async def acontent(self, request):
        self.calls.append(("content", request))
        return FakeSdkResponse()


class FakeSdkClient:
    def __init__(self):
        self.card = FakeSdkCardResource()
        self.card_element = FakeSdkCardElementResource()
        self.cardkit = SimpleNamespace(
            v1=SimpleNamespace(card=self.card, card_element=self.card_element)
        )


class _FakeSdkModelBuilder:
    def __init__(self, model_type):
        self.model = model_type()

    def type(self, value):
        self.model.type = value
        return self

    def data(self, value):
        self.model.data = value
        return self

    def request_body(self, value):
        self.model.request_body = value
        return self

    def card_id(self, value):
        self.model.card_id = value
        return self

    def element_id(self, value):
        self.model.element_id = value
        return self

    def content(self, value):
        self.model.content = value
        return self

    def sequence(self, value):
        self.model.sequence = value
        return self

    def uuid(self, value):
        self.model.uuid = value
        return self

    def settings(self, value):
        self.model.settings = value
        return self

    def build(self):
        return self.model


class _FakeSdkModel:
    @classmethod
    def builder(cls):
        return _FakeSdkModelBuilder(cls)


class FakeCreateCardRequestBody(_FakeSdkModel):
    pass


class FakeCreateCardRequest(_FakeSdkModel):
    pass


class FakeContentCardElementRequestBody(_FakeSdkModel):
    pass


class FakeContentCardElementRequest(_FakeSdkModel):
    pass


class FakeSettingsCardRequestBody(_FakeSdkModel):
    pass


class FakeSettingsCardRequest(_FakeSdkModel):
    pass


def install_fake_sdk_models(monkeypatch, module):
    monkeypatch.setattr(module, "CreateCardRequestBody", FakeCreateCardRequestBody)
    monkeypatch.setattr(module, "CreateCardRequest", FakeCreateCardRequest)
    monkeypatch.setattr(
        module, "ContentCardElementRequestBody", FakeContentCardElementRequestBody
    )
    monkeypatch.setattr(module, "ContentCardElementRequest", FakeContentCardElementRequest)
    monkeypatch.setattr(module, "SettingsCardRequestBody", FakeSettingsCardRequestBody)
    monkeypatch.setattr(module, "SettingsCardRequest", FakeSettingsCardRequest)


def test_cardkit_client_uses_lark_sdk_card_resources(monkeypatch):
    import gateway.platforms.feishu_streaming_card as streaming_card

    install_fake_sdk_models(monkeypatch, streaming_card)
    sdk_client = FakeSdkClient()
    client = streaming_card.FeishuCardKitClient(sdk_client)

    card_id = asyncio.run(client.create_card())
    asyncio.run(client.update_element_content(card_id, "content", "hello", 7))
    asyncio.run(client.close_card(card_id, "final text", 8))

    assert card_id == "card_123"
    create_request = sdk_client.card.calls[0][1]
    create_body = create_request.request_body
    assert create_body.type == "card_json"
    assert json.loads(create_body.data)["schema"] == "2.0"

    content_request = sdk_client.card_element.calls[0][1]
    content_body = content_request.request_body
    assert content_request.card_id == "card_123"
    assert content_request.element_id == "content"
    assert content_body.content == "hello"
    assert content_body.sequence == 7
    assert content_body.uuid == "s_card_123_7"

    settings_request = sdk_client.card.calls[1][1]
    settings_body = settings_request.request_body
    settings = json.loads(settings_body.settings)
    assert settings_request.card_id == "card_123"
    assert settings["config"]["streaming_mode"] is False
    assert settings["config"]["summary"]["content"] == "final text"
    assert settings_body.sequence == 8
    assert settings_body.uuid == "c_card_123_8"


def test_cardkit_client_rejects_over_boundary_content_without_truncating(monkeypatch):
    import gateway.platforms.feishu_streaming_card as streaming_card

    install_fake_sdk_models(monkeypatch, streaming_card)
    sdk_client = FakeSdkClient()
    client = streaming_card.FeishuCardKitClient(sdk_client)

    with pytest.raises(ValueError, match="CardKit content exceeds"):
        asyncio.run(
            client.update_element_content(
                "card_123",
                "content",
                "x" * (streaming_card.MAX_CARD_TEXT_LENGTH + 1),
                7,
            )
        )

    assert sdk_client.card_element.calls == []


def test_cardkit_client_raises_on_sdk_failure(monkeypatch):
    import gateway.platforms.feishu_streaming_card as streaming_card

    class FailingSdkCardResource(FakeSdkCardResource):
        async def acreate(self, request):
            self.calls.append(("create", request))
            return FakeSdkResponse(code=999, msg="no card permission")

    class FailingSdkClient(FakeSdkClient):
        def __init__(self):
            super().__init__()
            self.card = FailingSdkCardResource()
            self.cardkit = SimpleNamespace(
                v1=SimpleNamespace(
                    card=self.card,
                    card_element=self.card_element,
                )
            )

    install_fake_sdk_models(monkeypatch, streaming_card)
    client = streaming_card.FeishuCardKitClient(FailingSdkClient())

    with pytest.raises(RuntimeError, match="no card permission"):
        asyncio.run(client.create_card())
