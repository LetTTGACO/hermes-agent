"""CardKit sessions follow the adapter's SDK through official reconnects."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu import adapter as adapter_module
from plugins.platforms.feishu.feishu_cardkit import FeishuCardKitClient


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
