"""Personal Feishu CardKit progress integration coverage."""

import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    _reply_anchor_for_event,
)
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner


class _ProgressAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.FEISHU)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.progress_sent = threading.Event()
        self.progress_edited = threading.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        self.progress_sent.set()
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ) -> SendResult:
        self.edits.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "content": content,
            "finalize": finalize,
            "metadata": metadata,
        })
        self.progress_edited.set()
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _agent_type(progress_sent):
    class _Agent:
        def __init__(self, **kwargs):
            self.tool_progress_callback = kwargs.get("tool_progress_callback")
            self.tools = []

        def run_conversation(
            self, message, conversation_history=None, task_id=None, **kwargs
        ):
            assert kwargs["persist_user_platform_id"] == "om-inbound"
            callback = self.tool_progress_callback
            assert callback is not None
            callback("tool.started", "terminal", "pwd", {})
            assert progress_sent.wait(timeout=5), "first progress send did not arrive"
            callback(
                "tool.started", "browser_navigate", "https://example.com", {}
            )
            return {"final_response": "done", "messages": [], "api_calls": 1}

    return _Agent


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._service_tier = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._draining = False
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def test_feishu_dm_drops_reply_anchor():
    source = SimpleNamespace(
        platform=Platform.FEISHU,
        chat_type="dm",
        thread_id=None,
    )
    event = SimpleNamespace(
        source=source,
        raw_message=None,
        message_id="om_triggering_user_message",
        reply_to_message_id=None,
    )

    assert _reply_anchor_for_event(event) is None


def test_feishu_group_keeps_reply_anchor():
    source = SimpleNamespace(
        platform=Platform.FEISHU,
        chat_type="group",
        thread_id=None,
    )
    event = SimpleNamespace(
        source=source,
        raw_message=None,
        message_id="om_triggering_user_message",
        reply_to_message_id=None,
    )

    assert _reply_anchor_for_event(event) == "om_triggering_user_message"


@pytest.mark.asyncio
async def test_run_agent_feishu_progress_replies_inside_existing_thread(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    adapter = _ProgressAdapter()
    adapter.REQUIRES_EDIT_FINALIZE = True
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _agent_type(adapter.progress_sent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="topic_17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-feishu-progress",
        session_key="agent:main:feishu:group:oc_chat:topic_17585",
        event_message_id="om_triggering_user_message",
        inbound_message_id="om-inbound",
    )

    assert result["final_response"] == "done"
    assert adapter.progress_sent.is_set()
    assert adapter.progress_edited.is_set()
    assert adapter.sent[0]["reply_to"] == "om_triggering_user_message"
    assert adapter.sent[0]["metadata"] == {
        "thread_id": "topic_17585",
        "message_kind": "tool_progress",
        "_cardkit_turn_id": "om-inbound",
    }
    assert adapter.edits[0]["message_id"] == "progress-1"
    assert adapter.edits[0]["finalize"] is True
    assert adapter.edits[0]["metadata"] == {
        "thread_id": "topic_17585",
        "message_kind": "tool_progress",
        "_cardkit_turn_id": "om-inbound",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type,thread_id,anchor", [
    ("group", "topic-A", "om-trigger"),
    ("group", None, "om-trigger"),
    ("dm", None, None),
])
async def test_feishu_wiring_preserves_routing_and_true_inbound_owner(
    chat_type, thread_id, anchor
):
    runner = _make_runner(_ProgressAdapter())
    source = SessionSource(
        platform=Platform.FEISHU, chat_id="oc-chat", chat_type=chat_type,
        thread_id=thread_id,
    )
    ctx = TurnContext(source=source, inbound_message_id="om-inbound")
    status = runner._run_agent_bind_turn_wiring(
        ctx, TurnRunner(runner, ctx), source, anchor, False
    )
    progress = ctx._progress_metadata
    assert progress["message_kind"] == "tool_progress"
    assert progress["_cardkit_turn_id"] == status["_cardkit_turn_id"] == "om-inbound"
    assert "message_kind" not in status
    if thread_id:
        assert progress["thread_id"] == "topic-A"
        assert ctx._progress_reply_to == "om-trigger"
        assert status["reply_to_message_id"] == "om-trigger"
    else:
        assert ctx._progress_reply_to is None
        assert "thread_id" not in progress


@pytest.mark.asyncio
async def test_feishu_wiring_without_inbound_id_does_not_invent_owner():
    runner = _make_runner(_ProgressAdapter())
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", chat_type="dm")
    ctx = TurnContext(source=source)
    status = runner._run_agent_bind_turn_wiring(
        ctx, TurnRunner(runner, ctx), source, "om-trigger", False
    )
    assert ctx._progress_metadata == {"message_kind": "tool_progress"}
    assert status == {}


@pytest.mark.asyncio
async def test_feishu_wiring_clones_input_metadata(monkeypatch):
    runner = _make_runner(_ProgressAdapter())
    source = SessionSource(platform=Platform.FEISHU, chat_id="oc-chat", chat_type="group")
    original = {"thread_id": "topic-A"}
    monkeypatch.setattr(runner, "_run_agent_progress_threading", lambda *_: (original, "om-trigger", original))
    ctx = TurnContext(source=source, inbound_message_id="om-inbound")
    status = runner._run_agent_bind_turn_wiring(
        ctx, TurnRunner(runner, ctx), source, "om-trigger", False
    )
    assert original == {"thread_id": "topic-A"}
    assert status is not original
    assert ctx._progress_metadata is not original
    assert status is not ctx._progress_metadata
    assert "message_kind" not in status


@pytest.mark.asyncio
async def test_telegram_wiring_does_not_add_cardkit_metadata():
    adapter = _ProgressAdapter()
    adapter.platform = Platform.TELEGRAM
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="group", thread_id="topic-A")
    expected = runner._run_agent_progress_threading(source, "om-trigger", False)
    ctx = TurnContext(source=source, inbound_message_id="om-inbound")
    status = runner._run_agent_bind_turn_wiring(
        ctx, TurnRunner(runner, ctx), source, "om-trigger", False
    )
    assert (ctx._progress_metadata, ctx._progress_reply_to, status) == expected
    for metadata in (ctx._progress_metadata, status):
        assert "message_kind" not in (metadata or {})
        assert "_cardkit_turn_id" not in (metadata or {})


def test_feishu_dm_explicit_reply_anchor_override_wins():
    source = SimpleNamespace(platform=Platform.FEISHU, chat_type="dm", thread_id=None)
    event = SimpleNamespace(source=source, message_id="om-inbound", reply_anchor_override="om-redirect")
    assert _reply_anchor_for_event(event) == "om-redirect"
