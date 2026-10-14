"""Feishu CardKit streaming-card helper.

Ported from the hermes-agent project's gateway/platforms/feishu_streaming_card.py.
Provides Card JSON 2.0 construction and text utilities for Feishu textual
message delivery via the CardKit streaming API.

lark_oapi is NOT imported at the top level; it will be imported lazily where
needed (Task 2) to avoid a hard dependency for modules that only use these
pure-Python helpers.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import json
import re
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.feishu.cardkit")

CARD_CONTENT_ELEMENT_ID = "content"
MAX_CARD_TEXT_LENGTH = 30000

CARDKIT_ASSISTANT_PROFILE = "assistant"
CARDKIT_TOOL_PROGRESS_PROFILE = "tool_progress"
CARDKIT_STATIC_PROFILE = "static"

CARDKIT_STREAMING_PROFILES: dict[str, dict[str, Any]] = {
    CARDKIT_ASSISTANT_PROFILE: {
        "print_frequency_ms": {"default": 50},
        "print_step": {"default": 5},
        "print_strategy": "fast",
    },
    CARDKIT_TOOL_PROGRESS_PROFILE: {
        "print_frequency_ms": {"default": 30},
        "print_step": {"default": 80},
        "print_strategy": "fast",
    },
}


def strip_streaming_cursor(text: str) -> str:
    """Strip a trailing " ▉" or "▉" streaming cursor from *text*."""
    if text.endswith(" ▉"):
        return text[:-2].rstrip()
    if text.endswith("▉"):
        return text[:-1].rstrip()
    return text


def truncate_summary(text: str, max_chars: int = 50) -> str:
    """Truncate *text* to at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _summary_for_content(content: str, *, streaming_mode: bool) -> str:
    """Return a short summary string for a card's streaming summary config.

    Empty content yields "[Generating...]" when streaming, "" when static.
    """
    # build_card already strips the streaming cursor before calling us,
    # so we use the content as-is here.
    if content:
        return truncate_summary(content)
    if streaming_mode:
        return "[Generating...]"
    return ""


def build_card(
    content: str = "",
    *,
    streaming_mode: bool,
    profile: str = CARDKIT_ASSISTANT_PROFILE,
) -> dict[str, Any]:
    """Construct a Card JSON 2.0 dict for CardKit delivery.

    *streaming_mode* toggles the streaming_config block.  *profile* selects
    which streaming parameters to use (ignored when streaming_mode is False).
    """
    visible_content = strip_streaming_cursor(content or "")
    config: dict[str, Any] = {
        "streaming_mode": bool(streaming_mode),
        "summary": {
            "content": _summary_for_content(
                visible_content,
                streaming_mode=streaming_mode,
            )
        },
    }
    if streaming_mode:
        config["streaming_config"] = copy.deepcopy(
            CARDKIT_STREAMING_PROFILES.get(profile)
            or CARDKIT_STREAMING_PROFILES[CARDKIT_ASSISTANT_PROFILE]
        )
    return {
        "schema": "2.0",
        "config": config,
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": visible_content,
                    "element_id": CARD_CONTENT_ELEMENT_ID,
                }
            ]
        },
    }


def _import_cardkit_sdk():
    """Lazy-import CardKit SDK request/model classes.

    Called only when CardKit is actually used, so the adapter's
    import-time lark_oapi guard stays clean.
    """
    from lark_oapi.api.cardkit.v1 import (
        ContentCardElementRequest,
        ContentCardElementRequestBody,
        CreateCardRequest,
        CreateCardRequestBody,
        SettingsCardRequest,
        SettingsCardRequestBody,
    )
    return (
        ContentCardElementRequest,
        ContentCardElementRequestBody,
        CreateCardRequest,
        CreateCardRequestBody,
        SettingsCardRequest,
        SettingsCardRequestBody,
    )


class FeishuCardKitClient:
    """Wraps lark_oapi CardKit API calls for card create/update/close."""

    def __init__(self, sdk_client: Any):
        self.sdk_client = sdk_client

    @staticmethod
    def _ensure_success(response: Any, action: str) -> None:
        success = getattr(response, "success", None)
        if callable(success) and success():
            return
        code = getattr(response, "code", None)
        if code in (0, None) and not callable(success):
            return
        message = (
            getattr(response, "msg", "")
            or getattr(response, "message", "")
            or "unknown error"
        )
        raise RuntimeError(f"Feishu CardKit {action} failed: {message}")

    def _card_resource(self) -> Any:
        return self.sdk_client.cardkit.v1.card

    def _card_element_resource(self) -> Any:
        return self.sdk_client.cardkit.v1.card_element

    async def create_card(
        self,
        *,
        content: str = "",
        streaming_mode: bool = True,
        profile: str = CARDKIT_ASSISTANT_PROFILE,
    ) -> str:
        """Create a streaming or static card; return the card_id."""
        (
            _,
            _,
            CreateCardRequest,
            CreateCardRequestBody,
            _,
            _,
        ) = _import_cardkit_sdk()

        body = (
            CreateCardRequestBody.builder()
            .type("card_json")
            .data(
                json.dumps(
                    build_card(
                        content,
                        streaming_mode=streaming_mode,
                        profile=profile,
                    ),
                    ensure_ascii=False,
                )
            )
            .build()
        )
        request = CreateCardRequest.builder().request_body(body).build()
        response = await self._card_resource().acreate(request)
        self._ensure_success(response, "create card")
        card_id = getattr(getattr(response, "data", None), "card_id", None)
        if not card_id:
            raise RuntimeError(
                "Feishu CardKit create card failed: missing card_id"
            )
        return str(card_id)

    async def update_element_content(
        self, card_id, element_id, content, sequence
    ) -> None:
        """Push an updated text payload to a card element."""
        if len(content) > MAX_CARD_TEXT_LENGTH:
            raise ValueError(
                f"CardKit content exceeds {MAX_CARD_TEXT_LENGTH} characters"
            )

        (
            ContentCardElementRequest,
            ContentCardElementRequestBody,
            _,
            _,
            _,
            _,
        ) = _import_cardkit_sdk()

        body = (
            ContentCardElementRequestBody.builder()
            .content(content)
            .sequence(sequence)
            .uuid(f"s_{card_id}_{sequence}")
            .build()
        )
        request = (
            ContentCardElementRequest.builder()
            .card_id(card_id)
            .element_id(element_id)
            .request_body(body)
            .build()
        )
        response = await self._card_element_resource().acontent(request)
        self._ensure_success(response, "update card")

    async def close_card(self, card_id, final_text, sequence) -> None:
        """Close a streaming card with a final summary."""
        (
            _,
            _,
            _,
            _,
            SettingsCardRequest,
            SettingsCardRequestBody,
        ) = _import_cardkit_sdk()

        settings = {
            "config": {
                "streaming_mode": False,
                "summary": {"content": truncate_summary(final_text)},
            }
        }
        body = (
            SettingsCardRequestBody.builder()
            .settings(json.dumps(settings, ensure_ascii=False))
            .sequence(sequence)
            .uuid(f"c_{card_id}_{sequence}")
            .build()
        )
        request = (
            SettingsCardRequest.builder()
            .card_id(card_id)
            .request_body(body)
            .build()
        )
        response = await self._card_resource().asettings(request)
        self._ensure_success(response, "close card")


SendCardReference = Callable[..., Awaitable[SendResult]]


class FeishuStreamingCardSession:
    """Manages a single CardKit streaming card lifecycle.

    Wraps :class:`FeishuCardKitClient` to create a card, send a reference
    message into the chat, push incremental text updates, and close the card
    when streaming ends.  All public methods are idempotent and return a
    :class:`SendResult`.
    """

    def __init__(
        self,
        *,
        client: Any,
        chat_id: str,
        send_card_reference: SendCardReference,
        profile: str = CARDKIT_ASSISTANT_PROFILE,
    ):
        self.client = client
        self.chat_id = chat_id
        self.send_card_reference = send_card_reference
        self.profile = profile
        self.card_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.sequence = 1
        self.current_text = ""
        self.closed = False
        self._operation_lock = asyncio.Lock()

    async def start(
        self,
        initial_text: str,
        *,
        reply_to: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> SendResult:
        """Create the card, send the reference message, push initial content."""
        if self.card_id and self.message_id:
            return SendResult(success=True, message_id=self.message_id)

        self.card_id = await self.client.create_card(
            content=strip_streaming_cursor(initial_text),
            streaming_mode=True,
            profile=self.profile,
        )

        try:
            send_result = await self.send_card_reference(
                card_id=self.card_id,
                chat_id=self.chat_id,
                reply_to=reply_to,
                metadata=metadata,
            )
        except Exception:
            # Reference send raised: the card exists but has no chat message
            # yet (self.message_id is still None), so self.close() cannot be
            # used. Close the orphaned card best-effort, then re-raise.
            await self._best_effort_close_created_card()
            raise

        if not send_result.success:
            # Reference send failed without raising: same orphaned-card
            # situation. Close best-effort and propagate the failed result
            # (preserving its raw_response).
            await self._best_effort_close_created_card()
            return send_result

        self.message_id = send_result.message_id

        text = strip_streaming_cursor(initial_text)
        if text:
            try:
                await self._push_update(text, force=True)
            except Exception as exc:
                logger.warning(
                    "[Feishu] CardKit initial update failed for %s: %s",
                    self.card_id,
                    exc,
                    exc_info=True,
                )
                await self.close(None)
                return SendResult(
                    success=False,
                    message_id=self.message_id,
                    error=str(exc),
                    raw_response=getattr(send_result, "raw_response", None),
                )

        return SendResult(
            success=True,
            message_id=self.message_id,
            raw_response=getattr(send_result, "raw_response", None),
        )

    async def update(self, text: str) -> SendResult:
        """Push a full-text snapshot to the card."""
        async with self._operation_lock:
            if self.closed or not self.card_id or not self.message_id:
                return SendResult(
                    success=False, error="CardKit session is not active"
                )

            next_text = strip_streaming_cursor(text)

            if not next_text or next_text == self.current_text:
                return SendResult(success=True, message_id=self.message_id)

            await self._push_update(next_text, force=True)
            return SendResult(success=True, message_id=self.message_id)

    async def close(self, final_text: Optional[str] = None) -> SendResult:
        """Push final text and disable streaming mode."""
        async with self._operation_lock:
            if self.closed:
                return SendResult(success=True, message_id=self.message_id)

            if not self.card_id or not self.message_id:
                return SendResult(
                    success=False, error="CardKit session is not active"
                )

            final_snapshot = (
                strip_streaming_cursor(final_text) if final_text else ""
            )
            merged = final_snapshot or self.current_text

            if merged and merged != self.current_text:
                try:
                    await self._push_update(merged, force=True)
                except Exception as exc:
                    logger.warning(
                        "[Feishu] CardKit final update failed for %s: %s",
                        self.card_id,
                        exc,
                        exc_info=True,
                    )
                    return SendResult(
                        success=False,
                        message_id=self.message_id,
                        error=str(exc),
                    )

            self.sequence += 1
            try:
                await self.client.close_card(
                    self.card_id,
                    merged or self.current_text,
                    self.sequence,
                )
            except Exception as exc:
                logger.warning(
                    "[Feishu] CardKit close failed for %s: %s",
                    self.card_id,
                    exc,
                    exc_info=True,
                )
                return SendResult(
                    success=False,
                    message_id=self.message_id,
                    error=str(exc),
                )

            self.closed = True
            return SendResult(success=True, message_id=self.message_id)

    async def _best_effort_close_created_card(self) -> None:
        """Best-effort close of a created card whose reference send failed.

        Used from :meth:`start` when ``self.message_id`` is still ``None``
        (so :meth:`close` cannot be used, since it requires an active
        session).  Swallows close errors so the original reference-send
        failure is never masked.
        """
        if not self.card_id:
            return
        self.sequence += 1
        try:
            await self.client.close_card(
                self.card_id,
                self.current_text,
                self.sequence,
            )
        except Exception:
            logger.debug(
                "[Feishu] CardKit best-effort close failed for %s",
                self.card_id,
                exc_info=True,
            )
        self.closed = True

    async def _push_update(self, text: str, *, force: bool) -> None:
        """Push a text snapshot to the card element."""
        if not self.card_id:
            return
        if not force and text == self.current_text:
            return
        self.sequence += 1
        await self.client.update_element_content(
            self.card_id,
            CARD_CONTENT_ELEMENT_ID,
            text,
            self.sequence,
        )
        self.current_text = text


_CARDKIT_CARDID_INVALID_RE = re.compile(
    r"cardid is invalid|ErrCode:\s*11310", re.IGNORECASE
)
_CARDKIT_REFERENCE_INITIAL_DELAY_SECONDS = 0.3
_CARDKIT_REFERENCE_BACKOFF_SECONDS = (0.5, 1.0)


class FeishuCardKitRuntime:
    """Owns all active CardKit sessions for a Feishu adapter instance."""

    def __init__(self, *, client, send_card_reference: SendCardReference) -> None:
        self._client = client
        self._send_card_reference = send_card_reference
        self._sessions: dict[str, FeishuStreamingCardSession] = {}
        self._registry_lock = asyncio.Lock()
        self._close_all_lock = asyncio.Lock()
        self._closing = False
        self._in_flight_sends = 0
        self._sends_drained = asyncio.Event()
        self._sends_drained.set()

    async def try_send(self, chat_id, content, *, reply_to, metadata):
        if self._client is None:
            return None
        async with self._registry_lock:
            if self._closing:
                return None
            self._in_flight_sends += 1
            self._sends_drained.clear()
        try:
            await self._close_tool_progress_siblings(chat_id)
            streaming, profile = self._streaming_mode(metadata)
            if streaming:
                session = FeishuStreamingCardSession(
                    client=self._client,
                    chat_id=chat_id,
                    send_card_reference=self._send_reference_with_retry,
                    profile=profile,
                )
                result = await session.start(
                    content, reply_to=reply_to, metadata=metadata
                )
                if result.success and result.message_id:
                    async with self._registry_lock:
                        close_after_start = self._closing
                        if not close_after_start:
                            self._sessions[result.message_id] = session
                    if close_after_start:
                        await self._best_effort_close_session(
                            result.message_id, session
                        )
                return result
            return await self._send_static(
                chat_id, content, reply_to=reply_to, metadata=metadata
            )
        except Exception as exc:
            logger.warning(
                "[Feishu] CardKit send failed: %s", exc, exc_info=True
            )
            return SendResult(success=False, error=str(exc))
        finally:
            async with self._registry_lock:
                self._in_flight_sends -= 1
                if self._in_flight_sends == 0:
                    self._sends_drained.set()

    async def edit_if_managed(
        self, chat_id, message_id, content, *, finalize=False
    ):
        async with self._registry_lock:
            session = self._sessions.get(message_id)
        if session is None:
            return None
        try:
            if finalize and session.profile != CARDKIT_TOOL_PROGRESS_PROFILE:
                result = await session.close(content)
                if result.success:
                    await self._forget_session(message_id, session)
                return result
            return await session.update(content)
        except Exception as exc:
            logger.error(
                "[Feishu] Failed to edit CardKit message %s: %s",
                message_id,
                exc,
                exc_info=True,
            )
            return SendResult(success=False, message_id=message_id, error=str(exc))

    async def close_all(self) -> None:
        async with self._close_all_lock:
            async with self._registry_lock:
                self._closing = True
                sessions = list(self._sessions.items())
            for message_id, session in sessions:
                await self._best_effort_close_session(message_id, session)
            await self._sends_drained.wait()
            async with self._registry_lock:
                self._sessions.clear()

    @staticmethod
    def _streaming_mode(metadata):
        metadata = metadata or {}
        if metadata.get("message_kind") == "tool_progress":
            return True, CARDKIT_TOOL_PROGRESS_PROFILE
        if metadata.get("expect_edits"):
            return True, CARDKIT_ASSISTANT_PROFILE
        return False, CARDKIT_STATIC_PROFILE

    async def _send_reference_with_retry(
        self, *, card_id, chat_id, reply_to, metadata
    ):
        await asyncio.sleep(_CARDKIT_REFERENCE_INITIAL_DELAY_SECONDS)
        result = None
        for attempt in range(len(_CARDKIT_REFERENCE_BACKOFF_SECONDS) + 1):
            result = await self._send_card_reference(
                card_id=card_id,
                chat_id=chat_id,
                reply_to=reply_to,
                metadata=metadata,
            )
            if result.success or not self._is_cardid_invalid(result):
                return result
            if attempt < len(_CARDKIT_REFERENCE_BACKOFF_SECONDS):
                await asyncio.sleep(_CARDKIT_REFERENCE_BACKOFF_SECONDS[attempt])
        return result

    @staticmethod
    def _is_cardid_invalid(result: SendResult) -> bool:
        response = result.raw_response
        code = str(getattr(response, "code", ""))
        message = result.error or str(getattr(response, "msg", "") or "")
        return code == "230099" and bool(
            _CARDKIT_CARDID_INVALID_RE.search(message)
        )

    async def _send_static(self, chat_id, content, *, reply_to, metadata):
        card_id = await self._client.create_card(
            content=content,
            streaming_mode=False,
            profile=CARDKIT_STATIC_PROFILE,
        )
        try:
            result = await self._send_reference_with_retry(
                card_id=card_id,
                chat_id=chat_id,
                reply_to=reply_to,
                metadata=metadata,
            )
        except Exception:
            await self._best_effort_close(card_id, content)
            raise
        if not result.success:
            await self._best_effort_close(card_id, content)
        return result

    async def _best_effort_close(self, card_id, content) -> None:
        try:
            await self._client.close_card(card_id, content, 1)
        except Exception:
            logger.debug(
                "[Feishu] CardKit best-effort close failed for %s",
                card_id,
                exc_info=True,
            )

    async def _best_effort_close_session(
        self, message_id: str, session: FeishuStreamingCardSession
    ) -> None:
        try:
            result = await session.close(None)
            if not result.success:
                logger.debug(
                    "[Feishu] CardKit disconnect close failed for %s: %s",
                    message_id,
                    result.error,
                )
        except Exception as exc:
            logger.debug(
                "[Feishu] CardKit disconnect close failed for %s: %s",
                message_id,
                exc,
                exc_info=True,
            )

    async def _forget_session(
        self, message_id: str, session: FeishuStreamingCardSession
    ) -> None:
        async with self._registry_lock:
            if self._sessions.get(message_id) is session:
                self._sessions.pop(message_id, None)

    async def _close_tool_progress_siblings(self, chat_id) -> None:
        async with self._registry_lock:
            sessions = list(self._sessions.items())
        for message_id, session in sessions:
            if session.chat_id != chat_id:
                continue
            if session.profile != CARDKIT_TOOL_PROGRESS_PROFILE:
                continue
            try:
                result = await session.close(None)
            except Exception as exc:
                logger.debug(
                    "[Feishu] CardKit sibling close raised for %s: %s",
                    message_id,
                    exc,
                    exc_info=True,
                )
                continue
            if result.success:
                await self._forget_session(message_id, session)
