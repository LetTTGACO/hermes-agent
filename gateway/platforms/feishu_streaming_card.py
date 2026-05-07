"""Feishu CardKit streaming-card helper."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gateway.platforms.base import SendResult


logger = logging.getLogger("gateway.feishu.streaming_card")

CARD_CONTENT_ELEMENT_ID = "content"
STREAMING_UPDATE_THROTTLE_MS = 160
STREAMING_SIGNIFICANT_DELTA_CHARS = 18
TOKEN_EXPIRY_SAFETY_SECONDS = 60
MAX_CARD_TEXT_LENGTH = 30000
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}

_NATURAL_STREAMING_BOUNDARIES = "\n.!?;:。！？；："


def resolve_api_base(domain: Optional[str]) -> str:
    if domain == "lark":
        return "https://open.larksuite.com/open-apis"
    if domain == "feishu" or not domain:
        return "https://open.feishu.cn/open-apis"
    if domain.startswith(("http://", "https://")):
        return f"{domain.rstrip('/')}/open-apis"
    return "https://open.feishu.cn/open-apis"


def resolve_receive_id_type(chat_id: str) -> str:
    return "open_id" if chat_id.startswith("ou_") else "chat_id"


def build_streaming_card() -> Dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {
            "streaming_mode": True,
            "summary": {"content": "[Generating...]"},
            "streaming_config": {
                "print_frequency_ms": {"default": 70},
                "print_step": {"default": 1},
                "print_strategy": "fast",
            },
        },
        "body": {
            "elements": [
                {"tag": "markdown", "content": "", "element_id": CARD_CONTENT_ELEMENT_ID}
            ]
        },
    }


def truncate_summary(text: str, max_chars: int = 50) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def has_natural_streaming_boundary(text: str) -> bool:
    return bool(text) and text[-1] in _NATURAL_STREAMING_BOUNDARIES


def should_push_streaming_update(
    previous_text: str, next_text: str, *, block_streaming: bool
) -> bool:
    if not previous_text:
        return True
    if not block_streaming:
        return True
    delta_chars = max(0, len(next_text) - len(previous_text))
    return (
        has_natural_streaming_boundary(next_text)
        or delta_chars >= STREAMING_SIGNIFICANT_DELTA_CHARS
    )


def merge_streaming_text(previous_text: Optional[str], next_text: Optional[str]) -> str:
    previous = previous_text or ""
    next_value = next_text or ""
    if not previous:
        return next_value
    if not next_value:
        return previous
    if next_value.startswith(previous):
        return next_value
    if previous.startswith(next_value):
        return previous
    if previous.endswith(next_value):
        return previous

    max_overlap = min(len(previous), len(next_value))
    for overlap in range(max_overlap, 0, -1):
        if previous.endswith(next_value[:overlap]):
            return previous + next_value[overlap:]
    return previous + next_value


def strip_streaming_cursor(text: str) -> str:
    if text.endswith(" ▉"):
        return text[:-2].rstrip()
    if text.endswith("▉"):
        return text[:-1].rstrip()
    return text


@dataclass(frozen=True)
class FeishuCardKitCredentials:
    app_id: str
    app_secret: str
    domain: str = "feishu"


class RequestJson(Protocol):
    async def __call__(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ...


def _request_json_sync(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, method=method)
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload or "{}")
    except HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu CardKit HTTP {exc.code}: {payload[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Feishu CardKit request failed: {exc}") from exc


class FeishuCardKitClient:
    def __init__(
        self,
        credentials: FeishuCardKitCredentials,
        *,
        request_json: Optional[RequestJson] = None,
    ):
        self.credentials = credentials
        self.api_base = resolve_api_base(credentials.domain)
        self._request_json = request_json

    async def _call(
        self,
        method: str,
        path: str,
        *,
        token: Optional[str],
        body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"{self.api_base}{path}"
        if self._request_json is not None:
            return await self._request_json(method, url, headers=headers, body=body)
        return await asyncio.to_thread(
            _request_json_sync,
            method,
            url,
            headers=headers,
            body=body,
        )

    async def get_token(self) -> str:
        cache_key = f"{self.credentials.domain}|{self.credentials.app_id}"
        cached = _TOKEN_CACHE.get(cache_key)
        now = time.time()
        if cached and cached[1] > now + TOKEN_EXPIRY_SAFETY_SECONDS:
            return cached[0]
        payload = await self._call(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            token=None,
            body={
                "app_id": self.credentials.app_id,
                "app_secret": self.credentials.app_secret,
            },
        )
        if payload.get("code") != 0 or not payload.get("tenant_access_token"):
            raise RuntimeError(
                f"Feishu token error: {payload.get('msg', 'unknown error')}"
            )
        token = str(payload["tenant_access_token"])
        expire = float(payload.get("expire") or 7200)
        _TOKEN_CACHE[cache_key] = (token, now + expire)
        return token

    async def create_card(self) -> str:
        payload = await self._call(
            "POST",
            "/cardkit/v1/cards",
            token=await self.get_token(),
            body={
                "type": "card_json",
                "data": json.dumps(build_streaming_card(), ensure_ascii=False),
            },
        )
        card_id = (payload.get("data") or {}).get("card_id")
        if payload.get("code") != 0 or not card_id:
            raise RuntimeError(
                f"Feishu create card failed: {payload.get('msg', 'missing card_id')}"
            )
        return str(card_id)

    async def update_element_content(
        self,
        card_id: str,
        element_id: str,
        content: str,
        sequence: int,
    ) -> None:
        payload = await self._call(
            "PUT",
            f"/cardkit/v1/cards/{card_id}/elements/{element_id}/content",
            token=await self.get_token(),
            body={
                "content": content[:MAX_CARD_TEXT_LENGTH],
                "sequence": sequence,
                "uuid": f"s_{card_id}_{sequence}",
            },
        )
        if payload.get("code", 0) not in (0, None):
            raise RuntimeError(
                f"Feishu update card failed: {payload.get('msg', 'unknown error')}"
            )

    async def close_card(self, card_id: str, final_text: str, sequence: int) -> None:
        payload = await self._call(
            "PATCH",
            f"/cardkit/v1/cards/{card_id}/settings",
            token=await self.get_token(),
            body={
                "settings": json.dumps(
                    {
                        "config": {
                            "streaming_mode": False,
                            "summary": {"content": truncate_summary(final_text)},
                        }
                    },
                    ensure_ascii=False,
                ),
                "sequence": sequence,
                "uuid": f"c_{card_id}_{sequence}",
            },
        )
        if payload.get("code", 0) not in (0, None):
            raise RuntimeError(
                f"Feishu close card failed: {payload.get('msg', 'unknown error')}"
            )


SendCardReference = Callable[..., Awaitable[SendResult]]


class FeishuStreamingCardSession:
    def __init__(
        self,
        *,
        client: Any,
        chat_id: str,
        send_card_reference: SendCardReference,
        block_streaming: bool = True,
    ):
        self.client = client
        self.chat_id = chat_id
        self.send_card_reference = send_card_reference
        self.block_streaming = block_streaming
        self.card_id: Optional[str] = None
        self.message_id: Optional[str] = None
        self.sequence = 1
        self.current_text = ""
        self.pending_text: Optional[str] = None
        self.closed = False
        self.last_update_time = 0.0

    async def start(
        self,
        initial_text: str,
        *,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        if self.card_id and self.message_id:
            return SendResult(success=True, message_id=self.message_id)
        self.card_id = await self.client.create_card()
        send_result = await self.send_card_reference(
            card_id=self.card_id,
            chat_id=self.chat_id,
            reply_to=reply_to,
            metadata=metadata,
        )
        if not send_result.success:
            return send_result
        self.message_id = send_result.message_id
        text = strip_streaming_cursor(initial_text)
        if text:
            await self._push_update(text, force=True)
        return SendResult(
            success=True,
            message_id=self.message_id,
            raw_response=getattr(send_result, "raw_response", None),
        )

    async def update(self, text: str) -> SendResult:
        if self.closed or not self.card_id or not self.message_id:
            return SendResult(success=False, error="CardKit session is not active")
        next_text = merge_streaming_text(
            self.pending_text or self.current_text,
            strip_streaming_cursor(text),
        )
        if not next_text or next_text == self.current_text:
            return SendResult(success=True, message_id=self.message_id)
        self.pending_text = next_text
        if not should_push_streaming_update(
            self.current_text,
            next_text,
            block_streaming=self.block_streaming,
        ):
            return SendResult(success=True, message_id=self.message_id)
        now_ms = time.monotonic() * 1000
        if (
            self.block_streaming
            and now_ms - self.last_update_time < STREAMING_UPDATE_THROTTLE_MS
        ):
            return SendResult(success=True, message_id=self.message_id)
        await self._push_update(next_text, force=True)
        return SendResult(success=True, message_id=self.message_id)

    async def close(self, final_text: Optional[str] = None) -> SendResult:
        if self.closed:
            return SendResult(success=True, message_id=self.message_id)
        if not self.card_id or not self.message_id:
            return SendResult(success=False, error="CardKit session is not active")
        merged = merge_streaming_text(self.current_text, self.pending_text)
        if final_text:
            merged = merge_streaming_text(merged, strip_streaming_cursor(final_text))
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
        self.closed = True
        return SendResult(success=True, message_id=self.message_id)

    async def _push_update(self, text: str, *, force: bool) -> None:
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
        self.pending_text = None
        self.last_update_time = time.monotonic() * 1000
