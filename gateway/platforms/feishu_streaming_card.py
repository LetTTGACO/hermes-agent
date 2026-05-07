"""Feishu CardKit streaming-card helper."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional
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
