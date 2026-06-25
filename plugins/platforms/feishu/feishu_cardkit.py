"""Feishu CardKit streaming-card helper.

Ported from the hermes-agent project's gateway/platforms/feishu_streaming_card.py.
Provides Card JSON 2.0 construction and text utilities for Feishu textual
message delivery via the CardKit streaming API.

lark_oapi is NOT imported at the top level; it will be imported lazily where
needed (Task 2) to avoid a hard dependency for modules that only use these
pure-Python helpers.
"""

from __future__ import annotations

import copy
import logging
import json
from typing import Any, Dict, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.feishu.cardkit")

CARD_CONTENT_ELEMENT_ID = "content"
MAX_CARD_TEXT_LENGTH = 30000

CARDKIT_ASSISTANT_PROFILE = "assistant"
CARDKIT_TOOL_PROGRESS_PROFILE = "tool_progress"
CARDKIT_STATIC_PROFILE = "static"

CARDKIT_STREAMING_PROFILES: Dict[str, Dict[str, Any]] = {
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


def merge_streaming_text(
    previous_text: Optional[str], next_text: Optional[str]
) -> str:
    """Merge two chunks of streaming text, handling overlap.

    - If *next_text* starts with *previous_text*, *next_text* is the newer
      full snapshot -> return it.
    - If *previous_text* starts with *next_text*, the previous snapshot is
      longer -> return it.
    - If *previous_text* ends with *next_text*, the previous snapshot already
      contains the new chunk -> return it.
    - Otherwise find the longest suffix/prefix overlap and join on it.
    - If none of the above, concatenate.
    """
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
) -> Dict[str, Any]:
    """Construct a Card JSON 2.0 dict for CardKit delivery.

    *streaming_mode* toggles the streaming_config block.  *profile* selects
    which streaming parameters to use (ignored when streaming_mode is False).
    """
    visible_content = strip_streaming_cursor(content or "")
    config: Dict[str, Any] = {
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
