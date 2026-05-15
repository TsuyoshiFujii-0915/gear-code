from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatLine:
    """Visible chat line in the Gear Code TUI.

    Attributes:
        speaker: Display speaker label.
        text: Message text.
    """

    speaker: str
    text: str


def collect_chat_lines(events: list[dict[str, Any]]) -> list[ChatLine]:
    """Collects visible chat messages from stored session events.

    Args:
        events: Stored session events.

    Returns:
        Chat lines for user and assistant messages.

    Raises:
        ValueError: If a visible chat event has an invalid payload shape.
    """

    chat_lines: list[ChatLine] = []
    for event in events:
        kind = event.get("kind")
        if kind not in {"user_input", "assistant_message", "compaction_summary"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} payload must be an object.")
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{kind} payload.text must be a string.")
        if kind == "user_input":
            chat_lines.append(ChatLine("you", text))
        if kind == "assistant_message":
            chat_lines.append(ChatLine("assistant", text))
        if kind == "compaction_summary":
            chat_lines.append(ChatLine("assistant", f"summary saved: {_single_line(text)}"))
    return chat_lines


def collect_token_usage(events: list[dict[str, Any]]) -> int | None:
    """Collects total token usage from model response events.

    Args:
        events: Stored session events.

    Returns:
        Total token count, or None when no response usage was reported.

    Raises:
        ValueError: If a response usage object exists but lacks integer total_tokens.
    """

    total = 0
    found_usage = False
    for event in events:
        if event.get("kind") != "model_response":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or "usage" not in payload:
            continue
        usage = payload["usage"]
        if usage is None:
            continue
        if not isinstance(usage, dict):
            raise ValueError("model_response usage must be an object when present.")
        total_tokens = usage.get("total_tokens")
        if not isinstance(total_tokens, int):
            raise ValueError(
                "model_response usage.total_tokens must be an integer when usage is present."
            )
        total += total_tokens
        found_usage = True
    if not found_usage:
        return None
    return total


def compact_path(path: Path) -> str:
    """Formats a path with home directory replaced by ~.

    Args:
        path: Filesystem path to format.

    Returns:
        Compact string representation of the path.
    """

    text = str(path)
    home = str(Path.home())
    if text == home:
        return "~"
    if text.startswith(f"{home}/"):
        return f"~/{text[len(home) + 1:]}"
    return text


def compact_session(session_id: str) -> str:
    """Truncates a session ID for display.

    Args:
        session_id: Full session identifier.

    Returns:
        Session ID truncated to 8 characters if longer than 12.
    """

    if len(session_id) <= 12:
        return session_id
    return session_id[:8]


def format_tokens(token_usage: int | None) -> str:
    """Formats a token count for display.

    Args:
        token_usage: Token count, or None when unavailable.

    Returns:
        Human-readable token count string.
    """

    if token_usage is None:
        return "unavailable"
    if token_usage < 1000:
        return str(token_usage)
    return f"{token_usage / 1000:.1f}k"


def _single_line(text: str) -> str:
    return " ".join(text.split())
