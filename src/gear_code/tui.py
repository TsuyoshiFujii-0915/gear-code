from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from gear_code.agent.events import (
    AgentLoopEvent,
    ModelRequestStarted,
    ToolUseFinished,
    ToolUseStarted,
)


_MAX_TOOL_PAYLOAD_CHARS = 1200
_MAX_STRING_VALUE_CHARS = 900


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
        if kind not in {
            "user_input",
            "assistant_message",
            "compaction_summary",
            "turn_error",
            "tool_call",
            "tool_result",
        }:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} payload must be an object.")
        if kind == "user_input":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("you", text))
        if kind == "assistant_message":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("assistant", text))
        if kind == "compaction_summary":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("assistant", f"summary saved: {_single_line(text)}"))
        if kind == "turn_error":
            text = _required_text_payload(kind, payload)
            chat_lines.append(ChatLine("error", text))
        if kind == "tool_call":
            chat_lines.append(ChatLine("tool", _format_tool_call_payload(payload)))
        if kind == "tool_result":
            chat_lines.append(ChatLine("tool", _format_tool_result_payload(payload)))
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


def format_progress_event(event: AgentLoopEvent) -> str:
    """Formats an agent loop progress event for TUI display.

    Args:
        event: Agent loop progress event.

    Returns:
        TUI display text.

    Raises:
        ValueError: If the event type is unsupported.
    """

    if isinstance(event, ModelRequestStarted):
        return _format_model_request_started(event)
    if isinstance(event, ToolUseStarted):
        return _format_tool_use_started(event)
    if isinstance(event, ToolUseFinished):
        return _format_tool_use_finished(event)
    raise ValueError(f"Unsupported agent loop event: {type(event).__name__}")


def _required_text_payload(kind: str, payload: dict[str, Any]) -> str:
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError(f"{kind} payload.text must be a string.")
    return text


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _format_tool_call_payload(payload: dict[str, Any]) -> str:
    call_id = payload.get("call_id")
    iteration = payload.get("iteration")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(call_id, str):
        raise ValueError("tool_call payload.call_id must be a string.")
    if not isinstance(name, str):
        raise ValueError("tool_call payload.name must be a string.")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call payload.arguments must be an object.")
    return _format_tool_use(
        _format_iteration(iteration),
        "start",
        name,
        "args",
        arguments,
    )


def _format_tool_result_payload(payload: dict[str, Any]) -> str:
    call_id = payload.get("call_id")
    iteration = payload.get("iteration")
    name = payload.get("name")
    result = payload.get("result")
    if not isinstance(call_id, str):
        raise ValueError("tool_result payload.call_id must be a string.")
    if not isinstance(name, str):
        raise ValueError("tool_result payload.name must be a string.")
    if not isinstance(result, dict):
        raise ValueError("tool_result payload.result must be an object.")
    return _format_tool_use(
        _format_iteration(iteration),
        "finish",
        name,
        "result",
        result,
    )


def _format_tool_use_started(event: ToolUseStarted) -> str:
    return _format_tool_use(
        str(event.iteration),
        "start",
        event.name,
        "args",
        event.arguments,
    )


def _format_tool_use_finished(event: ToolUseFinished) -> str:
    return _format_tool_use(
        str(event.iteration),
        "finish",
        event.name,
        "result",
        event.result,
    )


def _format_model_request_started(event: ModelRequestStarted) -> str:
    return f"loop {event.iteration} model request"


def _format_tool_use(
    iteration: str,
    phase: str,
    tool_name: str,
    payload_label: str,
    payload: dict[str, object],
) -> str:
    return f"loop {iteration} {phase} {tool_name}\n{payload_label} {_json_preview(payload)}"


def _format_iteration(value: object) -> str:
    if value is None:
        return "?"
    if not isinstance(value, int):
        raise ValueError("tool event payload.iteration must be an integer when present.")
    return str(value)


def _json_preview(payload: dict[str, object]) -> str:
    preview_payload = _preview_value(payload)
    text = json.dumps(preview_payload, ensure_ascii=False, sort_keys=True)
    if len(text) <= _MAX_TOOL_PAYLOAD_CHARS:
        return text
    return f"{text[:_MAX_TOOL_PAYLOAD_CHARS]}... [truncated]"


def _preview_value(value: object) -> object:
    if isinstance(value, str):
        return _preview_string(value)
    if isinstance(value, dict):
        return {key: _preview_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_preview_value(item) for item in value]
    return value


def _preview_string(value: str) -> str:
    if len(value) <= _MAX_STRING_VALUE_CHARS:
        return value
    return f"{value[:_MAX_STRING_VALUE_CHARS]}... [truncated]"
