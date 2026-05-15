import unittest
from typing import Any

from gear_code.tui import ChatLine, collect_chat_lines, collect_token_usage, format_progress_event
from gear_code.agent.events import ModelRequestStarted, ToolUseFinished, ToolUseStarted


class CollectTokenUsageTests(unittest.TestCase):
    def test_sums_total_tokens_across_model_responses(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "model_response",
                "payload": {"usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200}},
            },
            {
                "kind": "model_response",
                "payload": {"usage": {"input_tokens": 1500, "output_tokens": 300, "total_tokens": 1800}},
            },
        ]

        usage = collect_token_usage(events)

        self.assertEqual(usage, 3000)

    def test_returns_none_when_no_model_response_has_usage(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "hello"}},
        ]

        usage = collect_token_usage(events)

        self.assertIsNone(usage)


class CollectChatLinesTests(unittest.TestCase):
    def test_maps_user_input_to_you_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "user_input", "payload": {"text": "Refactor the CLI"}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(lines, [ChatLine("you", "Refactor the CLI")])

    def test_maps_assistant_message_to_assistant_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "assistant_message", "payload": {"text": "Sure, let me look."}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(lines, [ChatLine("assistant", "Sure, let me look.")])

    def test_skips_events_that_are_not_chat(self) -> None:
        events: list[dict[str, Any]] = [
            {"kind": "model_response", "payload": {"usage": {"total_tokens": 100}}},
            {"kind": "user_input", "payload": {"text": "hello"}},
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].speaker, "you")

    def test_maps_tool_call_to_tool_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "tool_call",
                "payload": {
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": {"path": "src/gear_code/agent/loop.py"},
                },
            },
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(
            lines,
            [
                ChatLine(
                    "tool",
                    'loop ? start read_file\nargs {"path": "src/gear_code/agent/loop.py"}',
                )
            ],
        )

    def test_maps_tool_error_result_to_tool_speaker(self) -> None:
        events: list[dict[str, Any]] = [
            {
                "kind": "tool_result",
                "payload": {
                    "call_id": "call_1",
                    "name": "read_file",
                    "result": {
                        "error": {
                            "type": "file_not_found",
                            "message": "File does not exist.",
                            "origin": "file_read",
                            "details": {"path": "missing.py"},
                        }
                    },
                },
            },
        ]

        lines = collect_chat_lines(events)

        self.assertEqual(
            lines,
            [
                ChatLine(
                    "tool",
                    (
                        "loop ? finish read_file\n"
                        'result {"error": {"details": {"path": "missing.py"}, '
                        '"message": "File does not exist.", "origin": "file_read", '
                        '"type": "file_not_found"}}'
                    ),
                )
            ],
        )


class FormatProgressEventTests(unittest.TestCase):
    def test_formats_tool_use_started(self) -> None:
        text = format_progress_event(
            ToolUseStarted(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="shell",
                arguments={"cmd": "pytest"},
            )
        )

        self.assertEqual(text, 'loop 2 start shell\nargs {"cmd": "pytest"}')

    def test_formats_model_request_started(self) -> None:
        text = format_progress_event(ModelRequestStarted(session_id="session-1", iteration=2))

        self.assertEqual(text, "loop 2 model request")

    def test_formats_tool_use_finished(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=2,
                call_id="call_1",
                name="shell",
                result={"exit_code": 0},
            )
        )

        self.assertEqual(text, 'loop 2 finish shell\nresult {"exit_code": 0}')

    def test_truncates_large_tool_result_before_rendering(self) -> None:
        text = format_progress_event(
            ToolUseFinished(
                session_id="session-1",
                iteration=1,
                call_id="call_1",
                name="file_read",
                result={"path": "large.txt", "content": f"head{'x' * 3000}tail"},
            )
        )

        self.assertIn("head", text)
        self.assertIn("[truncated]", text)
        self.assertNotIn("tail", text)


if __name__ == "__main__":
    unittest.main()
