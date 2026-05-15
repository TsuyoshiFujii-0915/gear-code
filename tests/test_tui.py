import unittest
from typing import Any

from gear_code.tui import collect_chat_lines, collect_token_usage, ChatLine


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


if __name__ == "__main__":
    unittest.main()
