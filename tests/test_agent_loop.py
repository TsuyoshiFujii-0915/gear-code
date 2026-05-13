import unittest
from typing import Any

from gear_code.agent.loop import AgentLoop
from gear_code.config import ModelConfig
from gear_code.errors import GearError, gear_error
from gear_code.model.client import ModelClient
from gear_code.model.transport import HttpTransport
from gear_code.store.memory import MemoryContextStore
from gear_code.tools.base import Tool


class SequencedTransport(HttpTransport):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.payloads: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.responses.pop(0)


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": "echo",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        return {"text": arguments["text"]}


class RecoverableFailingTool(Tool):
    @property
    def name(self) -> str:
        return "recoverable_fail"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Fail recoverably.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raise gear_error(
            "path_outside_workspace",
            "Path is outside the workspace.",
            self.name,
            True,
            {"path": "/testbed"},
        )


class UnrecoverableFailingTool(Tool):
    @property
    def name(self) -> str:
        return "unrecoverable_fail"

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": "Fail unrecoverably.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def run(self, arguments: dict[str, object]) -> dict[str, object]:
        raise gear_error("fatal_tool_error", "Fatal tool error.", self.name, False, {})


class AgentLoopTests(unittest.TestCase):
    def test_runs_tool_call_and_returns_final_text(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text": "ok"}',
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        loop = AgentLoop(client, config, [EchoTool()], store)

        result = loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(result.final_text, "done")
        self.assertEqual(len(transport.payloads), 2)
        self.assertIn("function_call_output", str(transport.payloads[1]["input"]))
        self.assertIn("Use workspace-relative paths", str(transport.payloads[0]["instructions"]))

    def test_recoverable_tool_error_is_returned_to_model(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "recoverable_fail",
                            "arguments": "{}",
                        }
                    ]
                },
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "retried"}],
                        }
                    ]
                },
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        loop = AgentLoop(client, config, [RecoverableFailingTool()], store)

        result = loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(result.final_text, "retried")
        self.assertEqual(len(transport.payloads), 2)
        second_input = transport.payloads[1]["input"]
        self.assertIn("function_call_output", str(second_input))
        self.assertIn("path_outside_workspace", str(second_input))
        self.assertIn("/testbed", str(second_input))

    def test_unrecoverable_tool_error_is_not_returned_to_model(self) -> None:
        transport = SequencedTransport(
            [
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "unrecoverable_fail",
                            "arguments": "{}",
                        }
                    ]
                }
            ]
        )
        client = ModelClient(transport)
        config = ModelConfig(
            url="http://localhost:1234/v1/responses",
            model="local-model-id",
            api_key=None,
        )
        store = MemoryContextStore()
        loop = AgentLoop(client, config, [UnrecoverableFailingTool()], store)

        with self.assertRaises(GearError):
            loop.run_turn("session-1", "hello", 4, 30)

        self.assertEqual(len(transport.payloads), 1)


if __name__ == "__main__":
    unittest.main()
