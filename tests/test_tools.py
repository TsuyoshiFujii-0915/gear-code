import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gear_code.errors import GearError
from gear_code.tools.filesystem import FileReadTool, FileWriteTool
from gear_code.tools.registry import ToolRegistry
from gear_code.tools.runtimes import DockerShellRuntime, ShellRuntime
from gear_code.tools.shell import ShellTool


class FakeShellRuntime(ShellRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, int]] = []

    def run(self, command: str, workdir: Path, timeout_seconds: int) -> dict[str, object]:
        self.calls.append((command, workdir, timeout_seconds))
        return {
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }


class ToolTests(unittest.TestCase):
    def test_unknown_tool_name_fails(self) -> None:
        registry = ToolRegistry([])

        with self.assertRaises(GearError) as error:
            registry.run("missing_tool", {})

        self.assertEqual(error.exception.origin, "tool_registry")

    def test_file_read_rejects_path_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            outside_file = Path(temp_dir) / "secret.txt"
            outside_file.write_text("secret", encoding="utf-8")
            tool = FileReadTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"path": "../secret.txt"})

            self.assertEqual(error.exception.origin, "file_read")

    def test_file_write_requires_existing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tool = FileWriteTool(workspace)

            with self.assertRaises(GearError) as error:
                tool.run({"path": "missing/file.txt", "content": "hello"})

            self.assertEqual(error.exception.origin, "file_write")

    def test_shell_requires_timeout_seconds(self) -> None:
        runtime = FakeShellRuntime()
        tool = ShellTool(Path.cwd(), runtime)

        with self.assertRaises(GearError) as error:
            tool.run({"command": "echo ok", "workdir": "."})

        self.assertEqual(error.exception.origin, "shell")

    def test_docker_shell_timeout_returns_json_serializable_text_output(self) -> None:
        runtime = DockerShellRuntime(Path.cwd(), "python:3.11-slim", False)
        timeout = subprocess.TimeoutExpired(
            cmd=["docker"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

        with patch("subprocess.run", side_effect=timeout):
            result = runtime.run("sleep 2", Path.cwd(), 1)

        self.assertEqual(result["stdout"], "partial stdout")
        self.assertEqual(result["stderr"], "partial stderr")
        self.assertIs(result["timed_out"], True)
        json.dumps(result)

    def test_tool_schemas_explain_workspace_relative_paths(self) -> None:
        runtime = FakeShellRuntime()
        workspace = Path.cwd()
        shell_schema = ShellTool(workspace, runtime).schema()
        read_schema = FileReadTool(workspace).schema()
        write_schema = FileWriteTool(workspace).schema()

        shell_parameters = shell_schema["parameters"]
        self.assertIsInstance(shell_parameters, dict)
        shell_properties = shell_parameters["properties"]
        self.assertIsInstance(shell_properties, dict)
        shell_workdir = shell_properties["workdir"]
        self.assertIsInstance(shell_workdir, dict)
        self.assertIn("relative", str(shell_workdir["description"]))
        self.assertIn("'.'", str(shell_workdir["description"]))
        self.assertIn("Absolute paths are rejected", str(shell_workdir["description"]))

        read_parameters = read_schema["parameters"]
        write_parameters = write_schema["parameters"]
        self.assertIsInstance(read_parameters, dict)
        self.assertIsInstance(write_parameters, dict)
        read_properties = read_parameters["properties"]
        write_properties = write_parameters["properties"]
        self.assertIsInstance(read_properties, dict)
        self.assertIsInstance(write_properties, dict)
        self.assertIn("relative", str(read_properties["path"]["description"]))
        self.assertIn("relative", str(write_properties["path"]["description"]))

    def test_apply_patch_rejects_parent_directory_target(self) -> None:
        from gear_code.tools.patch import ApplyPatchTool

        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ApplyPatchTool(Path(temp_dir))
            patch = "\n".join(
                [
                    "--- ../outside.txt",
                    "+++ ../outside.txt",
                    "@@ -1 +1 @@",
                    "-old",
                    "+new",
                    "",
                ]
            )

            with self.assertRaises(GearError) as error:
                tool.run({"patch": patch})

            self.assertEqual(error.exception.origin, "apply_patch")


if __name__ == "__main__":
    unittest.main()
