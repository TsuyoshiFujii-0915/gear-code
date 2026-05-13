from __future__ import annotations

from pathlib import Path

from gear_code.config import ToolConfig
from gear_code.tools.base import Tool
from gear_code.tools.filesystem import FileReadTool, FileWriteTool
from gear_code.tools.patch import ApplyPatchTool
from gear_code.tools.runtimes import ShellRuntime
from gear_code.tools.shell import ShellTool


def build_configured_tools(
    config: ToolConfig,
    workspace: Path,
    shell_runtime: ShellRuntime,
) -> list[Tool]:
    """Builds model-callable tools enabled in config order.

    Args:
        config: Parsed tool availability configuration.
        workspace: Workspace root for file and shell tools.
        shell_runtime: Runtime used by the shell tool when enabled.

    Returns:
        Enabled tool instances in deterministic schema order.
    """

    tools: list[Tool] = []
    if config.shell_tool:
        tools.append(ShellTool(workspace, shell_runtime))
    if config.file_read:
        tools.append(FileReadTool(workspace))
    if config.file_write:
        tools.append(FileWriteTool(workspace))
    if config.apply_patch:
        tools.append(ApplyPatchTool(workspace))
    return tools
