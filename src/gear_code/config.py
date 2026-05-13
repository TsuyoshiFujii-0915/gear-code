from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import tomllib

from gear_code.errors import GearError, gear_error


DEFAULT_DOCKER_IMAGE = "python:3.11-slim"

DEFAULT_CONFIG_TEXT = """[model]
url = "http://localhost:1234/v1/responses"
model = "local-model-id"
api_key_env = ""

[runtime]
workdir = "."
session_dir = ".gear/sessions"
network = "disabled"
max_iterations = 8
model_timeout_seconds = 120
"""


@dataclass(frozen=True)
class ModelConfig:
    """Model endpoint configuration.

    Attributes:
        url: Complete Responses API-compatible endpoint URL.
        model: Model identifier sent in the request body.
        api_key: Bearer token value, or None when no auth header should be sent.
    """

    url: str
    model: str
    api_key: str | None


@dataclass(frozen=True)
class AppConfig:
    """Top-level Gear Code configuration.

    Attributes:
        model: Model endpoint configuration.
        runtime: Runtime configuration.
    """

    model: ModelConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration loaded from config.toml.

    Attributes:
        workdir: Workspace directory.
        session_dir: Directory for JSONL session logs.
        network_enabled: Whether sandbox shell commands can access the network.
        max_iterations: Maximum model calls per user turn.
        model_timeout_seconds: Model request timeout in seconds.
    """

    workdir: Path
    session_dir: Path
    network_enabled: bool
    max_iterations: int
    model_timeout_seconds: int


def load_config(path: Path, environment: Mapping[str, str]) -> AppConfig:
    """Loads Gear Code configuration from TOML.

    Args:
        path: Path to the TOML configuration file.
        environment: Environment mapping used to resolve API keys.

    Returns:
        Parsed application configuration.

    Raises:
        GearError: If the file cannot be read or required values are invalid.
    """

    try:
        raw_data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise gear_error(
            "config_read_failed",
            f"Failed to read config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise gear_error(
            "config_parse_failed",
            f"Failed to parse config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc

    model_table = _required_table(raw_data, "model")
    url = _required_string(model_table, "url", "model")
    model = _required_string(model_table, "model", "model")
    api_key_env = _required_string(model_table, "api_key_env", "model")
    api_key = _resolve_api_key(api_key_env, environment)
    runtime_table = _required_table(raw_data, "runtime")
    runtime = RuntimeConfig(
        workdir=Path(_required_string(runtime_table, "workdir", "runtime")),
        session_dir=Path(_required_string(runtime_table, "session_dir", "runtime")),
        network_enabled=_required_network(runtime_table),
        max_iterations=_required_positive_int(runtime_table, "max_iterations", "runtime"),
        model_timeout_seconds=_required_positive_int(
            runtime_table,
            "model_timeout_seconds",
            "runtime",
        ),
    )
    return AppConfig(ModelConfig(url=url, model=model, api_key=api_key), runtime)


def discover_config_path(start_dir: Path, home_dir: Path) -> Path:
    """Finds the effective Gear Code config file.

    Project-scoped config files are searched first, walking from start_dir
    toward the filesystem root. User-scoped config is used only when no
    project config exists.

    Args:
        start_dir: Directory used as the project discovery starting point.
        home_dir: User home directory.

    Returns:
        Path to the selected config file.

    Raises:
        GearError: If no config file exists in either scope.
    """

    current = start_dir.absolute()
    for directory in [current, *current.parents]:
        candidate = directory / ".gear" / "config.toml"
        if candidate.is_file():
            return candidate

    user_candidate = user_config_path(home_dir)
    if user_candidate.is_file():
        return user_candidate

    raise gear_error(
        "config_not_found",
        "No Gear Code config found. Run 'gear init' or 'gear init --scope user'.",
        "config",
        True,
        {
            "project_candidate": str(project_config_path(start_dir)),
            "user_candidate": str(user_candidate),
        },
    )


def initialize_config(scope: str, project_root: Path, home_dir: Path) -> Path:
    """Creates a default config file for a scope.

    Args:
        scope: Either "project" or "user".
        project_root: Project root used for project-scoped config.
        home_dir: Home directory used for user-scoped config.

    Returns:
        Created config path.

    Raises:
        GearError: If scope is invalid or the target already exists.
    """

    if scope == "project":
        path = project_config_path(project_root)
    elif scope == "user":
        path = user_config_path(home_dir)
    else:
        raise gear_error(
            "config_scope_invalid",
            "Invalid config scope. Expected 'project' or 'user'.",
            "config",
            True,
            {"scope": scope},
        )

    if path.exists():
        raise gear_error(
            "config_already_exists",
            f"Config already exists: {path}",
            "config",
            True,
            {"path": str(path)},
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    except OSError as exc:
        raise gear_error(
            "config_write_failed",
            f"Failed to write config file: {path}",
            "config",
            True,
            {"path": str(path), "reason": str(exc)},
        ) from exc
    return path


def project_config_path(project_root: Path) -> Path:
    """Returns the project-scoped config path.

    Args:
        project_root: Project root directory.

    Returns:
        Project-scoped config path.
    """

    return project_root.absolute() / ".gear" / "config.toml"


def user_config_path(home_dir: Path) -> Path:
    """Returns the user-scoped config path.

    Args:
        home_dir: User home directory.

    Returns:
        User-scoped config path.
    """

    return home_dir.absolute() / ".gear" / "config.toml"


def _required_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid [{key}] table.",
            "config",
            True,
            {"key": key},
        )
    return value


def _required_string(data: dict[str, object], key: str, table_name: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid string value: {table_name}.{key}",
            "config",
            True,
            {"table": table_name, "key": key},
        )
    return value


def _resolve_api_key(api_key_env: str, environment: Mapping[str, str]) -> str | None:
    if api_key_env == "":
        return None
    value = environment.get(api_key_env)
    if value is None or value == "":
        raise gear_error(
            "config_secret_missing",
            f"Environment variable is required but not set: {api_key_env}",
            "config",
            True,
            {"env": api_key_env},
        )
    return value


def _required_network(data: dict[str, object]) -> bool:
    value = _required_string(data, "network", "runtime")
    if value == "enabled":
        return True
    if value == "disabled":
        return False
    raise gear_error(
        "config_value_invalid",
        "Invalid value for runtime.network. Expected 'enabled' or 'disabled'.",
        "config",
        True,
        {"table": "runtime", "key": "network", "value": value},
    )


def _required_positive_int(data: dict[str, object], key: str, table_name: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 1:
        raise gear_error(
            "config_value_invalid",
            f"Missing or invalid positive integer value: {table_name}.{key}",
            "config",
            True,
            {"table": table_name, "key": key},
        )
    return value
