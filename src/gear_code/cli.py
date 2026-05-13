from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Mapping
from pathlib import Path
from uuid import uuid4
import os
import sys

from gear_code.agent.compaction import CompactionService
from gear_code.agent.loop import AgentLoop
from gear_code.config import (
    DEFAULT_DOCKER_IMAGE,
    RuntimeConfig,
    discover_config_path,
    initialize_config,
    load_config,
)
from gear_code.errors import GearError
from gear_code.model.client import ModelClient
from gear_code.model.transport import UrllibHttpTransport
from gear_code.store.jsonl import JsonlContextStore
from gear_code.tools.configured import build_configured_tools
from gear_code.tools.runtimes import DockerShellRuntime


def main() -> None:
    """Entrypoint for the gear command."""

    exit_code = run_cli(sys.argv[1:], os.environ)
    raise SystemExit(exit_code)


def run_cli(argv: list[str], environment: Mapping[str, str]) -> int:
    """Runs Gear Code command-line entrypoint.

    Args:
        argv: Command-line arguments without program name.
        environment: Environment variables.

    Returns:
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            _run_init(args, environment)
        else:
            _run_tui(args, environment)
    except GearError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="gear")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to gear TOML config. Overrides scoped config discovery.",
    )
    parser.add_argument("--workdir", type=Path, help="Override runtime.workdir.")
    parser.add_argument("--session-dir", type=Path, help="Override runtime.session_dir.")
    parser.add_argument(
        "--network",
        choices=["enabled", "disabled"],
        help="Override runtime.network.",
    )
    parser.add_argument("--max-iterations", type=int, help="Override runtime.max_iterations.")
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        help="Override runtime.model_timeout_seconds.",
    )
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser("init", help="Create a Gear Code config file.")
    init_parser.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help="Config scope to initialize. Defaults to project.",
    )
    return parser


def _run_tui(args: Namespace, environment: Mapping[str, str]) -> None:
    config_path = args.config
    if config_path is None:
        config_path = discover_config_path(Path.cwd(), _home_dir(environment))
    config = load_config(config_path, environment)
    runtime = _runtime_from_args(config.runtime, args)
    workspace = runtime.workdir.resolve()
    session_id = str(uuid4())
    shell_runtime = DockerShellRuntime(workspace, DEFAULT_DOCKER_IMAGE, runtime.network_enabled)
    tools = build_configured_tools(config.tool, workspace, shell_runtime)
    store = JsonlContextStore(runtime.session_dir)
    loop = AgentLoop(
        ModelClient(UrllibHttpTransport()),
        config.model,
        tools,
        store,
    )
    compaction = CompactionService(ModelClient(UrllibHttpTransport()))

    _print_header(config.model.model, config.model.url, session_id)
    while True:
        user_text = input("you> ")
        if user_text.strip() in {"/quit", "/exit"}:
            print("status> exiting")
            return
        if user_text.strip() == "/compact":
            print("status> compacting")
            summary = compaction.compact(
                session_id,
                store,
                config.model,
                runtime.model_timeout_seconds,
            )
            print(f"summary> {summary}")
            continue
        if user_text.strip() == "":
            print("status> empty message ignored")
            continue
        print("status> running")
        event_count = len(store.load(session_id))
        try:
            result = loop.run_turn(
                session_id,
                user_text,
                runtime.max_iterations,
                runtime.model_timeout_seconds,
            )
        except KeyboardInterrupt:
            print("status> interrupted")
            continue
        _print_turn_events(store.load(session_id)[event_count:])
        print(f"assistant> {result.final_text}")


def _run_init(args: Namespace, environment: Mapping[str, str]) -> None:
    path = initialize_config(args.scope, Path.cwd(), _home_dir(environment))
    print(f"created> {path}")


def _print_header(model: str, url: str, session_id: str) -> None:
    print("Gear Code")
    print(f"model> {model}")
    print(f"url> {url}")
    print(f"session> {session_id}")
    print("commands> /compact /quit /exit")


def _runtime_from_args(runtime: RuntimeConfig, args: Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        workdir=args.workdir if args.workdir is not None else runtime.workdir,
        session_dir=args.session_dir if args.session_dir is not None else runtime.session_dir,
        network_enabled=_network_enabled_from_arg(args.network, runtime.network_enabled),
        max_iterations=(
            args.max_iterations if args.max_iterations is not None else runtime.max_iterations
        ),
        model_timeout_seconds=(
            args.model_timeout_seconds
            if args.model_timeout_seconds is not None
            else runtime.model_timeout_seconds
        ),
    )


def _network_enabled_from_arg(network: str | None, config_value: bool) -> bool:
    if network is None:
        return config_value
    return network == "enabled"


def _home_dir(environment: Mapping[str, str]) -> Path:
    home = environment.get("HOME")
    if home is None or home == "":
        raise GearError(
            "home_missing",
            "HOME environment variable is required for user-scoped config discovery.",
            "config",
            True,
            {},
        )
    return Path(home)


def _print_turn_events(events: list[dict[str, Any]]) -> None:
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload")
        if kind == "tool_call" and isinstance(payload, dict):
            print(f"tool> {payload.get('name')}")
        if kind == "tool_result" and isinstance(payload, dict):
            result = payload.get("result")
            print(f"tool-result> {_summarize_result(result)}")


def _summarize_result(result: object) -> str:
    text = str(result)
    if len(text) <= 240:
        return text
    return f"{text[:240]}..."


if __name__ == "__main__":
    main()
