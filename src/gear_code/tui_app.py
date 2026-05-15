from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Rule, Static

from gear_code.agent.compaction import CompactionService
from gear_code.agent.loop import AgentLoop
from gear_code.config import ModelConfig, RuntimeConfig
from gear_code.store.jsonl import JsonlContextStore
from gear_code.tui import (
    ChatLine,
    collect_chat_lines,
    collect_token_usage,
    compact_path,
    compact_session,
    format_tokens,
)

_BG = "#0d0d11"
_SURFACE = "#13131c"
_BRAND = "#a78bfa"
_USER = "#34d399"
_ASST = "#7dd3fc"
_META = "#4b5675"
_RULE = "#1c1c28"
_DIM = "#2d3657"

_CSS = f"""
Screen {{
    background: {_BG};
    layout: vertical;
}}

#header {{
    height: 1;
    background: {_SURFACE};
    padding: 0 2;
}}

#workdir {{
    height: 1;
    background: {_BG};
    color: {_META};
    padding: 0 4;
}}

Rule {{
    background: {_BG};
    color: {_RULE};
}}

#chat {{
    height: 1fr;
    background: {_BG};
    padding: 0 2;
    scrollbar-color: {_RULE};
    scrollbar-color-hover: {_DIM};
    scrollbar-background: {_BG};
}}

#input-bar {{
    height: 1;
    background: {_BG};
    layout: horizontal;
}}

#prompt {{
    width: 4;
    background: {_BG};
    color: {_USER};
    padding: 0 2;
}}

Input {{
    height: 1;
    width: 1fr;
    border: none;
    padding: 0;
    background: {_BG};
    color: {_USER};
}}

Input:focus {{
    border: none;
}}
"""


class GearApp(App[None]):
    """Gear Code interactive TUI powered by Textual."""

    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("ctrl+d", "quit", "exit"),
    ]

    def __init__(
        self,
        model: str,
        session_id: str,
        workspace: Path,
        agent_loop: AgentLoop,
        compaction: CompactionService,
        store: JsonlContextStore,
        runtime: RuntimeConfig,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self._model = model
        self._session_id = session_id
        self._workspace = workspace
        self._agent_loop = agent_loop
        self._compaction = compaction
        self._store = store
        self._runtime = runtime
        self._model_config = model_config
        self._token_usage: int | None = None

    def compose(self) -> ComposeResult:
        yield Static(self._build_header(), id="header")
        yield Static(compact_path(self._workspace), id="workdir")
        yield Rule()
        yield RichLog(id="chat", markup=True, highlight=False, wrap=True, auto_scroll=True)
        yield Rule()
        with Horizontal(id="input-bar"):
            yield Static("▸", id="prompt")
            yield Input(id="input")
        yield Rule()

    def on_mount(self) -> None:
        events = self._store.load(self._session_id)
        self._token_usage = collect_token_usage(events)
        self._render_history(collect_chat_lines(events))
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        event.input.clear()
        if not user_text:
            return
        if user_text in {"/quit", "/exit"}:
            self.exit()
            return
        if user_text == "/compact":
            self._set_busy(True)
            self._do_compact()
            return
        self._write_message(self.query_one("#chat", RichLog), ChatLine("you", user_text))
        self._set_busy(True)
        self._do_run_agent(user_text)

    @work(thread=True)
    def _do_run_agent(self, user_text: str) -> None:
        try:
            self._agent_loop.run_turn(
                self._session_id,
                user_text,
                self._runtime.max_iterations,
                self._runtime.model_timeout_seconds,
            )
        except Exception:
            pass
        self._finish_work()

    @work(thread=True)
    def _do_compact(self) -> None:
        self._compaction.compact(
            self._session_id,
            self._store,
            self._model_config,
            self._runtime.model_timeout_seconds,
        )
        self._finish_work()

    def _finish_work(self) -> None:
        events = self._store.load(self._session_id)
        chat_lines = collect_chat_lines(events)
        token_usage = collect_token_usage(events)
        self.call_from_thread(self._apply_result, chat_lines, token_usage)

    def _apply_result(self, chat_lines: list[ChatLine], token_usage: int | None) -> None:
        self._token_usage = token_usage
        self._render_history(chat_lines)
        self.query_one("#header", Static).update(self._build_header())
        self._set_busy(False)

    def _render_history(self, chat_lines: list[ChatLine]) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        for line in chat_lines:
            self._write_message(chat, line)

    def _write_message(self, chat: RichLog, line: ChatLine) -> None:
        if line.speaker == "you":
            chat.write(f"[bold {_USER}]you[/bold {_USER}]")
            chat.write(line.text)
        else:
            chat.write(f"[bold {_ASST}]assistant[/bold {_ASST}]")
            chat.write(RichMarkdown(line.text, code_theme="dracula"))
        chat.write("")

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.query_one("#chat", RichLog).write(f"[{_DIM}]  ● thinking…[/{_DIM}]")
        inp = self.query_one(Input)
        inp.disabled = busy
        if not busy:
            inp.focus()

    def _build_header(self) -> str:
        sep = f"  [{_RULE}]│[/{_RULE}]  "
        return sep.join(
            [
                f"[bold {_BRAND}]◈ GEAR CODE[/bold {_BRAND}]",
                f"[{_META}]{self._model}[/{_META}]",
                f"[{_META}]#{compact_session(self._session_id)}[/{_META}]",
                f"[{_META}]{format_tokens(self._token_usage)}[/{_META}]",
            ]
        )
