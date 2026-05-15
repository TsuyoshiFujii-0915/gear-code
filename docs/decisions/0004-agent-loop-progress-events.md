# ADR-0004: Publish agent loop progress events for TUI rendering

## Status

accepted

## Context

The Textual TUI currently updates chat history after `AgentLoop.run_turn()` finishes.
Tool calls and tool results are already written to the session store, but users
cannot see which tools were used while a turn is running. This hides important
agent behavior during long tool executions.

Recoverable tool failures are part of the model feedback loop. They should be
returned as `function_call_output` so the model can correct its next action.
Unrecoverable tool failures should stop the turn and surface an explicit error.

## Decision

Add explicit agent loop progress events. `AgentLoop` publishes events when a
tool starts and when it finishes. A finished event may contain a normal result
or the explicit recoverable error result that is also returned to the model.

The TUI subscribes to these events and appends them to the `RichLog` from the
Textual app thread. Stored `tool_call` and `tool_result` events are rendered in
history so progress shown during a running turn remains visible after the final
history refresh.

## Consequences

Tool usage becomes visible during each turn without making the TUI poll the
store. Recoverable tool failures remain useful model inputs and are shown to the
user as explicit error results. Unrecoverable failures are not converted into
model input; they are displayed as explicit turn errors and the input is
re-enabled.
