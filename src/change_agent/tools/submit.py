"""Submission, behind an explicit human approval."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command, interrupt

from ..assessment import evaluate
from ..payload import build_payload, idempotency_key
from ..rendering import render_change, render_findings
from ..rules.catalog import blockers
from ..rules.critic import critique_fields
from ..servicenow.base import ServiceNowError
from ..servicenow.factory import get_client
from ..state import Submission, get_fields, get_submission


def _thread_id(runtime: ToolRuntime) -> str | None:
    config: dict[str, Any] = getattr(runtime, "config", None) or {}
    return (config.get("configurable") or {}).get("thread_id")


def _state_value(runtime: ToolRuntime, key: str) -> Any:
    state = runtime.state
    return state.get(key) if hasattr(state, "get") else getattr(state, key, None)


def _message(runtime: ToolRuntime, content: str, **update: Any) -> Command:
    return Command(
        update={
            **update,
            "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def submit_change(runtime: ToolRuntime) -> Command:
    """Submit the change to ServiceNow.

    Only call this once the user has said they want to submit. The tool runs the
    full review again, refuses while anything blocking is open, and then pauses
    for the user to approve the exact content that will be created -- their
    approval is required and cannot be given on their behalf.
    """
    fields = get_fields(runtime.state)
    if not fields:
        return _message(runtime, "There is nothing to submit; the change is empty.")

    existing = get_submission(runtime.state)
    if existing and existing.number:
        return _message(
            runtime,
            f"This change has already been submitted as {existing.number}. "
            f"Creating it a second time would duplicate it.",
        )

    override_reason: str | None = _state_value(runtime, "override_reason")
    findings = evaluate(fields)
    open_blockers = blockers(findings)

    if open_blockers and not override_reason:
        return _message(
            runtime,
            "Not submitted. These have to be resolved first:\n"
            + render_findings(open_blockers)
            + "\n\nWork through them with the user. If they decide to submit anyway, "
            "they must give a reason, which is recorded on the change; set it with "
            "record_override_reason.",
            findings=findings,
        )

    quality = critique_fields(fields, render_change(fields, verbose=True))
    payload = build_payload(fields, override_reason=override_reason)
    key = idempotency_key(payload, _thread_id(runtime))

    preview = render_change(fields, verbose=True)
    warnings = [f for f in findings if f.severity == "warning"] + quality

    decision = interrupt(
        {
            "type": "approve_change_submission",
            "preview": preview,
            "warnings": [f.model_dump() for f in warnings],
            "blockers": [f.model_dump() for f in open_blockers],
            "override_reason": override_reason,
            "payload": payload,
            "instructions": (
                "Reply with {'action': 'approve'} to create this change, "
                "{'action': 'edit', 'notes': '...'} to keep working on it, or "
                "{'action': 'cancel'}."
            ),
        }
    )

    action, notes = _read_decision(decision)

    if action != "approve":
        return _message(
            runtime,
            f"The user did not approve submission (action: {action})."
            + (f" What they said: {notes}" if notes else "")
            + " Nothing was created. Keep working on the change with them.",
        )

    try:
        created = get_client().create_change(payload, key)
    except ServiceNowError as exc:
        return _message(
            runtime,
            f"ServiceNow rejected the change: {exc}. Nothing was created. "
            f"Tell the user what failed.",
        )

    submission = Submission(
        idempotency_key=key,
        number=created.number,
        sys_id=created.sys_id,
        submitted_at=datetime.now(UTC),
        payload=payload,
    )
    return _message(
        runtime,
        f"Created {created.number} (sys_id {created.sys_id}). Tell the user the "
        f"number and, if any warnings were still open, which ones they accepted.",
        submission=submission,
    )


def _read_decision(decision: Any) -> tuple[str, str | None]:
    """Accept both a structured answer and a plain word from the client."""
    if isinstance(decision, dict):
        return str(decision.get("action", "cancel")).lower(), decision.get("notes")
    if isinstance(decision, bool):
        return ("approve" if decision else "cancel"), None
    text = str(decision).strip().lower()
    if text in {"approve", "approved", "yes", "y", "ok", "oui"}:
        return "approve", None
    return text or "cancel", None


@tool
def record_override_reason(reason: str, runtime: ToolRuntime) -> Command:
    """Record the user's reason for submitting with blocking findings still open.

    Only call this when the user has been shown the blockers and has explicitly
    decided to go ahead anyway. The reason is written into the change's work
    notes, so whoever approves it can see what was overridden and why.
    """
    if not reason.strip():
        return _message(runtime, "An override needs a reason. Ask the user for one.")
    return _message(
        runtime,
        f"Override recorded: {reason}. It will appear in the change's work notes. "
        f"submit_change will now proceed to the approval step.",
        override_reason=reason.strip(),
    )
