"""Keeping the model grounded in the change it is actually writing.

Without this the model reconstructs the change from a long tail of tool
messages and drifts: it re-asks for fields that are already set, or forgets that
a value was confirmed. Re-stating the current state on every turn is far cheaper
than the confusion it prevents.
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from .prompts import SYSTEM_PROMPT
from .rendering import render_change, render_findings
from .state import get_fields, get_findings, get_pending_lookup, get_submission


@dynamic_prompt
def ground_in_current_change(request: ModelRequest) -> str:
    """Append the live change and its open findings to the system prompt."""
    state = request.state
    fields = get_fields(state)
    findings = get_findings(state)
    pending = get_pending_lookup(state)

    sections = [SYSTEM_PROMPT, "# The change as it stands\n" + render_change(fields)]

    if findings:
        sections.append("# Open findings\n" + render_findings(findings))

    awaiting = [field for field, candidates in pending.items() if candidates]
    if awaiting:
        sections.append(
            "# Awaiting the user's choice\n"
            + "\n".join(
                f"- {field}: candidates offered, waiting for the user to pick one"
                for field in awaiting
            )
        )

    submission = get_submission(state)
    if submission and submission.number:
        sections.append(
            f"# Already submitted\nThis change was created as {submission.number}. "
            f"Do not submit it again."
        )

    return "\n\n".join(sections)
