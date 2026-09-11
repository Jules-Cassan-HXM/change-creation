"""The explicit review pass."""

from __future__ import annotations

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..assessment import evaluate
from ..rendering import render_change, render_findings
from ..rules.critic import critique_fields
from ..state import get_fields


@tool
def review_change(runtime: ToolRuntime) -> Command:
    """Review the whole change: consistency, completeness, and whether the
    free-text fields would make sense to someone who was not part of the
    conversation.

    Call this when the user thinks they are finished, when they ask for a
    review, and before proposing to submit. It is slower than the checks that
    run on every update because it reads the prose properly.

    Findings come back with suggested rewrites. Put the suggestions to the user
    as proposals in their own language -- do not apply them silently.
    """
    fields = get_fields(runtime.state)
    if not fields:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="The change is empty; there is nothing to review yet.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    summary = render_change(fields, verbose=True)
    findings = evaluate(fields)
    quality = critique_fields(fields, summary)

    report = (
        f"The change as it stands:\n{summary}\n\n"
        f"Consistency and completeness:\n{render_findings(findings)}\n\n"
        f"How it reads to someone with no context:\n{render_findings(quality)}"
    )
    # Only the deterministic findings are kept in state: they are recomputed on
    # every update, so they cannot go stale. The critique is a snapshot of text
    # that is about to change, and is left in the conversation instead.
    return Command(
        update={
            "findings": findings,
            "messages": [ToolMessage(content=report, tool_call_id=runtime.tool_call_id)],
        }
    )
