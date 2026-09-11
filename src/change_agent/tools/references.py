"""Setting the fields that must point at an exact record.

A configuration item, a group or a person cannot be approximated: a change
pointed at the wrong CI is worse than one pointed at nothing. So no ``sys_id``
ever reaches the change from text the model wrote. It can only arrive by way of
a lookup whose results were put in front of the user.
"""

from __future__ import annotations

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from ..assessment import evaluate
from ..rendering import render_candidates, render_findings
from ..schema import FIELD_SPECS, REFERENCE_FIELDS, Reference
from ..servicenow.base import ServiceNowError
from ..servicenow.factory import get_client
from ..state import Candidate, FieldValue, get_fields, get_pending_lookup


def _reference_spec(field: str):
    spec = FIELD_SPECS.get(field)
    if spec is None or not spec.is_reference:
        raise ValueError(
            f"{field!r} is not a reference field. Reference fields: "
            f"{', '.join(REFERENCE_FIELDS)}"
        )
    return spec


def _apply(field: str, candidate: Candidate, runtime: ToolRuntime, note: str) -> Command:
    """Write a chosen candidate into the change and re-run the rules."""
    value = FieldValue(
        value=Reference(
            sys_id=candidate.sys_id,
            display_value=candidate.display_value,
            table=candidate.table,
            context=candidate.context,
        ),
        source="lookup",
        confirmed=True,
    )
    merged = {**get_fields(runtime.state), field: value}
    findings = evaluate(merged)
    return Command(
        update={
            "fields": {field: value},
            # The offer has been taken up; leaving it open invites a later
            # selection against a stale list.
            "pending_lookup": {field: []},
            "findings": findings,
            "messages": [
                ToolMessage(
                    content=f"{note}\n\nOpen findings:\n{render_findings(findings)}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def search_reference(field: str, query: str, runtime: ToolRuntime) -> Command:
    """Look up the real records that could match what the user said, for a
    reference field such as cmdb_ci, assignment_group, assigned_to,
    requested_by or business_service.

    Show the candidates to the user with the detail that tells them apart --
    environment, support group, criticality -- and let them choose. Never pick
    between a production and a non-production record on the user's behalf.
    Then call select_reference with the sys_id they chose.
    """
    try:
        spec = _reference_spec(field)
    except ValueError as exc:
        return Command(
            update={"messages": [ToolMessage(content=str(exc), tool_call_id=runtime.tool_call_id)]}
        )

    try:
        records = get_client().search_reference(spec.reference_table or "", query, limit=5)
    except ServiceNowError as exc:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"The lookup failed: {exc}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    candidates = [
        Candidate(
            sys_id=r.sys_id,
            display_value=r.display_value,
            table=r.table,
            context=r.context,
            score=r.score,
        )
        for r in records
    ]

    if not candidates:
        return Command(
            update={
                "pending_lookup": {field: []},
                "messages": [
                    ToolMessage(
                        content=(
                            f"Nothing in {spec.reference_table} matches {query!r}. "
                            f"Ask the user for the exact name as it appears in "
                            f"ServiceNow, or for a different identifier."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    # One unambiguous, exact name match is not worth a round trip. Anything less
    # certain than that goes to the user.
    exact = [c for c in candidates if c.display_value.lower() == query.strip().lower()]
    if len(exact) == 1 and len(candidates) == 1:
        return _apply(
            field,
            exact[0],
            runtime,
            f"{spec.label} set to {exact[0].summary()} (single exact match).",
        )

    listing = render_candidates(field, candidates)
    return Command(
        update={
            "pending_lookup": {field: candidates},
            "messages": [
                ToolMessage(
                    content=(
                        f"{listing}\n\nAsk the user which one they mean, quoting the "
                        f"detail that distinguishes them, then call select_reference."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def select_reference(field: str, choice: str, runtime: ToolRuntime) -> Command:
    """Set a reference field to one of the candidates the user just chose from.

    `choice` is the sys_id, or the exact display value, of a candidate returned
    by the most recent search_reference for this field. Only call this once the
    user has told you which one they mean.
    """
    try:
        spec = _reference_spec(field)
    except ValueError as exc:
        return Command(
            update={"messages": [ToolMessage(content=str(exc), tool_call_id=runtime.tool_call_id)]}
        )

    offered = get_pending_lookup(runtime.state).get(field, [])
    if not offered:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"No candidates are on offer for {field}. Run "
                            f"search_reference('{field}', ...) first -- a reference can "
                            f"only be set from a lookup result."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    wanted = choice.strip().lower()
    match = next(
        (
            c
            for c in offered
            if c.sys_id.lower() == wanted or c.display_value.lower() == wanted
        ),
        None,
    )
    if match is None:
        # The guard that makes fabrication structurally impossible rather than
        # merely discouraged.
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"{choice!r} is not one of the candidates offered for {field}. "
                            f"Choose from:\n{render_candidates(field, offered)}"
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return _apply(field, match, runtime, f"{spec.label} set to {match.summary()}.")
