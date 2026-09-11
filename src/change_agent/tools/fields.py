"""Reading and writing the change's non-reference fields."""

from __future__ import annotations

from typing import Any, Literal

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..assessment import evaluate
from ..rendering import format_value, render_findings
from ..schema import FIELD_SPECS, REQUIRED_FIELDS, FieldSpec
from ..servicenow.base import ServiceNowError
from ..servicenow.factory import get_client
from ..state import FieldValue, get_fields
from ..validation import FieldRejected, coerce_value


class FieldUpdate(BaseModel):
    """One proposed value for one field."""

    field: str = Field(description="The change field to set, e.g. 'short_description'.")
    value: str | bool | int | float = Field(
        description="The value. Dates are UTC, 'YYYY-MM-DD HH:MM:SS'."
    )
    source: Literal["user", "inferred"] = Field(
        default="inferred",
        description=(
            "'user' when the user stated this value themselves; 'inferred' when you "
            "derived it from what they wrote. Be honest: inferred values are shown to "
            "the user as unconfirmed and are the ones they will want to check."
        ),
    )
    rationale: str | None = Field(
        default=None, description="For inferred values, what you based it on."
    )
    force: bool = Field(
        default=False,
        description=(
            "Only set true to overwrite a value the user already confirmed, and only "
            "after they have agreed to the change."
        ),
    )


def _describe(spec: FieldSpec) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "field": spec.name,
        "label": spec.label,
        "kind": spec.kind,
        "required_for_submit": spec.required_for_submit,
    }
    if spec.guidance:
        detail["guidance"] = spec.guidance
    if spec.reference_table:
        detail["reference_table"] = spec.reference_table
        detail["how_to_set"] = f"search_reference('{spec.name}', <what the user said>)"
    if spec.choice_source:
        try:
            detail["allowed_values"] = [
                {"value": c.value, "label": c.label}
                for c in get_client().get_choices(spec.choice_source)
            ]
        except ServiceNowError:
            pass
    return detail


@tool
def describe_fields(fields: list[str] | None = None) -> str:
    """Explain change fields: what they mean, whether they are required, and
    which values are allowed.

    Call this before asking the user about a field you are unsure of, and to get
    the allowed values for a choice field. With no argument it describes every
    field that is required for submission.
    """
    names = fields or list(REQUIRED_FIELDS)
    unknown = [n for n in names if n not in FIELD_SPECS]
    if unknown:
        return (
            f"Unknown field(s): {', '.join(unknown)}. "
            f"Known fields: {', '.join(FIELD_SPECS)}"
        )
    described = [_describe(FIELD_SPECS[n]) for n in names]
    return "\n".join(
        "\n".join(f"{k}: {v}" for k, v in detail.items()) + "\n" for detail in described
    )


@tool
def propose_change_fields(
    updates: list[FieldUpdate], runtime: ToolRuntime
) -> Command:
    """Set one or more fields on the change.

    Use this for everything the user tells you, whether they answered a single
    question or pasted a whole description at once. Reference fields
    (configuration item, groups, people) cannot be set here -- use
    search_reference for those.

    The result tells you what was accepted, what was rejected and why, what needs
    the user's confirmation before it can be overwritten, and every open finding
    against the change afterwards. Raise the findings with the user in their own
    language; do not just list codes at them.
    """
    client = get_client()
    current = get_fields(runtime.state)

    accepted: dict[str, FieldValue] = {}
    accepted_lines: list[str] = []
    rejected_lines: list[str] = []
    conflict_lines: list[str] = []

    for update in updates:
        try:
            value = coerce_value(update.field, update.value, client)
        except FieldRejected as rejection:
            detail = ", ".join(f"{k}={v}" for k, v in rejection.data.items())
            rejected_lines.append(
                f"- {rejection.field}: {rejection.reason}" + (f" ({detail})" if detail else "")
            )
            continue

        existing = current.get(update.field)
        # A value the user confirmed is not overwritten behind their back. This
        # is the difference between a copilot and something that quietly loses
        # what you told it three turns ago.
        if (
            existing
            and existing.is_set
            and existing.confirmed
            and existing.value != value
            and not update.force
        ):
            conflict_lines.append(
                f"- {update.field}: currently {format_value(existing.value)!r} "
                f"(confirmed by the user), you proposed {format_value(value)!r}. "
                f"Ask the user which is right, then repeat the update with force=true."
            )
            continue

        accepted[update.field] = FieldValue(
            value=value,
            source=update.source,
            confirmed=update.source == "user",
            rationale=update.rationale,
        )
        accepted_lines.append(
            f"- {update.field} = {format_value(value)} ({update.source})"
        )

    merged = {**current, **accepted}
    findings = evaluate(merged)

    report = []
    report.append("Accepted:\n" + ("\n".join(accepted_lines) or "- nothing"))
    if conflict_lines:
        report.append("Needs the user's confirmation before overwriting:\n" + "\n".join(conflict_lines))
    if rejected_lines:
        report.append("Rejected:\n" + "\n".join(rejected_lines))
    report.append("Open findings:\n" + render_findings(findings))

    return Command(
        update={
            "fields": accepted,
            "findings": findings,
            "messages": [
                ToolMessage(
                    content="\n\n".join(report), tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )
