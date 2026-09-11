"""The judgment half of the review.

The deterministic catalog can tell that a backout plan is three words long. It
cannot tell that a plan is precise about the wrong thing, or that a description
explains the *what* at length and never the *why*. That reading is what this
pass adds, against a fixed rubric so the answer does not drift with the model's
mood.

It costs a model call, so it runs on explicit review and before submission --
not on every field update.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm import get_model
from ..schema import FIELD_SPECS
from ..state import FieldValue
from .base import Finding

#: Free-text fields worth critiquing, in the order a reader meets them.
CRITIQUED_FIELDS = (
    "short_description",
    "description",
    "justification",
    "implementation_plan",
    "backout_plan",
    "test_plan",
    "risk_impact_analysis",
)

#: Below this, the field is reported as a finding rather than left alone.
ACCEPTABLE_SCORE = 4

RUBRIC = """\
You are reviewing one field of a ServiceNow change request on behalf of the \
people who did not write it: the approver deciding whether to let it run, and \
the engineer who may have to execute or reverse it at 3am with no context and \
no way to reach the author.

Score the field from 1 to 5 against four criteria:
- audience: understandable to someone who was not in the room. No unexplained \
internal shorthand, no "as discussed".
- actionable: concrete enough to act on -- named hosts, commands, systems, \
people. Not "update the server".
- verifiable: says how anyone would know it worked, or went wrong.
- complete: covers why, not only what.

5 means a stranger could act on it unaided. 3 means it is recognisable but \
would generate questions. 1 means it is a placeholder.

Judge only what the field is for; do not ask a short description to carry a \
plan. Write the issues and the rewrite in the same language the author used. \
Suggest a rewrite only when you can improve it using facts already present in \
the change -- never invent hostnames, versions, times or names. If you have \
nothing to add, return the field with no rewrite.\
"""


class FieldCritique(BaseModel):
    field: str
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    suggested_rewrite: str | None = None


class Critique(BaseModel):
    critiques: list[FieldCritique] = Field(default_factory=list)


def critique_fields(
    fields: dict[str, FieldValue], change_summary: str
) -> list[Finding]:
    """Return one finding per free-text field that falls short of the rubric."""
    present = [
        name
        for name in CRITIQUED_FIELDS
        if name in fields and fields[name].is_set and isinstance(fields[name].value, str)
    ]
    if not present:
        return []

    blocks = "\n\n".join(
        f"### {name} ({FIELD_SPECS[name].label})\n"
        f"Purpose: {FIELD_SPECS[name].guidance or 'n/a'}\n"
        f"Current text:\n{fields[name].value}"
        for name in present
    )
    prompt = (
        f"{RUBRIC}\n\n"
        f"The change as it stands, for context only:\n{change_summary}\n\n"
        f"Review each of these fields:\n\n{blocks}"
    )

    model = get_model("critic").with_structured_output(Critique)
    result = model.invoke(prompt)
    critique = result if isinstance(result, Critique) else Critique.model_validate(result)

    findings: list[Finding] = []
    for item in critique.critiques:
        if item.field not in FIELD_SPECS or item.score >= ACCEPTABLE_SCORE:
            continue
        findings.append(
            Finding(
                code="Q001",
                severity="warning",
                field=item.field,
                message=(
                    f"{FIELD_SPECS[item.field].label} scores {item.score}/5 for a reader "
                    f"with no context."
                ),
                data={
                    "score": item.score,
                    "issues": item.issues,
                    "suggested_rewrite": item.suggested_rewrite,
                },
            )
        )
    return findings
