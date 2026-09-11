"""The change request field model.

This module is the single source of truth for *what* a change request contains.
Field metadata (label, guidance, whether it blocks submission, which table a
reference points at) lives in ``json_schema_extra`` on the field itself, so
there is no parallel spec file that can drift away from the model.

Every field is optional: a change under construction is legitimately incomplete.
Validation of *completeness* is the job of the rule catalog, not of pydantic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FieldKind = Literal["text", "longtext", "choice", "reference", "datetime", "boolean", "integer"]

# ServiceNow serialises date/times in this format, in UTC.
SN_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class Reference(BaseModel):
    """A pointer to an exact record in another table.

    The ``sys_id`` may only ever come from a lookup result -- see
    ``change_agent.tools.references``. Nothing in the system constructs one from
    model-authored text.
    """

    model_config = ConfigDict(frozen=True)

    sys_id: str
    display_value: str
    table: str
    # Extra columns kept for disambiguation and for rules (e.g. a CI's
    # environment or support group).
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:  # what the user sees in a preview
        return self.display_value


def _f(
    kind: FieldKind,
    label: str,
    *,
    required: bool = False,
    guidance: str = "",
    reference_table: str | None = None,
    choice_source: str | None = None,
    sn_field: str | None = None,
) -> dict[str, Any]:
    """Build the metadata blob attached to a field."""
    return {
        "kind": kind,
        "label": label,
        "required_for_submit": required,
        "guidance": guidance,
        "reference_table": reference_table,
        "choice_source": choice_source,
        "sn_field": sn_field,
    }


class ChangeRequest(BaseModel):
    """A normal change request.

    Curated core only. Anything outside this set goes in ``extra`` and is passed
    through to ServiceNow untouched.
    """

    model_config = ConfigDict(extra="forbid")

    # -- What and why -----------------------------------------------------
    short_description: str | None = Field(
        None,
        json_schema_extra=_f(
            "text",
            "Short description",
            required=True,
            guidance=(
                "One line naming the system and the action taken on it. "
                "A reader who knows nothing about this work should be able to tell "
                "what is changing. Not a restatement of the category."
            ),
        ),
    )
    description: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Description",
            required=True,
            guidance=(
                "What is changing and *why* it is changing. The why is the part "
                "reviewers need and the part authors leave out."
            ),
        ),
    )
    justification: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Justification",
            guidance="Why this must happen now, and what happens if it does not.",
        ),
    )
    risk_impact_analysis: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Risk and impact analysis",
            guidance=(
                "What could go wrong, who is affected while it does, and how "
                "likely it is. Naming no risk at all is itself a finding."
            ),
        ),
    )

    # -- How --------------------------------------------------------------
    implementation_plan: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Implementation plan",
            required=True,
            guidance=(
                "Numbered steps someone else could execute at 3am without calling "
                "you: named hosts, commands, and the expected result of each step."
            ),
        ),
    )
    backout_plan: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Backout plan",
            required=True,
            guidance=(
                "The concrete sequence that returns the system to its current "
                "state, plus the point of no return if there is one. "
                "'Restore the backup' is not a backout plan unless it says which "
                "backup and how."
            ),
        ),
    )
    test_plan: str | None = Field(
        None,
        json_schema_extra=_f(
            "longtext",
            "Test plan",
            guidance=(
                "How we will know it worked, stated as checks with expected "
                "outcomes -- not 'verify the application is up'."
            ),
        ),
    )

    # -- Classification ---------------------------------------------------
    type: str | None = Field(
        None,
        json_schema_extra=_f("choice", "Type", choice_source="change_request.type"),
    )
    category: str | None = Field(
        None,
        json_schema_extra=_f("choice", "Category", choice_source="change_request.category"),
    )
    risk: str | None = Field(
        None,
        json_schema_extra=_f(
            "choice",
            "Risk",
            required=True,
            choice_source="change_request.risk",
            guidance="How likely this is to go wrong and how bad it is when it does.",
        ),
    )
    impact: str | None = Field(
        None,
        json_schema_extra=_f("choice", "Impact", choice_source="change_request.impact"),
    )
    urgency: str | None = Field(
        None,
        json_schema_extra=_f("choice", "Urgency", choice_source="change_request.urgency"),
    )
    priority: str | None = Field(
        None,
        json_schema_extra=_f("choice", "Priority", choice_source="change_request.priority"),
    )

    # -- When -------------------------------------------------------------
    start_date: datetime | None = Field(
        None,
        json_schema_extra=_f(
            "datetime",
            "Planned start date",
            required=True,
            sn_field="start_date",
            guidance="When work begins, in UTC.",
        ),
    )
    end_date: datetime | None = Field(
        None,
        json_schema_extra=_f(
            "datetime",
            "Planned end date",
            required=True,
            sn_field="end_date",
            guidance="When the system is back to a steady state, including verification.",
        ),
    )

    # -- Who and what it touches -----------------------------------------
    cmdb_ci: Reference | None = Field(
        None,
        json_schema_extra=_f(
            "reference",
            "Configuration item",
            required=True,
            reference_table="cmdb_ci",
            guidance="The exact CI being changed. Must be picked from a lookup.",
        ),
    )
    business_service: Reference | None = Field(
        None,
        json_schema_extra=_f(
            "reference", "Business service", reference_table="cmdb_ci_service"
        ),
    )
    assignment_group: Reference | None = Field(
        None,
        json_schema_extra=_f(
            "reference",
            "Assignment group",
            required=True,
            reference_table="sys_user_group",
            guidance="The group that will carry out the work.",
        ),
    )
    assigned_to: Reference | None = Field(
        None,
        json_schema_extra=_f("reference", "Assigned to", reference_table="sys_user"),
    )
    requested_by: Reference | None = Field(
        None,
        json_schema_extra=_f("reference", "Requested by", reference_table="sys_user"),
    )

    # -- Flags ------------------------------------------------------------
    production_system: bool | None = Field(
        None,
        json_schema_extra=_f("boolean", "Affects a production system"),
    )
    outage_expected: bool | None = Field(
        None,
        json_schema_extra=_f(
            "boolean",
            "Outage expected",
            sn_field="u_outage_expected",
            guidance="True if any user-visible interruption is expected, however brief.",
        ),
    )
    cab_required: bool | None = Field(
        None,
        json_schema_extra=_f("boolean", "CAB approval required"),
    )

    # Escape hatch for the ~80 fields the curated core does not model.
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "start_date", "end_date", mode="before"
    )
    @classmethod
    def _parse_datetime(cls, v: Any) -> Any:
        """Accept ISO 8601 and ServiceNow's own format."""
        if isinstance(v, str):
            text = v.strip()
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                return datetime.strptime(text, SN_DATETIME_FORMAT)
        return v


class FieldSpec(BaseModel):
    """Flattened, LLM-facing view of one field's metadata."""

    name: str
    kind: FieldKind
    label: str
    required_for_submit: bool
    guidance: str
    reference_table: str | None
    choice_source: str | None
    sn_field: str

    @property
    def is_reference(self) -> bool:
        return self.kind == "reference"


def _spec_from_model_field(name: str, info: Any) -> FieldSpec:
    meta: dict[str, Any] = dict(info.json_schema_extra or {})
    return FieldSpec(
        name=name,
        kind=meta.get("kind", "text"),
        label=meta.get("label", name.replace("_", " ").capitalize()),
        required_for_submit=bool(meta.get("required_for_submit")),
        guidance=meta.get("guidance") or "",
        reference_table=meta.get("reference_table"),
        choice_source=meta.get("choice_source"),
        sn_field=meta.get("sn_field") or name,
    )


FIELD_SPECS: dict[str, FieldSpec] = {
    name: _spec_from_model_field(name, info)
    for name, info in ChangeRequest.model_fields.items()
    if name != "extra"
}

FIELD_NAMES: tuple[str, ...] = tuple(FIELD_SPECS)
REQUIRED_FIELDS: tuple[str, ...] = tuple(
    n for n, s in FIELD_SPECS.items() if s.required_for_submit
)
REFERENCE_FIELDS: tuple[str, ...] = tuple(
    n for n, s in FIELD_SPECS.items() if s.is_reference
)


def spec(name: str) -> FieldSpec:
    """Look up one field's spec, raising a helpful error for unknown names."""
    try:
        return FIELD_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown change field {name!r}. Known fields: {', '.join(FIELD_NAMES)}"
        ) from None
