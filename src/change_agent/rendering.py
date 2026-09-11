"""Turning state into text for the model, the tools and the submit preview.

One module so that what the model is told about the change, what a tool reports
back, and what the user approves before submission cannot drift apart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .rules.base import Finding
from .schema import FIELD_SPECS, REQUIRED_FIELDS, SN_DATETIME_FORMAT, Reference
from .state import Candidate, FieldValue

#: Marks how much weight a value carries: confirmed by the user, or merely read
#: out of what they wrote.
_SOURCE_MARK = {True: "confirmed", False: "unconfirmed"}


def format_value(value: object) -> str:
    if isinstance(value, Reference):
        return f"{value.display_value} [{value.sys_id}]"
    if isinstance(value, datetime):
        return value.strftime(SN_DATETIME_FORMAT)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_change(fields: dict[str, FieldValue], *, verbose: bool = False) -> str:
    """The current change, in field order, with provenance."""
    set_fields = [(name, fields[name]) for name in FIELD_SPECS if name in fields and fields[name].is_set]
    if not set_fields:
        return "The change is empty; nothing has been captured yet."

    lines = []
    for name, fv in set_fields:
        spec = FIELD_SPECS[name]
        value = format_value(fv.value)
        if not verbose and len(value) > 160:
            value = value[:157] + "..."
        lines.append(f"- {spec.label} ({name}): {value}  [{_SOURCE_MARK[fv.confirmed]}]")

    empty_required = [
        FIELD_SPECS[n].label
        for n in REQUIRED_FIELDS
        if n not in fields or not fields[n].is_set
    ]
    if empty_required:
        lines.append("")
        lines.append("Still required and empty: " + ", ".join(empty_required))
    return "\n".join(lines)


def render_findings(findings: Iterable[Finding]) -> str:
    findings = list(findings)
    if not findings:
        return "No open findings."
    lines = []
    for f in findings:
        detail = ", ".join(f"{k}={v}" for k, v in f.data.items() if v not in (None, ""))
        where = f" ({f.field})" if f.field else ""
        lines.append(
            f"- [{f.severity.upper()}] {f.code}{where}: {f.message}"
            + (f" | {detail}" if detail else "")
        )
    return "\n".join(lines)


def render_candidates(field: str, candidates: list[Candidate]) -> str:
    if not candidates:
        return f"No match found for {field}."
    lines = [f"Candidates for {FIELD_SPECS[field].label} ({field}):"]
    lines += [
        f"  {i}. {c.summary()}  sys_id={c.sys_id}  match={c.score:.0f}"
        for i, c in enumerate(candidates, start=1)
    ]
    return "\n".join(lines)
