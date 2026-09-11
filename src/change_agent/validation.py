"""Coercing and checking a single proposed field value.

Kept apart from the tools so the rules that govern what may enter the change are
testable without going anywhere near an agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ValidationError

from .schema import FIELD_SPECS, ChangeRequest, FieldSpec
from .servicenow.base import ServiceNowClient, ServiceNowError


class FieldRejected(ValueError):
    """A proposed value cannot be accepted, with a reason for the model."""

    def __init__(self, field: str, reason: str, **data: Any) -> None:
        super().__init__(reason)
        self.field = field
        self.reason = reason
        self.data = data


def _match_choice(spec: FieldSpec, value: Any, client: ServiceNowClient) -> str:
    """Accept either the stored value or the label a human would recognise."""
    try:
        choices = client.get_choices(spec.choice_source or "")
    except ServiceNowError:
        # An unknown choice list should not block data entry; the instance will
        # have the final say on submission.
        return str(value)

    text = str(value).strip().lower()
    for choice in choices:
        if text in (choice.value.lower(), choice.label.lower()):
            return choice.value
    raise FieldRejected(
        spec.name,
        f"{value!r} is not an allowed value for {spec.label}.",
        allowed=[{"value": c.value, "label": c.label} for c in choices],
    )


def _coerce_bool(spec: FieldSpec, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "oui"}:
        return True
    if text in {"false", "no", "n", "0", "non"}:
        return False
    raise FieldRejected(spec.name, f"{value!r} is not a yes/no value for {spec.label}.")


def coerce_value(field: str, value: Any, client: ServiceNowClient) -> Any:
    """Validate and normalise one proposed value.

    Raises :class:`FieldRejected` with something the model can act on, rather
    than letting a bad value into state and discovering it at submission time.
    """
    if field not in FIELD_SPECS:
        raise FieldRejected(
            field,
            f"{field!r} is not a field on the change.",
            known_fields=list(FIELD_SPECS),
        )
    spec = FIELD_SPECS[field]

    if spec.is_reference:
        # The only path to a reference is a lookup the user confirmed. Accepting
        # a name here is how an invented sys_id would get in.
        raise FieldRejected(
            field,
            f"{spec.label} points at an exact record and cannot be set from text. "
            f"Use search_reference('{field}', ...) and have the user choose.",
            reference_table=spec.reference_table,
        )

    if value in (None, ""):
        raise FieldRejected(field, f"No value given for {spec.label}.")

    if spec.kind == "choice":
        value = _match_choice(spec, value, client)
    elif spec.kind == "boolean":
        value = _coerce_bool(spec, value)
    elif spec.kind == "integer":
        try:
            value = int(str(value).strip())
        except ValueError as exc:
            raise FieldRejected(field, f"{value!r} is not a whole number.") from exc

    # Let pydantic have the last word, which also parses date/times.
    try:
        validated = ChangeRequest.model_validate({field: value})
    except ValidationError as exc:
        detail = exc.errors()[0].get("msg", "invalid value")
        raise FieldRejected(field, f"{spec.label}: {detail}") from exc

    coerced = getattr(validated, field)
    if isinstance(coerced, datetime):
        return coerced
    return coerced
