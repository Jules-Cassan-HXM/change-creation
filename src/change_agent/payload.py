"""Turning the field map into what the Table API expects."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .schema import FIELD_SPECS, SN_DATETIME_FORMAT, Reference
from .state import FieldValue


def _serialise(value: Any) -> Any:
    if isinstance(value, Reference):
        return value.sys_id
    if isinstance(value, datetime):
        # ServiceNow stores date/times as UTC strings without an offset.
        return value.strftime(SN_DATETIME_FORMAT)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def build_payload(
    fields: dict[str, FieldValue], *, override_reason: str | None = None
) -> dict[str, Any]:
    """The change as ServiceNow wants it, using each field's instance name."""
    payload: dict[str, Any] = {"type": "normal"}
    for name, fv in fields.items():
        spec = FIELD_SPECS.get(name)
        if spec is None or not fv.is_set:
            continue
        payload[spec.sn_field] = _serialise(fv.value)

    if override_reason:
        # An override is part of the record, not a private note between the user
        # and the agent: whoever approves this needs to see it.
        payload["work_notes"] = (
            "Submitted with known blocking findings, at the requester's direction. "
            f"Reason given: {override_reason}"
        )
    return payload


def idempotency_key(payload: dict[str, Any], thread_id: str | None = None) -> str:
    """A key that survives a replay.

    It has to be derived, not generated: when a graph resumes after an
    interrupt the node runs again from the top, so a fresh uuid here would be a
    different key on the second pass and the guard would protect nothing. Same
    conversation plus same content therefore means same key, and a retry after a
    crash mid-POST resolves to the change that was already created.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(f"{thread_id or ''}|{canonical}".encode()).hexdigest()
    return digest[:32]
