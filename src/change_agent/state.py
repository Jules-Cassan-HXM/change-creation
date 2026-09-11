"""Agent state.

The change is held as a map of field name -> :class:`FieldValue` rather than as
a bare ``ChangeRequest`` instance, for two reasons:

1. Several tool calls can land in the same model turn. A dict with a merge
   reducer combines them; a whole-object write would have them clobber each
   other.
2. We need to know *where* each value came from. Without provenance the agent
   cannot tell a value the user explicitly confirmed from one it inferred out of
   prose three turns ago -- and it will happily overwrite the former.

The typed :class:`~change_agent.schema.ChangeRequest` is a pure validator,
materialised from this map on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from langchain.agents import AgentState
from pydantic import BaseModel, Field, field_validator

from .rules.base import Finding
from .schema import ChangeRequest, Reference

#: Where a value came from. ``user`` and ``lookup`` are first-hand; ``inferred``
#: means the agent read it out of prose and the user has not confirmed it.
ValueSource = Literal["user", "inferred", "lookup", "default"]


class FieldValue(BaseModel):
    """One field's value plus how we came to believe it."""

    value: Any
    source: ValueSource = "inferred"
    confirmed: bool = False
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rationale: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _restore_reference(cls, value: Any) -> Any:
        """Rebuild a :class:`Reference` that a checkpoint round trip flattened.

        ``value`` is deliberately untyped, so the checkpointer hands it back as
        a plain dict. Left alone it would reach the Table API as a nested object
        instead of a sys_id -- which is why this is fixed here, at the one place
        every value passes through, rather than at each point of use.
        """
        if isinstance(value, dict) and {"sys_id", "display_value", "table"} <= value.keys():
            return Reference.model_validate(value)
        return value

    @property
    def is_set(self) -> bool:
        return self.value not in (None, "", [], {})


class Candidate(BaseModel):
    """One option returned by a reference lookup."""

    sys_id: str
    display_value: str
    table: str
    context: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

    def summary(self) -> str:
        """Human-readable line, including whatever disambiguates it."""
        bits = [f"{k}={v}" for k, v in self.context.items() if v]
        return self.display_value + (f" ({', '.join(bits)})" if bits else "")


class Submission(BaseModel):
    """The outcome of a submit attempt."""

    idempotency_key: str
    number: str | None = None
    sys_id: str | None = None
    submitted_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def merge_fields(
    left: dict[str, FieldValue] | None, right: dict[str, FieldValue] | None
) -> dict[str, FieldValue]:
    """Reducer: later writes win per key, untouched keys survive.

    Merging per key (rather than replacing the dict) is what makes two tool
    calls in the same turn safe.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def replace_lookups(
    left: dict[str, list[Candidate]] | None,
    right: dict[str, list[Candidate]] | None,
) -> dict[str, list[Candidate]]:
    """Reducer for pending lookups: per field, the newest search wins."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ChangeAgentState(AgentState):
    """``AgentState`` brings ``messages``; the rest is ours."""

    fields: Annotated[dict[str, FieldValue], merge_fields]
    # Candidates last offered for a reference field. A sys_id is only accepted
    # by `select_reference` if it appears here -- this is what makes fabricating
    # a reference structurally impossible rather than merely discouraged.
    pending_lookup: Annotated[dict[str, list[Candidate]], replace_lookups]
    findings: list[Finding]
    override_reason: str | None
    submission: Submission | None


def _read(state: Any, key: str) -> Any:
    if state is None:
        return None
    if hasattr(state, "get"):
        return state.get(key)
    return getattr(state, key, None)


def _revive(model: type[BaseModel], value: Any) -> Any:
    """Restore a model that came back from a checkpoint as a plain dict."""
    return value if isinstance(value, model) else model.model_validate(value)


def get_fields(state: Any) -> dict[str, FieldValue]:
    """Read the field map out of state, tolerating a first turn with no keys."""
    return {
        name: _revive(FieldValue, value)
        for name, value in (_read(state, "fields") or {}).items()
    }


def get_findings(state: Any) -> list[Finding]:
    return [_revive(Finding, f) for f in (_read(state, "findings") or [])]


def get_pending_lookup(state: Any) -> dict[str, list[Candidate]]:
    return {
        field: [_revive(Candidate, c) for c in candidates]
        for field, candidates in (_read(state, "pending_lookup") or {}).items()
    }


def get_submission(state: Any) -> Submission | None:
    raw = _read(state, "submission")
    return _revive(Submission, raw) if raw else None


def to_change(fields: dict[str, FieldValue]) -> ChangeRequest:
    """Materialise the validated change from the field map.

    Values were validated on the way in, so this should not fail; if it does the
    error is a genuine bug and should surface rather than be swallowed.
    """
    payload = {name: fv.value for name, fv in fields.items() if fv.is_set}
    return ChangeRequest.model_validate(payload)
