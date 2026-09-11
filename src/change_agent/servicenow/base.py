"""The ServiceNow surface the agent depends on.

Three operations is all the copilot needs. Keeping the Protocol this small is
what lets the whole application run offline against fixtures.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class Record(BaseModel):
    """A row returned by a reference lookup."""

    sys_id: str
    display_value: str
    table: str
    context: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class Choice(BaseModel):
    value: str
    label: str


class CreatedChange(BaseModel):
    number: str
    sys_id: str


class ServiceNowClient(Protocol):
    def search_reference(
        self, table: str, query: str, limit: int = 5
    ) -> list[Record]:
        """Return the closest matches for ``query`` in ``table``, best first."""
        ...

    def get_choices(self, choice_source: str) -> list[Choice]:
        """Return the allowed values for a ``table.field`` choice list."""
        ...

    def create_change(
        self, payload: dict[str, Any], idempotency_key: str
    ) -> CreatedChange:
        """Create the change. Must be safe to call twice with the same key."""
        ...


class ServiceNowError(RuntimeError):
    """Raised when the instance rejects a call."""
