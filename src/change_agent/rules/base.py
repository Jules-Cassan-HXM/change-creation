"""Primitives for the challenge rules.

A *finding* is the machine-readable form of a challenge. It carries a stable
code and structured data rather than a user-facing sentence, so the agent can
relay it in whatever language the conversation is happening in.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["blocker", "warning"]


class Finding(BaseModel):
    """One thing the agent should raise with the user."""

    code: str
    severity: Severity
    field: str | None = None
    # English, and deliberately terse: it is a fallback and a log line, not the
    # sentence the user reads. The agent rephrases from `code` + `data`.
    message: str
    data: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        where = f" [{self.field}]" if self.field else ""
        return f"{self.code} {self.severity.upper()}{where}: {self.message}"


class Policy(BaseModel):
    """Instance policy the rules are checked against."""

    model_config = ConfigDict(frozen=True)

    # Minimum notice a normal change needs before CAB can review it.
    cab_lead_days: int = 5
    # A backout plan shorter than this is not a plan.
    min_backout_plan_chars: int = 80
    min_implementation_plan_chars: int = 80


class RuleContext(BaseModel):
    """Everything a rule is allowed to look at.

    Deliberately narrow: rules are pure functions of the change, the clock and
    policy. Anything a rule needs about a referenced record (a CI's environment,
    a group's name) travels inside ``Reference.context``, captured at lookup
    time, so no rule ever performs I/O.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    change: Any  # ChangeRequest; typed as Any to keep this module dependency-free
    now: datetime
    policy: Policy = Field(default_factory=Policy)


@runtime_checkable
class Rule(Protocol):
    """A check over a change. Returns nothing when it has no complaint."""

    code: str
    severity: Severity

    def __call__(self, ctx: RuleContext) -> Finding | list[Finding] | None: ...
