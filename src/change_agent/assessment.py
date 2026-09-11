"""Running the challenge rules over the current state."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from .rules.base import Finding, Policy, RuleContext
from .rules.catalog import run_rules
from .state import FieldValue, to_change


def policy_from_env() -> Policy:
    return Policy(cab_lead_days=int(os.environ.get("CAB_LEAD_DAYS", "5")))


def evaluate(fields: dict[str, FieldValue], *, now: datetime | None = None) -> list[Finding]:
    """Every deterministic finding open against the change as it stands."""
    return run_rules(
        RuleContext(
            change=to_change(fields),
            now=now or datetime.now(UTC),
            policy=policy_from_env(),
        )
    )
