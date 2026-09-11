from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from change_agent.rules.base import Policy, RuleContext
from change_agent.schema import ChangeRequest, Reference

NOW = datetime(2026, 9, 11, 9, 0, tzinfo=UTC)

PROD_CI = Reference(
    sys_id="ci0000000000000000000000000000a1",
    display_value="SRV-APP-01",
    table="cmdb_ci",
    context={"environment": "production", "support_group": "Unix Platform Team"},
)
DEV_CI = Reference(
    sys_id="ci0000000000000000000000000000a2",
    display_value="SRV-APP-01-DEV",
    table="cmdb_ci",
    context={"environment": "development", "support_group": "Unix Platform Team"},
)
UNIX_GROUP = Reference(
    sys_id="gr00000000000000000000000000001a",
    display_value="Unix Platform Team",
    table="sys_user_group",
)
DBA_GROUP = Reference(
    sys_id="gr00000000000000000000000000003a",
    display_value="Database Administration",
    table="sys_user_group",
)
CRITICAL_SERVICE = Reference(
    sys_id="sv00000000000000000000000000001a",
    display_value="Online Banking",
    table="cmdb_ci_service",
    context={"criticality": "1 - most critical"},
)

GOOD_IMPLEMENTATION_PLAN = """
1. Drain SRV-APP-01 from the F5 pool and confirm zero active sessions.
2. Apply the 6.4.2 package with `dnf update app-server`.
3. Restart the service and confirm it reports healthy on :8080/health.
4. Return the node to the pool and watch error rates for 10 minutes.
""".strip()

GOOD_BACKOUT_PLAN = """
1. Drain the node from the F5 pool again.
2. Roll back with `dnf downgrade app-server-6.4.1`.
3. Restart and confirm :8080/health reports 6.4.1, then return to the pool.
Point of no return: once the schema migration in step 2 has run, roll forward instead.
""".strip()


def make_change(**overrides: Any) -> ChangeRequest:
    """A change that passes every rule, so each test can break exactly one thing."""
    base: dict[str, Any] = {
        "short_description": "Patch app-server to 6.4.2 on SRV-APP-01",
        "description": (
            "SRV-APP-01 runs app-server 6.4.1, which is affected by CVE-2026-1234. "
            "We are patching to 6.4.2 to close it before the quarterly audit."
        ),
        "implementation_plan": GOOD_IMPLEMENTATION_PLAN,
        "backout_plan": GOOD_BACKOUT_PLAN,
        "test_plan": "Confirm :8080/health returns 6.4.2 and error rate stays under 0.1%.",
        "risk": "moderate",
        "type": "normal",
        "start_date": NOW + timedelta(days=10),
        "end_date": NOW + timedelta(days=10, hours=2),
        "cmdb_ci": PROD_CI,
        "assignment_group": UNIX_GROUP,
    }
    base.update(overrides)
    return ChangeRequest(**base)


@pytest.fixture
def ctx_factory():
    def build(change: ChangeRequest, *, now: datetime = NOW, **policy: Any) -> RuleContext:
        return RuleContext(change=change, now=now, policy=Policy(**policy))

    return build


@pytest.fixture
def runtime_factory():
    """Build a ToolRuntime so tools can be exercised without running the graph."""
    from langchain.tools import ToolRuntime

    def build(state: dict[str, Any] | None = None, *, thread_id: str = "test-thread"):
        return ToolRuntime(
            state=state or {},
            context=None,
            config={"configurable": {"thread_id": thread_id}},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

    return build


def field_map(**values: Any) -> dict[str, Any]:
    """Field map built straight from values, all treated as user-confirmed."""
    from change_agent.state import FieldValue

    return {
        name: FieldValue(value=value, source="user", confirmed=True)
        for name, value in values.items()
    }


def seeded_fields() -> dict[str, Any]:
    """A change with nothing blocking left, as a conversation would have built it."""
    from change_agent.state import FieldValue

    change = make_change()
    fields = {
        name: FieldValue(value=getattr(change, name), source="user", confirmed=True)
        for name in (
            "short_description", "description", "implementation_plan", "backout_plan",
            "test_plan", "risk", "type", "start_date", "end_date",
        )
    }
    fields["cmdb_ci"] = FieldValue(value=PROD_CI, source="lookup", confirmed=True)
    fields["assignment_group"] = FieldValue(value=UNIX_GROUP, source="lookup", confirmed=True)
    return fields
