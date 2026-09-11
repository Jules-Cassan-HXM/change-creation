"""Each rule gets a case that fires it and a case that must not."""

from __future__ import annotations

from datetime import timedelta

import pytest

from change_agent.rules.catalog import blockers, run_rules

from conftest import (
    CRITICAL_SERVICE,
    DBA_GROUP,
    DEV_CI,
    NOW,
    make_change,
)


def codes(ctx) -> set[str]:
    return {f.code for f in run_rules(ctx)}


def test_a_complete_change_raises_nothing(ctx_factory):
    assert run_rules(ctx_factory(make_change())) == []


@pytest.mark.parametrize("field", ["short_description", "cmdb_ci", "risk", "start_date"])
def test_r001_flags_each_missing_required_field(ctx_factory, field):
    findings = run_rules(ctx_factory(make_change(**{field: None})))
    assert any(f.code == "R001" and f.field == field for f in findings)


def test_r002_end_must_be_after_start(ctx_factory):
    change = make_change(end_date=NOW + timedelta(days=9))
    assert "R002" in codes(ctx_factory(change))


def test_r002_silent_when_window_is_sane(ctx_factory):
    assert "R002" not in codes(ctx_factory(make_change()))


def test_r003_rejects_a_one_liner_backout_plan(ctx_factory):
    change = make_change(backout_plan="Restore the backup.")
    findings = run_rules(ctx_factory(change))
    assert "R003" in {f.code for f in findings}
    assert any(f.code == "R003" for f in blockers(findings))


def test_r004_rejects_a_prose_implementation_plan(ctx_factory):
    change = make_change(implementation_plan="Update the package on the server.")
    assert "R004" in codes(ctx_factory(change))


def test_r005_flags_a_start_in_the_past(ctx_factory):
    change = make_change(
        start_date=NOW - timedelta(days=1), end_date=NOW - timedelta(hours=22)
    )
    assert "R005" in codes(ctx_factory(change))


def test_r006_challenges_low_risk_on_a_production_ci(ctx_factory):
    assert "R006" in codes(ctx_factory(make_change(risk="low")))


def test_r006_accepts_low_risk_off_production(ctx_factory):
    change = make_change(risk="low", cmdb_ci=DEV_CI)
    assert "R006" not in codes(ctx_factory(change))


def test_r006_also_triggers_on_a_critical_business_service(ctx_factory):
    change = make_change(risk="low", cmdb_ci=DEV_CI, business_service=CRITICAL_SERVICE)
    assert "R006" in codes(ctx_factory(change))


def test_r007_flags_no_impact_claim_against_a_flagged_outage(ctx_factory):
    change = make_change(
        description="Routine patch with no impact for users, applied in place.",
        outage_expected=True,
    )
    assert "R007" in codes(ctx_factory(change))


def test_r007_flags_no_impact_claim_against_a_restart_step(ctx_factory):
    change = make_change(
        description="Routine patch, no downtime expected, applied during the evening.",
        outage_expected=False,
    )
    assert "R007" in codes(ctx_factory(change))


def test_r007_works_in_french(ctx_factory):
    change = make_change(
        description="Mise a jour de securite sans interruption pour les utilisateurs.",
        outage_expected=True,
    )
    assert "R007" in codes(ctx_factory(change))


def test_r007_silent_when_the_outage_is_acknowledged(ctx_factory):
    assert "R007" not in codes(ctx_factory(make_change(outage_expected=True)))


def test_r008_wants_a_test_plan_for_a_risky_change(ctx_factory):
    change = make_change(risk="high", test_plan=None)
    assert "R008" in codes(ctx_factory(change))


def test_r008_silent_for_a_low_risk_change(ctx_factory):
    change = make_change(risk="low", cmdb_ci=DEV_CI, test_plan=None)
    assert "R008" not in codes(ctx_factory(change))


def test_r009_flags_a_description_that_repeats_the_one_liner(ctx_factory):
    change = make_change(
        short_description="Patch app-server to 6.4.2 on SRV-APP-01",
        description="Patch app-server to 6.4.2 on SRV-APP-01.",
    )
    assert "R009" in codes(ctx_factory(change))


def test_r010_spots_emergency_wording_in_a_normal_change(ctx_factory):
    change = make_change(
        justification="We need this deployed asap, the audit is next week."
    )
    assert "R010" in codes(ctx_factory(change))


def test_r010_silent_for_an_emergency_change(ctx_factory):
    change = make_change(
        type="emergency",
        justification="We need this deployed asap, the audit is next week.",
    )
    assert "R010" not in codes(ctx_factory(change))


def test_r011_flags_a_group_that_does_not_support_the_ci(ctx_factory):
    findings = run_rules(ctx_factory(make_change(assignment_group=DBA_GROUP)))
    finding = next(f for f in findings if f.code == "R011")
    assert finding.data["ci_support_group"] == "Unix Platform Team"


def test_r011_silent_when_the_group_matches(ctx_factory):
    assert "R011" not in codes(ctx_factory(make_change()))


def test_r012_flags_a_window_shorter_than_the_plan(ctx_factory):
    change = make_change(
        implementation_plan=(
            "1. Drain the node from the pool, allow 30 minutes for sessions to end.\n"
            "2. Apply the package, 2 hours.\n"
            "3. Verify and return to the pool, 45 minutes."
        ),
        end_date=NOW + timedelta(days=10, hours=1),
    )
    finding = next(f for f in run_rules(ctx_factory(change)) if f.code == "R012")
    assert finding.data["plan_minutes"] == 195
    assert finding.data["window_minutes"] == 60


def test_r012_silent_when_the_window_is_long_enough(ctx_factory):
    change = make_change(
        implementation_plan=(
            "1. Drain the node, 10 minutes.\n2. Patch, 20 minutes.\n3. Verify, 10 minutes."
        )
    )
    assert "R012" not in codes(ctx_factory(change))


def test_r013_flags_a_start_inside_the_cab_lead_time(ctx_factory):
    change = make_change(
        start_date=NOW + timedelta(days=2), end_date=NOW + timedelta(days=2, hours=2)
    )
    assert "R013" in codes(ctx_factory(change))


def test_r013_respects_a_configured_lead_time(ctx_factory):
    change = make_change(
        start_date=NOW + timedelta(days=2), end_date=NOW + timedelta(days=2, hours=2)
    )
    assert "R013" not in codes(ctx_factory(change, cab_lead_days=1))


def test_findings_are_ordered_blockers_first(ctx_factory):
    change = make_change(risk="low", backout_plan="nope", cmdb_ci=None)
    severities = [f.severity for f in run_rules(ctx_factory(change))]
    assert severities == sorted(severities, key=lambda s: s != "blocker")
