"""State has to survive the checkpointer.

Values are stored untyped, so anything that went in as a pydantic model comes
back from a checkpoint as a plain dict. Left unhandled, a configuration item
reaches the Table API as a nested object instead of a sys_id.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from change_agent.payload import build_payload
from change_agent.rules.base import Finding
from change_agent.schema import Reference
from change_agent.state import (
    Candidate,
    FieldValue,
    Submission,
    get_fields,
    get_findings,
    get_pending_lookup,
    get_submission,
    to_change,
)

from conftest import PROD_CI


def round_trip(value):
    serde = JsonPlusSerializer()
    return serde.loads_typed(serde.dumps_typed(value))


def test_a_reference_survives_a_checkpoint_round_trip():
    restored = round_trip({"fields": {"cmdb_ci": FieldValue(value=PROD_CI)}})
    fields = get_fields(restored)
    assert isinstance(fields["cmdb_ci"].value, Reference)
    assert fields["cmdb_ci"].value.sys_id == PROD_CI.sys_id


def test_a_restored_reference_still_serialises_to_a_sys_id():
    restored = get_fields(round_trip({"fields": {"cmdb_ci": FieldValue(value=PROD_CI)}}))
    assert build_payload(restored)["cmdb_ci"] == PROD_CI.sys_id


def test_a_restored_reference_is_still_readable_by_the_rules():
    restored = get_fields(round_trip({"fields": {"cmdb_ci": FieldValue(value=PROD_CI)}}))
    assert to_change(restored).cmdb_ci.context["environment"] == "production"


def test_a_datetime_survives_a_checkpoint_round_trip():
    when = datetime(2026, 10, 1, 22, 0, tzinfo=UTC)
    restored = get_fields(round_trip({"fields": {"start_date": FieldValue(value=when)}}))
    assert restored["start_date"].value == when


def test_findings_survive_as_findings():
    finding = Finding(code="R001", severity="blocker", field="risk", message="missing")
    restored = get_findings(round_trip({"findings": [finding]}))
    assert restored[0].severity == "blocker"


def test_candidates_survive_as_candidates():
    candidate = Candidate(sys_id="a" * 32, display_value="SRV-APP-01", table="cmdb_ci")
    restored = get_pending_lookup(round_trip({"pending_lookup": {"cmdb_ci": [candidate]}}))
    assert restored["cmdb_ci"][0].display_value == "SRV-APP-01"


def test_a_submission_survives_as_a_submission():
    restored = get_submission(
        round_trip({"submission": Submission(idempotency_key="k", number="CHG0030001")})
    )
    assert restored.number == "CHG0030001"


def test_the_readers_tolerate_an_empty_first_turn():
    assert get_fields({}) == {}
    assert get_findings(None) == []
    assert get_pending_lookup({}) == {}
    assert get_submission({}) is None
