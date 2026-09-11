"""The invariants that keep the agent honest: provenance and exact references."""

from __future__ import annotations

import pytest

from change_agent.state import Candidate, FieldValue, merge_fields
from change_agent.tools.fields import FieldUpdate, propose_change_fields
from change_agent.tools.references import search_reference, select_reference

from conftest import PROD_CI, field_map


def run(tool, args: dict, runtime):
    """Invoke a tool function directly and return its Command update."""
    return tool.func(**args, runtime=runtime).update


def message_of(update) -> str:
    return update["messages"][0].content


# -- provenance -----------------------------------------------------------

def test_a_stated_value_is_recorded_as_confirmed(runtime_factory):
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="Low", source="user")]},
        runtime_factory(),
    )
    assert update["fields"]["risk"].value == "low"
    assert update["fields"]["risk"].confirmed is True


def test_an_inferred_value_is_recorded_as_unconfirmed(runtime_factory):
    update = run(
        propose_change_fields,
        {
            "updates": [
                FieldUpdate(
                    field="risk", value="low", source="inferred",
                    rationale="they called it routine",
                )
            ]
        },
        runtime_factory(),
    )
    assert update["fields"]["risk"].confirmed is False
    assert update["fields"]["risk"].rationale == "they called it routine"


def test_a_confirmed_value_is_not_silently_overwritten(runtime_factory):
    state = {"fields": field_map(risk="low")}
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="high", source="inferred")]},
        runtime_factory(state),
    )
    assert "risk" not in update["fields"]
    assert "confirmation" in message_of(update).lower()


def test_an_unconfirmed_value_is_overwritten_freely(runtime_factory):
    state = {"fields": {"risk": FieldValue(value="low", source="inferred")}}
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="high", source="user")]},
        runtime_factory(state),
    )
    assert update["fields"]["risk"].value == "high"


def test_force_overwrites_a_confirmed_value(runtime_factory):
    state = {"fields": field_map(risk="low")}
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="high", source="user", force=True)]},
        runtime_factory(state),
    )
    assert update["fields"]["risk"].value == "high"


def test_repeating_a_confirmed_value_is_not_a_conflict(runtime_factory):
    state = {"fields": field_map(risk="low")}
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="low", source="user")]},
        runtime_factory(state),
    )
    assert update["fields"]["risk"].value == "low"


# -- validation -----------------------------------------------------------

def test_a_reference_cannot_be_set_from_text(runtime_factory):
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="cmdb_ci", value="SRV-APP-01", source="user")]},
        runtime_factory(),
    )
    assert update["fields"] == {}
    assert "search_reference" in message_of(update)


def test_an_invalid_choice_is_rejected_with_the_allowed_values(runtime_factory):
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="enormous", source="user")]},
        runtime_factory(),
    )
    assert update["fields"] == {}
    assert "very_high" in message_of(update)


def test_a_choice_label_is_accepted_and_stored_as_its_value(runtime_factory):
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="risk", value="Very High", source="user")]},
        runtime_factory(),
    )
    assert update["fields"]["risk"].value == "very_high"


def test_one_bad_update_does_not_discard_the_good_ones(runtime_factory):
    update = run(
        propose_change_fields,
        {
            "updates": [
                FieldUpdate(field="risk", value="low", source="user"),
                FieldUpdate(field="not_a_field", value="x", source="user"),
            ]
        },
        runtime_factory(),
    )
    assert "risk" in update["fields"]
    assert "not_a_field" in message_of(update)


def test_findings_are_returned_with_every_update(runtime_factory):
    update = run(
        propose_change_fields,
        {"updates": [FieldUpdate(field="short_description", value="Patch it", source="user")]},
        runtime_factory(),
    )
    assert any(f.code == "R001" for f in update["findings"])


# -- references -----------------------------------------------------------

def test_search_offers_the_production_and_non_production_twins(runtime_factory):
    update = run(search_reference, {"field": "cmdb_ci", "query": "SRV-APP-01"}, runtime_factory())
    offered = {c.display_value for c in update["pending_lookup"]["cmdb_ci"]}
    assert {"SRV-APP-01", "SRV-APP-01-DEV"} <= offered
    assert "fields" not in update  # nothing chosen on the user's behalf


def test_search_does_not_auto_pick_when_twins_exist(runtime_factory):
    update = run(search_reference, {"field": "cmdb_ci", "query": "PAY-API-PROD"}, runtime_factory())
    assert "fields" not in update
    assert len(update["pending_lookup"]["cmdb_ci"]) > 1


def test_search_reports_an_empty_result_without_guessing(runtime_factory):
    update = run(search_reference, {"field": "cmdb_ci", "query": "zzzz-nothing"}, runtime_factory())
    assert update["pending_lookup"]["cmdb_ci"] == []
    assert "fields" not in update


def test_search_rejects_a_non_reference_field(runtime_factory):
    update = run(search_reference, {"field": "risk", "query": "low"}, runtime_factory())
    assert "not a reference field" in message_of(update)


def test_select_writes_the_sys_id_from_the_offered_candidate(runtime_factory):
    offered = [
        Candidate(
            sys_id=PROD_CI.sys_id, display_value="SRV-APP-01", table="cmdb_ci",
            context={"environment": "production"},
        )
    ]
    update = run(
        select_reference,
        {"field": "cmdb_ci", "choice": PROD_CI.sys_id},
        runtime_factory({"pending_lookup": {"cmdb_ci": offered}}),
    )
    reference = update["fields"]["cmdb_ci"].value
    assert reference.sys_id == PROD_CI.sys_id
    assert reference.context["environment"] == "production"
    assert update["fields"]["cmdb_ci"].confirmed is True
    assert update["pending_lookup"]["cmdb_ci"] == []


def test_select_accepts_the_display_value_of_an_offered_candidate(runtime_factory):
    offered = [Candidate(sys_id=PROD_CI.sys_id, display_value="SRV-APP-01", table="cmdb_ci")]
    update = run(
        select_reference,
        {"field": "cmdb_ci", "choice": "srv-app-01"},
        runtime_factory({"pending_lookup": {"cmdb_ci": offered}}),
    )
    assert update["fields"]["cmdb_ci"].value.sys_id == PROD_CI.sys_id


def test_an_invented_sys_id_is_refused(runtime_factory):
    offered = [Candidate(sys_id=PROD_CI.sys_id, display_value="SRV-APP-01", table="cmdb_ci")]
    update = run(
        select_reference,
        {"field": "cmdb_ci", "choice": "ff" * 16},
        runtime_factory({"pending_lookup": {"cmdb_ci": offered}}),
    )
    assert "fields" not in update
    assert "not one of the candidates" in message_of(update)


def test_a_reference_cannot_be_selected_before_a_search(runtime_factory):
    update = run(
        select_reference,
        {"field": "cmdb_ci", "choice": PROD_CI.sys_id},
        runtime_factory({}),
    )
    assert "fields" not in update
    assert "search_reference" in message_of(update)


# -- the reducer ----------------------------------------------------------

def test_concurrent_updates_merge_without_loss():
    left = {"risk": FieldValue(value="low")}
    right = {"short_description": FieldValue(value="Patch it")}
    assert set(merge_fields(left, right)) == {"risk", "short_description"}


def test_the_later_write_wins_for_the_same_field():
    left = {"risk": FieldValue(value="low")}
    right = {"risk": FieldValue(value="high")}
    assert merge_fields(left, right)["risk"].value == "high"
