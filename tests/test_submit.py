"""Submission: the blocker gate, the approval pause, and no double-creation."""

from __future__ import annotations

import json

import pytest

from change_agent.payload import build_payload, idempotency_key
from change_agent.schema import Reference
from change_agent.servicenow.fake import FakeServiceNowClient
from change_agent.state import Submission
from change_agent.tools import submit as submit_module
from change_agent.tools.submit import record_override_reason, submit_change

from conftest import PROD_CI, UNIX_GROUP, make_change

pytestmark = pytest.mark.usefixtures("runtime_factory")


def complete_field_map() -> dict:
    """A change with no blockers left open."""
    from change_agent.state import FieldValue

    change = make_change()
    values = {
        name: getattr(change, name)
        for name in (
            "short_description", "description", "implementation_plan", "backout_plan",
            "test_plan", "risk", "type", "start_date", "end_date",
        )
    }
    values["cmdb_ci"] = PROD_CI
    values["assignment_group"] = UNIX_GROUP
    return {n: FieldValue(value=v, source="user", confirmed=True) for n, v in values.items()}


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeServiceNowClient(submitted_dir=tmp_path / "submitted")
    monkeypatch.setattr(submit_module, "get_client", lambda: fake)
    return fake


@pytest.fixture
def no_critic(monkeypatch):
    """The review pass needs a model; the submit path itself does not."""
    monkeypatch.setattr(submit_module, "critique_fields", lambda *a, **k: [])


def run(tool, runtime, **args):
    return tool.func(**args, runtime=runtime).update


def message_of(update) -> str:
    return update["messages"][0].content


# -- payload --------------------------------------------------------------

def test_payload_uses_instance_field_names_and_sys_ids():
    from change_agent.state import FieldValue

    payload = build_payload(
        {
            "cmdb_ci": FieldValue(value=PROD_CI),
            "outage_expected": FieldValue(value=True),
        }
    )
    assert payload["cmdb_ci"] == PROD_CI.sys_id
    assert payload["u_outage_expected"] == "true"


def test_payload_records_an_override_in_the_work_notes():
    payload = build_payload({}, override_reason="audit deadline is Friday")
    assert "audit deadline is Friday" in payload["work_notes"]


def test_the_idempotency_key_survives_a_replay():
    payload = build_payload(complete_field_map())
    assert idempotency_key(payload, "t1") == idempotency_key(payload, "t1")


def test_different_conversations_get_different_keys():
    payload = build_payload(complete_field_map())
    assert idempotency_key(payload, "t1") != idempotency_key(payload, "t2")


# -- the blocker gate -----------------------------------------------------

def test_submission_is_refused_while_a_blocker_is_open(runtime_factory, client, no_critic):
    fields = complete_field_map()
    del fields["backout_plan"]
    update = run(submit_change, runtime_factory({"fields": fields}))
    assert "Not submitted" in message_of(update)
    assert not list((client.submitted_dir).glob("*.json")) if client.submitted_dir.exists() else True


def test_an_override_reason_is_required_to_be_non_empty(runtime_factory):
    update = run(record_override_reason, runtime_factory(), reason="   ")
    assert "override_reason" not in update


def test_an_override_lets_submission_reach_approval(
    runtime_factory, client, no_critic, monkeypatch
):
    monkeypatch.setattr(submit_module, "interrupt", lambda payload: {"action": "approve"})
    fields = complete_field_map()
    del fields["backout_plan"]
    update = run(
        submit_change,
        runtime_factory({"fields": fields, "override_reason": "audit deadline"}),
    )
    assert update["submission"].number.startswith("CHG")
    written = json.loads(next(client.submitted_dir.glob("*.json")).read_text())
    assert "audit deadline" in written["payload"]["work_notes"]


# -- the approval pause ---------------------------------------------------

def test_the_user_sees_the_exact_payload_before_approving(
    runtime_factory, client, no_critic, monkeypatch
):
    seen = {}

    def capture(payload):
        seen.update(payload)
        return {"action": "approve"}

    monkeypatch.setattr(submit_module, "interrupt", capture)
    update = run(submit_change, runtime_factory({"fields": complete_field_map()}))

    assert seen["type"] == "approve_change_submission"
    assert seen["payload"]["cmdb_ci"] == PROD_CI.sys_id
    assert "SRV-APP-01" in seen["preview"]
    written = json.loads(next(client.submitted_dir.glob("*.json")).read_text())
    assert written["payload"] == seen["payload"]


@pytest.mark.parametrize("decision", [{"action": "cancel"}, {"action": "edit", "notes": "dates"}])
def test_nothing_is_created_without_approval(
    runtime_factory, client, no_critic, monkeypatch, decision
):
    monkeypatch.setattr(submit_module, "interrupt", lambda payload: decision)
    update = run(submit_change, runtime_factory({"fields": complete_field_map()}))
    assert "submission" not in update
    assert not client.submitted_dir.exists() or not list(client.submitted_dir.glob("*.json"))


# -- no double creation ---------------------------------------------------

def test_an_already_submitted_change_is_not_submitted_again(
    runtime_factory, client, no_critic
):
    state = {
        "fields": complete_field_map(),
        "submission": Submission(idempotency_key="k", number="CHG0030001"),
    }
    update = run(submit_change, runtime_factory(state))
    assert "already been submitted" in message_of(update)
    assert "submission" not in update


def test_replaying_the_same_key_returns_the_original_change(tmp_path):
    client = FakeServiceNowClient(submitted_dir=tmp_path)
    first = client.create_change({"short_description": "x"}, "same-key")
    second = client.create_change({"short_description": "x"}, "same-key")
    assert first.number == second.number
    assert len(list(tmp_path.glob("*.json"))) == 1
