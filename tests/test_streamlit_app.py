"""The Streamlit front end, driven headlessly with a scripted model.

Streamlit's AppTest runs the real script, so this covers the wiring that is easy
to get wrong: that the approval gate appears instead of the chat input when the
graph is interrupted, and that choosing a candidate goes through the agent
rather than writing a sys_id into state behind its back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from conftest import PROD_CI, seeded_fields
from test_graph import ScriptedModel, call

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Build an AppTest whose agent runs on a scripted model, in a temp cwd."""
    monkeypatch.chdir(tmp_path)

    def build(script: list[AIMessage], *, fake_servicenow=True):
        st.cache_resource.clear()
        monkeypatch.setattr(
            "change_agent.llm.get_model", lambda role="agent": ScriptedModel(script=script)
        )
        if fake_servicenow:
            from change_agent.servicenow.fake import FakeServiceNowClient
            from change_agent.tools import submit as submit_module

            monkeypatch.setattr(submit_module, "critique_fields", lambda *a, **k: [])
            monkeypatch.setattr(
                submit_module,
                "get_client",
                lambda: FakeServiceNowClient(submitted_dir=tmp_path / "submitted"),
            )
        return AppTest.from_file(APP, default_timeout=60)

    return build


def texts(at) -> str:
    """Everything rendered, flattened, for coarse assertions.

    Headings are their own element type in Streamlit, so they have to be asked
    for by name or an assertion about a section title silently never matches.
    """
    kinds = ("markdown", "caption", "subheader", "header", "title", "info", "error",
             "warning", "success", "text")
    return "\n".join(
        str(element.value) for kind in kinds for element in at.get(kind)
    )


def test_the_app_starts_empty(app):
    at = app([AIMessage(content="What are we changing?")])
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled is False


def test_a_message_reaches_the_agent_and_the_reply_is_shown(app):
    at = app([AIMessage(content="Which server is this on?")])
    at.run()
    at.chat_input[0].set_value("I need to patch an app server").run()
    assert not at.exception
    assert "Which server is this on?" in texts(at)


def test_candidates_are_offered_as_buttons_and_not_chosen_for_the_user(app):
    at = app(
        [
            call("search_reference", {"field": "cmdb_ci", "query": "app server"}, "c1"),
            AIMessage(content="Which of these?"),
        ]
    )
    at.run()
    at.chat_input[0].set_value("it's on the app server").run()
    assert not at.exception

    labels = [b.label for b in at.button]
    assert labels.count("Use this one") >= 2, "the twins must both be offered"
    assert "None of these" in labels
    # The agent must not have picked one.
    assert "SRV-APP-01-DEV" in texts(at)


def test_choosing_a_candidate_goes_back_through_the_agent(app):
    at = app(
        [
            call("search_reference", {"field": "cmdb_ci", "query": "SRV-APP-01"}, "c1"),
            AIMessage(content="Which of these do you mean?"),
            call("select_reference", {"field": "cmdb_ci", "choice": PROD_CI.sys_id}, "c2"),
            AIMessage(content="Set to SRV-APP-01."),
        ]
    )
    at.run()
    at.chat_input[0].set_value("the app server").run()
    at.button(key=f"pick-cmdb_ci-{PROD_CI.sys_id}").click().run()
    assert not at.exception
    assert "Set to SRV-APP-01." in texts(at)
    # The buttons are gone because the offer was taken up, not because it expired.
    assert "None of these" not in [b.label for b in at.button]


def seed(tmp_path, thread_id: str) -> None:
    """Put a submittable change on the app's thread, as a conversation would."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from change_agent.graph import TOOLS
    from change_agent.middleware import ground_in_current_change
    from change_agent.state import ChangeAgentState
    from langchain.agents import create_agent

    db = tmp_path / ".local" / "threads.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    agent = create_agent(
        model=ScriptedModel(script=[AIMessage(content="")]),
        tools=TOOLS,
        middleware=[ground_in_current_change],
        state_schema=ChangeAgentState,
        checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)),
    )
    agent.update_state(
        {"configurable": {"thread_id": thread_id}}, {"fields": seeded_fields()}
    )


def at_the_approval_gate(app, tmp_path):
    at = app([call("submit_change", {}, "c1"), AIMessage(content="Created.")])
    at.session_state["thread_id"] = "t-approve"
    at.run()
    seed(tmp_path, "t-approve")
    at.chat_input[0].set_value("submit it").run()
    return at


def test_submission_replaces_the_chat_input_with_an_approval_gate(app, tmp_path):
    at = at_the_approval_gate(app, tmp_path)

    assert not at.exception
    assert at.chat_input[0].disabled is True, "no chatting past the approval"
    assert "Approve before this is created" in texts(at)
    assert "Create this change" in [b.label for b in at.button]
    assert not (tmp_path / "submitted").exists(), "nothing may be created yet"


def test_the_gate_shows_the_payload_that_will_be_sent(app, tmp_path):
    at = at_the_approval_gate(app, tmp_path)
    # The page also renders tool-call arguments as JSON, so pick out the payload
    # rather than trusting its position.
    blobs = [json.loads(e.value) for e in at.get("json")]
    payload = next(b for b in blobs if "short_description" in b)
    assert payload["cmdb_ci"] == PROD_CI.sys_id
    assert payload["type"] == "normal"


def test_approving_creates_the_change(app, tmp_path):
    at = at_the_approval_gate(app, tmp_path)

    next(b for b in at.button if b.label == "Create this change").click().run()

    assert not at.exception
    assert len(list((tmp_path / "submitted").glob("*.json"))) == 1
    assert at.chat_input[0].disabled is False, "the gate releases once decided"


def test_cancelling_creates_nothing(app, tmp_path):
    at = at_the_approval_gate(app, tmp_path)
    next(b for b in at.button if b.label == "Cancel submission").click().run()

    assert not at.exception
    assert not (tmp_path / "submitted").exists()
