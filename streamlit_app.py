"""A chat front end for the change copilot, with the approval steps as buttons.

Run with:

    uv run streamlit run streamlit_app.py

Two moments in this agent need a person, and they work differently underneath:

* **Submission** is a real LangGraph interrupt. The graph stops inside
  `submit_change`, the pending approval shows up on the state snapshot, and
  nothing reaches ServiceNow until the user resumes it with a decision.
* **Choosing a referenced record** is not an interrupt. The agent asks in
  conversation, and the candidates it offered sit in `pending_lookup` on the
  state. This screen renders them as buttons, but pressing one sends an ordinary
  message naming the sys_id, so the choice still goes through `select_reference`
  and the same validation as a typed answer. Keeping it out of the interrupt
  machinery is what lets the agent narrow a search or ask a question of its own
  instead of being forced to stop dead on every lookup.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from change_agent.graph import TOOLS
from change_agent.llm import get_model
from change_agent.middleware import ground_in_current_change
from change_agent.rendering import format_value
from change_agent.schema import FIELD_SPECS, REQUIRED_FIELDS
from change_agent.state import (
    ChangeAgentState,
    get_fields,
    get_findings,
    get_pending_lookup,
    get_submission,
)

load_dotenv()

DB_PATH = Path(".local/threads.sqlite")
LONG_TEXT = 140

st.set_page_config(page_title="Change copilot", page_icon="🛠️", layout="wide")


# -- wiring ---------------------------------------------------------------

@st.cache_resource
def get_agent():
    """One compiled agent per server process, with durable threads."""
    from langchain.agents import create_agent

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    return create_agent(
        model=get_model("agent"),
        tools=TOOLS,
        middleware=[ground_in_current_change],
        state_schema=ChangeAgentState,
        checkpointer=SqliteSaver(connection),
        name="change_copilot",
    )


def thread_config() -> dict[str, Any]:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def advance(agent, config: dict[str, Any], inputs: Any) -> None:
    """Run the graph until it stops, showing what it does on the way."""
    with st.status("Working...", expanded=True) as status:
        try:
            for chunk in agent.stream(inputs, config, stream_mode="updates"):
                for node, update in (chunk or {}).items():
                    if node == "__interrupt__":
                        status.update(label="Waiting for your approval", state="complete")
                        continue
                    for message in (update or {}).get("messages", []) or []:
                        for call in getattr(message, "tool_calls", None) or []:
                            st.write(f"`{call['name']}`")
            status.update(label="Done", state="complete")
        except Exception as exc:  # surfaced rather than swallowed
            status.update(label="Failed", state="error")
            st.exception(exc)
            st.stop()
    st.rerun()


# -- pieces of the screen -------------------------------------------------

def provenance(confirmed: bool) -> str:
    return ":green[confirmed]" if confirmed else ":orange[inferred, unconfirmed]"


def render_change(fields: dict[str, Any], *, key_prefix: str) -> None:
    """The change itself, long text folded away, provenance on every value."""
    shown = [(n, fields[n]) for n in FIELD_SPECS if n in fields and fields[n].is_set]
    if not shown:
        st.caption("Nothing captured yet.")
        return

    for name, value in shown:
        label = FIELD_SPECS[name].label
        text = format_value(value.value)
        if len(text) > LONG_TEXT or "\n" in text:
            with st.expander(f"**{label}** — {provenance(value.confirmed)}"):
                st.text(text)
        else:
            st.markdown(f"**{label}**: {text} · {provenance(value.confirmed)}")


def render_findings(findings: list[Any], *, empty: str = "Nothing outstanding.") -> None:
    if not findings:
        st.caption(empty)
        return
    for finding in findings:
        where = FIELD_SPECS[finding.field].label if finding.field in FIELD_SPECS else None
        headline = f"**{where}** — {finding.message}" if where else finding.message
        if finding.severity == "blocker":
            st.error(headline, icon="⛔")
        else:
            st.warning(headline, icon="⚠️")
        rewrite = finding.data.get("suggested_rewrite")
        issues = finding.data.get("issues")
        if issues:
            st.caption(" · ".join(issues))
        if rewrite:
            with st.expander(f"Suggested rewrite for {where or finding.field}"):
                st.text(rewrite)


def render_sidebar(snapshot) -> None:
    fields = get_fields(snapshot.values)
    findings = get_findings(snapshot.values)
    submission = get_submission(snapshot.values)

    with st.sidebar:
        st.subheader("The change")

        filled = [n for n in REQUIRED_FIELDS if n in fields and fields[n].is_set]
        st.progress(
            len(filled) / len(REQUIRED_FIELDS),
            text=f"{len(filled)} of {len(REQUIRED_FIELDS)} required fields",
        )

        if submission and submission.number:
            st.success(f"Submitted as **{submission.number}**", icon="✅")

        render_change(fields, key_prefix="sidebar")

        st.divider()
        st.subheader("Open findings")
        render_findings(findings)

        st.divider()
        mode = os.environ.get("SERVICENOW_MODE", "fake")
        st.caption(
            f"ServiceNow: **{mode}**"
            + ("  ·  changes are written to `.local/submitted/`" if mode == "fake" else "")
        )
        if st.button("Start a new change", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


def render_history(snapshot) -> None:
    for message in snapshot.values.get("messages", []) or []:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            if message.content:
                with st.chat_message("assistant"):
                    st.markdown(
                        message.content
                        if isinstance(message.content, str)
                        else str(message.content)
                    )
            for call in message.tool_calls or []:
                with st.chat_message("assistant"):
                    with st.expander(f"Used `{call['name']}`"):
                        st.json(call["args"])
        elif isinstance(message, ToolMessage):
            continue  # the outcome is already in the sidebar and the reply


def render_candidate_picker(agent, config, snapshot) -> None:
    """Buttons for a reference the agent has just looked up."""
    pending = get_pending_lookup(snapshot.values)
    awaiting = {field: c for field, c in pending.items() if c}
    if not awaiting:
        return

    for field, candidates in awaiting.items():
        label = FIELD_SPECS[field].label
        st.info(f"Which **{label}** do you mean? These came back from ServiceNow.", icon="🔎")
        columns = st.columns(min(len(candidates), 3))
        for index, candidate in enumerate(candidates):
            with columns[index % len(columns)]:
                detail = " · ".join(f"{k}: {v}" for k, v in candidate.context.items() if v)
                st.markdown(f"**{candidate.display_value}**")
                st.caption(detail or "no further detail")
                if st.button(
                    "Use this one",
                    key=f"pick-{field}-{candidate.sys_id}",
                    use_container_width=True,
                ):
                    advance(
                        agent,
                        config,
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        f"For {field}, use {candidate.display_value} "
                                        f"(sys_id {candidate.sys_id})."
                                    ),
                                }
                            ]
                        },
                    )
        if st.button("None of these", key=f"pick-none-{field}", use_container_width=True):
            advance(
                agent,
                config,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"None of those are the right {label}.",
                        }
                    ]
                },
            )


def render_approval(agent, config, snapshot, request: dict[str, Any]) -> None:
    """The submission gate: everything that will be created, then a decision."""
    st.subheader("Approve before this is created")
    st.caption(
        "Nothing has been sent to ServiceNow. This is exactly what will be created."
    )

    blockers = request.get("blockers") or []
    warnings = request.get("warnings") or []
    override = request.get("override_reason")

    if override:
        st.error(
            f"Submitting with **{len(blockers)}** blocking finding(s) still open. "
            f"Recorded reason: _{override}_ — this goes into the change's work notes.",
            icon="⛔",
        )

    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### The change")
        render_change(get_fields(snapshot.values), key_prefix="approval")

    with right:
        if blockers:
            st.markdown("##### Still open")
            render_findings([_as_finding(f) for f in blockers])
        st.markdown("##### Worth a second look")
        render_findings(
            [_as_finding(f) for f in warnings], empty="Nothing flagged."
        )

    with st.expander("Exactly what will be sent to the Table API"):
        st.json(request.get("payload", {}))

    st.divider()
    approve, edit, cancel = st.columns(3)

    with approve:
        if st.button("Create this change", type="primary", use_container_width=True):
            advance(agent, config, Command(resume={"action": "approve"}))
    with edit:
        if st.button("Keep editing", use_container_width=True):
            st.session_state.show_edit_note = True
    with cancel:
        if st.button("Cancel submission", use_container_width=True):
            advance(agent, config, Command(resume={"action": "cancel"}))

    if st.session_state.get("show_edit_note"):
        notes = st.text_input("What still needs changing?", key="edit_notes")
        if st.button("Send that back to the agent", disabled=not notes):
            st.session_state.show_edit_note = False
            advance(agent, config, Command(resume={"action": "edit", "notes": notes}))


def _as_finding(raw: dict[str, Any]):
    from change_agent.rules.base import Finding

    return Finding.model_validate(raw)


# -- the page -------------------------------------------------------------

def main() -> None:
    st.title("Change copilot")

    agent = get_agent()
    config = thread_config()
    snapshot = agent.get_state(config)

    render_sidebar(snapshot)
    render_history(snapshot)

    pending_approval = snapshot.interrupts
    if pending_approval:
        render_approval(agent, config, snapshot, pending_approval[0].value)
        st.chat_input("Decide above before carrying on", disabled=True)
        return

    render_candidate_picker(agent, config, snapshot)

    if prompt := st.chat_input("Describe the change, or answer the last question"):
        advance(agent, config, {"messages": [{"role": "user", "content": prompt}]})


main()
