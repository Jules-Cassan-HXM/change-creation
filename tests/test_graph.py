"""End-to-end through the compiled agent, with a scripted model.

No API key required: the model is replaced, everything else -- the tools, the
state reducers, the checkpointer, the interrupt and its resume -- is the real
thing.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import PrivateAttr

from change_agent.graph import TOOLS
from change_agent.middleware import ground_in_current_change
from change_agent.servicenow.fake import FakeServiceNowClient
from change_agent.state import ChangeAgentState
from change_agent.tools import submit as submit_module

from conftest import PROD_CI, seeded_fields

CI_SYS_ID = PROD_CI.sys_id


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of assistant turns."""

    script: list[AIMessage] = []
    _calls: int = PrivateAttr(default=0)
    _seen: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def seen(self) -> list[list[BaseMessage]]:
        """Every message list the agent sent, for asserting on the prompt."""
        return self._seen

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._seen.append(list(messages))
        index = min(self._calls, len(self.script) - 1)
        self._calls += 1
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def build(script: list[AIMessage]):
    model = ScriptedModel(script=script)
    agent = create_agent(
        model=model,
        tools=TOOLS,
        middleware=[ground_in_current_change],
        state_schema=ChangeAgentState,
        checkpointer=InMemorySaver(),
    )
    return agent, model


@pytest.fixture(autouse=True)
def no_critic(monkeypatch):
    """The review pass needs a live model; submission itself does not."""
    monkeypatch.setattr(submit_module, "critique_fields", lambda *a, **k: [])


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeServiceNowClient(submitted_dir=tmp_path / "submitted")
    monkeypatch.setattr(submit_module, "get_client", lambda: fake)
    return fake


def test_a_reference_is_searched_then_chosen_through_the_graph():
    agent, _ = build(
        [
            call("propose_change_fields", {
                "updates": [
                    {"field": "short_description",
                     "value": "Patch app-server to 6.4.2 on SRV-APP-01",
                     "source": "user"}
                ]
            }, "c1"),
            call("search_reference", {"field": "cmdb_ci", "query": "SRV-APP-01"}, "c2"),
            call("select_reference", {"field": "cmdb_ci", "choice": CI_SYS_ID}, "c3"),
            AIMessage(content="Set to SRV-APP-01 (production)."),
        ]
    )
    config = {"configurable": {"thread_id": "t-ref"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Patch SRV-APP-01 to 6.4.2"}]}, config
    )

    fields = result["fields"]
    assert fields["short_description"].confirmed is True
    assert fields["cmdb_ci"].value.sys_id == CI_SYS_ID
    assert fields["cmdb_ci"].source == "lookup"
    # The offer was consumed, and the rules ran on the way through.
    assert result["pending_lookup"]["cmdb_ci"] == []
    assert any(f.code == "R001" for f in result["findings"])


def test_the_model_is_told_the_current_state_of_the_change():
    """The grounding middleware, checked where it matters: at the model call."""
    agent, model = build(
        [
            call("search_reference", {"field": "cmdb_ci", "query": "SRV-APP-01"}, "c1"),
            call("select_reference", {"field": "cmdb_ci", "choice": CI_SYS_ID}, "c2"),
            AIMessage(content="Done."),
        ]
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "the app server"}]},
        {"configurable": {"thread_id": "t-prompt"}},
    )

    final_prompt = model.seen[-1][0].content
    assert "SRV-APP-01" in final_prompt, "the model must see what is already set"
    assert "confirmed" in final_prompt, "and how much weight each value carries"
    assert "Still required and empty" in final_prompt, "and what is left to ask about"


def test_submission_pauses_for_approval_and_resumes(client):
    agent, _ = build(
        [
            call("submit_change", {}, "c1"),
            AIMessage(content="Created."),
        ]
    )
    config = {"configurable": {"thread_id": "t-submit"}}

    # Seed a change that has nothing blocking left, the way a real conversation
    # would have built it up.
    agent.update_state(config, {"fields": seeded_fields()})

    paused = agent.invoke({"messages": [{"role": "user", "content": "submit it"}]}, config)

    interrupts = paused["__interrupt__"]
    assert interrupts, "submission must pause for the user"
    request = interrupts[0].value
    assert request["type"] == "approve_change_submission"
    assert request["payload"]["cmdb_ci"] == CI_SYS_ID
    assert not client.submitted_dir.exists(), "nothing may be created before approval"

    resumed = agent.invoke(Command(resume={"action": "approve"}), config)

    assert resumed["submission"].number.startswith("CHG")
    assert len(list(client.submitted_dir.glob("*.json"))) == 1


def test_a_declined_approval_creates_nothing(client):
    agent, _ = build([call("submit_change", {}, "c1"), AIMessage(content="Left as a draft.")])
    config = {"configurable": {"thread_id": "t-decline"}}

    agent.update_state(config, {"fields": seeded_fields()})

    agent.invoke({"messages": [{"role": "user", "content": "submit it"}]}, config)
    resumed = agent.invoke(Command(resume={"action": "cancel"}), config)

    assert resumed.get("submission") is None
    assert not client.submitted_dir.exists() or not list(client.submitted_dir.glob("*.json"))
