"""The agent, assembled.

Exposed as ``graph`` for ``langgraph dev`` (see ``langgraph.json``).
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain.agents import create_agent

from .llm import get_model
from .middleware import ground_in_current_change
from .state import ChangeAgentState
from .tools.fields import describe_fields, propose_change_fields
from .tools.references import search_reference, select_reference
from .tools.review import review_change
from .tools.submit import record_override_reason, submit_change

load_dotenv()

TOOLS = [
    propose_change_fields,
    describe_fields,
    search_reference,
    select_reference,
    review_change,
    record_override_reason,
    submit_change,
]


def build_graph():
    """Build the compiled agent.

    No checkpointer is passed: the LangGraph dev server and platform supply
    their own, and `submit_change` interrupts, which requires one. When
    embedding this elsewhere, compile with a checkpointer of your own.
    """
    return create_agent(
        model=get_model("agent"),
        tools=TOOLS,
        # The system prompt is assembled per turn by the middleware so it can
        # carry the current state of the change.
        middleware=[ground_in_current_change],
        state_schema=ChangeAgentState,
        name="change_copilot",
    )


graph = build_graph()
