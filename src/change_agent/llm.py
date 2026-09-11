"""Model selection, kept in one place and driven by the environment.

``init_chat_model`` takes a ``provider:model`` string, so swapping provider is a
configuration change rather than a code change.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-sonnet-5"


@lru_cache(maxsize=4)
def get_model(role: str = "agent") -> BaseChatModel:
    """The model for a role.

    ``LLM_MODEL`` sets the conversational model; ``LLM_CRITIC_MODEL`` optionally
    points the review pass at a different one, since critique is a single
    structured call and can afford a different trade-off.
    """
    if role == "critic":
        name = os.environ.get("LLM_CRITIC_MODEL") or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    else:
        name = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    return init_chat_model(name)
