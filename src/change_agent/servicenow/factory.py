"""Client selection. Defaults to the fake so the agent always starts."""

from __future__ import annotations

import os
from functools import lru_cache

from .base import ServiceNowClient
from .fake import FakeServiceNowClient


@lru_cache(maxsize=1)
def get_client() -> ServiceNowClient:
    """Return the configured client.

    ``SERVICENOW_MODE=rest`` opts in to the live instance; anything else (and
    the absence of the variable) gets the fixture-backed fake.
    """
    if os.environ.get("SERVICENOW_MODE", "fake").lower() == "rest":
        from .rest import from_env

        return from_env()
    return FakeServiceNowClient()


def reset_client_cache() -> None:
    """Drop the memoised client; used by tests that switch modes."""
    get_client.cache_clear()
