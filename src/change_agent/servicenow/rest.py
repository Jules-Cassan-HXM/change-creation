"""Table API client for a real instance.

Written against the documented ServiceNow Table API but **not yet verified
against a live instance** -- development runs on :mod:`change_agent.servicenow.fake`.
Treat the query strings here as the first thing to check when something 404s.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from rapidfuzz import fuzz, utils

from .base import Choice, CreatedChange, Record, ServiceNowError

#: The column shown to a user for each referenced table.
DISPLAY_FIELD: dict[str, str] = {
    "cmdb_ci": "name",
    "cmdb_ci_service": "name",
    "sys_user_group": "name",
    "sys_user": "name",
}

#: Extra columns worth pulling back so the user can tell near-identical rows
#: apart, and so rules can reason about what is being touched.
CONTEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "cmdb_ci": ("sys_class_name", "environment", "support_group", "operational_status"),
    "cmdb_ci_service": ("busines_criticality", "owned_by"),
    "sys_user_group": ("manager", "email"),
    "sys_user": ("email", "title", "department"),
}


class RestServiceNowClient:
    """Thin, synchronous Table API wrapper."""

    def __init__(
        self,
        instance_url: str,
        username: str,
        password: str,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = instance_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            auth=(username, password),
            timeout=timeout,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    # -- plumbing ---------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            response = self._client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ServiceNowError(f"GET {path} failed: {exc}") from exc
        return response.json().get("result", [])

    @staticmethod
    def _flatten(value: Any) -> str:
        """Reference columns come back as ``{'display_value': ..., 'value': ...}``."""
        if isinstance(value, dict):
            return str(value.get("display_value") or value.get("value") or "")
        return str(value or "")

    # -- reads ------------------------------------------------------------
    def search_reference(self, table: str, query: str, limit: int = 5) -> list[Record]:
        display = DISPLAY_FIELD.get(table, "name")
        context_fields = CONTEXT_FIELDS.get(table, ())

        rows = self._get(
            f"/api/now/table/{table}",
            {
                # Ask the instance for a generous superset, then rank locally:
                # `LIKE` has no notion of which of twenty near-identical rows the
                # user actually meant.
                "sysparm_query": f"{display}LIKE{query}^ORDERBY{display}",
                "sysparm_fields": ",".join(("sys_id", display, *context_fields)),
                "sysparm_display_value": "all",
                "sysparm_limit": max(limit * 10, 50),
            },
        )

        scored: list[tuple[float, Record]] = []
        for row in rows:
            name = self._flatten(row.get(display))
            context = {
                field: self._flatten(row.get(field))
                for field in context_fields
                if self._flatten(row.get(field))
            }
            score = float(
                fuzz.WRatio(query, name, processor=utils.default_process)
            )
            scored.append(
                (
                    score,
                    Record(
                        sys_id=self._flatten(row.get("sys_id")),
                        display_value=name,
                        table=table,
                        context=context,
                        score=round(score, 1),
                    ),
                )
            )

        scored.sort(key=lambda pair: -pair[0])
        return [record for _, record in scored[:limit]]

    def get_choices(self, choice_source: str) -> list[Choice]:
        table, _, element = choice_source.partition(".")
        rows = self._get(
            "/api/now/table/sys_choice",
            {
                "sysparm_query": (
                    f"name={table}^element={element}^inactive=false^ORDERBYsequence"
                ),
                "sysparm_fields": "value,label",
                "sysparm_limit": 100,
            },
        )
        return [
            Choice(value=self._flatten(r.get("value")), label=self._flatten(r.get("label")))
            for r in rows
        ]

    # -- writes -----------------------------------------------------------
    def create_change(
        self, payload: dict[str, Any], idempotency_key: str
    ) -> CreatedChange:
        """Create the change, keyed on ``correlation_id``.

        ServiceNow has no idempotency header, so the key is written into
        ``correlation_id`` and we look for it first. A replay after a crash
        mid-POST therefore returns the original change instead of creating a
        duplicate -- the failure mode that matters most here, because nobody
        notices a second change until CAB does.
        """
        existing = self._get(
            "/api/now/table/change_request",
            {
                "sysparm_query": f"correlation_id={idempotency_key}",
                "sysparm_fields": "number,sys_id",
                "sysparm_limit": 1,
            },
        )
        if existing:
            return CreatedChange(
                number=self._flatten(existing[0].get("number")),
                sys_id=self._flatten(existing[0].get("sys_id")),
            )

        body = {**payload, "correlation_id": idempotency_key}
        try:
            response = self._client.post("/api/now/table/change_request", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ServiceNowError(f"Creating the change failed: {exc}") from exc

        result = response.json().get("result", {})
        return CreatedChange(
            number=self._flatten(result.get("number")),
            sys_id=self._flatten(result.get("sys_id")),
        )

    def close(self) -> None:
        self._client.close()


def from_env() -> RestServiceNowClient:
    missing = [
        name
        for name in ("SERVICENOW_INSTANCE_URL", "SERVICENOW_USER", "SERVICENOW_PASSWORD")
        if not os.environ.get(name)
    ]
    if missing:
        raise ServiceNowError(
            "Missing environment variables for the real client: " + ", ".join(missing)
        )
    return RestServiceNowClient(
        instance_url=os.environ["SERVICENOW_INSTANCE_URL"],
        username=os.environ["SERVICENOW_USER"],
        password=os.environ["SERVICENOW_PASSWORD"],
    )
