"""Fixture-backed ServiceNow, so the whole agent runs with no instance.

The fixtures are deliberately awkward: near-identical CI names, a production and
a non-production twin for most systems, and groups whose names overlap. The
disambiguation path is the one most likely to quietly do the wrong thing, so it
gets exercised on every single run rather than only in a dedicated test.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, utils

from .base import Choice, CreatedChange, Record, ServiceNowError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
#: Below this rapidfuzz score a row is not worth showing the user.
MATCH_FLOOR = 55.0


@lru_cache(maxsize=None)
def _load(table: str) -> tuple[dict[str, Any], ...]:
    path = FIXTURE_DIR / f"{table}.json"
    if not path.exists():
        raise ServiceNowError(f"No fixture for table {table!r}")
    return tuple(json.loads(path.read_text()))


#: Words that carry no discriminating power when matching a record name.
_STOPWORDS = frozenset(
    {"the", "a", "an", "our", "my", "on", "in", "of", "for",
     "le", "la", "les", "un", "une", "du", "de", "des", "sur"}
)


def _tokens(text: str) -> list[str]:
    """Split on anything that is not a letter or digit, dropping noise."""
    return [
        tok
        for tok in re.split(r"[^a-z0-9]+", text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _coverage(query: str, haystack: str) -> float:
    """Fraction of the query's meaningful tokens that the row accounts for.

    Plain fuzzy ratios are far too generous with short queries: "app server"
    scores highly against "Exchange mailbox server" purely on the shared word.
    Requiring the query's words to be accounted for is what keeps that row out.
    """
    q_tokens = _tokens(query)
    hay_tokens = _tokens(haystack)
    if not q_tokens or not hay_tokens:
        return 0.0
    hits = sum(
        1 for q in q_tokens if max(fuzz.partial_ratio(q, h) for h in hay_tokens) >= 80
    )
    return hits / len(q_tokens)


def _ratio(query: str, text: str) -> float:
    """Similarity that tolerates both abbreviation and reordering.

    ``WRatio`` alone punishes a short query against a longer name
    ("payment api" vs "Payment API UAT" scores 54), which would hide exactly the
    production/UAT twins the user most needs to choose between.
    """
    return max(
        fuzz.WRatio(query, text, processor=utils.default_process),
        fuzz.token_set_ratio(query, text, processor=utils.default_process) * 0.95,
    )


def _score(query: str, row: dict[str, Any]) -> float:
    """How well a fixture row answers ``query``.

    A match on the record's own name or alias outranks one that came only from
    its context (class, environment, owning service), so an exact hostname is
    never displaced by a category match.
    """
    names = [row["name"], *row.get("aliases", [])]
    context_values = [str(v) for v in row.get("context", {}).values()]
    full_text = " ".join(names + context_values)

    by_name = max(_ratio(query, name) for name in names)
    by_context = (
        fuzz.token_set_ratio(query, full_text, processor=utils.default_process) * 0.8
    )

    # Name matching saturates -- several unrelated rows can all land on the
    # same score -- so a slice of the context score is mixed in to break ties in
    # favour of the row whose surroundings also fit.
    raw = 0.88 * max(by_name, by_context) + 0.12 * by_context

    # Scale rather than hard-filter: a partially covered query still ranks, it
    # just ranks below anything that accounts for every word the user typed.
    return min(100.0, raw) * (0.35 + 0.65 * _coverage(query, full_text))


class FakeServiceNowClient:
    """In-memory stand-in for a real instance."""

    def __init__(self, submitted_dir: Path | None = None) -> None:
        self.submitted_dir = submitted_dir or Path(".local/submitted")

    # -- reads ------------------------------------------------------------
    def search_reference(self, table: str, query: str, limit: int = 5) -> list[Record]:
        rows = _load(table)
        scored = [
            (score, row)
            for row in rows
            if (score := _score(query, row)) >= MATCH_FLOOR
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        return [
            Record(
                sys_id=row["sys_id"],
                display_value=row["name"],
                table=table,
                context=row.get("context", {}),
                score=round(score, 1),
            )
            for score, row in scored[:limit]
        ]

    def get_choices(self, choice_source: str) -> list[Choice]:
        choices = json.loads((FIXTURE_DIR / "choices.json").read_text())
        if choice_source not in choices:
            raise ServiceNowError(f"No choice list for {choice_source!r}")
        return [Choice(**c) for c in choices[choice_source]]

    # -- writes -----------------------------------------------------------
    def create_change(
        self, payload: dict[str, Any], idempotency_key: str
    ) -> CreatedChange:
        """Write the payload to disk and hand back a synthetic number.

        Keyed by ``idempotency_key`` so a replay after a crash returns the
        original result instead of creating a second change -- the same contract
        the real client has to honour.
        """
        self.submitted_dir.mkdir(parents=True, exist_ok=True)
        path = self.submitted_dir / f"{idempotency_key}.json"

        if path.exists():
            existing = json.loads(path.read_text())
            return CreatedChange(
                number=existing["number"], sys_id=existing["sys_id"]
            )

        seq = 30000 + len(list(self.submitted_dir.glob("*.json"))) + 1
        created = CreatedChange(number=f"CHG00{seq}", sys_id=uuid.uuid4().hex)
        path.write_text(
            json.dumps(
                {
                    "number": created.number,
                    "sys_id": created.sys_id,
                    "idempotency_key": idempotency_key,
                    "created_at": datetime.now(UTC).isoformat(),
                    "payload": payload,
                },
                indent=2,
                default=str,
            )
        )
        return created
