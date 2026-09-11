"""The deterministic challenge rules.

These run on every field update: they are pure, fast, and always fire the same
way. Judgment calls that genuinely need a language model (is this plan
intelligible to somebody who was not in the room?) live in
:mod:`change_agent.rules.critic` instead.

Each rule is a plain function carrying ``code`` and ``severity`` attributes.
Findings carry structured data rather than a finished sentence, so the agent can
raise them in whichever language the conversation is happening in.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

from rapidfuzz import fuzz, utils

from ..schema import FIELD_SPECS, REQUIRED_FIELDS, ChangeRequest, Reference
from .base import Finding, RuleContext, Severity

RuleFn = Callable[[RuleContext], "Finding | list[Finding] | None"]

ALL_RULES: list[RuleFn] = []

#: Ratings we read as "the author says this is small". Deliberately narrow:
#: "moderate" is a defensible rating for a routine production change, and a rule
#: that fires on every such change is a rule people learn to click past.
_LOW_RISK = {"low"}
_LOW_IMPACT = {"3"}

#: Phrases claiming the change is invisible to users, in both working languages.
_NO_IMPACT_PHRASES = (
    "no outage", "no impact", "no downtime", "no interruption", "zero downtime",
    "non-disruptive", "transparent for users", "seamless",
    "aucune interruption", "aucun impact", "sans coupure", "sans interruption",
    "pas d'interruption", "pas de coupure", "transparent pour les utilisateurs",
)

#: Actions that normally imply a user-visible interruption.
_DISRUPTIVE_ACTIONS = (
    "reboot", "restart", "failover", "shutdown", "stop the service", "drain",
    "take offline", "cutover", "migrate", "redémarr", "redemarr", "bascule",
    "arrêt", "arret", "coupure", "redémarrage",
)

#: Language that belongs to an incident or an emergency change, not a normal one.
_URGENCY_PHRASES = (
    "asap", "as soon as possible", "urgent", "urgently", "emergency", "incident",
    "outage ongoing", "p1", "p2", "right now", "immediately", "critical issue",
    "dès que possible", "des que possible", "au plus vite", "en urgence",
    "immédiatement", "immediatement", "critique",
)

_DURATION_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(minutes?|mins?|m\b|hours?|hrs?|h\b|heures?|heure)",
    re.IGNORECASE,
)


def rule(code: str, severity: Severity) -> Callable[[RuleFn], RuleFn]:
    """Register a rule and stamp it with its code and severity."""

    def decorate(fn: RuleFn) -> RuleFn:
        fn.code = code  # type: ignore[attr-defined]
        fn.severity = severity  # type: ignore[attr-defined]
        ALL_RULES.append(fn)
        return fn

    return decorate


# -- helpers --------------------------------------------------------------

def _as_utc(value: datetime | None) -> datetime | None:
    """Treat a naive datetime as UTC, which is what ServiceNow stores."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _text(value: str | None) -> str:
    return (value or "").strip()


def _contains(haystack: str, needles: tuple[str, ...]) -> str | None:
    """Return the first phrase found, so the finding can quote it back."""
    lowered = haystack.lower()
    return next((n for n in needles if n in lowered), None)


def _ci_context(change: ChangeRequest) -> dict[str, Any]:
    ci = change.cmdb_ci
    return ci.context if isinstance(ci, Reference) else {}


def _looks_like_steps(text: str) -> bool:
    """Whether a plan reads as a sequence rather than a single sentence."""
    if re.search(r"(?m)^\s*(\d+[.)]|[-*•])\s+", text):
        return True
    return text.count("\n") >= 2 or len(re.findall(r"(?i)\bthen\b|\bpuis\b|\bensuite\b", text)) >= 1


# -- completeness ---------------------------------------------------------

@rule("R001", "blocker")
def missing_required_fields(ctx: RuleContext) -> list[Finding]:
    """Every field the instance refuses to accept a change without."""
    findings = []
    for name in REQUIRED_FIELDS:
        if getattr(ctx.change, name, None) in (None, "", [], {}):
            findings.append(
                Finding(
                    code="R001",
                    severity="blocker",
                    field=name,
                    message=f"{FIELD_SPECS[name].label} is required and is empty.",
                    data={"label": FIELD_SPECS[name].label,
                          "guidance": FIELD_SPECS[name].guidance},
                )
            )
    return findings


# -- scheduling -----------------------------------------------------------

@rule("R002", "blocker")
def end_before_start(ctx: RuleContext) -> Finding | None:
    start, end = _as_utc(ctx.change.start_date), _as_utc(ctx.change.end_date)
    if start and end and end <= start:
        return Finding(
            code="R002",
            severity="blocker",
            field="end_date",
            message="The planned end is not after the planned start.",
            data={"start_date": start.isoformat(), "end_date": end.isoformat()},
        )
    return None


@rule("R005", "warning")
def start_in_the_past(ctx: RuleContext) -> Finding | None:
    start = _as_utc(ctx.change.start_date)
    if start and start < ctx.now:
        return Finding(
            code="R005",
            severity="warning",
            field="start_date",
            message="The planned start is in the past.",
            data={"start_date": start.isoformat(), "now": ctx.now.isoformat()},
        )
    return None


@rule("R013", "warning")
def inside_cab_lead_time(ctx: RuleContext) -> Finding | None:
    """A normal change needs enough notice for CAB to actually review it."""
    start = _as_utc(ctx.change.start_date)
    if not start:
        return None
    earliest = ctx.now + timedelta(days=ctx.policy.cab_lead_days)
    if ctx.now <= start < earliest:
        return Finding(
            code="R013",
            severity="warning",
            field="start_date",
            message=(
                f"The start is inside the {ctx.policy.cab_lead_days}-day lead time a "
                f"normal change needs for CAB review."
            ),
            data={
                "start_date": start.isoformat(),
                "earliest_compliant": earliest.isoformat(),
                "cab_lead_days": ctx.policy.cab_lead_days,
            },
        )
    return None


@rule("R012", "warning")
def window_shorter_than_plan(ctx: RuleContext) -> Finding | None:
    """The implementation plan claims more time than the window allows."""
    start, end = _as_utc(ctx.change.start_date), _as_utc(ctx.change.end_date)
    plan = _text(ctx.change.implementation_plan)
    if not (start and end and plan):
        return None

    minutes = 0.0
    for amount, unit in _DURATION_PATTERN.findall(plan):
        value = float(amount.replace(",", "."))
        minutes += value * (60 if unit.lower().startswith(("h", "hr", "heure")) else 1)
    if minutes <= 0:
        return None

    window = (end - start).total_seconds() / 60
    if minutes > window:
        return Finding(
            code="R012",
            severity="warning",
            field="end_date",
            message=(
                "The steps in the implementation plan add up to more time than the "
                "change window allows."
            ),
            data={"plan_minutes": round(minutes), "window_minutes": round(window)},
        )
    return None


# -- plan quality (structural, not stylistic) -----------------------------

@rule("R003", "blocker")
def backout_plan_not_actionable(ctx: RuleContext) -> Finding | None:
    """Present but not a plan. Absence is R001's job."""
    text = _text(ctx.change.backout_plan)
    if not text:
        return None
    if len(text) < ctx.policy.min_backout_plan_chars or not _looks_like_steps(text):
        return Finding(
            code="R003",
            severity="blocker",
            field="backout_plan",
            message=(
                "The backout plan is too thin to follow: it needs the concrete "
                "sequence that returns the system to its current state."
            ),
            data={"length": len(text), "excerpt": text[:200],
                  "minimum": ctx.policy.min_backout_plan_chars},
        )
    return None


@rule("R004", "blocker")
def implementation_plan_not_actionable(ctx: RuleContext) -> Finding | None:
    text = _text(ctx.change.implementation_plan)
    if not text:
        return None
    if len(text) < ctx.policy.min_implementation_plan_chars or not _looks_like_steps(text):
        return Finding(
            code="R004",
            severity="blocker",
            field="implementation_plan",
            message=(
                "The implementation plan is not a sequence of steps somebody else "
                "could execute without asking questions."
            ),
            data={"length": len(text), "excerpt": text[:200],
                  "minimum": ctx.policy.min_implementation_plan_chars},
        )
    return None


@rule("R008", "warning")
def missing_test_plan_for_risky_change(ctx: RuleContext) -> Finding | None:
    risk = (ctx.change.risk or "").lower()
    if risk in {"very_high", "high", "moderate"} and not _text(ctx.change.test_plan):
        return Finding(
            code="R008",
            severity="warning",
            field="test_plan",
            message=f"Risk is {risk} but there is no test plan.",
            data={"risk": risk},
        )
    return None


@rule("R009", "warning")
def description_adds_nothing(ctx: RuleContext) -> Finding | None:
    """A description that restates the one-liner has dropped the *why*."""
    short, long = _text(ctx.change.short_description), _text(ctx.change.description)
    if not (short and long):
        return None
    similarity = fuzz.token_set_ratio(short, long, processor=utils.default_process)
    if similarity >= 88 or len(long) <= len(short) + 20:
        return Finding(
            code="R009",
            severity="warning",
            field="description",
            message=(
                "The description repeats the short description without explaining "
                "why the change is being made."
            ),
            data={"similarity": round(similarity), "short_description": short},
        )
    return None


# -- coherence ------------------------------------------------------------

@rule("R006", "warning")
def risk_understated_for_production(ctx: RuleContext) -> Finding | None:
    """Low risk on a production CI or a critical service is worth a question."""
    risk = (ctx.change.risk or "").lower()
    impact = (ctx.change.impact or "").lower()
    if risk not in _LOW_RISK and impact not in _LOW_IMPACT:
        return None

    ci_context = _ci_context(ctx.change)
    environment = str(ci_context.get("environment", "")).lower()
    service = ctx.change.business_service
    criticality = (
        str(service.context.get("criticality", "")).lower()
        if isinstance(service, Reference)
        else ""
    )

    on_production = environment == "production" or ctx.change.production_system is True
    on_critical_service = "most critical" in criticality
    if not (on_production or on_critical_service):
        return None

    return Finding(
        code="R006",
        severity="warning",
        field="risk",
        message=(
            "Risk or impact is rated low, but this touches production or a "
            "critical business service."
        ),
        data={
            "risk": risk or None,
            "impact": impact or None,
            "ci": ctx.change.cmdb_ci.display_value if ctx.change.cmdb_ci else None,
            "environment": environment or None,
            "business_service_criticality": criticality or None,
        },
    )


@rule("R007", "warning")
def outage_claim_contradicts_plan(ctx: RuleContext) -> list[Finding]:
    """Both directions of the "will users notice?" contradiction."""
    findings: list[Finding] = []
    prose = " ".join(
        filter(None, [ctx.change.description, ctx.change.risk_impact_analysis,
                      ctx.change.short_description])
    )
    plan = _text(ctx.change.implementation_plan)

    claim = _contains(prose, _NO_IMPACT_PHRASES)
    if claim and ctx.change.outage_expected is True:
        findings.append(
            Finding(
                code="R007",
                severity="warning",
                field="outage_expected",
                message="The change claims no user impact but an outage is flagged.",
                data={"phrase": claim},
            )
        )

    action = _contains(plan, _DISRUPTIVE_ACTIONS)
    if claim and action and ctx.change.outage_expected is not True:
        findings.append(
            Finding(
                code="R007",
                severity="warning",
                field="implementation_plan",
                message=(
                    "The change claims no user impact, but the implementation plan "
                    "contains a step that normally interrupts service."
                ),
                data={"phrase": claim, "action": action},
            )
        )
    return findings


@rule("R010", "warning")
def urgency_language_in_normal_change(ctx: RuleContext) -> Finding | None:
    """Emergency work filed as a normal change is a governance problem."""
    change_type = (ctx.change.type or "normal").lower()
    if change_type != "normal":
        return None
    prose = " ".join(
        filter(None, [ctx.change.short_description, ctx.change.description,
                      ctx.change.justification])
    )
    phrase = _contains(prose, _URGENCY_PHRASES)
    if phrase:
        return Finding(
            code="R010",
            severity="warning",
            field="type",
            message=(
                "This is filed as a normal change but the wording describes "
                "something urgent."
            ),
            data={"phrase": phrase},
        )
    return None


@rule("R011", "warning")
def assignment_group_not_ci_support_group(ctx: RuleContext) -> Finding | None:
    group, ci = ctx.change.assignment_group, ctx.change.cmdb_ci
    if not (isinstance(group, Reference) and isinstance(ci, Reference)):
        return None
    support_group = str(ci.context.get("support_group", "")).strip()
    if not support_group or support_group.lower() == group.display_value.lower():
        return None
    return Finding(
        code="R011",
        severity="warning",
        field="assignment_group",
        message=(
            "The assignment group is not the support group registered for this "
            "configuration item."
        ),
        data={
            "assignment_group": group.display_value,
            "ci": ci.display_value,
            "ci_support_group": support_group,
        },
    )


# -- runner ---------------------------------------------------------------

def run_rules(ctx: RuleContext) -> list[Finding]:
    """Run every rule, blockers first, in stable code order."""
    findings: list[Finding] = []
    for check in ALL_RULES:
        result = check(ctx)
        if result is None:
            continue
        findings.extend(result if isinstance(result, list) else [result])

    order = {"blocker": 0, "warning": 1}
    findings.sort(key=lambda f: (order[f.severity], f.code, f.field or ""))
    return findings


def blockers(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "blocker"]
