# Change request copilot

A LangGraph agent that writes ServiceNow **normal change requests** with you,
and argues with you while you do it.

Filling in the form is the easy part. The hard part is that a change written at
5pm by the person who knows the system is often unreadable to the two people who
need it: the approver deciding whether it may run, and whoever has to execute or
reverse it at 3am. This agent holds the change in its state, fills it in through
conversation or from a single pasted dump, challenges what does not add up, and
will not submit anything structurally broken without an explicit, recorded
override.

## Quickstart

```bash
uv sync
cp .env.example .env                      # set LLM_MODEL and the matching provider key

uv run streamlit run streamlit_app.py     # the chat UI
uv run langgraph dev                      # or LangGraph Studio, for debugging
```

No ServiceNow instance is needed. It runs against fixtures by default.

```bash
uv run pytest             # 82 tests, no API key required
```

## The UI

`streamlit_app.py` is a chat window with the two approval moments surfaced as
buttons. They look alike on screen and work differently underneath, which is
worth knowing before changing either:

**Choosing a referenced record** is *not* an interrupt. The agent asks in
conversation and the candidates it offered sit in `pending_lookup` on the state;
the UI renders them as buttons showing what distinguishes them -- environment,
support group, criticality -- so the production and development twins are
visibly different things. Pressing one sends an ordinary message naming the
sys_id, so the choice still goes through `select_reference` and exactly the same
validation as a typed answer. Keeping lookups out of the interrupt machinery is
what lets the agent narrow a search or ask a question of its own rather than
being forced to stop dead on every one.

**Submitting** is a real LangGraph interrupt. The graph stops inside
`submit_change`, the chat input is disabled, and the whole change is laid out
with each value's provenance, the findings still open, and the exact Table API
payload in an expander. Create / keep editing / cancel resume the graph with the
user's decision. Nothing reaches ServiceNow until then.

Threads are kept in `.local/threads.sqlite`, so a change survives a restart.

## What it does

**Everything the user says goes through one path.** Answering a question and
pasting three paragraphs are the same operation: extract what is there, validate
it, report what is still missing. There is no separate bulk-import mode.

**References cannot be fabricated.** A configuration item, group or person is a
real record, and a change pointed at the wrong CI is worse than one pointed at
nothing. No `sys_id` can reach the change from model-authored text: the only
route in is `search_reference` → the user chooses → `select_reference`, which
refuses any id that was not in the candidate list it just offered. The agent is
told never to pick between a production and a non-production record itself, and
the fixtures are full of such twins so that path gets exercised constantly.

**Values remember where they came from.** Each field records whether the user
stated it or the agent inferred it. A value the user confirmed is never
overwritten silently -- the update comes back asking which version is right.
Without this the agent quietly destroys what you told it three turns ago, which
is the characteristic failure of this kind of assistant.

**Challenging is a rule catalog, not a paragraph of prompt.** Thirteen
deterministic rules run on every update: end before start, a backout plan that
is three words long, low risk on a production CI serving a critical service,
"no user impact" next to a plan that restarts the service, emergency wording in
a normal change, an assignment group that is not the CI's support group, a
window shorter than the steps it contains, a start inside the CAB lead time.
They are pure functions and always fire the same way.

On top of that, `review_change` runs a model pass that scores each free-text
field against a fixed rubric -- would a stranger understand this, could they act
on it, could they tell whether it worked -- and proposes rewrites. That judgment
needs a model; the thirteen rules do not, and are cheap enough to run constantly.

Findings carry a code and structured data rather than a finished sentence, so
the agent raises them in whatever language the conversation is in.

**Two tiers, and the user decides.** Blockers stop submission. Warnings are
raised once and never block -- the user knows their system. If they want to
submit with blockers open they must give a reason, which is recorded in the
change's work notes for the approver to see, not kept between them and the agent.

**Submission pauses for a human.** `submit_change` renders exactly what will be
created and interrupts. Nothing is sent until the user approves. The idempotency
key is *derived* from the content and thread rather than generated, because a
resumed graph re-runs the node from the top -- a fresh uuid would be a different
key on the second pass and would protect nothing.

## Layout

```
src/change_agent/
  schema.py        the change model; field metadata lives on the fields themselves
  state.py         field map with provenance, merge reducer, reviving readers
  validation.py    coercing and checking one proposed value
  assessment.py    running the rule catalog over current state
  rendering.py     one place that turns state into text
  payload.py       field map -> Table API payload, idempotency key
  prompts.py       the system prompt
  middleware.py    re-states the live change to the model every turn
  graph.py         the assembled agent
streamlit_app.py   chat UI, with both approval steps as buttons
  rules/           base.py, catalog.py (R001-R013), critic.py (the rubric pass)
  tools/           fields.py, references.py, review.py, submit.py
  servicenow/      Protocol, fixture-backed fake, Table API client, factory
```

There is no YAML field spec. Field metadata (label, guidance, required, which
table a reference points at) lives in `json_schema_extra` on the pydantic model,
so there is nothing to drift out of sync. Choice *values* are instance-specific
and come from `fixtures/choices.json`, regenerable from `sys_choice`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `anthropic:claude-sonnet-5` | `provider:model` for `init_chat_model` |
| `LLM_CRITIC_MODEL` | falls back to `LLM_MODEL` | model for the review pass |
| `SERVICENOW_MODE` | `fake` | `rest` to talk to a real instance |
| `SERVICENOW_INSTANCE_URL` / `_USER` / `_PASSWORD` | -- | required when `rest` |
| `CAB_LEAD_DAYS` | `5` | notice a normal change needs before CAB |

Submitted changes in fake mode are written to `.local/submitted/` so you can see
exactly what would have been sent.

## Trying it by hand

In Studio, these are the behaviours worth checking:

1. **Dump** a messy paragraph describing a change. Most fields should fill, and
   the agent should ask only about what is genuinely ambiguous.
2. **Say "the app server"**. It must search, show the production and development
   twins with their environments, and refuse to choose for you.
3. **Claim low risk on a production CI**, or write "no user impact" next to a
   plan that restarts a service. It should push back rather than accept it.
4. **Confirm a value, then contradict it.** It should ask which is right.
5. **Submit without a backout plan.** It should refuse, then ask for a reason if
   you insist -- and that reason should appear in the work notes.
6. **Approve a submission.** Check the change number, and that the JSON in
   `.local/submitted/` matches what the preview showed you.
7. **Run a whole conversation in French.** Findings should come back in French.

## Status

Built and tested against the fixtures. The `rest` client is written against the
documented Table API but has **not been run against a live instance** -- expect
to adjust the query strings in `servicenow/rest.py` on first contact, and check
`CONTEXT_FIELDS` matches your instance's dictionary.

Scope is normal changes only. Standard changes (template-driven) and emergency
changes (different approval path, its own challenge rules) are not implemented.
