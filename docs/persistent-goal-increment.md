# Persistent goal state: first runtime increment

Continue the existing temp.js durable runtime from local reviewed ownership commit
93123b8548161ba2d6d38ff9ead47d48f99c0c9a. This is an unpublished candidate, not main.
The first merchant journey needs a goal and its corrections to outlive an individual
model run. Do not create another agent loop or move Orders/Payments rules here.

## Source placement

Extract the existing beater-agent journal into a small workspace crate,
beater-journal, preserving its public API and existing tests. beater-agent re-exports
the same types and uses that single journal implementation. This enables direct tests
of the real persistence component without building unrelated V8/browser/Python tools.
Keep the exact existing dependency versions; no broad lockfile updates.

Add generic typed goal storage in that same SQLite journal. The goal has a stable
ID, four-coordinate scope, revision, objective, exact playbook identity/digests and
bounded parameters. These are declared local records. Scope supplied to this library
is not authenticated native admission; no HTTP endpoint or default tenant is added.

## Required behavior

- Create, read and revise a goal using full organization/project/environment/site
  scope. Require a bounded actor and caller-supplied request identity for mutations.
  Bind durable replay to scope, actor, operation, target and exact request content.
  Same key with changed content conflicts; replay does not refresh old state.
- Require expected revision on changes. Commit current state, immutable revision
  history, event and replay receipt in one transaction. Two competing corrections
  have one winner. Separate connections and reopening the journal preserve results.
- Persist milestone evidence references and the parameter dependencies they used.
  Supplier/budget corrections invalidate only dependent preparations; unrelated
  completed work remains recorded. Objective or playbook changes invalidate all
  affected milestones. Retain original evidence and history when invalidating.
- Evidence records remain declared/unverified references. This component cannot
  manufacture provider verification, balances, purchase authority or completed
  business outcomes. A model ending a turn does not complete the goal.
- Bound input sizes, identifiers, revisions, pagination and journal snapshots.
  Reject missing scope, malformed records and inconsistent dependency references.
  Do not serialize raw provider credentials or fetch resource URLs.
- Preserve legacy run/step journal behavior. Public runtime execution remains behind
  the existing run ownership guard. This first storage increment must not expose a
  new unqualified route to run/resume effects or a caller-controlled verified flag.

## Qualification

Run existing journal tests plus real SQLite tests for persistence/reopen, exact replay,
changed-body conflict, simultaneous revision changes, scope/actor isolation, atomic
rollback, immutable history and dependency invalidation. A merchant-shaped test saves
supplier A and a budget, records unaffected order mapping and a supplier-dependent
preview, reopens, switches to supplier B and lowers the budget, then proves the mapping
and historical preview survive while current preview requires refresh.

Run the real standalone journal crate, not a copied implementation or alternate mock
store. Limit builds to the owned target and selected lightweight crate. Full agent
linking, shared task HTTP/SDK/MCP/native admission, worker/event integration, actual
provider effects and end-to-end goal completion remain separate required steps.
This increment is a prerequisite for them, not a substitute for that end state.
