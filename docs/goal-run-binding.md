# Goal-run revision binding

`beater-journal` can atomically create a run bound to one exact, scoped goal
revision. The binding is versioned JSON in `runs.goal_binding`; `NULL` means a
legacy unbound run. A non-NULL value is never treated as unbound: malformed,
missing, cross-scope, or stale bindings make the run `needs_review`.

`create_goal_bound_run` checks the stored goal column and serialized body, then
inserts both the running row and immutable binding in one SQLite immediate
transaction. It has no rebind API. `gate_goal_run`, `start_step`, and
transitions to `running` or `completed` require that same exact current goal
revision. This stops future journaled work and resume/completion from silently
using a changed goal.

`beater-agent::run_for_goal` accepts a caller-chosen run ID, full storage scope,
goal ID, expected revision, agent name and prompt in `GoalRunRequest`. It acquires
the same process ownership lock as `run`/`resume`, creates the binding before
configuration loading, and enters the existing worker. Duplicate creation is an
error: after setup failure or interruption, call the existing `resume` with that
saved run ID. There is no second scheduler or agent loop.

Resume checks the goal before configuration, browser cleanup or the completed-run
fast path. It checks the loaded agent name before registry construction. Changes
during configuration are caught before starting a model step; changes during a
model call are caught before another tool step or run completion. The already
received result remains in the historical step journal. Even a previously completed
run is placed in review when explicitly resumed against a later goal revision.

Any goal revision change, including a milestone update, invalidates the old run's
binding. Continuing the new revision requires a separately created bound run; this
API does not silently rebind or rebuild the plan. Old NULL bindings remain the
legacy unbound path and confer no goal-isolation guarantee.

SQL extraction rejects non-text, malformed and oversized bindings and malformed
current goals before parsing. Limits are 4 KiB of binding JSON and 64 KiB of goal
JSON, counted as bytes including embedded NUL and multibyte characters. These
checks do not make the SQLite store tamper-proof.

Validation passes 116 `beater-agent` tests plus 26 `beater-journal` tests, with no
ignores, and all-target clippy with warnings denied. The worker tests use the
existing local model fixture for deterministic timing. They do not replace real
NVIDIA gateway or merchant-provider qualification.

This is trusted-local storage linkage only. It is not Auth admission, an
execution authority, provider identity, source proof, or completion evidence.
A correction can commit after a journal step is committed but before an
external provider observes the call; SQLite must not be held over that effect,
so this component does not claim to fence that unavoidable interval.
