# Local run ownership increment

Base798646f6839a9af9bf80a24272ab205b28a3cb5f, isolated source worktree.
Use Rust1.96.1 std::fs::File::try_lock to exclusively own one run in one trusted
local app directory. Retain the file inode permanently; drop/OS process death releases
the advisory lock. No heartbeat/expiry takeover of a live owner and no new dependency.
Acquire before journal open/create/read, config load, browser cleanup or provider
construction, holding through the execution and cleanup/trace-export lifetime.
Validate bounded run ID; canonicalize existing app root; reject symlinked lock directory
or lock file. The app directory must be local and controlled by its operator. This
is not a hostile shared-filesystem, distributed lease, Auth, or tenant isolation system.
The old30-second timestamp check is not ownership; remove that heuristic once OS lock
owns liveness. A stale running journal can resume immediately after actual owner death.

Proof: use exact installed toolchain and compile only dependency-free ownership module
with rustc --test. Actual child processes prove exclusivity for same run, distinct run
independence, release on graceful drop/crash, and inode retention; malformed IDs and
symlink paths fail. Runner source integration verified by inspection and rustfmt only
unless a lightweight check proves feasible. Do not claim full agent build/tests: V8,
Python/Wasmtime/browser dependencies are excluded under3.2GiB free. No provider calls.

## Implementation and local proof

`ownership.rs` is dependency-free. Its retained File guard encloses `run` and
`resume`; resume acquires before journal read, browser cleanup, config loader and
provider construction. New run accepts a pre-materialized configuration value under
its existing API; caller-side extraction precedes this boundary. Ownership starts
before executable registry/model construction and journal creation. No CLI API expanded.
Direct Journal APIs remain persistence primitives, not execution authority; other
writers must adopt the same ownership protocol before joining this execution lane.

Pinned `rustc1.96.1 (31fca3adb 2026-06-26)` compiled this exact module using std only.
Five standalone tests passed, including actual subprocess contention and killed-owner
release. The same five also passed through a wrapper module to verify the child-test
selector works when nested as in the agent crate. Runner fixture updates are written
but NOT executed: no Cargo/full-agent compilation or linked runtime proof was attempted.
Rustfmt checks on owned Rust paths and git diff --check pass. No model/provider calls.

To reproduce without Cargo (output under a chosen scratch directory):
`rustc +1.96.1 --edition 2024 --test crates/beater-agent/src/ownership.rs -o /tmp/tempera-run-ownership-tests`
then `/tmp/tempera-run-ownership-tests --test-threads=1`.

This local-filesystem advisory-lock slice does not pin model/tool configuration,
resolve remote unknown outcomes, enforce lifetime spend budgets, add tenant Auth/site
claims, implement editable goals or establish distributed execution safety. Those
remain separate reviewed followups. Existing locked files are never unlinked by the
runtime. The app directory and its ancestors must be operator-controlled; rejecting
static symlink paths does not claim protection from a hostile concurrent filesystem
administrator replacing paths/inodes or from network filesystem lock semantics.

## Required rollout condition

Quiesce all older pre-lock run/resume writers before deploying or resuming with this
version. Older binaries do not participate in the advisory-lock protocol; the new
lock (and removal of the old30-second heuristic) cannot protect mixed-version writers.
An operator must establish that old writers are stopped and account for their in-flight
work. This patch performs no automatic process kill, orphan subprocess cleanup, or
provider cancellation. Unknown provider/subprocess outcomes still require reconciliation;
configuration drift/replay policy is not fixed by owning the local run.
