# Interrupted tool replay contracts

An interrupted tool call can have an unknown external outcome. The runner now
journals a versioned declaration fingerprint before tool dispatch and requires an
exact original/current match before replaying that call. Changing the current tool
to `idempotent: true` cannot authorize replay of a formerly non-idempotent call.

The replay gate checks every prior call for the tool-use ID: tool name, exact input,
stable run/tool-use idempotency key, positive non-overflowing attempt, supported
contract version, valid digest and the original idempotence declaration. Missing,
legacy, malformed, changed or non-idempotent contracts leave the run `needs_review`.
Already completed tool results continue to be reused from the journal.

The fingerprint binds the tool kind, name, input schema and selected effective
configuration. Python binds its contained file path, bounded source digest and
timeout. Remote MCP binds endpoint, remote tool, retry, timeout, egress, auth
selector and session; discovered children also bind their parent declaration.
Browser tools bind provider, origins, timeout, session and secret selector names.
Sandbox/Wasmtime bind source, policy and entrypoint, plus the effective sandbox
service URL and lane where applicable. Rust builtins bind their declaration and
schema. Runtime secret values are not read or fingerprinted for this contract.

Canonical JSON sorts object keys and retains array order. Serialization checks
depth 32, 16,384 nodes and 1 MiB of output. Source acceptance is capped at 2 MiB and
requires regular contained files for local paths; Base64 validates decoded size
before allocating its exact output. Unsupported or oversized descriptors produce
no replay contract. This does not change fresh execution semantics, but a later
interruption without a usable contract requires review.

This is a trusted-local declaration/configuration drift check, not proof of
idempotence, authenticated provider implementation, effect authority or a complete
execution dependency pin. It does not cover transitive Python imports, secret
rotation, all runtime/environment changes, or mutation after registry construction.
Registry setup can load Python metadata and perform provider discovery before the
replay gate. The gate stops resumed tool dispatch; it does not guarantee no setup
work or network discovery. The local journal is not tamper-proof.

Validation: the actual `beater-agent` suite passes 110 tests with no ignores,
including successful unchanged replay, original false/current true rejection,
endpoint/schema/source drift, missing and malformed rows, ambiguous non-idempotent
calls, exact source-size boundaries and malformed Base64. All-target clippy with
warnings denied also passes. No external merchant provider, phone transport or
production deployment is qualified by these tests.
