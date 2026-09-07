# Local persistent-goal read projection

`beater-journal` exposes a metadata-only local projection over the existing
SQLite goal journal. `goal_summary` returns the full storage scope, stable goal ID, revision,
playbook identity and declared preparation counts in one SQLite snapshot.
`goal_activity_page` returns at most 100 immutable revision/event metadata
receipts, the same explicit scope/goal identity, and a scope/target-bound local continuation cursor.

Neither projection exposes objectives, parameters, evidence handles, actor IDs,
request IDs, request bodies or transcripts. They do not infer worker state,
completion, verification, provider outcome, business authority or execution
authority. The cursor is not signed, authenticated or cross-device portable.

All output remains trusted-local declaration data. The caller-provided scope is a
storage key, not Auth admission. An admitted projection must still establish native
identity, apply redaction, bind an authenticated cursor and reconcile verified
provider/worker receipts before presenting consequential state.

Activity pages read and validate at most 100 contiguous revisions plus one sentinel,
and report both their fixed local snapshot and returned range. A page that has not
reached its snapshot is explicitly incomplete; callers must follow its local cursor
before consuming later history. Missing, duplicate, gapped or malformed records fail
closed when their bounded page is requested rather than being reported as no activity
or completion.

`reached_snapshot` means this page ends at the chosen upper revision. It does not
prove that a caller consumed preceding pages: this local unsigned cursor can be
constructed by a caller. The returned range makes gaps visible to the consumer.
