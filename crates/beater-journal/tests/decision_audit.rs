use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

use beater_journal::{
    DecisionDomain, DecisionPackageV1, ExecutionAuthority, Goal, GoalMutation, GoalPatch,
    GoalScope, Journal, PlaybookIdentity, VerificationCeiling, verify_decision_audit_event,
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TestDir(std::path::PathBuf);
impl TestDir {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "beater-decision-audit-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scope() -> GoalScope {
    GoalScope {
        organization: "org-a".into(),
        project: "project-a".into(),
        environment: "test".into(),
        site: "site-a".into(),
    }
}
fn mutation(request_id: &str) -> GoalMutation {
    GoalMutation {
        actor: "operator-a".into(),
        request_id: request_id.into(),
        operation: "record".into(),
    }
}
fn goal() -> Goal {
    Goal {
        id: "goal-a".into(),
        scope: scope(),
        revision: 1,
        objective: "make a bounded decision".into(),
        playbook: PlaybookIdentity {
            id: "playbook-a".into(),
            digest: "a".repeat(64),
        },
        parameters: BTreeMap::new(),
    }
}
fn package(id: &str, revision: i64) -> DecisionPackageV1 {
    DecisionPackageV1 {
        version: 1,
        decision_id: id.into(),
        revision,
        domain: DecisionDomain::Software,
        question: "Should this local package be recorded?".into(),
        object_references: vec![],
        evidence: vec![],
        graph_context: None,
        policy_references: vec![],
        calculation_references: vec![],
        model_references: vec![],
        producer_references: vec![],
        options: vec!["record".into()],
        constraints: vec!["local-only".into()],
        rationale: "An append-only local record is useful for audit continuity.".into(),
        reviews: vec![],
        authorization_observations: vec![],
        effect_attempts: vec![],
        acknowledgements: vec![],
        outcomes: vec![],
        reconciliations: vec![],
        correction_of: None,
        successor_to: None,
        verification_ceiling: VerificationCeiling::RecordedOnly,
        execution_authority: ExecutionAuthority::None,
    }
}
fn setup(dir: &TestDir) -> (Journal, Goal) {
    let journal = Journal::open(dir.path()).unwrap();
    let goal = journal.create_goal(&mutation("create"), goal()).unwrap();
    journal
        .create_goal_bound_run(
            &goal.scope,
            &goal.id,
            goal.revision,
            "run-a",
            "agent-a",
            "decision work",
        )
        .unwrap();
    (journal, goal)
}

#[test]
fn create_reopen_correction_and_recorded_only_ceiling() {
    let dir = TestDir::new("reopen");
    let (journal, goal) = setup(&dir);
    let first = journal
        .append_decision_package("run-a", 0, "append-1", package("decision-a", 1))
        .unwrap();
    assert_eq!(first.current_revision, 1);
    assert_eq!(
        first.verification_ceiling,
        VerificationCeiling::RecordedOnly
    );
    assert_eq!(first.execution_authority, ExecutionAuthority::None);
    let mut correction = package("decision-a", 2);
    correction.correction_of = Some(beater_journal::decision_audit::DecisionRevisionReference {
        decision_id: "decision-a".into(),
        revision: 1,
    });
    let second = journal
        .append_decision_package("run-a", 1, "append-2", correction)
        .unwrap();
    assert_eq!(second.events.len(), 2);
    assert_eq!(second.events[0].package.revision, 1);
    drop(journal);
    let reopened = Journal::open(dir.path()).unwrap();
    let read = reopened.decision_audit(&goal.scope, "decision-a").unwrap();
    assert_eq!(read.current_revision, 2);
    assert_eq!(
        read.events[1].predecessor_event_digest.as_deref(),
        Some(read.events[0].event_digest.as_str())
    );
    let revised = reopened
        .revise_goal(
            &mutation("legitimate-goal-revision"),
            &goal.scope,
            &goal.id,
            goal.revision,
            GoalPatch {
                objective: Some("later goal revision".into()),
                playbook: None,
                parameters: None,
            },
        )
        .unwrap();
    assert_eq!(revised.revision, 2);
    assert_eq!(
        reopened
            .decision_audit(&goal.scope, "decision-a")
            .unwrap()
            .current_revision,
        2
    );
    assert!(
        reopened
            .append_decision_package("run-a", 2, "after-close", package("decision-a", 3))
            .is_err()
    );
}

#[test]
fn idempotency_conflict_and_rollback_leave_no_partial_rows() {
    let dir = TestDir::new("replay");
    let (journal, goal) = setup(&dir);
    journal
        .append_decision_package("run-a", 0, "same-key", package("decision-a", 1))
        .unwrap();
    assert_eq!(
        journal
            .append_decision_package("run-a", 0, "same-key", package("decision-a", 1))
            .unwrap()
            .current_revision,
        1
    );
    let mut changed = package("decision-a", 1);
    changed.question = "changed bytes".into();
    assert!(
        journal
            .append_decision_package("run-a", 0, "same-key", changed)
            .is_err()
    );
    assert!(
        journal
            .append_decision_package("run-a", 0, "new-key", package("decision-a", 1))
            .is_err()
    );
    assert!(
        journal
            .append_decision_package("run-a", 0, "bad-first", package("partial-a", 2))
            .is_err()
    );
    assert!(journal.decision_audit(&goal.scope, "partial-a").is_err());
}

#[test]
fn rejects_unbound_wrong_scope_stale_and_same_revision_goal_body_tamper() {
    let dir = TestDir::new("binding");
    let (journal, goal) = setup(&dir);
    journal.create_run("unbound-a", "agent-a", "work").unwrap();
    assert!(
        journal
            .append_decision_package("unbound-a", 0, "append", package("unbound-decision", 1))
            .is_err()
    );
    journal
        .append_decision_package("run-a", 0, "append", package("decision-a", 1))
        .unwrap();
    let wrong = GoalScope {
        site: "other-site".into(),
        ..goal.scope.clone()
    };
    assert!(journal.decision_audit(&wrong, "decision-a").is_err());
    let db = dir.path().join(".beater/journal.db");
    drop(journal);
    let connection = rusqlite::Connection::open(&db).unwrap();
    let mut changed = goal.clone();
    changed.objective = "same revision but different body".into();
    connection.execute("UPDATE goals SET body=?1 WHERE organization=?2 AND project=?3 AND environment=?4 AND site=?5 AND id=?6", rusqlite::params![serde_json::to_string(&changed).unwrap(), goal.scope.organization, goal.scope.project, goal.scope.environment, goal.scope.site, goal.id]).unwrap();
    drop(connection);
    let journal = Journal::open(dir.path()).unwrap();
    assert_eq!(
        journal
            .decision_audit(&goal.scope, "decision-a")
            .unwrap()
            .current_revision,
        1
    );
    // The live binding is still current by revision alone, but exact body digest
    // continuity rejects another append.
    assert!(
        journal
            .append_decision_package("run-a", 1, "stale", package("decision-a", 2))
            .is_err()
    );
}

#[test]
fn corruption_gaps_and_rewrites_fail_closed() {
    let dir = TestDir::new("corrupt");
    let (journal, goal) = setup(&dir);
    journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap();
    journal
        .append_decision_package("run-a", 1, "two", package("decision-a", 2))
        .unwrap();
    drop(journal);
    let connection = rusqlite::Connection::open(dir.path().join(".beater/journal.db")).unwrap();
    connection
        .execute(
            "DELETE FROM decision_audit_events WHERE decision_id='decision-a' AND sequence=1",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(
        Journal::open(dir.path())
            .unwrap()
            .decision_audit(&goal.scope, "decision-a")
            .is_err()
    );

    let rewrite_dir = TestDir::new("rewrite");
    let (journal, rewrite_goal) = setup(&rewrite_dir);
    journal
        .append_decision_package("run-a", 0, "one", package("rewritten-a", 1))
        .unwrap();
    drop(journal);
    let connection =
        rusqlite::Connection::open(rewrite_dir.path().join(".beater/journal.db")).unwrap();
    connection
        .execute(
            "UPDATE decision_audit_events SET package_body=x'7B7D' WHERE decision_id='rewritten-a'",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(
        Journal::open(rewrite_dir.path())
            .unwrap()
            .decision_audit(&rewrite_goal.scope, "rewritten-a")
            .is_err()
    );
}

#[test]
fn recorded_binding_tamper_and_event_relink_fail_closed() {
    let dir = TestDir::new("binding-tamper");
    let (journal, goal) = setup(&dir);
    journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap();
    journal
        .append_decision_package("run-a", 1, "two", package("decision-a", 2))
        .unwrap();
    drop(journal);
    let connection = rusqlite::Connection::open(dir.path().join(".beater/journal.db")).unwrap();
    connection
        .execute(
            "UPDATE decision_audit_events SET predecessor_event_digest=NULL WHERE decision_id='decision-a' AND sequence=2",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(
        Journal::open(dir.path())
            .unwrap()
            .decision_audit(&goal.scope, "decision-a")
            .is_err()
    );

    let snapshot_dir = TestDir::new("snapshot-tamper");
    let (journal, snapshot_goal) = setup(&snapshot_dir);
    journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap();
    drop(journal);
    let connection =
        rusqlite::Connection::open(snapshot_dir.path().join(".beater/journal.db")).unwrap();
    connection
        .execute(
            "UPDATE decision_audits SET goal_body=x'7B7D' WHERE decision_id='decision-a'",
            [],
        )
        .unwrap();
    drop(connection);
    assert!(
        Journal::open(snapshot_dir.path())
            .unwrap()
            .decision_audit(&snapshot_goal.scope, "decision-a")
            .is_err()
    );

    let transplant_dir = TestDir::new("scope-transplant");
    let (journal, transplant_goal) = setup(&transplant_dir);
    journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap();
    drop(journal);
    let connection =
        rusqlite::Connection::open(transplant_dir.path().join(".beater/journal.db")).unwrap();
    connection
        .execute(
            "UPDATE decision_audits SET site='other-site' WHERE decision_id='decision-a'",
            [],
        )
        .unwrap();
    connection
        .execute(
            "UPDATE decision_audit_events SET site='other-site' WHERE decision_id='decision-a'",
            [],
        )
        .unwrap();
    drop(connection);
    let other_scope = GoalScope {
        site: "other-site".into(),
        ..transplant_goal.scope.clone()
    };
    assert!(
        Journal::open(transplant_dir.path())
            .unwrap()
            .decision_audit(&other_scope, "decision-a")
            .is_err()
    );
}

#[test]
fn public_event_envelope_recomputes_and_rejects_tampering() {
    let dir = TestDir::new("envelope");
    let (journal, goal) = setup(&dir);
    let projection = journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap();
    let serialized = serde_json::to_string(&projection).unwrap();
    let decoded: beater_journal::DecisionAuditProjectionV1 =
        serde_json::from_str(&serialized).unwrap();
    verify_decision_audit_event(&decoded.events[0]).unwrap();
    for event in [
        {
            let mut value = decoded.events[0].clone();
            value.request_id = "other".into();
            value
        },
        {
            let mut value = decoded.events[0].clone();
            value.created_at += 1;
            value
        },
        {
            let mut value = decoded.events[0].clone();
            value.scope.site = "other-site".into();
            value
        },
        {
            let mut value = decoded.events[0].clone();
            value.goal_digest = "0".repeat(64);
            value
        },
    ] {
        assert!(verify_decision_audit_event(&event).is_err());
    }
    assert_eq!(goal.id, decoded.events[0].goal_id);
}

#[test]
fn rejects_unversioned_references_unsupported_authority_and_triggered_write_rollback() {
    let dir = TestDir::new("negative");
    let (journal, goal) = setup(&dir);
    let mut unversioned = package("unversioned-a", 1);
    unversioned
        .object_references
        .push(beater_journal::decision_audit::VersionedReference {
            locator: "object:a".into(),
            revision: None,
        });
    assert!(
        journal
            .append_decision_package("run-a", 0, "unversioned", unversioned)
            .is_err()
    );
    let mut value = serde_json::to_value(package("authority-a", 1)).unwrap();
    value["execution_authority"] = serde_json::json!("Granted");
    assert!(serde_json::from_value::<DecisionPackageV1>(value).is_err());
    journal
        .append_decision_package("run-a", 0, "initialize", package("initial-a", 1))
        .unwrap();
    let db = dir.path().join(".beater/journal.db");
    drop(journal);
    let connection = rusqlite::Connection::open(&db).unwrap();
    connection.execute("CREATE TRIGGER reject_decision_event BEFORE INSERT ON decision_audit_events BEGIN SELECT RAISE(ABORT, 'forced event write failure'); END", []).unwrap();
    drop(connection);
    let journal = Journal::open(dir.path()).unwrap();
    assert!(
        journal
            .append_decision_package("run-a", 0, "forced", package("rollback-a", 1))
            .is_err()
    );
    assert!(journal.decision_audit(&goal.scope, "rollback-a").is_err());
}

#[test]
fn standalone_verifier_rejects_self_consistent_invalid_envelopes() {
    use beater_journal::decision_audit::{DecisionAuditEventV1, verify_decision_audit_event};
    use sha2::{Digest, Sha256};

    // Independent caller-side encoding of the documented event preimage. A
    // matching hash alone must not validate inconsistent request/history fields.
    fn rehash(event: &mut DecisionAuditEventV1) {
        let mut preimage = serde_json::json!({
            "domain": "TEMPERA_DECISION_AUDIT_EVENT_V1",
            "scope": event.scope,
            "run_id": event.run_id,
            "goal_id": event.goal_id,
            "goal_revision": event.goal_revision,
            "goal_digest": event.goal_digest,
            "decision_id": event.package.decision_id,
            "sequence": event.sequence,
            "request_id": event.request_id,
            "request_digest": event.request_digest,
            "package_digest": event.package_digest,
            "predecessor_event_digest": event.predecessor_event_digest,
            "created_at": event.created_at,
        });
        preimage.sort_all_objects();
        event.event_digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&preimage).unwrap())
        );
    }

    let dir = TestDir::new("standalone-invariants");
    let (journal, _) = setup(&dir);
    let first = journal
        .append_decision_package("run-a", 0, "one", package("decision-a", 1))
        .unwrap()
        .events
        .remove(0);
    let second = journal
        .append_decision_package("run-a", 1, "two", package("decision-a", 2))
        .unwrap()
        .events
        .remove(1);
    for original in [&first, &second] {
        let mut recomputed = original.clone();
        rehash(&mut recomputed);
        assert_eq!(recomputed.event_digest, original.event_digest);
        verify_decision_audit_event(&recomputed).unwrap();
    }
    let mut invalid = Vec::new();
    let mut event = first.clone();
    event.request_digest = "b".repeat(64);
    invalid.push(event);
    let mut event = first.clone();
    event.goal_digest = "A".repeat(64);
    invalid.push(event);
    for sequence in [0, -1, 2, 1_000_000_001] {
        let mut event = first.clone();
        event.sequence = sequence;
        invalid.push(event);
    }
    let mut event = first.clone();
    event.predecessor_event_digest = Some("b".repeat(64));
    invalid.push(event);
    let mut event = second.clone();
    event.predecessor_event_digest = None;
    invalid.push(event);
    let mut event = second.clone();
    event.predecessor_event_digest = Some("not-a-digest".into());
    invalid.push(event);
    for mut event in invalid {
        rehash(&mut event);
        assert!(verify_decision_audit_event(&event).is_err());
    }
}

#[test]
fn concurrent_compare_and_swap_has_one_winner() {
    let dir = TestDir::new("cas");
    let (_, goal) = setup(&dir);
    let path = Arc::new(dir.path().to_path_buf());
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for request in ["left", "right"] {
        let path = path.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            Journal::open(&path)
                .unwrap()
                .append_decision_package("run-a", 0, request, package("decision-a", 1))
                .is_ok()
        }));
    }
    assert_eq!(
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|won| *won)
            .count(),
        1
    );
    assert_eq!(
        Journal::open(dir.path())
            .unwrap()
            .decision_audit(&goal.scope, "decision-a")
            .unwrap()
            .current_revision,
        1
    );
}
