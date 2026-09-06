//! Local, declaration-only read projections.  These records are not Auth
//! admission, worker state, completion evidence, or execution authority.

use anyhow::{Context, Result, ensure};
use rusqlite::{OptionalExtension, params};

use crate::{
    Goal, GoalScope, Journal, MAX_REVISION, Milestone, PlaybookIdentity, valid_identifier,
    validate_goal, validate_milestone_shape, validate_scope,
};

const MAX_RECEIPTS: usize = 100;

/// Counts application-declared preparation only.  There is no completed state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PreparationCounts {
    pub planned: u32,
    pub prepared: u32,
    pub needs_refresh: u32,
}

/// Metadata-only local snapshot; it intentionally excludes objective, parameters,
/// evidence handles, actors, request bodies, and every execution/completion claim.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GoalSummary {
    pub scope: GoalScope,
    pub id: String,
    pub revision: i64,
    pub playbook: PlaybookIdentity,
    pub preparations: PreparationCounts,
}

/// An unsigned, local SQLite continuation.  It is scope/target bound to prevent
/// accidental reuse, but does not authenticate a caller or travel across devices.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GoalActivityCursor {
    pub scope: GoalScope,
    pub goal_id: String,
    pub after_revision: i64,
    pub through_revision: i64,
}

/// Immutable journal metadata only; it omits revision body, actors and requests.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GoalActivityReceipt {
    pub revision: i64,
    pub event: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GoalActivityPage {
    pub scope: GoalScope,
    pub goal_id: String,
    pub snapshot_revision: i64,
    pub returned_after_revision: i64,
    pub returned_through_revision: i64,
    /// Reaching this local snapshot is not authenticated workflow completion.
    pub reached_snapshot: bool,
    pub receipts: Vec<GoalActivityReceipt>,
    pub next_cursor: Option<GoalActivityCursor>,
}

fn valid_goal_event(event: &str) -> bool {
    matches!(event, "created" | "revised" | "milestone_recorded")
}

fn validate_cursor(scope: &GoalScope, goal_id: &str, cursor: &GoalActivityCursor) -> Result<()> {
    validate_scope(scope)?;
    ensure!(valid_identifier(goal_id), "invalid goal id");
    ensure!(
        cursor.scope == *scope && cursor.goal_id == goal_id,
        "cursor scope or target mismatch"
    );
    ensure!(
        cursor.through_revision >= 1 && cursor.through_revision <= MAX_REVISION,
        "invalid cursor snapshot"
    );
    ensure!(
        cursor.after_revision >= 0 && cursor.after_revision < cursor.through_revision,
        "invalid cursor continuation"
    );
    Ok(())
}

fn count(milestone: &Milestone, counts: &mut PreparationCounts) -> Result<()> {
    match milestone.preparation.as_str() {
        "planned" => {
            counts.planned = counts
                .planned
                .checked_add(1)
                .context("preparation count overflow")?
        }
        "prepared" => {
            counts.prepared = counts
                .prepared
                .checked_add(1)
                .context("preparation count overflow")?
        }
        "needs_refresh" => {
            counts.needs_refresh = counts
                .needs_refresh
                .checked_add(1)
                .context("preparation count overflow")?
        }
        _ => unreachable!("validated milestone preparation"),
    }
    Ok(())
}

impl Journal {
    /// Reads a goal and bounded current milestones in one journal SQLite snapshot.
    pub fn goal_summary(&self, scope: &GoalScope, goal_id: &str) -> Result<GoalSummary> {
        validate_scope(scope)?;
        ensure!(valid_identifier(goal_id), "invalid goal id");
        // The journal connection is configured for immediate transactions. This
        // shares that established SQLite snapshot behavior without changing it.
        let tx = self.conn.unchecked_transaction()?;
        let (stored_revision, body): (i64, String) = tx.query_row(
            "SELECT revision,body FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5",
            params![scope.organization, scope.project, scope.environment, scope.site, goal_id], |r| Ok((r.get(0)?, r.get(1)?)),
        ).optional()?.with_context(|| format!("no goal {goal_id} in scope"))?;
        let goal: Goal = serde_json::from_str(&body)?;
        validate_goal(&goal)?;
        ensure!(
            goal.scope == *scope && goal.id == goal_id,
            "corrupt scoped goal row"
        );
        ensure!(
            goal.revision == stored_revision,
            "corrupt goal revision column"
        );
        let mut counts = PreparationCounts::default();
        let mut stmt = tx.prepare("SELECT milestone_id,body FROM goal_milestones WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 ORDER BY milestone_id LIMIT ?6")?;
        let rows = stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                goal_id,
                crate::MAX_MILESTONES as i64 + 1
            ],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )?;
        let mut seen = 0usize;
        for row in rows {
            let (milestone_id, body) = row?;
            seen += 1;
            ensure!(
                seen <= crate::MAX_MILESTONES,
                "too many persisted milestones"
            );
            let milestone: Milestone = serde_json::from_str(&body)?;
            validate_milestone_shape(&milestone)?;
            ensure!(milestone.id == milestone_id, "corrupt milestone id column");
            if !milestone.requires_refresh {
                ensure!(
                    milestone
                        .parameter_dependencies
                        .iter()
                        .all(|dependency| goal.parameters.contains_key(dependency)),
                    "inconsistent current milestone dependency"
                );
            }
            count(&milestone, &mut counts)?;
        }
        drop(stmt);
        tx.commit()?;
        Ok(GoalSummary {
            scope: goal.scope,
            id: goal.id,
            revision: goal.revision,
            playbook: goal.playbook,
            preparations: counts,
        })
    }

    /// Projects immutable goal-revision/event metadata. Missing, duplicate, gapped,
    /// or malformed history is an error rather than an inferred inactive state.
    pub fn goal_activity_page(
        &self,
        scope: &GoalScope,
        goal_id: &str,
        cursor: Option<GoalActivityCursor>,
    ) -> Result<GoalActivityPage> {
        validate_scope(scope)?;
        ensure!(valid_identifier(goal_id), "invalid goal id");
        let tx = self.conn.unchecked_transaction()?;
        let current: i64 = tx.query_row("SELECT revision FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5", params![scope.organization,scope.project,scope.environment,scope.site,goal_id], |r| r.get(0)).optional()?.with_context(|| format!("no goal {goal_id} in scope"))?;
        ensure!(
            (1..=MAX_REVISION).contains(&current),
            "corrupt goal revision"
        );
        let cursor = match cursor {
            Some(cursor) => {
                validate_cursor(scope, goal_id, &cursor)?;
                ensure!(
                    cursor.through_revision <= current,
                    "cursor snapshot exceeds current goal revision"
                );
                cursor
            }
            None => GoalActivityCursor {
                scope: scope.clone(),
                goal_id: goal_id.into(),
                after_revision: 0,
                through_revision: current,
            },
        };

        let mut stmt = tx.prepare("SELECT revision,created_at FROM goal_revisions WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND revision>?6 AND revision<=?7 ORDER BY revision LIMIT ?8")?;
        let rows = stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                goal_id,
                cursor.after_revision,
                cursor.through_revision,
                (MAX_RECEIPTS + 1) as i64
            ],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
        )?;
        let mut revisions = Vec::new();
        let mut expected = cursor
            .after_revision
            .checked_add(1)
            .context("history revision overflow")?;
        for row in rows {
            let (revision, created_at) = row?;
            ensure!(
                revision == expected && created_at >= 0,
                "goal history is missing, duplicate, or truncated"
            );
            revisions.push((revision, created_at));
            expected = expected
                .checked_add(1)
                .context("history revision overflow")?;
        }
        drop(stmt);
        ensure!(
            revisions.len()
                == (cursor.through_revision - cursor.after_revision).min((MAX_RECEIPTS + 1) as i64)
                    as usize,
            "goal history is missing, duplicate, or truncated"
        );
        let has_more = revisions.len() > MAX_RECEIPTS;
        revisions.truncate(MAX_RECEIPTS);
        let mut receipts = Vec::with_capacity(revisions.len());
        for (revision, created_at) in revisions {
            let mut events = tx.prepare("SELECT event,created_at FROM goal_events WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND revision=?6 LIMIT 2")?;
            let values = events
                .query_map(
                    params![
                        scope.organization,
                        scope.project,
                        scope.environment,
                        scope.site,
                        goal_id,
                        revision
                    ],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            ensure!(
                values.len() == 1 && valid_goal_event(&values[0].0) && values[0].1 == created_at,
                "corrupt goal activity receipt"
            );
            receipts.push(GoalActivityReceipt {
                revision,
                event: values.into_iter().next().context("missing goal event")?.0,
                created_at,
            });
        }
        let returned_through_revision = receipts
            .last()
            .context("bounded receipt page unexpectedly empty")?
            .revision;
        let reached_snapshot = !has_more && returned_through_revision == cursor.through_revision;
        let next_cursor = (!reached_snapshot).then(|| GoalActivityCursor {
            after_revision: returned_through_revision,
            ..cursor.clone()
        });
        tx.commit()?;
        Ok(GoalActivityPage {
            scope: scope.clone(),
            goal_id: goal_id.into(),
            snapshot_revision: cursor.through_revision,
            returned_after_revision: cursor.after_revision,
            returned_through_revision,
            reached_snapshot,
            receipts,
            next_cursor,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GoalMutation, Milestone};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "beater-projection-{}-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn scope() -> GoalScope {
        GoalScope {
            organization: "org".into(),
            project: "project".into(),
            environment: "dev".into(),
            site: "site".into(),
        }
    }
    fn mutation(id: usize) -> GoalMutation {
        GoalMutation {
            actor: "local-reader".into(),
            request_id: format!("request-{id}"),
            operation: "recorded".into(),
        }
    }
    fn goal(scope: GoalScope) -> Goal {
        Goal {
            id: "goal-1".into(),
            scope,
            revision: 1,
            objective: "private objective".into(),
            playbook: PlaybookIdentity {
                id: "playbook-v1".into(),
                digest: "a".repeat(64),
            },
            parameters: [("supplier".into(), serde_json::json!("private"))]
                .into_iter()
                .collect::<BTreeMap<_, _>>(),
        }
    }
    fn milestone() -> Milestone {
        Milestone {
            id: "preview".into(),
            preparation: "prepared".into(),
            evidence_references: vec!["artifact:private".into()],
            parameter_dependencies: BTreeSet::new(),
            requires_refresh: false,
        }
    }

    #[test]
    fn summary_is_single_snapshot_and_omits_raw_goal_data() {
        let temp = TempDir::new();
        let journal = Journal::open(temp.path()).unwrap();
        let scope = scope();
        let mut current = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        current = journal
            .record_milestone(
                &mutation(1),
                &scope,
                "goal-1",
                current.revision,
                milestone(),
            )
            .unwrap();
        let summary = journal.goal_summary(&scope, "goal-1").unwrap();
        assert_eq!(summary.scope, scope);
        assert_eq!(summary.id, "goal-1");
        assert_eq!(summary.revision, current.revision);
        assert_eq!(summary.preparations.prepared, 1);
        let exposed = serde_json::to_string(&summary).unwrap();
        for forbidden in [
            "private objective",
            "supplier",
            "artifact:private",
            "local-reader",
            "request-",
        ] {
            assert!(!exposed.contains(forbidden));
        }
    }

    #[test]
    fn receipt_pages_are_snapshot_stable_across_reopen_and_concurrent_append() {
        let temp = TempDir::new();
        let scope = scope();
        let journal = Journal::open(temp.path()).unwrap();
        let mut current = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        for index in 1..=120 {
            current = journal
                .record_milestone(
                    &mutation(index),
                    &scope,
                    "goal-1",
                    current.revision,
                    milestone(),
                )
                .unwrap();
        }
        let first = journal.goal_activity_page(&scope, "goal-1", None).unwrap();
        assert_eq!(first.scope, scope);
        assert_eq!(first.goal_id, "goal-1");
        assert_eq!(first.receipts.len(), 100);
        assert_eq!(first.returned_after_revision, 0);
        assert_eq!(first.returned_through_revision, 100);
        assert_eq!(first.snapshot_revision, current.revision);
        assert!(!first.reached_snapshot);
        let cursor = first.next_cursor.unwrap();
        let snapshot = cursor.through_revision;
        drop(journal);
        let app_path = temp.path().to_path_buf();
        let writer_scope = scope.clone();
        thread::spawn(move || {
            let writer = Journal::open(&app_path).unwrap();
            let current = writer.goal(&writer_scope, "goal-1").unwrap();
            writer
                .record_milestone(
                    &mutation(121),
                    &writer_scope,
                    "goal-1",
                    current.revision,
                    milestone(),
                )
                .unwrap();
        })
        .join()
        .unwrap();
        let reopened = Journal::open(temp.path()).unwrap();
        let second = reopened
            .goal_activity_page(&scope, "goal-1", Some(cursor))
            .unwrap();
        assert_eq!(second.receipts.first().unwrap().revision, 101);
        assert_eq!(second.receipts.last().unwrap().revision, snapshot);
        assert!(second.next_cursor.is_none());
        assert!(second.reached_snapshot);
        let foreign = GoalActivityCursor {
            scope: GoalScope {
                site: "other".into(),
                ..scope.clone()
            },
            goal_id: "goal-1".into(),
            after_revision: 0,
            through_revision: snapshot,
        };
        assert!(
            reopened
                .goal_activity_page(&scope, "goal-1", Some(foreign))
                .is_err()
        );
    }

    #[test]
    fn gaps_or_duplicate_events_fail_closed() {
        let temp = TempDir::new();
        let journal = Journal::open(temp.path()).unwrap();
        let scope = scope();
        let current = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        journal
            .record_milestone(
                &mutation(1),
                &scope,
                "goal-1",
                current.revision,
                milestone(),
            )
            .unwrap();
        journal
            .conn
            .execute("DELETE FROM goal_events WHERE revision=2", [])
            .unwrap();
        assert!(journal.goal_activity_page(&scope, "goal-1", None).is_err());
        journal.conn.execute("INSERT INTO goal_events(organization,project,environment,site,goal_id,revision,event,created_at) VALUES('org','project','dev','site','goal-1',2,'duplicate',0)",[]).unwrap();
        journal.conn.execute("INSERT INTO goal_events(organization,project,environment,site,goal_id,revision,event,created_at) VALUES('org','project','dev','site','goal-1',2,'duplicate-again',0)",[]).unwrap();
        assert!(journal.goal_activity_page(&scope, "goal-1", None).is_err());
    }

    #[test]
    fn later_history_corruption_waits_for_its_bounded_page() {
        let temp = TempDir::new();
        let journal = Journal::open(temp.path()).unwrap();
        let scope = scope();
        let mut current = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        for index in 1..=102 {
            current = journal
                .record_milestone(
                    &mutation(index),
                    &scope,
                    "goal-1",
                    current.revision,
                    milestone(),
                )
                .unwrap();
        }
        journal
            .conn
            .execute("DELETE FROM goal_events WHERE revision=102", [])
            .unwrap();
        let first = journal.goal_activity_page(&scope, "goal-1", None).unwrap();
        assert_eq!(first.receipts.len(), 100);
        assert!(!first.reached_snapshot);
        assert!(
            journal
                .goal_activity_page(&scope, "goal-1", first.next_cursor)
                .is_err()
        );
    }

    #[test]
    fn summary_rejects_column_mismatch_poisoned_count_and_invalid_event() {
        let temp = TempDir::new();
        let journal = Journal::open(temp.path()).unwrap();
        let scope = scope();
        let current = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        journal
            .conn
            .execute("UPDATE goals SET revision=9 WHERE id='goal-1'", [])
            .unwrap();
        assert!(journal.goal_summary(&scope, "goal-1").is_err());
        journal
            .conn
            .execute("UPDATE goals SET revision=1 WHERE id='goal-1'", [])
            .unwrap();
        journal.conn.execute("INSERT INTO goal_milestones(organization,project,environment,site,goal_id,milestone_id,body,created_at) VALUES('org','project','dev','site','goal-1','different',?1,0)", params![serde_json::to_string(&milestone()).unwrap()]).unwrap();
        assert!(journal.goal_summary(&scope, "goal-1").is_err());
        journal
            .conn
            .execute("DELETE FROM goal_milestones", [])
            .unwrap();
        for index in 0..=crate::MAX_MILESTONES {
            journal.conn.execute("INSERT INTO goal_milestones(organization,project,environment,site,goal_id,milestone_id,body,created_at) VALUES('org','project','dev','site','goal-1',?1,?2,0)", params![format!("m-{index}"), serde_json::to_string(&Milestone { id:format!("m-{index}"), ..milestone() }).unwrap()]).unwrap();
        }
        assert!(journal.goal_summary(&scope, "goal-1").is_err());
        journal
            .conn
            .execute("DELETE FROM goal_milestones", [])
            .unwrap();
        journal
            .conn
            .execute(
                "UPDATE goal_events SET event='unexpected' WHERE revision=1",
                [],
            )
            .unwrap();
        assert!(journal.goal_activity_page(&scope, "goal-1", None).is_err());
        assert_eq!(
            journal.goal(&scope, "goal-1").unwrap().revision,
            current.revision
        );
    }

    #[test]
    fn truncated_page_tail_and_mismatched_event_timestamp_fail_immediately() {
        let temp = TempDir::new();
        let journal = Journal::open(temp.path()).unwrap();
        let scope = scope();
        let initial = journal
            .create_goal(&mutation(0), goal(scope.clone()))
            .unwrap();
        journal
            .record_milestone(
                &mutation(1),
                &scope,
                "goal-1",
                initial.revision,
                milestone(),
            )
            .unwrap();
        journal
            .conn
            .execute(
                "UPDATE goal_events SET created_at=created_at+1 WHERE revision=2",
                [],
            )
            .unwrap();
        assert!(journal.goal_activity_page(&scope, "goal-1", None).is_err());
        journal
            .conn
            .execute(
                "UPDATE goal_events SET created_at=created_at-1 WHERE revision=2",
                [],
            )
            .unwrap();
        journal
            .conn
            .execute("DELETE FROM goal_revisions WHERE revision=2", [])
            .unwrap();
        // A short prefix cannot be returned with a misleading continuation when
        // its expected tail should fit within this same bounded page.
        assert!(journal.goal_activity_page(&scope, "goal-1", None).is_err());
    }
}
