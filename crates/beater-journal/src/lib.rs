//! The durability contract (ARCHITECTURE.md §5): a `started` row is committed
//! before anything executes; `completed` + result written after. Resume
//! rebuilds state from completed steps and re-runs only what's safe.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};

const JOURNAL_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

pub struct Journal {
    conn: Connection,
}

#[derive(Debug)]
pub struct RunRow {
    pub id: String,
    pub agent: String,
    pub status: String,
    pub input: String,
    #[allow(dead_code)]
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug)]
pub struct StepRow {
    #[allow(dead_code)]
    pub seq: i64,
    pub kind: String,   // llm_call | tool_call
    pub status: String, // started | completed | failed
    pub request: serde_json::Value,
    pub result: Option<serde_json::Value>,
    #[allow(dead_code)]
    pub tool_name: Option<String>,
    pub tool_use_id: Option<String>,
    pub attempt: i64,
}

#[derive(Debug)]
pub struct StepPartialRow {
    pub seq: i64,
    pub ordinal: i64,
    pub kind: String,
    pub payload: serde_json::Value,
    #[allow(dead_code)]
    pub created_at: i64,
}

/// The four coordinates are storage isolation keys.  They deliberately make no
/// claim that the caller has been admitted or authenticated by a native service.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct GoalScope {
    pub organization: String,
    pub project: String,
    pub environment: String,
    pub site: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlaybookIdentity {
    pub id: String,
    pub digest: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Goal {
    pub id: String,
    pub scope: GoalScope,
    pub revision: i64,
    pub objective: String,
    pub playbook: PlaybookIdentity,
    pub parameters: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GoalPatch {
    pub objective: Option<String>,
    pub playbook: Option<PlaybookIdentity>,
    pub parameters: Option<BTreeMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GoalMutation {
    pub actor: String,
    pub request_id: String,
    pub operation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Milestone {
    pub id: String,
    /// An application-declared preparation state.  There is intentionally no
    /// caller-controlled verified or business-completed state in this store.
    pub preparation: String,
    /// Opaque local handles only. Credential material and provider retrieval
    /// belong at an admitted runtime boundary; this lexical guard is not a
    /// secret detector or provider-verification mechanism.
    pub evidence_references: Vec<String>,
    pub parameter_dependencies: BTreeSet<String>,
    /// Store-owned and monotonic. Clearing it awaits an admitted verification
    /// operation, which this generic persistence component does not expose.
    pub requires_refresh: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MilestoneRevision {
    pub goal_revision: i64,
    pub milestone: Milestone,
    pub actor: String,
    pub request_id: String,
    pub operation: String,
    pub event: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GoalRevision {
    pub revision: i64,
    pub goal: Goal,
    pub actor: String,
    pub request_id: String,
    pub operation: String,
    pub created_at: i64,
}

/// A caller can capture `goal.revision` once as the upper bound, then use the
/// final returned revision as the next `after_revision` for a stable history.
#[derive(Debug, Clone, Copy)]
pub struct HistoryWindow {
    pub after_revision: i64,
    pub through_revision: i64,
    pub limit: usize,
}

const MAX_IDENTIFIER: usize = 128;
const MAX_TEXT: usize = 4_096;
const MAX_PARAMETERS: usize = 32;
const MAX_PARAMETER_JSON: usize = 8_192;
const MAX_ALL_PARAMETER_JSON: usize = 32_768;
const MAX_PARAMETER_DEPTH: usize = 16;
const MAX_PARAMETER_NODES: usize = 1_024;
const MAX_MILESTONES: usize = 128;
const MAX_EVIDENCE: usize = 32;
const MAX_REVISION: i64 = 1_000_000_000;

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
}

fn valid_playbook_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn validate_scope(scope: &GoalScope) -> Result<()> {
    ensure!(
        valid_identifier(&scope.organization),
        "invalid organization scope"
    );
    ensure!(valid_identifier(&scope.project), "invalid project scope");
    ensure!(
        valid_identifier(&scope.environment),
        "invalid environment scope"
    );
    ensure!(valid_identifier(&scope.site), "invalid site scope");
    Ok(())
}

fn validate_mutation(mutation: &GoalMutation) -> Result<()> {
    ensure!(valid_identifier(&mutation.actor), "invalid actor");
    ensure!(
        valid_identifier(&mutation.request_id),
        "invalid request identity"
    );
    ensure!(valid_identifier(&mutation.operation), "invalid operation");
    Ok(())
}

fn validate_parameters(parameters: &BTreeMap<String, serde_json::Value>) -> Result<()> {
    ensure!(
        parameters.len() <= MAX_PARAMETERS,
        "too many goal parameters"
    );
    let mut remaining_nodes = MAX_PARAMETER_NODES;
    for (key, value) in parameters {
        ensure!(valid_identifier(key), "invalid parameter key");
        ensure!(!credential_label(key), "credential-labelled parameter key");
        validate_parameter_value(value, 0, &mut remaining_nodes)?;
        ensure!(
            serde_json::to_string(value)?.len() <= MAX_PARAMETER_JSON,
            "parameter too large"
        );
    }
    ensure!(
        serde_json::to_string(parameters)?.len() <= MAX_ALL_PARAMETER_JSON,
        "goal parameters too large"
    );
    Ok(())
}

fn credential_label(value: &str) -> bool {
    let normalized: String = value
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    [
        "token",
        "secret",
        "password",
        "credential",
        "apikey",
        "authorization",
        "accesskey",
        "privatekey",
    ]
    .iter()
    .any(|word| normalized.contains(word))
}

fn validate_parameter_value(
    value: &serde_json::Value,
    depth: usize,
    remaining_nodes: &mut usize,
) -> Result<()> {
    ensure!(depth <= MAX_PARAMETER_DEPTH, "parameter nesting too deep");
    ensure!(*remaining_nodes > 0, "too many parameter values");
    *remaining_nodes -= 1;
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                ensure!(key.len() <= MAX_IDENTIFIER, "parameter object key too long");
                ensure!(!credential_label(key), "credential-labelled parameter key");
                validate_parameter_value(value, depth + 1, remaining_nodes)?;
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                validate_parameter_value(value, depth + 1, remaining_nodes)?;
            }
        }
        serde_json::Value::String(value) => ensure!(
            value.len() <= MAX_PARAMETER_JSON
                && !value
                    .trim_start()
                    .to_ascii_lowercase()
                    .starts_with("bearer ")
                && !value.trim_start().to_ascii_lowercase().starts_with("key="),
            "explicit credential marker in parameter"
        ),
        _ => {}
    }
    Ok(())
}

fn valid_evidence_reference(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
        && !credential_label(value)
}

fn validate_goal(goal: &Goal) -> Result<()> {
    validate_scope(&goal.scope)?;
    ensure!(valid_identifier(&goal.id), "invalid goal id");
    ensure!(
        !goal.objective.is_empty() && goal.objective.len() <= MAX_TEXT,
        "invalid objective"
    );
    ensure!(
        valid_identifier(&goal.playbook.id) && valid_playbook_digest(&goal.playbook.digest),
        "invalid playbook identity"
    );
    ensure!(
        (1..=MAX_REVISION).contains(&goal.revision),
        "invalid goal revision"
    );
    validate_parameters(&goal.parameters)
}

fn validate_milestone_shape(milestone: &Milestone) -> Result<()> {
    ensure!(valid_identifier(&milestone.id), "invalid milestone id");
    ensure!(
        matches!(
            milestone.preparation.as_str(),
            "planned" | "prepared" | "needs_refresh"
        ),
        "invalid milestone preparation"
    );
    ensure!(
        milestone.evidence_references.len() <= MAX_EVIDENCE,
        "too many evidence references"
    );
    ensure!(
        milestone.requires_refresh == (milestone.preparation == "needs_refresh"),
        "inconsistent milestone refresh state"
    );
    ensure!(
        milestone.parameter_dependencies.len() <= MAX_PARAMETERS,
        "too many milestone dependencies"
    );
    for reference in &milestone.evidence_references {
        ensure!(
            valid_evidence_reference(reference),
            "invalid opaque evidence reference"
        );
    }
    for dependency in &milestone.parameter_dependencies {
        ensure!(
            valid_identifier(dependency),
            "inconsistent milestone dependency"
        );
    }
    Ok(())
}

fn validate_patch(patch: &GoalPatch) -> Result<()> {
    if let Some(objective) = &patch.objective {
        ensure!(
            !objective.is_empty() && objective.len() <= MAX_TEXT,
            "invalid objective"
        );
    }
    if let Some(playbook) = &patch.playbook {
        ensure!(
            valid_identifier(&playbook.id) && valid_playbook_digest(&playbook.digest),
            "invalid playbook identity"
        );
    }
    if let Some(parameters) = &patch.parameters {
        validate_parameters(parameters)?;
    }
    Ok(())
}

fn validate_history_window(window: HistoryWindow) -> Result<()> {
    ensure!((1..=256).contains(&window.limit), "invalid history limit");
    ensure!(
        (0..=MAX_REVISION).contains(&window.after_revision)
            && (0..=MAX_REVISION).contains(&window.through_revision)
            && window.after_revision <= window.through_revision,
        "invalid history window"
    );
    Ok(())
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Journal {
    pub fn open(app_dir: &Path) -> Result<Self> {
        let dir = app_dir.join(".beater");
        std::fs::create_dir_all(&dir)?;
        let mut conn = Connection::open(dir.join("journal.db"))?;
        conn.busy_timeout(JOURNAL_BUSY_TIMEOUT)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        let journal_mode: String =
            conn.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        ensure!(
            journal_mode.eq_ignore_ascii_case("wal"),
            "failed to enable WAL journal mode: got {journal_mode}"
        );
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.set_transaction_behavior(TransactionBehavior::Immediate);
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS runs(
               id TEXT PRIMARY KEY, agent TEXT NOT NULL, status TEXT NOT NULL,
               input TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS steps(
               run_id TEXT NOT NULL, seq INTEGER NOT NULL,
               kind TEXT NOT NULL, status TEXT NOT NULL,
               request TEXT NOT NULL, result TEXT,
               tool_name TEXT, tool_use_id TEXT,
               attempt INTEGER NOT NULL DEFAULT 1,
               started_at INTEGER NOT NULL, finished_at INTEGER,
               PRIMARY KEY(run_id, seq));
             CREATE TABLE IF NOT EXISTS step_partials(
               run_id TEXT NOT NULL, seq INTEGER NOT NULL,
               ordinal INTEGER NOT NULL,
               kind TEXT NOT NULL,
               payload TEXT NOT NULL,
               created_at INTEGER NOT NULL,
               PRIMARY KEY(run_id, seq, ordinal),
               FOREIGN KEY(run_id, seq) REFERENCES steps(run_id, seq));",
        )?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS goals(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, updated_at INTEGER NOT NULL,
                PRIMARY KEY(organization, project, environment, site, id));
              CREATE TABLE IF NOT EXISTS goal_revisions(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                goal_id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL,
                actor TEXT NOT NULL, request_id TEXT NOT NULL, operation TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization, project, environment, site, goal_id, revision));
              CREATE TABLE IF NOT EXISTS goal_events(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                goal_id TEXT NOT NULL, revision INTEGER NOT NULL, event TEXT NOT NULL, created_at INTEGER NOT NULL);
              CREATE TABLE IF NOT EXISTS goal_replays(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                actor TEXT NOT NULL, operation TEXT NOT NULL, target TEXT NOT NULL, request_id TEXT NOT NULL,
                request_body TEXT NOT NULL, response_body TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization, project, environment, site, actor, operation, target, request_id));
              CREATE TABLE IF NOT EXISTS goal_milestones(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                goal_id TEXT NOT NULL, milestone_id TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization, project, environment, site, goal_id, milestone_id));
              CREATE TABLE IF NOT EXISTS goal_milestone_revisions(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                goal_id TEXT NOT NULL, milestone_id TEXT NOT NULL, goal_revision INTEGER NOT NULL,
                body TEXT NOT NULL, actor TEXT NOT NULL, request_id TEXT NOT NULL, operation TEXT NOT NULL,
                event TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization, project, environment, site, goal_id, milestone_id, goal_revision));"
        )?;
        Ok(Self { conn })
    }

    pub fn create_run(&self, id: &str, agent: &str, input: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO runs(id, agent, status, input, created_at, updated_at)
             VALUES(?1, ?2, 'running', ?3, ?4, ?4)",
            params![id, agent, input, now()],
        )?;
        Ok(())
    }

    pub fn set_run_status(&self, id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE runs SET status = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, status, now()],
        )?;
        Ok(())
    }

    pub fn run(&self, id: &str) -> Result<RunRow> {
        self.conn
            .query_row(
                "SELECT id, agent, status, input, created_at, updated_at FROM runs WHERE id = ?1",
                params![id],
                |r| {
                    Ok(RunRow {
                        id: r.get(0)?,
                        agent: r.get(1)?,
                        status: r.get(2)?,
                        input: r.get(3)?,
                        created_at: r.get(4)?,
                        updated_at: r.get(5)?,
                    })
                },
            )
            .optional()?
            .with_context(|| format!("no run {id} in journal"))
    }

    pub fn list_runs(&self) -> Result<Vec<(RunRow, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT r.id, r.agent, r.status, r.input, r.created_at, r.updated_at,
                    (SELECT COUNT(*) FROM steps s WHERE s.run_id = r.id)
             FROM runs r ORDER BY r.created_at DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    RunRow {
                        id: r.get(0)?,
                        agent: r.get(1)?,
                        status: r.get(2)?,
                        input: r.get(3)?,
                        created_at: r.get(4)?,
                        updated_at: r.get(5)?,
                    },
                    r.get(6)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Committed BEFORE the step executes — the crash-safety anchor.
    pub fn start_step(
        &self,
        run_id: &str,
        kind: &str,
        request: &serde_json::Value,
        tool_name: Option<&str>,
        tool_use_id: Option<&str>,
        attempt: i64,
    ) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        let seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM steps WHERE run_id = ?1",
            params![run_id],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO steps(run_id, seq, kind, status, request, tool_name, tool_use_id, attempt, started_at)
             VALUES(?1, ?2, ?3, 'started', ?4, ?5, ?6, ?7, ?8)",
            params![run_id, seq, kind, request.to_string(), tool_name, tool_use_id, attempt, now()],
        )?;
        tx.commit()?;
        Ok(seq)
    }

    pub fn complete_step(&self, run_id: &str, seq: i64, result: &serde_json::Value) -> Result<()> {
        self.conn.execute(
            "UPDATE steps SET status = 'completed', result = ?3, finished_at = ?4
             WHERE run_id = ?1 AND seq = ?2",
            params![run_id, seq, result.to_string(), now()],
        )?;
        Ok(())
    }

    pub fn fail_step(&self, run_id: &str, seq: i64, error: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE steps SET status = 'failed', result = ?3, finished_at = ?4
             WHERE run_id = ?1 AND seq = ?2",
            params![
                run_id,
                seq,
                serde_json::json!({"error": error}).to_string(),
                now()
            ],
        )?;
        Ok(())
    }

    pub fn append_step_partial(
        &self,
        run_id: &str,
        seq: i64,
        kind: &str,
        payload: &serde_json::Value,
    ) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        let ordinal: i64 = tx.query_row(
            "SELECT COALESCE(MAX(ordinal), 0) + 1
             FROM step_partials WHERE run_id = ?1 AND seq = ?2",
            params![run_id, seq],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO step_partials(run_id, seq, ordinal, kind, payload, created_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![run_id, seq, ordinal, kind, payload.to_string(), now()],
        )?;
        tx.commit()?;
        Ok(ordinal)
    }

    pub fn steps(&self, run_id: &str) -> Result<Vec<StepRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, kind, status, request, result, tool_name, tool_use_id, attempt
             FROM steps WHERE run_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt
            .query_map(params![run_id], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                    r.get::<_, i64>(7)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows.into_iter()
            .map(
                |(seq, kind, status, request, result, tool_name, tool_use_id, attempt)| {
                    Ok(StepRow {
                        seq,
                        kind,
                        status,
                        request: serde_json::from_str(&request)?,
                        result: result.map(|r| serde_json::from_str(&r)).transpose()?,
                        tool_name,
                        tool_use_id,
                        attempt,
                    })
                },
            )
            .collect()
    }

    pub fn step_partials(&self, run_id: &str, seq: i64) -> Result<Vec<StepPartialRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT seq, ordinal, kind, payload, created_at
             FROM step_partials WHERE run_id = ?1 AND seq = ?2 ORDER BY ordinal",
        )?;
        let rows = stmt
            .query_map(params![run_id, seq], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows.into_iter()
            .map(|(seq, ordinal, kind, payload, created_at)| {
                Ok(StepPartialRow {
                    seq,
                    ordinal,
                    kind,
                    payload: serde_json::from_str(&payload)?,
                    created_at,
                })
            })
            .collect()
    }

    fn replay_or_conflict<T: serde::Serialize>(
        tx: &rusqlite::Transaction<'_>,
        scope: &GoalScope,
        mutation: &GoalMutation,
        target: &str,
        request: &T,
    ) -> Result<Option<Goal>> {
        let request_body = serde_json::to_string(request)?;
        let found: Option<(String, String)> = tx.query_row(
            "SELECT request_body, response_body FROM goal_replays WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND actor=?5 AND operation=?6 AND target=?7 AND request_id=?8",
            params![scope.organization, scope.project, scope.environment, scope.site, mutation.actor, mutation.operation, target, mutation.request_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        match found {
            Some((existing, response)) => {
                ensure!(
                    existing == request_body,
                    "idempotency key conflicts with different request content"
                );
                Ok(Some(serde_json::from_str(&response)?))
            }
            None => Ok(None),
        }
    }

    fn store_goal_revision(
        tx: &rusqlite::Transaction<'_>,
        goal: &Goal,
        mutation: &GoalMutation,
        event: &str,
        request_body: &str,
    ) -> Result<()> {
        let body = serde_json::to_string(goal)?;
        let timestamp = now();
        tx.execute("INSERT INTO goals(organization,project,environment,site,id,revision,body,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
                    ON CONFLICT(organization,project,environment,site,id) DO UPDATE SET revision=excluded.revision,body=excluded.body,updated_at=excluded.updated_at",
            params![goal.scope.organization, goal.scope.project, goal.scope.environment, goal.scope.site, goal.id, goal.revision, body, timestamp])?;
        tx.execute("INSERT INTO goal_revisions(organization,project,environment,site,goal_id,revision,body,actor,request_id,operation,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![goal.scope.organization, goal.scope.project, goal.scope.environment, goal.scope.site, goal.id, goal.revision, body, mutation.actor, mutation.request_id, mutation.operation, timestamp])?;
        tx.execute("INSERT INTO goal_events(organization,project,environment,site,goal_id,revision,event,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![goal.scope.organization, goal.scope.project, goal.scope.environment, goal.scope.site, goal.id, goal.revision, event, timestamp])?;
        tx.execute("INSERT INTO goal_replays(organization,project,environment,site,actor,operation,target,request_id,request_body,response_body,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![goal.scope.organization, goal.scope.project, goal.scope.environment, goal.scope.site, mutation.actor, mutation.operation, goal.id, mutation.request_id, request_body, serde_json::to_string(goal)?, timestamp])?;
        Ok(())
    }

    pub fn create_goal(&self, mutation: &GoalMutation, goal: Goal) -> Result<Goal> {
        validate_mutation(mutation)?;
        validate_goal(&goal)?;
        ensure!(goal.revision == 1, "initial goal revision must be one");
        let request_body = serde_json::to_string(&goal)?;
        let tx = self.conn.unchecked_transaction()?;
        if let Some(replay) = Self::replay_or_conflict(&tx, &goal.scope, mutation, &goal.id, &goal)?
        {
            tx.commit()?;
            return Ok(replay);
        }
        let exists: Option<i64> = tx.query_row("SELECT revision FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5", params![goal.scope.organization,goal.scope.project,goal.scope.environment,goal.scope.site,goal.id], |r| r.get(0)).optional()?;
        ensure!(exists.is_none(), "goal already exists");
        Self::store_goal_revision(&tx, &goal, mutation, "created", &request_body)?;
        tx.commit()?;
        Ok(goal)
    }

    pub fn goal(&self, scope: &GoalScope, id: &str) -> Result<Goal> {
        validate_scope(scope)?;
        ensure!(valid_identifier(id), "invalid goal id");
        self.conn.query_row("SELECT body FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5", params![scope.organization,scope.project,scope.environment,scope.site,id], |r| r.get::<_,String>(0))
            .optional()?.with_context(|| format!("no goal {id} in scope"))
            .and_then(|body| Ok(serde_json::from_str(&body)?))
    }

    pub fn goal_history(
        &self,
        scope: &GoalScope,
        id: &str,
        limit: usize,
    ) -> Result<Vec<GoalRevision>> {
        self.goal_history_window(
            scope,
            id,
            HistoryWindow {
                after_revision: 0,
                through_revision: MAX_REVISION,
                limit,
            },
        )
    }

    pub fn goal_history_window(
        &self,
        scope: &GoalScope,
        id: &str,
        window: HistoryWindow,
    ) -> Result<Vec<GoalRevision>> {
        validate_scope(scope)?;
        ensure!(valid_identifier(id), "invalid goal id");
        validate_history_window(window)?;
        let mut stmt = self.conn.prepare("SELECT revision,body,actor,request_id,operation,created_at FROM goal_revisions WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND revision>?6 AND revision<=?7 ORDER BY revision LIMIT ?8")?;
        stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                id,
                window.after_revision,
                window.through_revision,
                window.limit as i64
            ],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            },
        )?
        .map(|row| {
            let (revision, body, actor, request_id, operation, created_at) = row?;
            Ok(GoalRevision {
                revision,
                goal: serde_json::from_str(&body)?,
                actor,
                request_id,
                operation,
                created_at,
            })
        })
        .collect()
    }

    pub fn revise_goal(
        &self,
        mutation: &GoalMutation,
        scope: &GoalScope,
        id: &str,
        expected_revision: i64,
        patch: GoalPatch,
    ) -> Result<Goal> {
        validate_mutation(mutation)?;
        validate_scope(scope)?;
        ensure!(
            valid_identifier(id) && (1..=MAX_REVISION).contains(&expected_revision),
            "invalid correction target"
        );
        validate_patch(&patch)?;
        let request = serde_json::json!({"expected_revision":expected_revision,"patch":patch});
        let request_body = serde_json::to_string(&request)?;
        let tx = self.conn.unchecked_transaction()?;
        if let Some(replay) = Self::replay_or_conflict(&tx, scope, mutation, id, &request)? {
            tx.commit()?;
            return Ok(replay);
        }
        let old: Goal=tx.query_row("SELECT body FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5",params![scope.organization,scope.project,scope.environment,scope.site,id],|r|r.get::<_,String>(0)) .optional()?.with_context(|| format!("no goal {id} in scope")).and_then(|body|Ok(serde_json::from_str(&body)?))?;
        ensure!(
            old.revision == expected_revision,
            "revision conflict: expected {expected_revision}, current {}",
            old.revision
        );
        let mut next = old.clone();
        if let Some(value) = patch.objective {
            next.objective = value;
        }
        if let Some(value) = patch.playbook {
            next.playbook = value;
        }
        if let Some(value) = patch.parameters {
            next.parameters = value;
        }
        next.revision = next
            .revision
            .checked_add(1)
            .filter(|revision| *revision <= MAX_REVISION)
            .context("goal revision limit reached")?;
        validate_goal(&next)?;
        let all = old.objective != next.objective || old.playbook != next.playbook;
        let changed: BTreeSet<String> = old
            .parameters
            .keys()
            .chain(next.parameters.keys())
            .filter(|k| old.parameters.get(*k) != next.parameters.get(*k))
            .cloned()
            .collect();
        let milestones = self.milestones_in_tx(&tx, scope, id)?;
        for mut milestone in milestones {
            if all || !milestone.parameter_dependencies.is_disjoint(&changed) {
                milestone.requires_refresh = true;
                milestone.preparation = "needs_refresh".into();
                Self::store_milestone_revision(
                    &tx,
                    scope,
                    id,
                    next.revision,
                    &milestone,
                    mutation,
                    "invalidated",
                )?;
            }
        }
        Self::store_goal_revision(&tx, &next, mutation, "revised", &request_body)?;
        tx.commit()?;
        Ok(next)
    }

    fn milestones_in_tx(
        &self,
        tx: &rusqlite::Transaction<'_>,
        scope: &GoalScope,
        id: &str,
    ) -> Result<Vec<Milestone>> {
        let mut stmt=tx.prepare("SELECT body FROM goal_milestones WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 ORDER BY milestone_id")?;
        stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                id
            ],
            |r| r.get::<_, String>(0),
        )?
        .map(|row| Ok(serde_json::from_str(&row?)?))
        .collect()
    }

    fn store_milestone_revision(
        tx: &rusqlite::Transaction<'_>,
        scope: &GoalScope,
        goal_id: &str,
        goal_revision: i64,
        milestone: &Milestone,
        mutation: &GoalMutation,
        event: &str,
    ) -> Result<()> {
        let body = serde_json::to_string(milestone)?;
        let timestamp = now();
        tx.execute("INSERT INTO goal_milestones(organization,project,environment,site,goal_id,milestone_id,body,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(organization,project,environment,site,goal_id,milestone_id) DO UPDATE SET body=excluded.body,created_at=excluded.created_at", params![scope.organization,scope.project,scope.environment,scope.site,goal_id,milestone.id,body,timestamp])?;
        tx.execute("INSERT INTO goal_milestone_revisions(organization,project,environment,site,goal_id,milestone_id,goal_revision,body,actor,request_id,operation,event,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)", params![scope.organization,scope.project,scope.environment,scope.site,goal_id,milestone.id,goal_revision,body,mutation.actor,mutation.request_id,mutation.operation,event,timestamp])?;
        Ok(())
    }

    pub fn record_milestone(
        &self,
        mutation: &GoalMutation,
        scope: &GoalScope,
        id: &str,
        expected_revision: i64,
        milestone: Milestone,
    ) -> Result<Goal> {
        validate_mutation(mutation)?;
        validate_scope(scope)?;
        ensure!(
            valid_identifier(id) && (1..=MAX_REVISION).contains(&expected_revision),
            "invalid milestone target"
        );
        validate_milestone_shape(&milestone)?;
        let request =
            serde_json::json!({"expected_revision":expected_revision,"milestone":milestone});
        let request_body = serde_json::to_string(&request)?;
        let tx = self.conn.unchecked_transaction()?;
        if let Some(replay) = Self::replay_or_conflict(&tx, scope, mutation, id, &request)? {
            tx.commit()?;
            return Ok(replay);
        }
        let mut goal:Goal=tx.query_row("SELECT body FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5",params![scope.organization,scope.project,scope.environment,scope.site,id],|r|r.get::<_,String>(0)) .optional()?.with_context(|| format!("no goal {id} in scope")).and_then(|body|Ok(serde_json::from_str(&body)?))?;
        ensure!(
            goal.revision == expected_revision,
            "revision conflict: expected {expected_revision}, current {}",
            goal.revision
        );
        ensure!(
            milestone
                .parameter_dependencies
                .iter()
                .all(|key| goal.parameters.contains_key(key)),
            "inconsistent milestone dependency"
        );
        let exists: Option<i64> = tx.query_row("SELECT 1 FROM goal_milestones WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND milestone_id=?6", params![scope.organization,scope.project,scope.environment,scope.site,id,milestone.id], |r| r.get(0)).optional()?;
        if exists.is_none() {
            let count:i64=tx.query_row("SELECT COUNT(*) FROM goal_milestones WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5", params![scope.organization,scope.project,scope.environment,scope.site,id],|r|r.get(0))?;
            ensure!(count < MAX_MILESTONES as i64, "too many milestones");
        }
        let mut milestone = milestone;
        if let Some(previous) = tx.query_row("SELECT body FROM goal_milestones WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND milestone_id=?6", params![scope.organization,scope.project,scope.environment,scope.site,id,milestone.id], |r| r.get::<_, String>(0)).optional()? {
            let previous: Milestone = serde_json::from_str(&previous)?;
            if previous.requires_refresh { milestone.requires_refresh = true; milestone.preparation = "needs_refresh".into(); }
        }
        goal.revision = goal
            .revision
            .checked_add(1)
            .filter(|revision| *revision <= MAX_REVISION)
            .context("goal revision limit reached")?;
        Self::store_milestone_revision(
            &tx,
            scope,
            id,
            goal.revision,
            &milestone,
            mutation,
            "recorded",
        )?;
        Self::store_goal_revision(&tx, &goal, mutation, "milestone_recorded", &request_body)?;
        tx.commit()?;
        Ok(goal)
    }

    pub fn milestones(&self, scope: &GoalScope, id: &str) -> Result<Vec<Milestone>> {
        validate_scope(scope)?;
        ensure!(valid_identifier(id), "invalid goal id");
        let tx = self.conn.unchecked_transaction()?;
        let rows = self.milestones_in_tx(&tx, scope, id)?;
        tx.commit()?;
        Ok(rows)
    }

    pub fn milestone_history(
        &self,
        scope: &GoalScope,
        goal_id: &str,
        milestone_id: &str,
        limit: usize,
    ) -> Result<Vec<MilestoneRevision>> {
        self.milestone_history_window(
            scope,
            goal_id,
            milestone_id,
            HistoryWindow {
                after_revision: 0,
                through_revision: MAX_REVISION,
                limit,
            },
        )
    }

    pub fn milestone_history_window(
        &self,
        scope: &GoalScope,
        goal_id: &str,
        milestone_id: &str,
        window: HistoryWindow,
    ) -> Result<Vec<MilestoneRevision>> {
        validate_scope(scope)?;
        ensure!(
            valid_identifier(goal_id) && valid_identifier(milestone_id),
            "invalid milestone history target"
        );
        validate_history_window(window)?;
        let mut stmt = self.conn.prepare("SELECT goal_revision,body,actor,request_id,operation,event,created_at FROM goal_milestone_revisions WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND goal_id=?5 AND milestone_id=?6 AND goal_revision>?7 AND goal_revision<=?8 ORDER BY goal_revision LIMIT ?9")?;
        stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                goal_id,
                milestone_id,
                window.after_revision,
                window.through_revision,
                window.limit as i64
            ],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            },
        )?
        .map(|row| {
            let (goal_revision, body, actor, request_id, operation, event, created_at) = row?;
            Ok(MilestoneRevision {
                goal_revision,
                milestone: serde_json::from_str(&body)?,
                actor,
                request_id,
                operation,
                event,
                created_at,
            })
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Journal;
    use serde_json::json;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Barrier};
    use std::thread;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "beater-journal-{name}-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn records_run_and_step_lifecycle_in_order() {
        let app = TempDir::new("lifecycle");
        let journal = Journal::open(app.path()).unwrap();
        journal.create_run("run-1", "support", "hello").unwrap();

        let llm = journal
            .start_step(
                "run-1",
                "llm_call",
                &json!({"messages": [{"role": "user", "content": "hello"}]}),
                None,
                None,
                1,
            )
            .unwrap();
        journal
            .complete_step("run-1", llm, &json!({"stop_reason": "tool_use"}))
            .unwrap();
        let tool = journal
            .start_step(
                "run-1",
                "tool_call",
                &json!({"name": "summarize_numbers"}),
                Some("summarize_numbers"),
                Some("toolu_1"),
                2,
            )
            .unwrap();
        journal.fail_step("run-1", tool, "boom").unwrap();
        journal.set_run_status("run-1", "needs_review").unwrap();

        let run = journal.run("run-1").unwrap();
        assert_eq!(run.status, "needs_review");

        let steps = journal.steps("run-1").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].seq, 1);
        assert_eq!(steps[0].kind, "llm_call");
        assert_eq!(steps[0].status, "completed");
        assert_eq!(steps[1].seq, 2);
        assert_eq!(steps[1].kind, "tool_call");
        assert_eq!(steps[1].status, "failed");
        assert_eq!(steps[1].tool_name.as_deref(), Some("summarize_numbers"));
        assert_eq!(steps[1].tool_use_id.as_deref(), Some("toolu_1"));
        assert_eq!(steps[1].attempt, 2);
        assert_eq!(steps[1].result.as_ref().unwrap()["error"], "boom");
    }

    #[test]
    fn records_step_partials_before_final_result() {
        let app = TempDir::new("partials");
        let journal = Journal::open(app.path()).unwrap();
        journal.create_run("run-1", "support", "stream").unwrap();
        let seq = journal
            .start_step(
                "run-1",
                "llm_call",
                &json!({"stream": true, "messages": [{"role": "user", "content": "hi"}]}),
                None,
                None,
                1,
            )
            .unwrap();

        let first = journal
            .append_step_partial("run-1", seq, "text_delta", &json!({"text": "hel"}))
            .unwrap();
        let second = journal
            .append_step_partial("run-1", seq, "text_delta", &json!({"text": "lo"}))
            .unwrap();
        assert_eq!((first, second), (1, 2));

        let before_complete = Journal::open(app.path())
            .unwrap()
            .step_partials("run-1", seq)
            .unwrap();
        assert_eq!(before_complete.len(), 2);
        assert_eq!(before_complete[0].seq, seq);
        assert_eq!(before_complete[0].ordinal, 1);
        assert_eq!(before_complete[0].kind, "text_delta");
        assert_eq!(before_complete[0].payload["text"], "hel");
        assert_eq!(before_complete[1].ordinal, 2);
        assert_eq!(before_complete[1].payload["text"], "lo");

        journal
            .complete_step(
                "run-1",
                seq,
                &json!({
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn"
                }),
            )
            .unwrap();

        let after_complete = Journal::open(app.path())
            .unwrap()
            .step_partials("run-1", seq)
            .unwrap();
        assert_eq!(after_complete.len(), 2);
        assert_eq!(after_complete[0].payload["text"], "hel");
        assert_eq!(after_complete[1].payload["text"], "lo");
    }

    #[test]
    fn open_configures_wal_and_busy_timeout() {
        let app = TempDir::new("pragma");
        let journal = Journal::open(app.path()).unwrap();

        let journal_mode: String = journal
            .conn
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        let busy_timeout_ms: i64 = journal
            .conn
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))
            .unwrap();

        assert_eq!(journal_mode.to_ascii_lowercase(), "wal");
        assert_eq!(busy_timeout_ms, 5_000);
    }

    #[test]
    fn concurrent_start_step_allocates_unique_sequences() {
        let app = TempDir::new("concurrent-start-step");
        let journal = Journal::open(app.path()).unwrap();
        journal.create_run("run-1", "support", "hello").unwrap();

        let workers = 8;
        let barrier = Arc::new(Barrier::new(workers));
        let app_path = Arc::new(app.path().to_path_buf());
        let handles = (0..workers)
            .map(|worker| {
                let barrier = Arc::clone(&barrier);
                let app_path = Arc::clone(&app_path);
                thread::spawn(move || {
                    let tool_use_id = format!("toolu_{worker}");
                    barrier.wait();
                    let journal = Journal::open(&app_path).unwrap();
                    journal
                        .start_step(
                            "run-1",
                            "tool_call",
                            &json!({"worker": worker}),
                            Some("echo"),
                            Some(&tool_use_id),
                            1,
                        )
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();

        let seqs = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<BTreeSet<_>>();
        let expected = (1..=workers as i64).collect::<BTreeSet<_>>();
        assert_eq!(seqs, expected);

        let steps = Journal::open(app.path()).unwrap().steps("run-1").unwrap();
        assert_eq!(steps.len(), workers);
        for (index, step) in steps.iter().enumerate() {
            assert_eq!(step.seq, index as i64 + 1);
            assert_eq!(step.status, "started");
        }
    }

    #[test]
    fn list_runs_reports_step_counts() {
        let app = TempDir::new("list");
        let journal = Journal::open(app.path()).unwrap();
        journal.create_run("run-1", "support", "one").unwrap();
        journal.create_run("run-2", "support", "two").unwrap();
        journal
            .start_step("run-2", "llm_call", &json!({"messages": []}), None, None, 1)
            .unwrap();

        let runs = journal.list_runs().unwrap();

        assert_eq!(runs.len(), 2);
        let run_2 = runs.iter().find(|(run, _)| run.id == "run-2").unwrap();
        assert_eq!(run_2.1, 1);
        let run_1 = runs.iter().find(|(run, _)| run.id == "run-1").unwrap();
        assert_eq!(run_1.1, 0);
    }

    fn scope() -> super::GoalScope {
        super::GoalScope {
            organization: "org-a".into(),
            project: "merchant".into(),
            environment: "test".into(),
            site: "site-a".into(),
        }
    }

    fn mutation(request_id: &str) -> super::GoalMutation {
        super::GoalMutation {
            actor: "operator-a".into(),
            request_id: request_id.into(),
            operation: "correct".into(),
        }
    }

    #[test]
    fn merchant_goal_reopens_preserves_history_and_invalidates_only_dependent_preparation() {
        use super::{Goal, GoalPatch, Milestone, PlaybookIdentity};
        use std::collections::{BTreeMap, BTreeSet};
        let app = TempDir::new("merchant-goal");
        let scope = scope();
        let mut params = BTreeMap::new();
        params.insert("supplier".into(), json!("supplierA"));
        params.insert("budget".into(), json!(1200));
        let goal = Goal {
            id: "order-17".into(),
            scope: scope.clone(),
            revision: 1,
            objective: "source a compliant order".into(),
            playbook: PlaybookIdentity {
                id: "purchase-v1".into(),
                digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            },
            parameters: params,
        };
        let journal = Journal::open(app.path()).unwrap();
        let goal = journal.create_goal(&mutation("create-1"), goal).unwrap();
        let mapping = Milestone {
            id: "order-mapping".into(),
            preparation: "prepared".into(),
            evidence_references: vec!["mapping-evidence-1".into()],
            parameter_dependencies: BTreeSet::new(),
            requires_refresh: false,
        };
        let goal = journal
            .record_milestone(
                &mutation("mapping-1"),
                &scope,
                "order-17",
                goal.revision,
                mapping,
            )
            .unwrap();
        let preview = Milestone {
            id: "supplier-preview".into(),
            preparation: "prepared".into(),
            evidence_references: vec!["preview-evidence-A".into()],
            parameter_dependencies: ["supplier".into(), "budget".into()].into_iter().collect(),
            requires_refresh: false,
        };
        let revision = journal
            .record_milestone(
                &mutation("preview-1"),
                &scope,
                "order-17",
                goal.revision,
                preview,
            )
            .unwrap()
            .revision;
        drop(journal);

        let journal = Journal::open(app.path()).unwrap();
        let mut corrected = BTreeMap::new();
        corrected.insert("supplier".into(), json!("supplierB"));
        corrected.insert("budget".into(), json!(900));
        let patch = GoalPatch {
            objective: None,
            playbook: None,
            parameters: Some(corrected),
        };
        let revised = journal
            .revise_goal(
                &mutation("correction-1"),
                &scope,
                "order-17",
                revision,
                patch.clone(),
            )
            .unwrap();
        assert_eq!(revised.revision, revision + 1);
        assert_eq!(
            journal
                .revise_goal(
                    &mutation("correction-1"),
                    &scope,
                    "order-17",
                    revision,
                    patch
                )
                .unwrap(),
            revised
        );
        let milestones = journal.milestones(&scope, "order-17").unwrap();
        let mapping = milestones.iter().find(|m| m.id == "order-mapping").unwrap();
        let preview = milestones
            .iter()
            .find(|m| m.id == "supplier-preview")
            .unwrap();
        assert!(!mapping.requires_refresh);
        assert!(preview.requires_refresh);
        assert_eq!(preview.evidence_references, vec!["preview-evidence-A"]);
        assert_eq!(
            journal.goal_history(&scope, "order-17", 16).unwrap().len(),
            4
        );
        let rerecorded = journal
            .record_milestone(
                &mutation("preview-rerecord"),
                &scope,
                "order-17",
                revised.revision,
                Milestone {
                    id: "supplier-preview".into(),
                    preparation: "prepared".into(),
                    evidence_references: vec!["preview-evidence-B".into()],
                    parameter_dependencies: ["supplier".into(), "budget".into()]
                        .into_iter()
                        .collect(),
                    requires_refresh: false,
                },
            )
            .unwrap();
        let current = journal
            .milestones(&scope, "order-17")
            .unwrap()
            .into_iter()
            .find(|m| m.id == "supplier-preview")
            .unwrap();
        assert!(current.requires_refresh);
        assert_eq!(current.preparation, "needs_refresh");
        let history = journal
            .milestone_history(&scope, "order-17", "supplier-preview", 8)
            .unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(
            history[0].milestone.evidence_references,
            vec!["preview-evidence-A"]
        );
        assert_eq!(history[1].event, "invalidated");
        assert_eq!(
            history[2].milestone.evidence_references,
            vec!["preview-evidence-B"]
        );
        assert_eq!(rerecorded.revision, revised.revision + 1);
        assert!(
            journal
                .revise_goal(
                    &mutation("correction-1"),
                    &scope,
                    "order-17",
                    revision,
                    GoalPatch {
                        objective: Some("different".into()),
                        playbook: None,
                        parameters: None
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn stale_or_other_scope_cannot_change_a_goal() {
        use super::{Goal, GoalPatch, PlaybookIdentity};
        let app = TempDir::new("goal-cas");
        let journal = Journal::open(app.path()).unwrap();
        let scope = scope();
        let goal = Goal {
            id: "g-1".into(),
            scope: scope.clone(),
            revision: 1,
            objective: "keep evidence".into(),
            playbook: PlaybookIdentity {
                id: "p".into(),
                digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            },
            parameters: BTreeMap::new(),
        };
        journal.create_goal(&mutation("create"), goal).unwrap();
        let changed = journal
            .revise_goal(
                &mutation("one"),
                &scope,
                "g-1",
                1,
                GoalPatch {
                    objective: Some("corrected".into()),
                    playbook: None,
                    parameters: None,
                },
            )
            .unwrap();
        assert_eq!(changed.revision, 2);
        assert!(
            journal
                .revise_goal(
                    &mutation("two"),
                    &scope,
                    "g-1",
                    1,
                    GoalPatch {
                        objective: Some("loser".into()),
                        playbook: None,
                        parameters: None
                    }
                )
                .is_err()
        );
        let other = super::GoalScope {
            site: "site-b".into(),
            ..scope.clone()
        };
        assert!(journal.goal(&other, "g-1").is_err());
    }

    #[test]
    fn concurrent_corrections_have_one_winner_and_invalid_patch_rolls_back() {
        use super::{Goal, GoalPatch, PlaybookIdentity};
        let app = TempDir::new("goal-concurrency");
        let scope = scope();
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_goal(
                &mutation("create"),
                Goal {
                    id: "g-1".into(),
                    scope: scope.clone(),
                    revision: 1,
                    objective: "keep evidence".into(),
                    playbook: PlaybookIdentity {
                        id: "p".into(),
                        digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                            .into(),
                    },
                    parameters: BTreeMap::new(),
                },
            )
            .unwrap();
        assert!(
            journal
                .revise_goal(
                    &mutation("bad"),
                    &scope,
                    "g-1",
                    1,
                    GoalPatch {
                        objective: Some("x".into()),
                        playbook: Some(PlaybookIdentity {
                            id: "bad space".into(),
                            digest:
                                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                                    .into()
                        }),
                        parameters: None
                    }
                )
                .is_err()
        );
        assert_eq!(journal.goal(&scope, "g-1").unwrap().revision, 1);
        assert_eq!(journal.goal_history(&scope, "g-1", 8).unwrap().len(), 1);
        let barrier = Arc::new(Barrier::new(2));
        let app_path = Arc::new(app.path().to_path_buf());
        let handles = (0..2)
            .map(|worker| {
                let barrier = Arc::clone(&barrier);
                let app_path = Arc::clone(&app_path);
                let scope = scope.clone();
                thread::spawn(move || {
                    barrier.wait();
                    Journal::open(&app_path)
                        .unwrap()
                        .revise_goal(
                            &super::GoalMutation {
                                actor: format!("actor-{worker}"),
                                request_id: format!("request-{worker}"),
                                operation: "correct".into(),
                            },
                            &scope,
                            "g-1",
                            1,
                            GoalPatch {
                                objective: Some(format!("winner-{worker}")),
                                playbook: None,
                                parameters: None,
                            },
                        )
                        .is_ok()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
        assert_eq!(
            Journal::open(app.path())
                .unwrap()
                .goal_history(&scope, "g-1", 8)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn rejects_unbounded_revisions_invalid_read_ids_and_credential_shaped_inputs() {
        use super::{Goal, PlaybookIdentity};
        let app = TempDir::new("goal-bounds");
        let journal = Journal::open(app.path()).unwrap();
        let scope = scope();
        for revision in [0, 2, i64::MAX] {
            assert!(journal.create_goal(&mutation(&format!("r-{revision}")), Goal { id: format!("g-{revision}"), scope:scope.clone(), revision, objective:"objective".into(), playbook:PlaybookIdentity{id:"p".into(),digest:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()}, parameters:BTreeMap::new() }).is_err());
        }
        let mut parameters = BTreeMap::new();
        parameters.insert("supplier_token".into(), json!("x"));
        assert!(
            journal
                .create_goal(
                    &mutation("credential-key"),
                    Goal {
                        id: "credential".into(),
                        scope: scope.clone(),
                        revision: 1,
                        objective: "objective".into(),
                        playbook: PlaybookIdentity {
                            id: "p".into(),
                            digest:
                                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                                    .into()
                        },
                        parameters
                    }
                )
                .is_err()
        );
        for digest in [
            "",
            "sha256-abc",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ] {
            assert!(
                journal
                    .create_goal(
                        &mutation(&format!("digest-{}", digest.len())),
                        Goal {
                            id: format!("digest-{}", digest.len()),
                            scope: scope.clone(),
                            revision: 1,
                            objective: "objective".into(),
                            playbook: PlaybookIdentity {
                                id: "p".into(),
                                digest: digest.into()
                            },
                            parameters: BTreeMap::new()
                        }
                    )
                    .is_err()
            );
        }
        assert!(journal.goal_history(&scope, "bad id", 1).is_err());
        assert!(journal.milestones(&scope, "bad id").is_err());
    }

    fn sample_goal() -> super::Goal {
        super::Goal {
            id: "bounded-goal".into(),
            scope: scope(),
            revision: 1,
            objective: "Preserve declared preparation across restarts".into(),
            playbook: super::PlaybookIdentity {
                id: "merchant-recovery-v1".into(),
                digest: "a".repeat(64),
            },
            parameters: BTreeMap::new(),
        }
    }

    fn sample_milestone() -> super::Milestone {
        super::Milestone {
            id: "preview".into(),
            preparation: "prepared".into(),
            evidence_references: vec!["artifact:declared-preview".into()],
            parameter_dependencies: BTreeSet::new(),
            requires_refresh: false,
        }
    }

    #[test]
    fn malformed_shapes_are_bounded_before_request_serialization_or_writes() {
        let app = TempDir::new("goal-input-shapes");
        let journal = Journal::open(app.path()).unwrap();
        let goal = journal
            .create_goal(&mutation("create"), sample_goal())
            .unwrap();
        let mut nested = json!(0);
        for _ in 0..40 {
            nested = json!([nested]);
        }
        for value in [
            nested,
            json!(vec![0; 1_100]),
            json!({"API-Key": "synthetic-only"}),
            json!("Bearer synthetic-only"),
        ] {
            let patch = super::GoalPatch {
                objective: None,
                playbook: None,
                parameters: Some([("configuration".into(), value)].into_iter().collect()),
            };
            assert!(
                journal
                    .revise_goal(&mutation("invalid-shape"), &goal.scope, &goal.id, 1, patch)
                    .is_err()
            );
        }
        for kind in 0..3 {
            let mut milestone = sample_milestone();
            match kind {
                0 => milestone.evidence_references = vec!["artifact:declared".into(); 33],
                1 => milestone.requires_refresh = true,
                _ => milestone.evidence_references = vec!["password=synthetic-only".into()],
            }
            assert!(
                journal
                    .record_milestone(
                        &mutation("invalid-milestone"),
                        &goal.scope,
                        &goal.id,
                        1,
                        milestone
                    )
                    .is_err()
            );
        }
        assert_eq!(journal.goal(&goal.scope, &goal.id).unwrap(), goal);
        assert_eq!(
            journal
                .goal_history(&goal.scope, &goal.id, 256)
                .unwrap()
                .len(),
            1
        );
        assert!(
            journal
                .milestones(&goal.scope, &goal.id)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn both_histories_continue_past_first_page_with_a_stable_upper_revision() {
        let app = TempDir::new("goal-history-window");
        let journal = Journal::open(app.path()).unwrap();
        let mut goal = journal
            .create_goal(&mutation("create"), sample_goal())
            .unwrap();
        for index in 0..260 {
            goal = journal
                .record_milestone(
                    &mutation(&format!("record-{index}")),
                    &goal.scope,
                    &goal.id,
                    goal.revision,
                    sample_milestone(),
                )
                .unwrap();
        }
        let upper = goal.revision;
        journal
            .record_milestone(
                &mutation("after-snapshot"),
                &goal.scope,
                &goal.id,
                goal.revision,
                sample_milestone(),
            )
            .unwrap();
        drop(journal);
        let journal = Journal::open(app.path()).unwrap();
        let mut cursor = 0;
        let mut revisions = Vec::new();
        loop {
            let page = journal
                .goal_history_window(
                    &goal.scope,
                    &goal.id,
                    super::HistoryWindow {
                        after_revision: cursor,
                        through_revision: upper,
                        limit: 17,
                    },
                )
                .unwrap();
            if page.is_empty() {
                break;
            }
            cursor = page.last().unwrap().revision;
            revisions.extend(page.into_iter().map(|entry| entry.revision));
        }
        assert_eq!(revisions, (1..=upper).collect::<Vec<_>>());
        cursor = 0;
        revisions.clear();
        loop {
            let page = journal
                .milestone_history_window(
                    &goal.scope,
                    &goal.id,
                    "preview",
                    super::HistoryWindow {
                        after_revision: cursor,
                        through_revision: upper,
                        limit: 19,
                    },
                )
                .unwrap();
            if page.is_empty() {
                break;
            }
            cursor = page.last().unwrap().goal_revision;
            revisions.extend(page.into_iter().map(|entry| entry.goal_revision));
        }
        assert_eq!(revisions, (2..=upper).collect::<Vec<_>>());
        let other_scope = super::GoalScope {
            site: "unrelated-site".into(),
            ..goal.scope.clone()
        };
        assert!(
            journal
                .goal_history_window(
                    &other_scope,
                    &goal.id,
                    super::HistoryWindow {
                        after_revision: 0,
                        through_revision: upper,
                        limit: 17,
                    }
                )
                .unwrap()
                .is_empty()
        );
        assert!(
            journal
                .goal_history_window(
                    &goal.scope,
                    &goal.id,
                    super::HistoryWindow {
                        after_revision: upper + 1,
                        through_revision: upper,
                        limit: 17,
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn failed_replay_insert_rolls_back_goal_milestone_histories_and_events() {
        let app = TempDir::new("goal-atomic-rollback");
        let journal = Journal::open(app.path()).unwrap();
        let goal = journal
            .create_goal(&mutation("create"), sample_goal())
            .unwrap();
        journal
            .conn
            .execute_batch(
                "CREATE TRIGGER reject_test_receipt BEFORE INSERT ON goal_replays
            WHEN NEW.request_id = 'rollback' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END;",
            )
            .unwrap();
        assert!(
            journal
                .record_milestone(
                    &mutation("rollback"),
                    &goal.scope,
                    &goal.id,
                    1,
                    sample_milestone()
                )
                .is_err()
        );
        assert_eq!(journal.goal(&goal.scope, &goal.id).unwrap(), goal);
        assert_eq!(
            journal
                .goal_history(&goal.scope, &goal.id, 32)
                .unwrap()
                .len(),
            1
        );
        assert!(
            journal
                .milestones(&goal.scope, &goal.id)
                .unwrap()
                .is_empty()
        );
        assert!(
            journal
                .milestone_history(&goal.scope, &goal.id, "preview", 32)
                .unwrap()
                .is_empty()
        );
        let events: i64 = journal
            .conn
            .query_row("SELECT COUNT(*) FROM goal_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(events, 1);
        journal
            .conn
            .execute_batch("DROP TRIGGER reject_test_receipt")
            .unwrap();
        assert_eq!(
            journal
                .record_milestone(
                    &mutation("rollback"),
                    &goal.scope,
                    &goal.id,
                    1,
                    sample_milestone()
                )
                .unwrap()
                .revision,
            2
        );
    }

    #[test]
    fn revision_limit_rejects_both_mutations_without_overflow_or_partial_history() {
        let app = TempDir::new("goal-revision-limit");
        let journal = Journal::open(app.path()).unwrap();
        let mut goal = journal
            .create_goal(&mutation("create"), sample_goal())
            .unwrap();
        // Fault-inject the far-future current row to exercise checked increments
        // without issuing a billion journal writes.
        goal.revision = super::MAX_REVISION;
        journal
            .conn
            .execute(
                "UPDATE goals SET revision=?1,body=?2",
                rusqlite::params![goal.revision, serde_json::to_string(&goal).unwrap()],
            )
            .unwrap();
        assert!(
            journal
                .revise_goal(
                    &mutation("revise-limit"),
                    &goal.scope,
                    &goal.id,
                    goal.revision,
                    super::GoalPatch {
                        objective: Some("changed".into()),
                        playbook: None,
                        parameters: None
                    }
                )
                .is_err()
        );
        assert!(
            journal
                .record_milestone(
                    &mutation("milestone-limit"),
                    &goal.scope,
                    &goal.id,
                    goal.revision,
                    sample_milestone()
                )
                .is_err()
        );
        assert_eq!(journal.goal(&goal.scope, &goal.id).unwrap(), goal);
        assert_eq!(
            journal
                .goal_history(&goal.scope, &goal.id, 32)
                .unwrap()
                .len(),
            1
        );
        assert!(
            journal
                .milestones(&goal.scope, &goal.id)
                .unwrap()
                .is_empty()
        );
    }
}
