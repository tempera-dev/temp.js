//! Immutable, local decision-audit records. These are references recorded by a
//! goal-bound run, not authorization, external verification, or effect control.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use rusqlite::{OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::{
    Goal, GoalRunGate, GoalScope, Journal, MAX_REVISION, valid_identifier, validate_goal,
    validate_scope,
};

const DECISION_VERSION: u8 = 1;
const MAX_DECISIONS_PER_SCOPE: i64 = 10_000;
const MAX_DECISION_TEXT: usize = 4_096;
const MAX_REFERENCE: usize = 512;
const MAX_REFS: usize = 64;
const MAX_OPTIONS: usize = 32;
const MAX_BODY: usize = 32_768;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionDomain {
    Software,
    Supply,
    Payments,
    Operations,
}

/// The only verification ceiling represented in this local journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VerificationCeiling {
    RecordedOnly,
}

/// The journal cannot grant execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionAuthority {
    None,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedReference {
    pub locator: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReference {
    pub reference: VersionedReference,
    pub missing: bool,
}

/// Attribution/context only; never an authority assertion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphContextReference {
    pub reference: VersionedReference,
    pub watermark: Option<String>,
}

macro_rules! typed_reference {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            pub reference: VersionedReference,
        }
    };
}

typed_reference!(ReviewReference);
typed_reference!(AuthorizationObservationReference);
typed_reference!(EffectAttemptReference);
typed_reference!(AcknowledgementReference);
typed_reference!(OutcomeReference);
typed_reference!(ReconciliationReference);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRevisionReference {
    pub decision_id: String,
    pub revision: i64,
}

/// Immutable package body. All external handles are local recorded references.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionPackageV1 {
    pub version: u8,
    pub decision_id: String,
    pub revision: i64,
    pub domain: DecisionDomain,
    pub question: String,
    pub object_references: Vec<VersionedReference>,
    pub evidence: Vec<EvidenceReference>,
    pub graph_context: Option<GraphContextReference>,
    pub policy_references: Vec<VersionedReference>,
    pub calculation_references: Vec<VersionedReference>,
    pub model_references: Vec<VersionedReference>,
    pub producer_references: Vec<VersionedReference>,
    pub options: Vec<String>,
    pub constraints: Vec<String>,
    pub rationale: String,
    pub reviews: Vec<ReviewReference>,
    pub authorization_observations: Vec<AuthorizationObservationReference>,
    pub effect_attempts: Vec<EffectAttemptReference>,
    pub acknowledgements: Vec<AcknowledgementReference>,
    pub outcomes: Vec<OutcomeReference>,
    pub reconciliations: Vec<ReconciliationReference>,
    pub correction_of: Option<DecisionRevisionReference>,
    pub successor_to: Option<DecisionRevisionReference>,
    pub verification_ceiling: VerificationCeiling,
    pub execution_authority: ExecutionAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAuditEventV1 {
    pub scope: GoalScope,
    pub run_id: String,
    pub goal_id: String,
    pub goal_revision: i64,
    pub goal_digest: String,
    pub sequence: i64,
    pub package: DecisionPackageV1,
    pub request_id: String,
    pub request_digest: String,
    pub package_digest: String,
    pub event_digest: String,
    pub predecessor_event_digest: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAuditProjectionV1 {
    pub scope: GoalScope,
    pub run_id: String,
    pub goal_id: String,
    pub goal_revision: i64,
    pub goal_digest: String,
    pub decision_id: String,
    pub current_revision: i64,
    pub verification_ceiling: VerificationCeiling,
    pub execution_authority: ExecutionAuthority,
    pub events: Vec<DecisionAuditEventV1>,
}

struct RecordedDecisionBinding<'a> {
    scope: &'a GoalScope,
    decision_id: &'a str,
    run_id: &'a str,
    goal_id: &'a str,
    goal_revision: i64,
    goal_digest: &'a str,
    goal_body: &'a [u8],
}

fn valid_reference(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REFERENCE
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@')
        })
}

fn validate_reference(reference: &VersionedReference) -> Result<()> {
    ensure!(
        valid_reference(&reference.locator),
        "invalid decision reference"
    );
    let revision = reference
        .revision
        .as_deref()
        .context("exact decision reference requires a revision or digest")?;
    ensure!(
        valid_reference(revision),
        "invalid decision reference revision"
    );
    Ok(())
}

fn validate_reference_list(references: &[VersionedReference], field: &str) -> Result<()> {
    ensure!(references.len() <= MAX_REFS, "too many {field}");
    for reference in references {
        validate_reference(reference)?;
    }
    Ok(())
}

fn validate_typed_references<T>(
    references: &[T],
    field: &str,
    get: impl Fn(&T) -> &VersionedReference,
) -> Result<()> {
    ensure!(references.len() <= MAX_REFS, "too many {field}");
    for reference in references {
        validate_reference(get(reference))?;
    }
    Ok(())
}

fn validate_revision_reference(
    reference: &DecisionRevisionReference,
    package: &DecisionPackageV1,
) -> Result<()> {
    ensure!(
        valid_identifier(&reference.decision_id),
        "invalid linked decision id"
    );
    ensure!(
        (1..=MAX_REVISION).contains(&reference.revision),
        "invalid linked decision revision"
    );
    ensure!(
        reference.decision_id != package.decision_id || reference.revision != package.revision,
        "decision cannot link to itself"
    );
    Ok(())
}

fn canonical_value(value: &serde_json::Value, output: &mut String) -> Result<()> {
    match value {
        serde_json::Value::Null => output.push_str("null"),
        serde_json::Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(number) => {
            // Canonical JSON in this journal accepts integer coordinates only.
            if let Some(value) = number.as_i64() {
                output.push_str(&value.to_string());
            } else if let Some(value) = number.as_u64() {
                output.push_str(&value.to_string());
            } else {
                anyhow::bail!("ambiguous floating-point value is not accepted");
            }
        }
        serde_json::Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        serde_json::Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_value(value, output)?;
            }
            output.push(']');
        }
        serde_json::Value::Object(values) => {
            output.push('{');
            let mut ordered: BTreeMap<&str, &serde_json::Value> = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key, value);
            }
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                canonical_value(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_bytes<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let mut output = String::new();
    canonical_value(&value, &mut output)?;
    ensure!(output.len() <= MAX_BODY, "decision body too large");
    Ok(output.into_bytes())
}

fn digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

#[derive(serde::Serialize)]
struct EventDigestInput<'a> {
    domain: &'static str,
    scope: &'a GoalScope,
    run_id: &'a str,
    goal_id: &'a str,
    goal_revision: i64,
    goal_digest: &'a str,
    decision_id: &'a str,
    sequence: i64,
    request_id: &'a str,
    request_digest: &'a str,
    package_digest: &'a str,
    predecessor_event_digest: &'a Option<String>,
    created_at: i64,
}

fn event_digest(event: &DecisionAuditEventV1) -> Result<String> {
    Ok(digest(&canonical_bytes(&EventDigestInput {
        domain: "TEMPERA_DECISION_AUDIT_EVENT_V1",
        scope: &event.scope,
        run_id: &event.run_id,
        goal_id: &event.goal_id,
        goal_revision: event.goal_revision,
        goal_digest: &event.goal_digest,
        decision_id: &event.package.decision_id,
        sequence: event.sequence,
        request_id: &event.request_id,
        request_digest: &event.request_digest,
        package_digest: &event.package_digest,
        predecessor_event_digest: &event.predecessor_event_digest,
        created_at: event.created_at,
    })?))
}

/// Recomputes the local event digest from a serialized event envelope. This is
/// integrity verification only; it grants neither external authority nor effects.
pub fn verify_decision_audit_event(event: &DecisionAuditEventV1) -> Result<()> {
    validate_scope(&event.scope)?;
    ensure!(
        valid_identifier(&event.run_id) && valid_identifier(&event.goal_id),
        "invalid event binding identity"
    );
    ensure!(
        (1..=MAX_REVISION).contains(&event.goal_revision) && event.created_at >= 0,
        "invalid event coordinates"
    );
    validate_package(&event.package)?;
    ensure!(
        (1..=MAX_REVISION).contains(&event.sequence) && event.package.revision == event.sequence,
        "invalid event sequence or package revision"
    );
    let valid_digest = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    };
    ensure!(
        valid_digest(&event.goal_digest) && valid_digest(&event.request_digest),
        "invalid event binding digest"
    );
    ensure!(
        match &event.predecessor_event_digest {
            None => event.sequence == 1,
            Some(previous) => event.sequence > 1 && valid_digest(previous),
        },
        "invalid event predecessor"
    );
    ensure!(
        valid_identifier(&event.request_id),
        "invalid event request id"
    );
    ensure!(
        digest(&canonical_bytes(&event.package)?) == event.package_digest,
        "event package digest mismatch"
    );
    ensure!(
        digest(&canonical_bytes(&(
            event.run_id.as_str(),
            event.sequence - 1,
            event.request_id.as_str(),
            &event.package,
        ))?) == event.request_digest,
        "event request digest mismatch"
    );
    ensure!(
        event_digest(event)? == event.event_digest,
        "event digest mismatch"
    );
    Ok(())
}

fn validate_package(package: &DecisionPackageV1) -> Result<()> {
    ensure!(
        package.version == DECISION_VERSION,
        "unsupported decision package version"
    );
    ensure!(
        valid_identifier(&package.decision_id),
        "invalid decision id"
    );
    ensure!(
        (1..=MAX_REVISION).contains(&package.revision),
        "invalid decision revision"
    );
    ensure!(
        !package.question.is_empty() && package.question.len() <= MAX_DECISION_TEXT,
        "invalid decision question"
    );
    ensure!(
        package.options.len() <= MAX_OPTIONS && package.constraints.len() <= MAX_OPTIONS,
        "too many decision options or constraints"
    );
    ensure!(
        !package.rationale.is_empty() && package.rationale.len() <= MAX_DECISION_TEXT,
        "invalid decision rationale"
    );
    for text in package.options.iter().chain(package.constraints.iter()) {
        ensure!(
            !text.is_empty() && text.len() <= MAX_DECISION_TEXT,
            "invalid decision text"
        );
    }
    validate_reference_list(&package.object_references, "object references")?;
    ensure!(
        package.evidence.len() <= MAX_REFS,
        "too many evidence references"
    );
    for evidence in &package.evidence {
        ensure!(
            valid_reference(&evidence.reference.locator),
            "invalid evidence reference"
        );
        if evidence.missing {
            if let Some(revision) = &evidence.reference.revision {
                ensure!(
                    valid_reference(revision),
                    "invalid missing-evidence reference revision"
                );
            }
        } else {
            validate_reference(&evidence.reference)?;
        }
    }
    if let Some(graph) = &package.graph_context {
        validate_reference(&graph.reference)?;
        let watermark = graph
            .watermark
            .as_deref()
            .context("graph context requires a watermark")?;
        ensure!(valid_reference(watermark), "invalid graph watermark");
    }
    validate_reference_list(&package.policy_references, "policy references")?;
    validate_reference_list(&package.calculation_references, "calculation references")?;
    validate_reference_list(&package.model_references, "model references")?;
    validate_reference_list(&package.producer_references, "producer references")?;
    validate_typed_references(&package.reviews, "review references", |item| {
        &item.reference
    })?;
    validate_typed_references(
        &package.authorization_observations,
        "authorization observation references",
        |item| &item.reference,
    )?;
    validate_typed_references(
        &package.effect_attempts,
        "effect attempt references",
        |item| &item.reference,
    )?;
    validate_typed_references(
        &package.acknowledgements,
        "acknowledgement references",
        |item| &item.reference,
    )?;
    validate_typed_references(&package.outcomes, "outcome references", |item| {
        &item.reference
    })?;
    validate_typed_references(
        &package.reconciliations,
        "reconciliation references",
        |item| &item.reference,
    )?;
    if let Some(reference) = &package.correction_of {
        validate_revision_reference(reference, package)?;
    }
    if let Some(reference) = &package.successor_to {
        validate_revision_reference(reference, package)?;
    }
    ensure!(
        package.verification_ceiling == VerificationCeiling::RecordedOnly,
        "decision verification must remain recorded only"
    );
    ensure!(
        package.execution_authority == ExecutionAuthority::None,
        "decision journal cannot grant execution authority"
    );
    canonical_bytes(package)?;
    Ok(())
}

/// Validates a closed v1 package without opening a journal transaction.
/// This is integrity and bounded-shape validation only; it grants no authority.
pub fn validate_decision_package_v1(package: &DecisionPackageV1) -> Result<()> {
    validate_package(package)
}

/// Validates append coordinates before a caller records a durable request.
pub fn validate_decision_append_request(expected_revision: i64) -> Result<()> {
    ensure!(
        (0..=MAX_REVISION).contains(&expected_revision),
        "invalid expected decision revision"
    );
    Ok(())
}

/// Validates an exact read coordinate before a caller records a durable request.
pub fn validate_decision_read_request(decision_id: &str, expected_revision: i64) -> Result<()> {
    ensure!(valid_identifier(decision_id), "invalid decision id");
    ensure!(
        (1..=MAX_REVISION).contains(&expected_revision),
        "invalid expected decision revision"
    );
    Ok(())
}

impl Journal {
    fn initialize_decision_audit(tx: &rusqlite::Transaction<'_>) -> Result<()> {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS decision_audits(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                decision_id TEXT NOT NULL, run_id TEXT NOT NULL, goal_id TEXT NOT NULL, goal_revision INTEGER NOT NULL,
                goal_digest TEXT NOT NULL, goal_body BLOB NOT NULL, current_revision INTEGER NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization,project,environment,site,decision_id));
              CREATE TABLE IF NOT EXISTS decision_audit_events(
                organization TEXT NOT NULL, project TEXT NOT NULL, environment TEXT NOT NULL, site TEXT NOT NULL,
                decision_id TEXT NOT NULL, sequence INTEGER NOT NULL, revision INTEGER NOT NULL,
                request_id TEXT NOT NULL, request_digest TEXT NOT NULL, package_body BLOB NOT NULL,
                package_digest TEXT NOT NULL, predecessor_event_digest TEXT, event_digest TEXT NOT NULL, created_at INTEGER NOT NULL,
                PRIMARY KEY(organization,project,environment,site,decision_id,sequence),
                UNIQUE(organization,project,environment,site,decision_id,revision),
                UNIQUE(organization,project,environment,site,decision_id,request_id));"
        )?;
        Ok(())
    }

    fn current_decision_binding(
        tx: &rusqlite::Transaction<'_>,
        run_id: &str,
    ) -> Result<(crate::GoalRunBinding, String, Vec<u8>)> {
        let binding = match Self::gate_goal_run_in_tx(tx, run_id)? {
            GoalRunGate::Current(binding) => binding,
            GoalRunGate::Unbound => {
                anyhow::bail!("decision audit requires a current goal-bound run")
            }
            GoalRunGate::NeedsReview => {
                anyhow::bail!("decision audit requires a current goal-bound run")
            }
        };
        let body: String = tx.query_row(
            "SELECT body FROM goals WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND id=?5",
            params![binding.scope.organization, binding.scope.project, binding.scope.environment, binding.scope.site, binding.goal_id],
            |row| row.get(0),
        )?;
        let goal: Goal = serde_json::from_str(&body).context("corrupt bound goal body")?;
        validate_goal(&goal)?;
        ensure!(
            goal.scope == binding.scope
                && goal.id == binding.goal_id
                && goal.revision == binding.goal_revision,
            "bound goal changed"
        );
        let body = canonical_bytes(&goal)?;
        Ok((binding, digest(&body), body))
    }

    fn verify_recorded_decision_binding(binding: RecordedDecisionBinding<'_>) -> Result<()> {
        validate_scope(binding.scope)?;
        ensure!(
            valid_identifier(binding.run_id) && valid_identifier(binding.goal_id),
            "corrupt decision binding identity"
        );
        ensure!(
            (1..=MAX_REVISION).contains(&binding.goal_revision),
            "corrupt decision goal revision"
        );
        ensure!(
            binding.goal_digest.len() == 64
                && binding
                    .goal_digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "corrupt decision goal digest"
        );
        let goal: Goal = serde_json::from_slice(binding.goal_body)
            .context("corrupt recorded decision goal snapshot")?;
        validate_goal(&goal)?;
        ensure!(
            goal.scope == *binding.scope
                && goal.id == binding.goal_id
                && goal.revision == binding.goal_revision,
            "recorded decision goal snapshot mismatch"
        );
        ensure!(
            canonical_bytes(&goal)? == binding.goal_body,
            "recorded decision goal snapshot is not canonical"
        );
        ensure!(
            digest(binding.goal_body) == binding.goal_digest,
            "recorded decision goal digest mismatch"
        );
        ensure!(valid_identifier(binding.decision_id), "invalid decision id");
        Ok(())
    }

    /// Appends an immutable revision after a goal/run, scope, revision and exact
    /// canonical goal body are all proven current in the same SQLite transaction.
    pub fn append_decision_package(
        &self,
        run_id: &str,
        expected_revision: i64,
        request_id: &str,
        package: DecisionPackageV1,
    ) -> Result<DecisionAuditProjectionV1> {
        ensure!(
            valid_identifier(run_id) && valid_identifier(request_id),
            "invalid decision append identity"
        );
        ensure!(
            (0..=MAX_REVISION).contains(&expected_revision),
            "invalid expected decision revision"
        );
        validate_package(&package)?;
        let package_bytes = canonical_bytes(&package)?;
        let package_digest = digest(&package_bytes);
        let request_bytes = canonical_bytes(&(run_id, expected_revision, request_id, &package))?;
        let request_digest = digest(&request_bytes);
        let tx = self.conn.unchecked_transaction()?;
        Self::initialize_decision_audit(&tx)?;
        let (binding, goal_digest, goal_body) = Self::current_decision_binding(&tx, run_id)?;
        let existing: Option<(String, String, i64, String, Vec<u8>, i64)> = tx.query_row(
            "SELECT run_id,goal_id,goal_revision,goal_digest,goal_body,current_revision FROM decision_audits WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5",
            params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?)),
        ).optional()?;
        if let Some((
            stored_run,
            goal_id,
            goal_revision,
            stored_digest,
            stored_goal_body,
            current_revision,
        )) = existing
        {
            ensure!(
                stored_run == run_id
                    && goal_id == binding.goal_id
                    && goal_revision == binding.goal_revision
                    && stored_digest == goal_digest,
                "decision audit continuity binding mismatch"
            );
            Self::verify_recorded_decision_binding(RecordedDecisionBinding {
                scope: &binding.scope,
                decision_id: &package.decision_id,
                run_id,
                goal_id: &goal_id,
                goal_revision,
                goal_digest: &stored_digest,
                goal_body: &stored_goal_body,
            })?;
            let replay: Option<(String, i64)> = tx.query_row(
                "SELECT request_digest,revision FROM decision_audit_events WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5 AND request_id=?6",
                params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id,request_id],
                |row| Ok((row.get(0)?,row.get(1)?)),
            ).optional()?;
            if let Some((stored_request, _)) = replay {
                ensure!(
                    stored_request == request_digest,
                    "decision request id replay has different content"
                );
                let result = Self::read_decision_projection_in_tx(
                    &tx,
                    &binding.scope,
                    &package.decision_id,
                )?;
                tx.commit()?;
                return Ok(result);
            }
            ensure!(
                expected_revision == current_revision && package.revision == current_revision + 1,
                "decision revision conflict"
            );
        } else {
            ensure!(
                expected_revision == 0 && package.revision == 1,
                "first decision revision must be one"
            );
            let count: i64 = tx.query_row("SELECT COUNT(*) FROM decision_audits WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4", params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site], |row| row.get(0))?;
            ensure!(
                count < MAX_DECISIONS_PER_SCOPE,
                "too many decisions in scope"
            );
            tx.execute("INSERT INTO decision_audits(organization,project,environment,site,decision_id,run_id,goal_id,goal_revision,goal_digest,goal_body,current_revision,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,0,?11)", params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id,run_id,binding.goal_id,binding.goal_revision,goal_digest,goal_body,crate::now()])?;
        }
        let predecessor_event_digest: Option<String> = tx.query_row(
            "SELECT event_digest FROM decision_audit_events WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5 ORDER BY sequence DESC LIMIT 1",
            params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id], |row| row.get(0),
        ).optional()?;
        let sequence: i64 = tx.query_row(
            "SELECT COALESCE(MAX(sequence),0)+1 FROM decision_audit_events WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5",
            params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id], |row| row.get(0),
        )?;
        let created_at = crate::now();
        let event = DecisionAuditEventV1 {
            scope: binding.scope.clone(),
            run_id: run_id.into(),
            goal_id: binding.goal_id.clone(),
            goal_revision: binding.goal_revision,
            goal_digest: goal_digest.clone(),
            sequence,
            package: package.clone(),
            request_id: request_id.into(),
            request_digest: request_digest.clone(),
            package_digest: package_digest.clone(),
            event_digest: String::new(),
            predecessor_event_digest: predecessor_event_digest.clone(),
            created_at,
        };
        let event_digest = event_digest(&event)?;
        tx.execute("INSERT INTO decision_audit_events(organization,project,environment,site,decision_id,sequence,revision,request_id,request_digest,package_body,package_digest,predecessor_event_digest,event_digest,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)", params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id,sequence,package.revision,request_id,request_digest,package_bytes,package_digest,predecessor_event_digest,event_digest,created_at])?;
        tx.execute("UPDATE decision_audits SET current_revision=?6 WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5", params![binding.scope.organization,binding.scope.project,binding.scope.environment,binding.scope.site,package.decision_id,package.revision])?;
        let result =
            Self::read_decision_projection_in_tx(&tx, &binding.scope, &package.decision_id)?;
        tx.commit()?;
        Ok(result)
    }

    pub fn decision_audit(
        &self,
        scope: &GoalScope,
        decision_id: &str,
    ) -> Result<DecisionAuditProjectionV1> {
        validate_scope(scope)?;
        ensure!(valid_identifier(decision_id), "invalid decision id");
        let tx = self.conn.unchecked_transaction()?;
        Self::initialize_decision_audit(&tx)?;
        let result = Self::read_decision_projection_in_tx(&tx, scope, decision_id)?;
        tx.commit()?;
        Ok(result)
    }

    fn read_decision_projection_in_tx(
        tx: &rusqlite::Transaction<'_>,
        scope: &GoalScope,
        decision_id: &str,
    ) -> Result<DecisionAuditProjectionV1> {
        let (run_id, goal_id, goal_revision, goal_digest, goal_body, current_revision): (String,String,i64,String,Vec<u8>,i64) = tx.query_row(
            "SELECT run_id,goal_id,goal_revision,goal_digest,goal_body,current_revision FROM decision_audits WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5",
            params![scope.organization,scope.project,scope.environment,scope.site,decision_id],
            |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?)),
        ).optional()?.with_context(|| format!("no decision {decision_id} in scope"))?;
        Self::verify_recorded_decision_binding(RecordedDecisionBinding {
            scope,
            decision_id,
            run_id: &run_id,
            goal_id: &goal_id,
            goal_revision,
            goal_digest: &goal_digest,
            goal_body: &goal_body,
        })?;
        ensure!(
            (1..=MAX_REVISION).contains(&current_revision),
            "corrupt decision revision"
        );
        let mut stmt = tx.prepare("SELECT sequence,revision,request_id,request_digest,package_body,package_digest,predecessor_event_digest,event_digest,created_at FROM decision_audit_events WHERE organization=?1 AND project=?2 AND environment=?3 AND site=?4 AND decision_id=?5 ORDER BY sequence")?;
        let rows = stmt.query_map(
            params![
                scope.organization,
                scope.project,
                scope.environment,
                scope.site,
                decision_id
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                ))
            },
        )?;
        let mut events = Vec::new();
        let mut predecessor_event_digest = None;
        for (index, row) in rows.enumerate() {
            let (
                sequence,
                revision,
                request_id,
                request_digest,
                bytes,
                package_digest,
                row_predecessor,
                row_event_digest,
                created_at,
            ) = row?;
            let expected = i64::try_from(index + 1).context("decision history length overflow")?;
            ensure!(
                sequence == expected && revision == expected,
                "decision history is missing, duplicate, or gapped"
            );
            ensure!(
                row_predecessor == predecessor_event_digest,
                "decision predecessor link mismatch"
            );
            ensure!(
                digest(&bytes) == package_digest,
                "decision package digest mismatch"
            );
            let package: DecisionPackageV1 =
                serde_json::from_slice(&bytes).context("corrupt decision package")?;
            validate_package(&package)?;
            ensure!(
                package.decision_id == decision_id && package.revision == revision,
                "decision package identity mismatch"
            );
            ensure!(
                canonical_bytes(&package)? == bytes,
                "decision package is not canonical"
            );
            let event = DecisionAuditEventV1 {
                scope: scope.clone(),
                run_id: run_id.clone(),
                goal_id: goal_id.clone(),
                goal_revision,
                goal_digest: goal_digest.clone(),
                sequence,
                package,
                request_id,
                request_digest,
                package_digest,
                event_digest: row_event_digest,
                predecessor_event_digest: row_predecessor,
                created_at,
            };
            verify_decision_audit_event(&event)?;
            predecessor_event_digest = Some(event.event_digest.clone());
            events.push(event);
        }
        ensure!(
            events.len() as i64 == current_revision,
            "decision history is missing, duplicate, or truncated"
        );
        let last = events.last().context("empty decision history")?;
        Ok(DecisionAuditProjectionV1 {
            scope: scope.clone(),
            run_id,
            goal_id,
            goal_revision,
            goal_digest,
            decision_id: decision_id.into(),
            current_revision,
            verification_ceiling: last.package.verification_ceiling,
            execution_authority: last.package.execution_authority,
            events,
        })
    }
}
