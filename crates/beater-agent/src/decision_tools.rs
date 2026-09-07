//! Goal-bound, runner-private decision journal tools.
//!
//! These declarations are deliberately not `ToolRegistry` entries: that
//! registry is also published by the dev server's MCP surface, whose synthetic
//! runs have no goal binding. They are only offered by the durable runner after
//! it proves the run's current local goal binding.

use anyhow::{Context, Result, ensure};
use beater_journal::{
    DecisionAuditProjectionV1, DecisionPackageV1, GoalRunGate, Journal,
    validate_decision_append_request, validate_decision_package_v1, validate_decision_read_request,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::resume_contract::ToolResumeContract;

pub const APPEND_TOOL_NAME: &str = "decision_audit_append";
pub const READ_TOOL_NAME: &str = "decision_audit_read";
const TOOL_VERSION: u8 = 1;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AppendInput {
    version: u8,
    expected_revision: i64,
    package: DecisionPackageV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadInput {
    version: u8,
    decision_id: String,
    /// A replay of this read may return only this exact historical revision.
    /// It must never present a newer current projection as the old result.
    expected_revision: i64,
}

pub fn rejects_configured_name(name: &str) -> bool {
    matches!(name, APPEND_TOOL_NAME | READ_TOOL_NAME)
}

pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": APPEND_TOOL_NAME,
            "description": "Record a typed local decision package for the current goal-bound run. This records references only and grants no authority or verification.",
            "input_schema": append_schema(),
        }),
        json!({
            "name": READ_TOOL_NAME,
            "description": "Read one exact local decision-history revision for the current goal-bound run. This is recorded history, not current authorization or verification.",
            "input_schema": read_schema(),
        }),
    ]
}

pub fn resume_contract(name: &str) -> Option<ToolResumeContract> {
    let schema = match name {
        APPEND_TOOL_NAME => append_schema(),
        READ_TOOL_NAME => read_schema(),
        _ => return None,
    };
    ToolResumeContract::new(
        name,
        true,
        &json!({
            "domain": "TEMPERA_DECISION_RUNTIME_TOOL_V1",
            "tool_version": TOOL_VERSION,
            "name": name,
            "input_schema": schema,
        }),
    )
}

pub fn append(
    journal: &Journal,
    run_id: &str,
    request_id: &str,
    input: &Value,
) -> Result<DecisionAuditProjectionV1> {
    let input = parse_append(input)?;
    // `append_decision_package` validates the entire bounded package before it
    // starts its SQLite write transaction and proves the current binding inside
    // that transaction.
    journal.append_decision_package(run_id, input.expected_revision, request_id, input.package)
}

pub fn validate_append(input: &Value) -> Result<()> {
    let input = parse_append(input)?;
    validate_decision_append_request(input.expected_revision)?;
    validate_decision_package_v1(&input.package)?;
    Ok(())
}

pub fn validate_read(input: &Value) -> Result<()> {
    let input = parse_read(input)?;
    validate_decision_read_request(&input.decision_id, input.expected_revision)?;
    Ok(())
}

pub fn read(journal: &Journal, run_id: &str, input: &Value) -> Result<DecisionAuditProjectionV1> {
    let input = parse_read(input)?;
    let scope = match journal.gate_goal_run(run_id)? {
        GoalRunGate::Current(binding) => binding.scope,
        GoalRunGate::Unbound => anyhow::bail!("decision read requires a current goal-bound run"),
        GoalRunGate::NeedsReview => return Err(beater_journal::GoalRunNeedsReview.into()),
    };
    let projection = journal.decision_audit(&scope, &input.decision_id)?;
    ensure!(
        projection.current_revision == input.expected_revision,
        "decision read revision changed; exact historical replay is unavailable"
    );
    Ok(projection)
}

fn parse_append(input: &Value) -> Result<AppendInput> {
    let input: AppendInput = serde_json::from_value(input.clone())
        .context("decision append input must match the closed v1 shape")?;
    ensure!(
        input.version == TOOL_VERSION,
        "unsupported decision append tool version"
    );
    Ok(input)
}

fn parse_read(input: &Value) -> Result<ReadInput> {
    let input: ReadInput = serde_json::from_value(input.clone())
        .context("decision read input must match the closed v1 shape")?;
    ensure!(
        input.version == TOOL_VERSION,
        "unsupported decision read tool version"
    );
    Ok(input)
}

fn append_schema() -> Value {
    // The nested package is parsed by `DecisionPackageV1` before a journaled
    // tool step is opened. Keep this descriptor closed instead of advertising
    // an open package blob as a model-facing contract.
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["version", "expected_revision", "package"],
        "properties": {
            "version": {"type": "integer", "const": TOOL_VERSION},
            "expected_revision": {"type": "integer", "minimum": 0, "maximum": 1_000_000_000},
            "package": decision_package_schema()
        }
    })
}

fn reference_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false,
        "required": ["locator", "revision"],
        "properties": {
            "locator": {"type": "string", "minLength": 1, "maxLength": 512},
            "revision": {"type": ["string", "null"], "maxLength": 512}
        }
    })
}

fn typed_reference_schema() -> Value {
    json!({
        "type": "object", "additionalProperties": false,
        "required": ["reference"],
        "properties": {"reference": reference_schema()}
    })
}

fn decision_package_schema() -> Value {
    let reference = reference_schema();
    let typed_reference = typed_reference_schema();
    let evidence = json!({
        "type": "object", "additionalProperties": false,
        "required": ["reference", "missing"],
        "properties": {"reference": reference.clone(), "missing": {"type": "boolean"}}
    });
    let graph_context = json!({
        "type": "object", "additionalProperties": false,
        "required": ["reference", "watermark"],
        "properties": {
            "reference": reference.clone(),
            "watermark": {"type": ["string", "null"], "maxLength": 512}
        }
    });
    let revision_reference = json!({
        "type": "object", "additionalProperties": false,
        "required": ["decision_id", "revision"],
        "properties": {
            "decision_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "revision": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000}
        }
    });
    json!({
        "type": "object", "additionalProperties": false,
        "required": [
            "version", "decision_id", "revision", "domain", "question", "object_references",
            "evidence", "graph_context", "policy_references", "calculation_references", "model_references",
            "producer_references", "options", "constraints", "rationale", "reviews",
            "authorization_observations", "effect_attempts", "acknowledgements", "outcomes", "reconciliations",
            "correction_of", "successor_to", "verification_ceiling", "execution_authority"
        ],
        "properties": {
            "version": {"type": "integer", "const": 1},
            "decision_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "revision": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000},
            "domain": {"type": "string", "enum": ["software", "supply", "payments", "operations"]},
            "question": {"type": "string", "minLength": 1, "maxLength": 4096},
            "object_references": {"type": "array", "maxItems": 64, "items": reference.clone()},
            "evidence": {"type": "array", "maxItems": 64, "items": evidence},
            "graph_context": {"anyOf": [graph_context, {"type": "null"}]},
            "policy_references": {"type": "array", "maxItems": 64, "items": reference.clone()},
            "calculation_references": {"type": "array", "maxItems": 64, "items": reference.clone()},
            "model_references": {"type": "array", "maxItems": 64, "items": reference.clone()},
            "producer_references": {"type": "array", "maxItems": 64, "items": reference.clone()},
            "options": {"type": "array", "maxItems": 32, "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
            "constraints": {"type": "array", "maxItems": 32, "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
            "rationale": {"type": "string", "minLength": 1, "maxLength": 4096},
            "reviews": {"type": "array", "maxItems": 64, "items": typed_reference.clone()},
            "authorization_observations": {"type": "array", "maxItems": 64, "items": typed_reference.clone()},
            "effect_attempts": {"type": "array", "maxItems": 64, "items": typed_reference.clone()},
            "acknowledgements": {"type": "array", "maxItems": 64, "items": typed_reference.clone()},
            "outcomes": {"type": "array", "maxItems": 64, "items": typed_reference},
            "reconciliations": {"type": "array", "maxItems": 64, "items": typed_reference_schema()},
            "correction_of": {"anyOf": [revision_reference.clone(), {"type": "null"}]},
            "successor_to": {"anyOf": [revision_reference, {"type": "null"}]},
            "verification_ceiling": {"type": "string", "const": "RecordedOnly"},
            "execution_authority": {"type": "string", "const": "None"}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{APPEND_TOOL_NAME, READ_TOOL_NAME, rejects_configured_name, tool_definitions};

    #[test]
    fn private_schemas_are_closed_and_versioned() {
        let definitions = tool_definitions();
        assert_eq!(definitions.len(), 2);
        assert!(rejects_configured_name(APPEND_TOOL_NAME));
        assert!(rejects_configured_name(READ_TOOL_NAME));
        let append = definitions
            .iter()
            .find(|tool| tool["name"] == APPEND_TOOL_NAME)
            .unwrap();
        assert_eq!(append["input_schema"]["additionalProperties"], false);
        assert_eq!(append["input_schema"]["properties"]["version"]["const"], 1);
        assert_eq!(
            append["input_schema"]["properties"]["package"]["additionalProperties"],
            false
        );
        let read = definitions
            .iter()
            .find(|tool| tool["name"] == READ_TOOL_NAME)
            .unwrap();
        assert_eq!(read["input_schema"]["additionalProperties"], false);
        assert_eq!(read["input_schema"]["properties"]["version"]["const"], 1);
    }
}

fn read_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["version", "decision_id", "expected_revision"],
        "properties": {
            "version": {"type": "integer", "const": TOOL_VERSION},
            "decision_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "expected_revision": {"type": "integer", "minimum": 1, "maximum": 1_000_000_000}
        }
    })
}
