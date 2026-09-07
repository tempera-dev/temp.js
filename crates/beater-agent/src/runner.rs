//! The agent loop. Lives in Rust — not the JS isolate — so it survives hot
//! reloads and every LLM/tool step is journaled before it executes.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::decision_tools;
use crate::llm::{LlmClient, LlmSelection};
use crate::ownership::RunOwnership;
use crate::registry::{
    AgentConfig, BeatboxConfig, ToolCallContext, ToolNeedsReview, ToolRegistry,
    browser_session_dir, cleanup_stale_browser_sessions,
};
use crate::resume_contract::ToolResumeContract;
use crate::trace_export;
use beater_journal::{GoalRunGate, GoalRunNeedsReview, GoalScope, Journal};

const MAX_TOKENS: u64 = 16000;
const MAX_LOOP_STEPS: usize = 50;

struct Ctx {
    journal: Journal,
    client: LlmClient,
    registry: ToolRegistry,
    config: AgentConfig,
    model: String,
    run_id: String,
}

pub struct JournaledToolCall {
    pub seq: i64,
    pub context: ToolCallContext,
}

/// Trusted-local linkage, not authenticated scope or effect authority. Keep the
/// caller-chosen run ID to resume the same durable work after a setup failure.
pub struct GoalRunRequest {
    pub run_id: String,
    pub scope: GoalScope,
    pub goal_id: String,
    pub expected_revision: i64,
    pub agent_name: String,
    pub prompt: String,
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

fn setup(
    app_dir: &Path,
    config_value: Value,
    venv: Option<&PathBuf>,
    beatbox: &BeatboxConfig,
) -> Result<(AgentConfig, ToolRegistry)> {
    if let Some(venv) = venv {
        if venv.is_dir() {
            beater_py::attach_venv(venv)?;
        } else {
            tracing::info!("no venv at {} — stdlib-only python tools", venv.display());
        }
    }
    let config: AgentConfig = serde_json::from_value(config_value)
        .context("agent.ts default export did not match defineAgent shape")?;
    anyhow::ensure!(
        config
            .tools
            .iter()
            .all(|tool| !decision_tools::rejects_configured_name(&tool.name)),
        "agent configuration cannot claim a runner-private decision tool name"
    );
    let agent_dir = app_dir.join("agents").join(&config.name);
    let registry = ToolRegistry::build_with_beatbox_and_browser_session_dir(
        &agent_dir,
        &config.tools,
        beatbox,
        Some(browser_session_dir(app_dir)),
    )?;
    Ok((config, registry))
}

pub fn run(
    app_dir: &Path,
    agent_name: &str,
    config_value: Value,
    venv: Option<PathBuf>,
    beatbox: BeatboxConfig,
    prompt: &str,
) -> Result<()> {
    let run_id = uuid::Uuid::new_v4().to_string();
    let _ownership = RunOwnership::acquire(app_dir, &run_id)?;
    let (config, registry) = setup(app_dir, config_value, venv.as_ref(), &beatbox)?;
    anyhow::ensure!(
        config.name == agent_name,
        "agent.ts declares name {:?} but directory is {agent_name:?}",
        config.name
    );
    let llm = LlmSelection::from_config(&config)?;
    let client = LlmClient::from_provider(&llm.provider)?;
    let journal = Journal::open(app_dir)?;
    journal.create_run(&run_id, agent_name, prompt)?;
    println!("run {run_id}");

    let ctx = Ctx {
        journal,
        client,
        registry,
        config,
        model: llm.model,
        run_id,
    };
    let messages = vec![json!({"role": "user", "content": prompt})];
    let result = runtime()?.block_on(async {
        let result = agent_loop(&ctx, messages, 1).await;
        finish_goal_review(&ctx, result).await
    });
    export_run_trace_best_effort(app_dir, &ctx.run_id);
    result
}

/// Atomically persist a goal-bound run, then use the existing worker under the
/// same ownership lock. A duplicate run ID is an error; use `resume` thereafter.
pub fn run_for_goal(
    app_dir: &Path,
    request: &GoalRunRequest,
    venv: Option<PathBuf>,
    beatbox: BeatboxConfig,
    load_config: impl Fn(&str) -> Result<Value>,
) -> Result<()> {
    let _ownership = RunOwnership::acquire(app_dir, &request.run_id)?;
    let journal = Journal::open(app_dir)?;
    journal.create_goal_bound_run(
        &request.scope,
        &request.goal_id,
        request.expected_revision,
        &request.run_id,
        &request.agent_name,
        &request.prompt,
    )?;
    println!("run {}", request.run_id);
    resume_owned(
        app_dir,
        &request.run_id,
        journal,
        venv,
        beatbox,
        load_config,
    )
}

pub fn resume(
    app_dir: &Path,
    run_id: &str,
    venv: Option<PathBuf>,
    beatbox: BeatboxConfig,
    load_config: impl Fn(&str) -> Result<Value>,
) -> Result<()> {
    let _ownership = RunOwnership::acquire(app_dir, run_id)?;
    let journal = Journal::open(app_dir)?;
    resume_owned(app_dir, run_id, journal, venv, beatbox, load_config)
}

fn resume_owned(
    app_dir: &Path,
    run_id: &str,
    journal: Journal,
    venv: Option<PathBuf>,
    beatbox: BeatboxConfig,
    load_config: impl Fn(&str) -> Result<Value>,
) -> Result<()> {
    let run = journal.run(run_id)?;
    if matches!(journal.gate_goal_run(run_id)?, GoalRunGate::NeedsReview) {
        println!("run {run_id} needs review: its goal binding is no longer current");
        return Ok(());
    }
    if run.status == "completed" {
        println!("run {run_id} already completed");
        return Ok(());
    }
    cleanup_stale_browser_sessions(app_dir, run_id)
        .with_context(|| format!("cleaning stale browser sessions for run {run_id}"))?;
    let config_value = load_config(&run.agent)?;
    anyhow::ensure!(
        config_value["name"].as_str() == Some(run.agent.as_str()),
        "resumed agent configuration does not match the saved agent"
    );
    let (config, registry) = setup(app_dir, config_value, venv.as_ref(), &beatbox)?;
    let steps = journal.steps(run_id)?;
    let llm = LlmSelection::from_config(&config)?;

    let ctx = Ctx {
        journal,
        client: LlmClient::from_provider(&llm.provider)?,
        registry,
        config,
        model: llm.model,
        run_id: run_id.to_string(),
    };
    let result = runtime()?.block_on(async {
        let result = resume_async(&ctx, run, steps).await;
        finish_goal_review(&ctx, result).await
    });
    export_run_trace_best_effort(app_dir, &ctx.run_id);
    result
}

async fn finish_goal_review(ctx: &Ctx, result: Result<()>) -> Result<()> {
    match result {
        Err(error) if error.downcast_ref::<GoalRunNeedsReview>().is_some() => {
            println!(
                "run {} needs review: its goal binding is no longer current",
                ctx.run_id
            );
            close_browser_sessions_best_effort(ctx).await;
            Ok(())
        }
        result => result,
    }
}

fn is_review_error(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ToolNeedsReview>().is_some()
        || error.downcast_ref::<GoalRunNeedsReview>().is_some()
}

fn export_run_trace_best_effort(app_dir: &Path, run_id: &str) {
    if let Err(error) = trace_export::export_run_if_configured(app_dir, run_id) {
        tracing::warn!("trace export for run {run_id} failed: {error:#}");
    }
}

async fn resume_async(
    ctx: &Ctx,
    run: beater_journal::RunRow,
    steps: Vec<beater_journal::StepRow>,
) -> Result<()> {
    let run_id = ctx.run_id.as_str();
    // Rebuild conversation state from the journal. The last llm_call's request
    // body carries the exact messages[] at that point — no delta replay needed.
    let last_llm = steps.iter().rev().find(|s| s.kind == "llm_call");
    let mut next_llm_attempt = 1;
    let messages = match last_llm {
        None => vec![json!({"role": "user", "content": run.input})],
        Some(step) if step.status != "completed" => {
            // Dangling LLM call: we own the request and it had no observable
            // side effect on our state — always safe to re-issue.
            next_llm_attempt = step.attempt + 1;
            println!(
                "resuming: re-issuing interrupted LLM call (attempt {})",
                next_llm_attempt
            );
            step.request["messages"]
                .as_array()
                .context("journaled llm_call request has no messages")?
                .clone()
        }
        Some(step) => {
            let response = step.result.as_ref().context("completed step has result")?;
            let content = response["content"].clone();
            let mut messages = step.request["messages"]
                .as_array()
                .context("journaled llm_call request has no messages")?
                .clone();
            messages.push(json!({"role": "assistant", "content": content}));
            let stop_reason = response["stop_reason"].as_str().unwrap_or_default();

            if stop_reason == "end_turn" {
                // The last response needed no tools; the run actually finished.
                ctx.journal.set_run_status(run_id, "completed")?;
                println!("run {run_id} was already finished — marked completed");
                close_browser_sessions_best_effort(ctx).await;
                return Ok(());
            }
            if stop_reason == "pause_turn" {
                // Server-side pause: assistant turn is already appended; ask
                // the model to continue from exactly the journaled state.
                messages
            } else if stop_reason != "tool_use" {
                ctx.journal.set_run_status(run_id, "failed")?;
                close_browser_sessions_best_effort(ctx).await;
                if stop_reason == "refusal" {
                    bail!(
                        "run {run_id} failed before resume: model refused: {}",
                        response["stop_details"]
                    );
                }
                bail!(
                    "run {run_id} failed before resume: unexpected stop_reason {stop_reason:?} — raise max_tokens or inspect the journal"
                );
            } else {
                let tool_uses: Vec<Value> = content
                    .as_array()
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter(|b| b["type"] == "tool_use")
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default();
                if tool_uses.is_empty() {
                    ctx.journal.set_run_status(run_id, "failed")?;
                    close_browser_sessions_best_effort(ctx).await;
                    bail!(
                        "run {run_id} failed before resume: stop_reason \"tool_use\" had no tool_use blocks"
                    );
                }

                // Fill in tool results: journaled ones verbatim; dangling ones
                // Replay requires the original declaration and current descriptor
                // to agree. Mutable current configuration cannot bless old work.
                let mut tool_results = Vec::new();
                for tu in &tool_uses {
                    let (id, name) = (
                        tu["id"].as_str().unwrap_or_default(),
                        tu["name"].as_str().unwrap_or_default(),
                    );
                    let done = steps.iter().find(|s| {
                        s.kind == "tool_call"
                            && s.status == "completed"
                            && s.tool_use_id.as_deref() == Some(id)
                    });
                    let tool_result = match done {
                        Some(s)
                            if completed_tool_matches_recorded_contract(
                                run_id,
                                name,
                                id,
                                &tu["input"],
                                s,
                                decision_tools::resume_contract(name)
                                    .as_ref()
                                    .or(ctx.registry.resume_contract(name).as_ref()),
                            ) =>
                        {
                            let content = s
                                .result
                                .as_ref()
                                .and_then(|r| r["content"].as_str())
                                .unwrap_or_default()
                                .to_string();
                            json!({"type": "tool_result", "tool_use_id": id, "content": content})
                        }
                        Some(_) => {
                            ctx.journal.set_run_status(run_id, "needs_review")?;
                            println!(
                                "run {run_id} needs review: completed tool {name} ({id}) does not exactly match its recorded declaration"
                            );
                            close_browser_sessions_best_effort(ctx).await;
                            return Ok(());
                        }
                        None => {
                            let private_contract = decision_tools::resume_contract(name);
                            let registry_contract = ctx.registry.resume_contract(name);
                            if !tool_replay_matches_recorded_contract(
                                run_id,
                                name,
                                id,
                                &tu["input"],
                                &steps,
                                private_contract.as_ref().or(registry_contract.as_ref()),
                            ) {
                                ctx.journal.set_run_status(run_id, "needs_review")?;
                                println!(
                                    "run {run_id} needs review: tool {name} ({id}) may have executed \
                                     before the crash and lacks an unchanged, originally idempotent replay contract"
                                );
                                close_browser_sessions_best_effort(ctx).await;
                                return Ok(());
                            }
                            let prior_attempts = steps
                                .iter()
                                .filter(|s| s.tool_use_id.as_deref() == Some(id))
                                .map(|s| s.attempt)
                                .max()
                                .unwrap_or(0);
                            println!(
                                "resuming: re-running interrupted tool {name} (attempt {})",
                                prior_attempts + 1
                            );
                            match execute_tool_step(ctx, name, id, &tu["input"], prior_attempts + 1)
                                .await
                            {
                                Ok(content) => {
                                    json!({"type": "tool_result", "tool_use_id": id, "content": content})
                                }
                                Err(e) if is_review_error(&e) => {
                                    println!("← needs review: {e:#}");
                                    ctx.journal.set_run_status(run_id, "needs_review")?;
                                    close_browser_sessions_best_effort(ctx).await;
                                    return Ok(());
                                }
                                Err(e) => {
                                    if decision_tools::resume_contract(name).is_some() {
                                        println!(
                                            "← tool {name} rejected (decision content redacted)"
                                        );
                                    } else {
                                        println!("← tool error: {e:#}");
                                    }
                                    json!({
                                        "type": "tool_result",
                                        "tool_use_id": id,
                                        "content": if decision_tools::resume_contract(name).is_some() { "Error: DECISION_AUDIT_REJECTED".to_string() } else { format!("Error: {e:#}") },
                                        "is_error": true,
                                    })
                                }
                            }
                        }
                    };
                    tool_results.push(tool_result);
                }
                messages.push(json!({"role": "user", "content": tool_results}));
                messages
            }
        }
    };

    ctx.journal.set_run_status(run_id, "running")?;
    agent_loop(ctx, messages, next_llm_attempt).await
}

pub fn list_runs(app_dir: &Path) -> Result<()> {
    let journal = Journal::open(app_dir)?;
    let runs = journal.list_runs()?;
    if runs.is_empty() {
        println!("no runs");
        return Ok(());
    }
    println!(
        "{:<38} {:<12} {:<13} {:>5}  input",
        "run", "agent", "status", "steps"
    );
    for (run, steps) in runs {
        let input: String = run.input.chars().take(40).collect();
        println!(
            "{:<38} {:<12} {:<13} {:>5}  {input}",
            run.id, run.agent, run.status, steps
        );
    }
    Ok(())
}

async fn agent_loop(ctx: &Ctx, mut messages: Vec<Value>, mut next_llm_attempt: i64) -> Result<()> {
    for _ in 0..MAX_LOOP_STEPS {
        let mut tools = ctx.registry.api_tools();
        match ctx.journal.gate_goal_run(&ctx.run_id)? {
            GoalRunGate::Current(_) => {
                let definitions = tools
                    .as_array_mut()
                    .context("tool registry returned a non-array definition")?;
                definitions.extend(decision_tools::tool_definitions());
            }
            GoalRunGate::Unbound => {}
            GoalRunGate::NeedsReview => return Err(GoalRunNeedsReview.into()),
        }
        let body = json!({
            "model": ctx.model,
            "max_tokens": MAX_TOKENS,
            "system": ctx.config.system,
            "thinking": {"type": "adaptive"},
            "tools": tools,
            "messages": messages,
        });

        let seq =
            ctx.journal
                .start_step(&ctx.run_id, "llm_call", &body, None, None, next_llm_attempt)?;
        next_llm_attempt = 1;
        let response = match ctx
            .client
            .create_message_streaming(&body, |partial| {
                let kind = partial["event"].as_str().unwrap_or("stream_event");
                ctx.journal
                    .append_step_partial(&ctx.run_id, seq, kind, partial)?;
                Ok(())
            })
            .await
        {
            Ok(r) => r,
            Err(e) => {
                ctx.journal.fail_step(&ctx.run_id, seq, &format!("{e:#}"))?;
                ctx.journal.set_run_status(&ctx.run_id, "failed")?;
                close_browser_sessions_best_effort(ctx).await;
                return Err(e);
            }
        };
        ctx.journal.complete_step(&ctx.run_id, seq, &response)?;

        let content = response["content"].clone();
        for block in content.as_array().into_iter().flatten() {
            if block["type"] == "text" {
                println!("{}", block["text"].as_str().unwrap_or_default());
            }
        }
        messages.push(json!({"role": "assistant", "content": content}));

        match response["stop_reason"].as_str().unwrap_or_default() {
            "tool_use" => {
                let mut tool_results = Vec::new();
                let mut saw_tool_use = false;
                for block in content.as_array().into_iter().flatten() {
                    if block["type"] != "tool_use" {
                        continue;
                    }
                    saw_tool_use = true;
                    let id = block["id"].as_str().unwrap_or_default();
                    let name = block["name"].as_str().unwrap_or_default();
                    if decision_tools::resume_contract(name).is_some() {
                        println!("→ tool {name} [runner-private decision input redacted]");
                    } else {
                        println!("→ tool {name} {}", block["input"]);
                    }
                    let result = execute_tool_step(ctx, name, id, &block["input"], 1).await;
                    match result {
                        Ok(content) => {
                            if decision_tools::resume_contract(name).is_some() {
                                println!("← tool {name} completed (decision content redacted)");
                            } else {
                                println!("← {content}");
                            }
                            tool_results.push(json!({
                                "type": "tool_result", "tool_use_id": id, "content": content,
                            }));
                        }
                        Err(e) if is_review_error(&e) => {
                            println!("← needs review: {e:#}");
                            ctx.journal.set_run_status(&ctx.run_id, "needs_review")?;
                            close_browser_sessions_best_effort(ctx).await;
                            return Ok(());
                        }
                        Err(e) => {
                            let private = decision_tools::resume_contract(name).is_some();
                            if private {
                                println!("← tool {name} rejected (decision content redacted)");
                            } else {
                                println!("← tool error: {e:#}");
                            }
                            tool_results.push(json!({
                                "type": "tool_result", "tool_use_id": id,
                                "content": if private { "Error: decision audit rejected".to_string() } else { format!("Error: {e:#}") }, "is_error": true,
                            }));
                        }
                    }
                }
                if !saw_tool_use {
                    ctx.journal.set_run_status(&ctx.run_id, "failed")?;
                    close_browser_sessions_best_effort(ctx).await;
                    bail!("model returned stop_reason \"tool_use\" with no tool_use blocks");
                }
                messages.push(json!({"role": "user", "content": tool_results}));
            }
            "end_turn" => {
                ctx.journal.set_run_status(&ctx.run_id, "completed")?;
                close_browser_sessions_best_effort(ctx).await;
                return Ok(());
            }
            // server-side pause: assistant turn is already appended; re-send as-is
            "pause_turn" => continue,
            "refusal" => {
                ctx.journal.set_run_status(&ctx.run_id, "failed")?;
                close_browser_sessions_best_effort(ctx).await;
                bail!("model refused: {}", response["stop_details"]);
            }
            other => {
                ctx.journal.set_run_status(&ctx.run_id, "failed")?;
                close_browser_sessions_best_effort(ctx).await;
                bail!("unexpected stop_reason {other:?} — raise max_tokens or inspect the journal");
            }
        }
    }
    ctx.journal.set_run_status(&ctx.run_id, "failed")?;
    close_browser_sessions_best_effort(ctx).await;
    bail!("agent exceeded {MAX_LOOP_STEPS} loop steps")
}

async fn close_browser_sessions_best_effort(ctx: &Ctx) {
    if let Err(error) = ctx.registry.close_browser_sessions(&ctx.run_id).await {
        tracing::warn!(
            "browser session cleanup for run {} failed: {error:#}",
            ctx.run_id
        );
    }
}

/// Journal-wrapped tool execution: started row committed before the tool runs.
pub fn start_journaled_tool_call(
    journal: &Journal,
    run_id: &str,
    name: &str,
    tool_use_id: &str,
    input: &Value,
    attempt: i64,
    idempotency_key: Option<String>,
) -> Result<JournaledToolCall> {
    start_journaled_tool_call_with_contract(
        journal,
        run_id,
        name,
        tool_use_id,
        input,
        attempt,
        idempotency_key,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn start_journaled_tool_call_with_contract(
    journal: &Journal,
    run_id: &str,
    name: &str,
    tool_use_id: &str,
    input: &Value,
    attempt: i64,
    idempotency_key: Option<String>,
    contract: Option<ToolResumeContract>,
) -> Result<JournaledToolCall> {
    let mut request = match &idempotency_key {
        Some(key) => json!({
            "name": name,
            "input": input,
            "tool_use_id": tool_use_id,
            "idempotency_key": key,
        }),
        None => json!({"name": name, "input": input, "tool_use_id": tool_use_id}),
    };
    if let Some(contract) = contract {
        anyhow::ensure!(contract.is_valid(), "invalid tool resume contract");
        request["resume_contract"] = serde_json::to_value(contract)?;
    }
    let seq = journal.start_step(
        run_id,
        "tool_call",
        &request,
        Some(name),
        Some(tool_use_id),
        attempt,
    )?;
    Ok(JournaledToolCall {
        seq,
        context: ToolCallContext {
            run_id: Some(run_id.to_string()),
            tool_use_id: Some(tool_use_id.to_string()),
            idempotency_key,
        },
    })
}

pub fn complete_journaled_tool_call(
    journal: &Journal,
    run_id: &str,
    seq: i64,
    result: &str,
) -> Result<()> {
    journal.complete_step(run_id, seq, &json!({"content": result}))
}

pub fn fail_journaled_tool_call(
    journal: &Journal,
    run_id: &str,
    seq: i64,
    error: &str,
) -> Result<()> {
    journal.fail_step(run_id, seq, error)
}

async fn execute_tool_step(
    ctx: &Ctx,
    name: &str,
    tool_use_id: &str,
    input: &Value,
    attempt: i64,
) -> Result<String> {
    let private_tool = decision_tools::resume_contract(name);
    if private_tool.is_some() {
        match redact_private_error(ctx.journal.gate_goal_run(&ctx.run_id))? {
            GoalRunGate::Current(_) => {}
            GoalRunGate::Unbound => {
                bail!("runner-private decision tools require a current goal-bound run")
            }
            GoalRunGate::NeedsReview => return Err(GoalRunNeedsReview.into()),
        }
        // Strict parsing happens before `start_step`, so malformed package
        // bodies cannot become durable tool requests.
        match name {
            decision_tools::APPEND_TOOL_NAME => {
                redact_private_error(decision_tools::validate_append(input))?
            }
            decision_tools::READ_TOOL_NAME => {
                redact_private_error(decision_tools::validate_read(input))?
            }
            _ => unreachable!(),
        }
    }
    let idempotency_key = tool_idempotency_key(&ctx.run_id, tool_use_id);
    let call = redact_private_error(start_journaled_tool_call_with_contract(
        &ctx.journal,
        &ctx.run_id,
        name,
        tool_use_id,
        input,
        attempt,
        idempotency_key,
        private_tool.or_else(|| ctx.registry.resume_contract(name)),
    ))?;
    if decision_tools::resume_contract(name).is_some() {
        let projection = match name {
            decision_tools::APPEND_TOOL_NAME => {
                decision_tools::append(&ctx.journal, &ctx.run_id, tool_use_id, input)
            }
            decision_tools::READ_TOOL_NAME => {
                decision_tools::read(&ctx.journal, &ctx.run_id, input)
            }
            _ => unreachable!(),
        };
        return match projection {
            Ok(projection) => {
                let result = redact_private_error(
                    serde_json::to_string(&projection).map_err(anyhow::Error::from),
                )?;
                redact_private_error(complete_journaled_tool_call(
                    &ctx.journal,
                    &ctx.run_id,
                    call.seq,
                    &result,
                ))?;
                Ok(result)
            }
            Err(error) => {
                redact_private_error(fail_journaled_tool_call(
                    &ctx.journal,
                    &ctx.run_id,
                    call.seq,
                    "decision audit rejected",
                ))?;
                if matches!(
                    redact_private_error(ctx.journal.gate_goal_run(&ctx.run_id))?,
                    GoalRunGate::NeedsReview
                ) {
                    return Err(GoalRunNeedsReview.into());
                }
                let _ = error;
                bail!("decision audit rejected")
            }
        };
    }
    match ctx
        .registry
        .execute_with_context(name, input, &call.context)
        .await
    {
        Ok(result) => {
            complete_journaled_tool_call(&ctx.journal, &ctx.run_id, call.seq, &result)?;
            Ok(result)
        }
        Err(e) => {
            fail_journaled_tool_call(&ctx.journal, &ctx.run_id, call.seq, &format!("{e:#}"))?;
            Err(e)
        }
    }
}

fn redact_private_error<T>(result: Result<T>) -> Result<T> {
    result.map_err(|error| {
        if is_review_error(&error) {
            error
        } else {
            anyhow::anyhow!("DECISION_AUDIT_REJECTED")
        }
    })
}

fn completed_tool_matches_recorded_contract(
    run_id: &str,
    name: &str,
    tool_use_id: &str,
    input: &Value,
    step: &beater_journal::StepRow,
    current: Option<&ToolResumeContract>,
) -> bool {
    step.status == "completed"
        && tool_replay_matches_recorded_contract(
            run_id,
            name,
            tool_use_id,
            input,
            std::slice::from_ref(step),
            current,
        )
}

fn tool_idempotency_key(run_id: &str, tool_use_id: &str) -> Option<String> {
    (!tool_use_id.is_empty()).then(|| format!("beater:{run_id}:tool:{tool_use_id}"))
}

fn tool_replay_matches_recorded_contract(
    run_id: &str,
    name: &str,
    tool_use_id: &str,
    input: &Value,
    steps: &[beater_journal::StepRow],
    current: Option<&ToolResumeContract>,
) -> bool {
    let Some(current) =
        current.filter(|value| value.is_valid() && value.idempotent && value.name == name)
    else {
        return false;
    };
    let Some(key) = tool_idempotency_key(run_id, tool_use_id) else {
        return false;
    };
    let mut found = false;
    for step in steps
        .iter()
        .filter(|step| step.kind == "tool_call" && step.tool_use_id.as_deref() == Some(tool_use_id))
    {
        found = true;
        if step.tool_name.as_deref() != Some(name)
            || step.request["name"].as_str() != Some(name)
            || step.request["tool_use_id"].as_str() != Some(tool_use_id)
            || step.request["input"] != *input
            || step.request["idempotency_key"].as_str() != Some(key.as_str())
            || step.attempt <= 0
            || step.attempt == i64::MAX
        {
            return false;
        }
        let Ok(recorded) =
            serde_json::from_value::<ToolResumeContract>(step.request["resume_contract"].clone())
        else {
            return false;
        };
        if !recorded.is_valid() || !recorded.idempotent || recorded != *current {
            return false;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::{resume, run, run_for_goal, tool_idempotency_key};
    use crate::registry::{AgentConfig, BeatboxConfig, ToolRegistry};
    use beater_journal::Journal;
    use rusqlite::params;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct TempApp {
        path: PathBuf,
    }

    impl TempApp {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "beater-runner-{name}-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
            ));
            fs::create_dir_all(path.join("agents/support/tools")).unwrap();
            fs::write(
                path.join("agents/support/tools/echo.py"),
                r#"
TOOL = {
    "description": "Echo a value.",
    "input_schema": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}

def run(input):
    return {"echo": input["value"]}
"#,
            )
            .unwrap();
            fs::write(
                path.join("agents/support/tools/fib.wat"),
                r#"
(module
  (func $fib (param $n i64) (result i64)
    local.get $n
    i64.const 2
    i64.lt_s
    if (result i64)
      local.get $n
    else
      local.get $n
      i64.const 1
      i64.sub
      call $fib
      local.get $n
      i64.const 2
      i64.sub
      call $fib
      i64.add
    end)
  (export "run" (func $fib)))
"#,
            )
            .unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempApp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    struct EnvGuard;

    impl EnvGuard {
        fn set(base_url: &str) -> Self {
            unsafe {
                std::env::set_var("ANTHROPIC_API_KEY", "test-key");
                std::env::set_var("ANTHROPIC_BASE_URL", base_url);
                std::env::set_var("BEATER_ANTHROPIC_ALLOW_INSECURE_LOOPBACK", "1");
            }
            Self
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("ANTHROPIC_API_KEY");
                std::env::remove_var("ANTHROPIC_BASE_URL");
                std::env::remove_var("BEATER_ANTHROPIC_ALLOW_INSECURE_LOOPBACK");
                std::env::remove_var("BEATER_TRACE_EXPORT_URL");
                std::env::remove_var("BEATER_TENANT_ID");
                std::env::remove_var("BEATER_PROJECT_ID");
                std::env::remove_var("BEATER_ENVIRONMENT_ID");
                std::env::remove_var("BEATER_API_KEY");
                std::env::remove_var("BEATER_OTLP_EXPORT_URL");
                std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
                std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT");
                std::env::remove_var("OTEL_EXPORTER_OTLP_HEADERS");
                std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_HEADERS");
            }
        }
    }

    struct MockAnthropic {
        base_url: String,
        requests: Arc<Mutex<Vec<String>>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    #[derive(Debug)]
    struct CapturedRequest {
        request_line: String,
        headers: String,
        body: String,
    }

    struct MockBeatbox {
        base_url: String,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl MockBeatbox {
        fn new(responses: Vec<Value>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server_requests = Arc::clone(&requests);
            let mut responses: VecDeque<String> = responses
                .into_iter()
                .map(|value| value.to_string())
                .collect();
            let handle = thread::spawn(move || {
                while let Some(response) = responses.pop_front() {
                    let (mut stream, _) = listener.accept().unwrap();
                    let request = read_http_request(&mut stream);
                    server_requests.lock().unwrap().push(request);
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        response.len(),
                        response
                    );
                    stream.write_all(reply.as_bytes()).unwrap();
                }
            });
            Self {
                base_url: format!("http://{addr}"),
                requests,
                handle: Some(handle),
            }
        }

        fn join(mut self) -> Vec<CapturedRequest> {
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
            Arc::try_unwrap(self.requests)
                .unwrap()
                .into_inner()
                .unwrap()
        }
    }

    impl MockAnthropic {
        fn new(responses: Vec<Value>) -> Self {
            Self::before_response(responses, || {})
        }

        fn before_response(
            responses: Vec<Value>,
            mut before_response: impl FnMut() + Send + 'static,
        ) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server_requests = Arc::clone(&requests);
            let mut responses: VecDeque<String> = responses
                .into_iter()
                .map(anthropic_stream_response)
                .collect();
            let handle = thread::spawn(move || {
                while let Some(response) = responses.pop_front() {
                    let (mut stream, _) = listener.accept().unwrap();
                    let body = read_http_body(&mut stream);
                    server_requests.lock().unwrap().push(body);
                    before_response();
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n{}",
                        response
                    );
                    stream.write_all(reply.as_bytes()).unwrap();
                }
            });
            Self {
                base_url: format!("http://{addr}"),
                requests,
                handle: Some(handle),
            }
        }

        fn join(mut self) -> Vec<String> {
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
            Arc::try_unwrap(self.requests)
                .unwrap()
                .into_inner()
                .unwrap()
        }
    }

    struct MockTraceIngest {
        base_url: String,
        requests: Arc<Mutex<Vec<CapturedRequest>>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl MockTraceIngest {
        fn new(expected_requests: usize) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let server_requests = Arc::clone(&requests);
            let handle = thread::spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(5);
                while server_requests.lock().unwrap().len() < expected_requests {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let request = read_http_request(&mut stream);
                            server_requests.lock().unwrap().push(request);
                            let response = json!({
                                "ack": {
                                    "accepted_raw": 1,
                                    "accepted_spans": 1,
                                    "duplicate_raw": 0,
                                    "duplicate_spans": 0,
                                },
                                "downstream_queued": false,
                            })
                            .to_string();
                            let reply = format!(
                                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                response.len(),
                                response
                            );
                            stream.write_all(reply.as_bytes()).unwrap();
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            if Instant::now() >= deadline {
                                break;
                            }
                            thread::sleep(Duration::from_millis(10));
                        }
                        Err(error) => panic!("mock trace ingest accept failed: {error}"),
                    }
                }
            });
            Self {
                base_url: format!("http://{addr}"),
                requests,
                handle: Some(handle),
            }
        }

        fn join(mut self) -> Vec<CapturedRequest> {
            if let Some(handle) = self.handle.take() {
                handle.join().unwrap();
            }
            Arc::try_unwrap(self.requests)
                .unwrap()
                .into_inner()
                .unwrap()
        }
    }

    fn anthropic_stream_response(response: Value) -> String {
        let content = response["content"].as_array().cloned().unwrap_or_default();
        let mut out = String::new();
        out.push_str(&sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": response["id"].as_str().unwrap_or("msg_mock"),
                    "type": "message",
                    "role": "assistant",
                    "model": response["model"].as_str().unwrap_or("mock"),
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 1, "output_tokens": 0}
                }
            }),
        ));
        for (index, block) in content.iter().enumerate() {
            let index = index as u64;
            let mut start_block = block.clone();
            match block["type"].as_str().unwrap_or_default() {
                "text" => {
                    let text = block["text"].as_str().unwrap_or_default();
                    start_block["text"] = Value::String(String::new());
                    out.push_str(&sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": start_block,
                        }),
                    ));
                    out.push_str(&sse(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        }),
                    ));
                }
                "tool_use" => {
                    let input = block["input"].clone();
                    start_block["input"] = json!({});
                    out.push_str(&sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": start_block,
                        }),
                    ));
                    out.push_str(&sse(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": input.to_string(),
                            },
                        }),
                    ));
                }
                _ => {
                    out.push_str(&sse(
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": start_block,
                        }),
                    ));
                }
            }
            out.push_str(&sse(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
        out.push_str(&sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": response["stop_reason"].as_str().unwrap_or("end_turn"),
                    "stop_sequence": response.get("stop_sequence").cloned().unwrap_or(Value::Null),
                },
                "usage": {"output_tokens": 1},
            }),
        ));
        out.push_str(&sse("message_stop", json!({"type": "message_stop"})));
        out
    }

    fn sse(event: &str, data: Value) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn read_http_body(stream: &mut std::net::TcpStream) -> String {
        read_http_request(stream).body
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> CapturedRequest {
        let mut bytes = Vec::new();
        let mut buf = [0_u8; 1024];
        let mut headers_end = None;
        let mut content_len = None;
        loop {
            let n = stream.read(&mut buf).unwrap();
            assert_ne!(n, 0, "client closed before sending a complete request");
            bytes.extend_from_slice(&buf[..n]);
            if headers_end.is_none() {
                headers_end = bytes.windows(4).position(|window| window == b"\r\n\r\n");
                if let Some(end) = headers_end {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    content_len = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    });
                }
            }
            if let Some(end) = headers_end
                && content_len.is_none()
            {
                let headers = String::from_utf8_lossy(&bytes[..end]).to_string();
                return CapturedRequest {
                    request_line: headers.lines().next().unwrap_or_default().to_string(),
                    headers,
                    body: String::new(),
                };
            }
            if let (Some(end), Some(len)) = (headers_end, content_len) {
                let body_start = end + 4;
                if bytes.len() >= body_start + len {
                    let headers = String::from_utf8_lossy(&bytes[..end]).to_string();
                    return CapturedRequest {
                        request_line: headers.lines().next().unwrap_or_default().to_string(),
                        headers,
                        body: String::from_utf8(bytes[body_start..body_start + len].to_vec())
                            .unwrap(),
                    };
                }
            }
        }
    }

    fn config(idempotent: bool) -> Value {
        json!({
            "name": "support",
            "provider": "anthropic",
            "model": "mock",
            "system": "test",
            "tools": [{
                "kind": "python",
                "name": "echo",
                "path": "./tools/echo.py",
                "idempotent": idempotent,
            }],
        })
    }

    fn browser_config() -> Value {
        json!({
            "name": "support",
            "provider": "anthropic",
            "model": "mock",
            "system": "test",
            "tools": [{
                "kind": "browser",
                "name": "browser.checkout",
                "provider": "mock_cdp",
                "description": "Verify checkout in a browser.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "task": {"type": "string"}
                    },
                    "required": ["url", "task"]
                },
                "session": {"scope": "run", "cleanup": "always"},
                "allowedOrigins": ["https://shop.example"],
                "timeoutMs": 1000,
                "idempotent": false
            }],
        })
    }

    fn sandbox_config(idempotent: bool) -> Value {
        json!({
            "name": "support",
            "provider": "anthropic",
            "model": "mock",
            "system": "test",
            "tools": [{
                "kind": "sandbox",
                "name": "fib_wasm",
                "path": "./tools/fib.wat",
                "idempotent": idempotent,
                "description": "Run fib in beatbox.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            }],
        })
    }

    fn remote_config(endpoint: &str, schema: Value) -> Value {
        let url = reqwest::Url::parse(endpoint).unwrap();
        let host = url.host_str().unwrap();
        let egress = url
            .port()
            .map(|port| format!("{host}:{port}"))
            .unwrap_or_else(|| host.to_string());
        json!({
            "name": "support",
            "provider": "anthropic",
            "model": "mock",
            "system": "test",
            "tools": [{
                "kind": "remote_mcp",
                "name": "crm.lookup",
                "description": "Look up a contact.",
                "inputSchema": schema,
                "endpoint": endpoint,
                "tool": "lookup",
                "timeoutMs": 1000,
                "egress": [egress],
                "idempotent": true,
            }],
        })
    }

    fn seed_interrupted_tool_run(app: &TempApp) {
        seed_interrupted_tool_run_for_config(
            app,
            "echo",
            json!({"value": "ok"}),
            config(true),
            BeatboxConfig::default(),
        );
    }

    fn seed_interrupted_tool_run_for(app: &TempApp, name: &str, input: Value) {
        seed_interrupted_tool_run_for_config(
            app,
            name,
            input,
            config(true),
            BeatboxConfig::default(),
        );
    }

    fn seed_interrupted_tool_run_for_config(
        app: &TempApp,
        name: &str,
        input: Value,
        config_value: Value,
        beatbox: BeatboxConfig,
    ) {
        let config: AgentConfig = serde_json::from_value(config_value).unwrap();
        let registry = ToolRegistry::build_with_beatbox(
            &app.path().join("agents").join(&config.name),
            &config.tools,
            &beatbox,
        )
        .unwrap();
        let contract = registry.resume_contract(name);
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_run("run-1", "support", &format!("call {name}"))
            .unwrap();
        let request = json!({
            "messages": [{"role": "user", "content": format!("call {name}")}],
        });
        let response = json!({
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": name,
                "input": input,
            }],
            "stop_reason": "tool_use",
        });
        let llm = journal
            .start_step("run-1", "llm_call", &request, None, None, 1)
            .unwrap();
        journal.complete_step("run-1", llm, &response).unwrap();
        super::start_journaled_tool_call_with_contract(
            &journal,
            "run-1",
            name,
            "toolu_1",
            &input,
            1,
            super::tool_idempotency_key("run-1", "toolu_1"),
            contract,
        )
        .unwrap();
        mark_run_stale(app, "run-1");
    }

    fn seed_interrupted_llm_run(app: &TempApp) {
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_run("run-1", "support", "continue interrupted llm")
            .unwrap();
        journal
            .start_step(
                "run-1",
                "llm_call",
                &json!({
                    "messages": [{
                        "role": "user",
                        "content": "continue interrupted llm",
                    }],
                }),
                None,
                None,
                2,
            )
            .unwrap();
        mark_run_stale(app, "run-1");
    }

    fn seed_completed_llm_run(app: &TempApp, status: &str, response: Value) {
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_run("run-1", "support", "continue previous response")
            .unwrap();
        let llm = journal
            .start_step(
                "run-1",
                "llm_call",
                &json!({
                    "messages": [{
                        "role": "user",
                        "content": "continue previous response",
                    }],
                }),
                None,
                None,
                1,
            )
            .unwrap();
        journal.complete_step("run-1", llm, &response).unwrap();
        journal.set_run_status("run-1", status).unwrap();
        if status == "running" {
            mark_run_stale(app, "run-1");
        }
    }

    fn mark_run_stale(app: &TempApp, run_id: &str) {
        let stale_updated_at = chrono::Utc::now().timestamp() - 31;
        let conn = rusqlite::Connection::open(app.path().join(".beater/journal.db")).unwrap();
        conn.execute(
            "UPDATE runs SET updated_at = ?2 WHERE id = ?1",
            params![run_id, stale_updated_at],
        )
        .unwrap();
    }

    fn execution_result_json(value: i64) -> Value {
        json!({
            "status": "ok",
            "value": value,
            "exit_code": null,
            "stdout": "",
            "stdout_truncated": false,
            "stderr": "",
            "stderr_truncated": false,
            "error": null,
            "metrics": {
                "wall_time_ms": 1,
                "cpu_time_ms": 1,
                "fuel_used": 42,
                "peak_memory_bytes": null,
            },
            "lane": "wasm",
            "deterministic": true,
            "inputs_digest": "sha256:test",
            "engine_version": "test",
            "beatbox_version": "test",
            "effective_isolation": {
                "os": "test",
                "mechanisms": ["wasmtime", "empty-linker"],
                "landlock_abi": null,
                "downgrades": [],
            },
            "egress": [],
        })
    }

    fn job_record_json(job_id: &str, result: Value) -> Value {
        json!({
            "job_id": job_id,
            "status": "succeeded",
            "request": {
                "lane": "wasm",
                "source": {"kind": "wasm_wat", "text": "(module)"},
                "entrypoint": null,
                "input": {"n": 10},
                "stdin": "",
                "policy": {},
                "idempotency_key": "beater:run-1:tool:toolu_1",
            },
            "result": result,
            "error": null,
            "created_at": "2026-07-02T00:00:00Z",
            "updated_at": "2026-07-02T00:00:00Z",
        })
    }

    #[test]
    fn goal_run_stale_creation_never_loads_configuration() {
        let app = TempApp::new("goal-stale-create");
        let journal = Journal::open(app.path()).unwrap();
        let mut request = create_goal_request(&journal);
        request.expected_revision = 2;
        assert!(
            super::run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
                panic!("stale goal must be rejected before configuration")
            })
            .is_err()
        );
        assert!(journal.run(&request.run_id).is_err());
        assert!(journal.list_runs().unwrap().is_empty());
    }

    #[test]
    fn goal_run_stale_resume_parks_before_configuration() {
        let app = TempApp::new("goal-stale-resume");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        journal
            .create_goal_bound_run(
                &request.scope,
                &request.goal_id,
                1,
                &request.run_id,
                &request.agent_name,
                &request.prompt,
            )
            .unwrap();
        journal
            .start_step(
                &request.run_id,
                "llm_call",
                &json!({"messages": []}),
                None,
                None,
                1,
            )
            .unwrap();
        revise_test_goal(&journal, &request.scope);
        resume(
            app.path(),
            &request.run_id,
            None,
            BeatboxConfig::default(),
            |_| panic!("stale goal must be parked before configuration"),
        )
        .unwrap();
        assert_eq!(journal.run(&request.run_id).unwrap().status, "needs_review");
        assert_eq!(journal.steps(&request.run_id).unwrap().len(), 1);
    }

    #[test]
    fn goal_run_current_uses_existing_worker_and_completes() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("goal-current-run");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let server = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "local test response"}],
            "stop_reason": "end_turn",
        })]);
        let _env = EnvGuard::set(&server.base_url);
        super::run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        assert_eq!(server.join().len(), 1);
        assert_eq!(journal.run(&request.run_id).unwrap().status, "completed");
        assert!(
            matches!(journal.gate_goal_run(&request.run_id).unwrap(), beater_journal::GoalRunGate::Current(binding) if binding.goal_revision == 1 && binding.scope == request.scope)
        );
        assert_eq!(journal.steps(&request.run_id).unwrap().len(), 1);
        revise_test_goal(&journal, &request.scope);
        resume(
            app.path(),
            &request.run_id,
            None,
            BeatboxConfig::default(),
            |_| panic!("historically completed work must be checked against the corrected goal"),
        )
        .unwrap();
        assert_eq!(journal.run(&request.run_id).unwrap().status, "needs_review");
        let steps = journal.steps(&request.run_id).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(
            steps[0].status, "completed",
            "retain historical response evidence"
        );
    }

    #[test]
    fn goal_run_changed_during_setup_never_starts_model_step() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("goal-setup-change");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let _env = EnvGuard::set("http://127.0.0.1:9");
        super::run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
            revise_test_goal(&journal, &request.scope);
            Ok(config(true))
        })
        .unwrap();
        assert_eq!(journal.run(&request.run_id).unwrap().status, "needs_review");
        assert!(journal.steps(&request.run_id).unwrap().is_empty());
    }

    #[test]
    fn goal_run_changed_during_model_call_cannot_dispatch_tool_or_mark_complete() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        for response in [
            json!({"content": [{"type": "text", "text": "old goal answer"}], "stop_reason": "end_turn"}),
            json!({"content": [{"type": "tool_use", "id": "toolu_1", "name": "echo", "input": {"value": "old"}}], "stop_reason": "tool_use"}),
        ] {
            let app = TempApp::new("goal-inflight-change");
            let journal = Journal::open(app.path()).unwrap();
            let request = create_goal_request(&journal);
            let app_path = app.path().to_path_buf();
            let scope = request.scope.clone();
            let server = MockAnthropic::before_response(vec![response], move || {
                revise_test_goal(&Journal::open(&app_path).unwrap(), &scope);
            });
            let _env = EnvGuard::set(&server.base_url);
            super::run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
                Ok(config(true))
            })
            .unwrap();
            assert_eq!(server.join().len(), 1);
            assert_eq!(journal.run(&request.run_id).unwrap().status, "needs_review");
            let steps = journal.steps(&request.run_id).unwrap();
            assert_eq!(steps.len(), 1);
            assert_eq!(steps[0].kind, "llm_call");
            assert_eq!(steps[0].status, "completed");
            assert!(
                steps[0].result.is_some(),
                "retain the actual old response without claiming current completion"
            );
        }
    }

    #[test]
    fn goal_run_rejects_different_agent_before_registry_setup() {
        let app = TempApp::new("goal-config-name");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let error =
            super::run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
                Ok(json!({"name": "other", "tools": [{"kind": "python", "path": "missing.py"}]}))
            })
            .unwrap_err();
        assert!(error.to_string().contains("saved agent"));
        assert!(journal.steps(&request.run_id).unwrap().is_empty());
    }

    fn create_goal_request(journal: &Journal) -> super::GoalRunRequest {
        use beater_journal::{Goal, GoalMutation, GoalScope, PlaybookIdentity};
        let scope = GoalScope {
            organization: "org".into(),
            project: "project".into(),
            environment: "test".into(),
            site: "site".into(),
        };
        journal
            .create_goal(
                &GoalMutation {
                    actor: "owner".into(),
                    request_id: "create".into(),
                    operation: "create".into(),
                },
                Goal {
                    id: "goal".into(),
                    scope: scope.clone(),
                    revision: 1,
                    objective: "Prepare a current proposal".into(),
                    playbook: PlaybookIdentity {
                        id: "generic-preparation".into(),
                        digest: "a".repeat(64),
                    },
                    parameters: Default::default(),
                },
            )
            .unwrap();
        super::GoalRunRequest {
            run_id: "goal-run".into(),
            scope,
            goal_id: "goal".into(),
            expected_revision: 1,
            agent_name: "support".into(),
            prompt: "Prepare current goal".into(),
        }
    }

    fn decision_package_input(revision: i64, correction_of: Option<i64>) -> Value {
        json!({
            "version": 1,
            "expected_revision": revision - 1,
            "package": {
                "version": 1,
                "decision_id": "decision-a",
                "revision": revision,
                "domain": "software",
                "question": "Record a local decision?",
                "object_references": [], "evidence": [], "graph_context": null,
                "policy_references": [], "calculation_references": [], "model_references": [], "producer_references": [],
                "options": ["record"], "constraints": ["local-only"],
                "rationale": "Preserve bounded local history.",
                "reviews": [], "authorization_observations": [], "effect_attempts": [], "acknowledgements": [], "outcomes": [], "reconciliations": [],
                "correction_of": correction_of.map(|previous| json!({"decision_id": "decision-a", "revision": previous})),
                "successor_to": null,
                "verification_ceiling": "RecordedOnly",
                "execution_authority": "None"
            }
        })
    }

    #[test]
    fn goal_runner_appends_corrects_reads_and_reopens_decision_history() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("decision-tools");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let server = MockAnthropic::new(vec![
            json!({"content": [{"type":"tool_use", "id":"decision-append-1", "name":"decision_audit_append", "input": decision_package_input(1, None)}], "stop_reason":"tool_use"}),
            json!({"content": [{"type":"tool_use", "id":"decision-append-2", "name":"decision_audit_append", "input": decision_package_input(2, Some(1))}], "stop_reason":"tool_use"}),
            json!({"content": [{"type":"tool_use", "id":"decision-read-2", "name":"decision_audit_read", "input": {"version":1, "decision_id":"decision-a", "expected_revision":2}}], "stop_reason":"tool_use"}),
            json!({"content": [{"type":"text", "text":"recorded"}], "stop_reason":"end_turn"}),
        ]);
        let _env = EnvGuard::set(&server.base_url);
        run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        assert_eq!(server.join().len(), 4);
        drop(journal);
        let reopened = Journal::open(app.path()).unwrap();
        let history = reopened
            .decision_audit(&request.scope, "decision-a")
            .unwrap();
        assert_eq!(history.current_revision, 2);
        assert_eq!(history.events.len(), 2);
        assert_eq!(
            history.events[1]
                .package
                .correction_of
                .as_ref()
                .unwrap()
                .revision,
            1
        );
        assert_eq!(reopened.run(&request.run_id).unwrap().status, "completed");
    }

    #[test]
    fn stale_goal_before_private_append_keeps_no_decision_event() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("decision-stale-before-append");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let app_path = app.path().to_path_buf();
        let scope = request.scope.clone();
        let server = MockAnthropic::before_response(
            vec![
                json!({"content": [{"type":"tool_use", "id":"decision-append", "name":"decision_audit_append", "input": decision_package_input(1, None)}], "stop_reason":"tool_use"}),
            ],
            move || revise_test_goal(&Journal::open(&app_path).unwrap(), &scope),
        );
        let _env = EnvGuard::set(&server.base_url);
        run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        assert_eq!(journal.run(&request.run_id).unwrap().status, "needs_review");
        assert!(
            journal
                .decision_audit(&request.scope, "decision-a")
                .is_err()
        );
    }

    #[test]
    fn invalid_private_package_never_creates_a_tool_step_or_decision_event() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("decision-invalid-before-step");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        let mut invalid = decision_package_input(1, None);
        invalid["package"]["question"] = json!("x".repeat(4_097));
        let server = MockAnthropic::new(vec![
            json!({"content": [{"type":"tool_use", "id":"invalid-private", "name":"decision_audit_append", "input": invalid}], "stop_reason":"tool_use"}),
            json!({"content": [{"type":"text", "text":"handled"}], "stop_reason":"end_turn"}),
        ]);
        let _env = EnvGuard::set(&server.base_url);
        run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        assert_eq!(journal.steps(&request.run_id).unwrap().len(), 2);
        assert!(
            journal
                .steps(&request.run_id)
                .unwrap()
                .iter()
                .all(|step| step.kind != "tool_call")
        );
        assert!(
            journal
                .decision_audit(&request.scope, "decision-a")
                .is_err()
        );
    }

    #[test]
    fn unsupported_private_authority_and_version_never_create_tool_steps() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        for (label, mut input) in [
            ("authority", decision_package_input(1, None)),
            ("version", decision_package_input(1, None)),
        ] {
            if label == "authority" {
                input["package"]["execution_authority"] = json!("Execute");
            } else {
                input["version"] = json!(2);
            }
            let app = TempApp::new(&format!("decision-invalid-{label}"));
            let journal = Journal::open(app.path()).unwrap();
            let request = create_goal_request(&journal);
            let server = MockAnthropic::new(vec![
                json!({"content":[{"type":"tool_use","id":"invalid","name":"decision_audit_append","input":input}],"stop_reason":"tool_use"}),
                json!({"content":[{"type":"text","text":"handled"}],"stop_reason":"end_turn"}),
            ]);
            let _env = EnvGuard::set(&server.base_url);
            run_for_goal(app.path(), &request, None, BeatboxConfig::default(), |_| {
                Ok(config(true))
            })
            .unwrap();
            assert!(
                journal
                    .steps(&request.run_id)
                    .unwrap()
                    .iter()
                    .all(|step| step.kind != "tool_call")
            );
        }
    }

    #[test]
    fn private_append_cas_and_replay_conflict_preserve_one_revision() {
        let app = TempApp::new("decision-cas-replay");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        journal
            .create_goal_bound_run(
                &request.scope,
                &request.goal_id,
                request.expected_revision,
                &request.run_id,
                &request.agent_name,
                &request.prompt,
            )
            .unwrap();
        let first = decision_package_input(1, None);
        crate::decision_tools::append(&journal, &request.run_id, "same-id", &first).unwrap();
        assert_eq!(
            crate::decision_tools::append(&journal, &request.run_id, "same-id", &first)
                .unwrap()
                .current_revision,
            1
        );
        let mut changed = first.clone();
        changed["package"]["question"] = json!("changed");
        assert!(
            crate::decision_tools::append(&journal, &request.run_id, "same-id", &changed).is_err()
        );
        let mut conflict = decision_package_input(2, Some(1));
        conflict["expected_revision"] = json!(0);
        assert!(
            crate::decision_tools::append(&journal, &request.run_id, "new-id", &conflict).is_err()
        );
        assert_eq!(
            journal
                .decision_audit(&request.scope, "decision-a")
                .unwrap()
                .current_revision,
            1
        );
    }

    #[test]
    fn private_read_refuses_a_different_current_goal_after_retaining_history() {
        let app = TempApp::new("decision-read-current-goal");
        let journal = Journal::open(app.path()).unwrap();
        let request = create_goal_request(&journal);
        journal
            .create_goal_bound_run(
                &request.scope,
                &request.goal_id,
                1,
                &request.run_id,
                &request.agent_name,
                &request.prompt,
            )
            .unwrap();
        crate::decision_tools::append(
            &journal,
            &request.run_id,
            "append",
            &decision_package_input(1, None),
        )
        .unwrap();
        revise_test_goal(&journal, &request.scope);
        assert!(
            crate::decision_tools::read(
                &journal,
                &request.run_id,
                &json!({"version": 1, "decision_id": "decision-a", "expected_revision": 1})
            )
            .is_err()
        );
        assert_eq!(
            journal
                .decision_audit(&request.scope, "decision-a")
                .unwrap()
                .events
                .len(),
            1
        );
    }

    #[test]
    fn configured_private_name_is_rejected_before_run_creation() {
        let app = TempApp::new("decision-name-collision");
        let error = run(app.path(), "support", json!({"name":"support","provider":"anthropic","model":"x","tools":[{"kind":"rust","name":"decision_audit_append"}]}), None, BeatboxConfig::default(), "x").unwrap_err();
        assert!(format!("{error:#}").contains("runner-private decision tool name"));
        assert!(
            Journal::open(app.path())
                .unwrap()
                .list_runs()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unbound_runner_does_not_advertise_or_execute_private_tools() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("decision-unbound");
        let server = MockAnthropic::new(vec![
            json!({"content":[{"type":"tool_use","id":"forged","name":"decision_audit_append","input":decision_package_input(1, None)}],"stop_reason":"tool_use"}),
            json!({"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}),
        ]);
        let _env = EnvGuard::set(&server.base_url);
        run(
            app.path(),
            "support",
            config(true),
            None,
            BeatboxConfig::default(),
            "x",
        )
        .unwrap();
        let requests = server.join();
        assert!(!requests[0].contains("decision_audit_append"));
        let journal = Journal::open(app.path()).unwrap();
        let (_, steps) = journal.list_runs().unwrap().pop().unwrap();
        assert_eq!(steps, 2);
    }

    fn revise_test_goal(journal: &Journal, scope: &beater_journal::GoalScope) {
        use beater_journal::{GoalMutation, GoalPatch};
        journal
            .revise_goal(
                &GoalMutation {
                    actor: "owner".into(),
                    request_id: "correct".into(),
                    operation: "correct".into(),
                },
                scope,
                "goal",
                1,
                GoalPatch {
                    objective: Some("Prepare a revised proposal".into()),
                    playbook: None,
                    parameters: None,
                },
            )
            .unwrap();
    }

    #[test]
    fn resume_reruns_interrupted_idempotent_tool_once_and_finishes() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("idempotent");
        seed_interrupted_tool_run(&app);
        let server = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
        })]);
        let _env = EnvGuard::set(&server.base_url);

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let requests = server.join();

        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_str(&requests[0]).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.last().unwrap()["role"], "user");
        assert_eq!(
            messages.last().unwrap()["content"][0]["tool_use_id"],
            "toolu_1"
        );

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        let steps = journal.steps("run-1").unwrap();
        let tool_steps: Vec<_> = steps
            .iter()
            .filter(|step| {
                step.kind == "tool_call" && step.tool_use_id.as_deref() == Some("toolu_1")
            })
            .collect();
        assert_eq!(tool_steps.len(), 2);
        assert_eq!(tool_steps[0].status, "started");
        assert_eq!(tool_steps[0].attempt, 1);
        assert_eq!(tool_steps[1].status, "completed");
        assert_eq!(tool_steps[1].attempt, 2);
        assert_eq!(
            tool_steps[1].request["idempotency_key"],
            "beater:run-1:tool:toolu_1"
        );
    }

    #[test]
    fn resume_turns_failed_idempotent_tool_rerun_into_error_result() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("idempotent-error");
        seed_interrupted_tool_run_for(&app, "echo", json!({"missing": "value"}));
        let server = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "handled tool error"}],
            "stop_reason": "end_turn",
        })]);
        let _env = EnvGuard::set(&server.base_url);

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let requests = server.join();

        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_str(&requests[0]).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let tool_result = &messages.last().unwrap()["content"][0];
        assert_eq!(tool_result["tool_use_id"], "toolu_1");
        assert_eq!(tool_result["is_error"], true);
        assert!(
            tool_result["content"].as_str().unwrap().contains("Error:"),
            "{tool_result}"
        );

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        let tool_steps: Vec<_> = journal
            .steps("run-1")
            .unwrap()
            .into_iter()
            .filter(|step| {
                step.kind == "tool_call" && step.tool_use_id.as_deref() == Some("toolu_1")
            })
            .collect();
        assert_eq!(tool_steps.len(), 2);
        assert_eq!(tool_steps[0].status, "started");
        assert_eq!(tool_steps[1].status, "failed");
        assert_eq!(tool_steps[1].attempt, 2);
    }

    #[test]
    fn resume_parks_removed_tool_without_replaying_legacy_step() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("removed-tool");
        seed_interrupted_tool_run_for(&app, "old.echo", json!({"value": "ok"}));
        let _env = EnvGuard::set("http://127.0.0.1:9");

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "needs_review");
        let tool_steps: Vec<_> = journal
            .steps("run-1")
            .unwrap()
            .into_iter()
            .filter(|step| step.kind == "tool_call")
            .collect();
        assert_eq!(tool_steps.len(), 1);
        assert_eq!(tool_steps[0].tool_name.as_deref(), Some("old.echo"));
        assert_eq!(tool_steps[0].status, "started");
    }

    #[test]
    fn run_completes_browser_tool_through_agent_loop() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("browser-tool");
        let server = MockAnthropic::new(vec![
            json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_browser",
                    "name": "browser.checkout",
                    "input": {
                        "url": "https://shop.example/cart",
                        "task": "verify checkout"
                    },
                }],
                "stop_reason": "tool_use",
            }),
            json!({
                "content": [{"type": "text", "text": "checkout verified"}],
                "stop_reason": "end_turn",
            }),
        ]);
        let _env = EnvGuard::set(&server.base_url);

        run(
            app.path(),
            "support",
            browser_config(),
            None,
            BeatboxConfig::default(),
            "verify checkout",
        )
        .unwrap();
        let requests = server.join();
        assert_eq!(requests.len(), 2);

        let body: Value = serde_json::from_str(&requests[1]).unwrap();
        let messages = body["messages"].as_array().unwrap();
        let tool_result = &messages.last().unwrap()["content"][0];
        assert_eq!(tool_result["tool_use_id"], "toolu_browser");
        assert!(
            tool_result["content"]
                .as_str()
                .unwrap()
                .contains("Mock Browser Page")
        );

        let journal = Journal::open(app.path()).unwrap();
        let (run, _) = journal.list_runs().unwrap().pop().unwrap();
        assert_eq!(run.status, "completed");
        let steps = journal.steps(&run.id).unwrap();
        let llm_steps: Vec<_> = steps
            .iter()
            .filter(|step| step.kind == "llm_call")
            .collect();
        assert_eq!(llm_steps.len(), 2);
        let first_partials = journal.step_partials(&run.id, llm_steps[0].seq).unwrap();
        assert!(
            first_partials.iter().any(|partial| {
                partial.kind == "content_block_delta"
                    && partial.payload["data"]["delta"]["type"] == "input_json_delta"
            }),
            "{first_partials:?}"
        );
        let final_partials = journal.step_partials(&run.id, llm_steps[1].seq).unwrap();
        assert!(
            final_partials.iter().any(|partial| {
                partial.kind == "content_block_delta"
                    && partial.payload["data"]["delta"]["text"] == "checkout verified"
            }),
            "{final_partials:?}"
        );
        let tool_step = steps
            .into_iter()
            .find(|step| step.kind == "tool_call")
            .expect("browser tool call step");
        assert_eq!(tool_step.status, "completed");
        assert_eq!(tool_step.tool_use_id.as_deref(), Some("toolu_browser"));
    }

    #[test]
    fn run_exports_beater_native_trace_when_configured() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("trace-export");
        let anthropic = MockAnthropic::new(vec![
            json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_browser",
                    "name": "browser.checkout",
                    "input": {
                        "url": "https://shop.example/cart",
                        "task": "verify checkout"
                    },
                }],
                "stop_reason": "tool_use",
            }),
            json!({
                "content": [{"type": "text", "text": "checkout verified"}],
                "stop_reason": "end_turn",
            }),
        ]);
        let trace_ingest = MockTraceIngest::new(4);
        let _env = EnvGuard::set(&anthropic.base_url);
        unsafe {
            std::env::set_var("BEATER_TRACE_EXPORT_URL", &trace_ingest.base_url);
            std::env::set_var("BEATER_TENANT_ID", "tenant");
            std::env::set_var("BEATER_PROJECT_ID", "project");
            std::env::set_var("BEATER_ENVIRONMENT_ID", "prod");
            std::env::set_var("BEATER_API_KEY", "trace-key");
        }

        run(
            app.path(),
            "support",
            browser_config(),
            None,
            BeatboxConfig::default(),
            "verify checkout",
        )
        .unwrap();
        let _anthropic_requests = anthropic.join();
        let trace_requests = trace_ingest.join();

        assert_eq!(trace_requests.len(), 4);
        assert!(
            trace_requests
                .iter()
                .all(|request| request.request_line == "POST /v1/traces/native HTTP/1.1")
        );
        assert!(
            trace_requests[0]
                .headers
                .to_ascii_lowercase()
                .contains("x-beater-api-key: trace-key")
        );
        let spans: Vec<Value> = trace_requests
            .iter()
            .map(|request| serde_json::from_str(&request.body).unwrap())
            .collect();
        assert!(spans.iter().any(|span| span["kind"] == "agent.run"));
        assert_eq!(
            spans
                .iter()
                .filter(|span| span["kind"] == "llm.call")
                .count(),
            2
        );
        let tool = spans
            .iter()
            .find(|span| span["kind"] == "tool.call")
            .expect("tool span");
        assert_eq!(tool["scope"]["tenant_id"], "tenant");
        assert_eq!(tool["scope"]["project_id"], "project");
        assert_eq!(tool["scope"]["environment_id"], "prod");
        assert_eq!(tool["parent_span_id"], "run");
        assert_eq!(tool["attributes"]["beater.tool_name"], "browser.checkout");
        assert_eq!(tool["attributes"]["beater.tool_use_id"], "toolu_browser");
    }

    #[test]
    fn run_exports_otlp_trace_when_configured() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("otlp-trace-export");
        let anthropic = MockAnthropic::new(vec![
            json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_browser",
                    "name": "browser.checkout",
                    "input": {
                        "url": "https://shop.example/cart",
                        "task": "verify checkout"
                    },
                }],
                "stop_reason": "tool_use",
            }),
            json!({
                "content": [{"type": "text", "text": "checkout verified"}],
                "stop_reason": "end_turn",
            }),
        ]);
        let trace_ingest = MockTraceIngest::new(1);
        let _env = EnvGuard::set(&anthropic.base_url);
        unsafe {
            std::env::set_var("BEATER_OTLP_EXPORT_URL", &trace_ingest.base_url);
            std::env::set_var("BEATER_TENANT_ID", "tenant");
            std::env::set_var("BEATER_PROJECT_ID", "project");
            std::env::set_var("BEATER_ENVIRONMENT_ID", "prod");
            std::env::set_var("BEATER_API_KEY", "trace-key");
            std::env::set_var("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-extra-trace=ok");
        }

        run(
            app.path(),
            "support",
            browser_config(),
            None,
            BeatboxConfig::default(),
            "verify checkout",
        )
        .unwrap();
        let _anthropic_requests = anthropic.join();
        let trace_requests = trace_ingest.join();

        assert_eq!(trace_requests.len(), 1);
        assert_eq!(trace_requests[0].request_line, "POST /v1/traces HTTP/1.1");
        let headers = trace_requests[0].headers.to_ascii_lowercase();
        assert!(headers.contains("x-beater-api-key: trace-key"));
        assert!(headers.contains("x-extra-trace: ok"));
        let payload: Value = serde_json::from_str(&trace_requests[0].body).unwrap();
        let spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
            .as_array()
            .unwrap();
        assert_eq!(spans.len(), 4);
        assert!(spans.iter().any(|span| {
            span["name"] == "beater.js agent run"
                && span["attributes"].as_array().unwrap().iter().any(|attr| {
                    attr["key"] == "beater.run_id"
                        && attr["value"]["stringValue"].as_str().unwrap().len() == 36
                })
        }));
        assert_eq!(
            spans
                .iter()
                .filter(
                    |span| span["attributes"].as_array().unwrap().iter().any(|attr| {
                        attr["key"] == "beater.span_kind"
                            && attr["value"]["stringValue"] == "llm.call"
                    })
                )
                .count(),
            2
        );
        let tool = spans
            .iter()
            .find(|span| {
                span["attributes"].as_array().unwrap().iter().any(|attr| {
                    attr["key"] == "beater.span_kind" && attr["value"]["stringValue"] == "tool.call"
                })
            })
            .expect("tool span");
        assert_eq!(tool["parentSpanId"], spans[0]["spanId"]);
        assert!(tool["attributes"].as_array().unwrap().iter().any(|attr| {
            attr["key"] == "beater.tool_use_id" && attr["value"]["stringValue"] == "toolu_browser"
        }));
    }

    #[test]
    fn resume_does_not_promote_original_non_idempotent_declaration() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("idempotence-promotion");
        seed_interrupted_tool_run_for_config(
            &app,
            "echo",
            json!({"value": "ok"}),
            config(false),
            BeatboxConfig::default(),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");
        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "needs_review");
        let steps = journal.steps("run-1").unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].status, "started");
        assert_eq!(steps[1].attempt, 1);
        assert_eq!(steps[1].request["resume_contract"]["idempotent"], false);
    }

    #[test]
    fn resume_parks_interrupted_non_idempotent_tool_for_review() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("non-idempotent");
        seed_interrupted_tool_run_for_config(
            &app,
            "echo",
            json!({"value": "ok"}),
            config(false),
            BeatboxConfig::default(),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(false))
        })
        .unwrap();

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "needs_review");
        let tool_steps: Vec<_> = journal
            .steps("run-1")
            .unwrap()
            .into_iter()
            .filter(|step| step.kind == "tool_call")
            .collect();
        assert_eq!(tool_steps.len(), 1);
        assert_eq!(tool_steps[0].status, "started");
        assert_eq!(tool_steps[0].attempt, 1);
    }

    #[test]
    fn resume_parks_config_or_schema_drift_before_any_replay() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        for (label, current) in [
            (
                "endpoint",
                remote_config(
                    "http://127.0.0.1:19002/mcp",
                    json!({"type": "object", "properties": {"email": {"type": "string"}}}),
                ),
            ),
            (
                "schema",
                remote_config(
                    "http://127.0.0.1:19001/mcp",
                    json!({"type": "object", "properties": {"id": {"type": "string"}}}),
                ),
            ),
        ] {
            let app = TempApp::new(label);
            let original = remote_config(
                "http://127.0.0.1:19001/mcp",
                json!({"type": "object", "properties": {"email": {"type": "string"}}}),
            );
            seed_interrupted_tool_run_for_config(
                &app,
                "crm.lookup",
                json!({"email": "a@example.test"}),
                original,
                BeatboxConfig::default(),
            );
            let _env = EnvGuard::set("http://127.0.0.1:9");
            resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
                Ok(current.clone())
            })
            .unwrap();
            let journal = Journal::open(app.path()).unwrap();
            assert_eq!(
                journal.run("run-1").unwrap().status,
                "needs_review",
                "{label}"
            );
            assert_eq!(journal.steps("run-1").unwrap().len(), 2, "{label}");
        }
    }

    #[test]
    fn resume_parks_legacy_or_malformed_contracts() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        for malformed in [false, true] {
            let app = TempApp::new(if malformed {
                "malformed-contract"
            } else {
                "legacy-contract"
            });
            seed_interrupted_tool_run(&app);
            let conn = rusqlite::Connection::open(app.path().join(".beater/journal.db")).unwrap();
            let raw_request: String = conn
                .query_row(
                    "SELECT request FROM steps WHERE run_id='run-1' AND kind='tool_call'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            let mut request: Value = serde_json::from_str(&raw_request).unwrap();
            if malformed {
                request["resume_contract"] =
                    json!({"version": 1, "name": "echo", "idempotent": true, "fingerprint": "bad"});
            } else {
                request.as_object_mut().unwrap().remove("resume_contract");
            }
            conn.execute(
                "UPDATE steps SET request=?1 WHERE run_id='run-1' AND kind='tool_call'",
                [request.to_string()],
            )
            .unwrap();
            let _env = EnvGuard::set("http://127.0.0.1:9");
            resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
                Ok(config(true))
            })
            .unwrap();
            assert_eq!(
                Journal::open(app.path())
                    .unwrap()
                    .run("run-1")
                    .unwrap()
                    .status,
                "needs_review"
            );
        }
    }

    #[test]
    fn resume_cleans_stale_browser_session_before_review() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("browser-stale-session");
        seed_interrupted_tool_run_for(
            &app,
            "browser.checkout",
            json!({"url": "https://shop.example/cart", "task": "verify checkout"}),
        );
        let session_dir = app.path().join(".beater/browser-sessions");
        fs::create_dir_all(&session_dir).unwrap();
        let wrapper_path = session_dir.join("run-1-wrapper.cjs");
        let marker_path = session_dir.join("run-1-marker.json");
        fs::write(&wrapper_path, "setTimeout(() => {}, 30000);\n").unwrap();
        fs::write(
            &marker_path,
            json!({
                "session_id": "run-1",
                "wrapper_script": wrapper_path.clone(),
                "runner_script": "/dev/null",
                "owner_pid": 1,
                "created_at": 1,
            })
            .to_string(),
        )
        .unwrap();
        let _env = EnvGuard::set("http://127.0.0.1:9");

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(browser_config())
        })
        .unwrap();

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "needs_review");
        assert!(!marker_path.exists());
        assert!(!wrapper_path.exists());
    }

    #[test]
    fn resume_preserves_failed_refusal_instead_of_marking_completed() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("refusal-stop");
        seed_completed_llm_run(
            &app,
            "failed",
            json!({
                "content": [{"type": "text", "text": "I can't help with that."}],
                "stop_reason": "refusal",
                "stop_details": {"reason": "safety"},
            }),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");

        let err = resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap_err();

        assert!(format!("{err:#}").contains("model refused"));
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "failed");
        assert_eq!(journal.steps("run-1").unwrap().len(), 1);
    }

    #[test]
    fn resume_marks_running_max_tokens_failed_and_does_not_run_truncated_tools() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("max-tokens-stop");
        seed_completed_llm_run(
            &app,
            "running",
            json!({
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_truncated",
                    "name": "echo",
                    "input": {"value": "possibly truncated"},
                }],
                "stop_reason": "max_tokens",
            }),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");

        let err = resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap_err();

        assert!(format!("{err:#}").contains("unexpected stop_reason \"max_tokens\""));
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "failed");
        let steps = journal.steps("run-1").unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].kind, "llm_call");
    }

    #[test]
    fn resume_reissues_interrupted_llm_with_incremented_attempt() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("interrupted-llm-attempt");
        seed_interrupted_llm_run(&app);
        let server = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "continued"}],
            "stop_reason": "end_turn",
        })]);
        let _env = EnvGuard::set(&server.base_url);

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let requests = server.join();

        assert_eq!(requests.len(), 1);
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        let llm_attempts: Vec<_> = journal
            .steps("run-1")
            .unwrap()
            .into_iter()
            .filter(|step| step.kind == "llm_call")
            .map(|step| (step.status, step.attempt))
            .collect();
        assert_eq!(
            llm_attempts,
            vec![("started".to_string(), 2), ("completed".to_string(), 3)]
        );
    }

    #[test]
    fn resume_refuses_owned_running_run_before_loading_config() {
        let app = TempApp::new("owned-running-run");
        let _owner = crate::ownership::RunOwnership::acquire(app.path(), "run-1").unwrap();
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_run("run-1", "support", "still active")
            .unwrap();

        let err = resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            panic!("owned run should reject before loading config")
        })
        .unwrap_err();

        assert!(format!("{err:#}").contains("run ownership unavailable"));
        assert_eq!(journal.run("run-1").unwrap().status, "running");
    }

    #[test]
    fn resume_without_owner_does_not_wait_for_timestamp_grace() {
        let app = TempApp::new("unowned-fresh-run");
        let journal = Journal::open(app.path()).unwrap();
        journal
            .create_run("run-1", "support", "interrupted")
            .unwrap();
        let err = resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            anyhow::bail!("config loader reached after exclusive ownership")
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("config loader reached"));
        assert!(crate::ownership::RunOwnership::acquire(app.path(), "run-1").is_ok());
    }

    #[test]
    fn resume_marks_completed_end_turn_finished_without_reissuing_llm() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("end-turn-stop");
        seed_completed_llm_run(
            &app,
            "running",
            json!({
                "content": [{"type": "text", "text": "already done"}],
                "stop_reason": "end_turn",
            }),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        assert_eq!(journal.steps("run-1").unwrap().len(), 1);
    }

    #[test]
    fn resume_reissues_pause_turn_instead_of_marking_completed() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("pause-turn-stop");
        seed_completed_llm_run(
            &app,
            "running",
            json!({
                "content": [{
                    "type": "server_tool_use",
                    "id": "srvu_1",
                    "name": "web_search",
                    "input": {"query": "beater"},
                }],
                "stop_reason": "pause_turn",
            }),
        );
        let server = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "continued"}],
            "stop_reason": "end_turn",
        })]);
        let _env = EnvGuard::set(&server.base_url);

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(config(true))
        })
        .unwrap();
        let requests = server.join();

        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_str(&requests[0]).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(
            messages[1]["content"],
            json!([{
                "type": "server_tool_use",
                "id": "srvu_1",
                "name": "web_search",
                "input": {"query": "beater"},
            }])
        );

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        assert_eq!(
            journal
                .steps("run-1")
                .unwrap()
                .iter()
                .filter(|step| step.kind == "llm_call")
                .count(),
            2
        );
    }

    #[test]
    fn resume_reruns_interrupted_idempotent_sandbox_tool_through_beatbox_job() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("sandbox-idempotent");
        let anthropic = MockAnthropic::new(vec![json!({
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
        })]);
        let beatbox = MockBeatbox::new(vec![
            json!({"job_id": "job-1"}),
            job_record_json("job-1", execution_result_json(55)),
        ]);
        let beatbox_config = BeatboxConfig {
            url: beatbox.base_url.clone(),
            api_key: None,
        };
        seed_interrupted_tool_run_for_config(
            &app,
            "fib_wasm",
            json!({"n": 10}),
            sandbox_config(true),
            beatbox_config.clone(),
        );
        let _env = EnvGuard::set(&anthropic.base_url);

        resume(app.path(), "run-1", None, beatbox_config, |_| {
            Ok(sandbox_config(true))
        })
        .unwrap();

        let beatbox_requests = beatbox.join();
        assert_eq!(beatbox_requests.len(), 2);
        assert!(
            beatbox_requests[0]
                .request_line
                .starts_with("POST /v1/jobs ")
        );
        let body: Value = serde_json::from_str(&beatbox_requests[0].body).unwrap();
        assert_eq!(body["idempotency_key"], "beater:run-1:tool:toolu_1");
        assert!(
            beatbox_requests[1]
                .request_line
                .starts_with("GET /v1/jobs/job-1 ")
        );

        let _anthropic_requests = anthropic.join();
        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "completed");
        let steps = journal.steps("run-1").unwrap();
        let completed = steps
            .iter()
            .find(|step| step.kind == "tool_call" && step.status == "completed")
            .expect("completed sandbox tool step");
        let content = completed.result.as_ref().unwrap()["content"]
            .as_str()
            .unwrap();
        let result: Value = serde_json::from_str(content).unwrap();
        assert_eq!(result["value"], 55);
    }

    #[test]
    fn resume_parks_interrupted_non_idempotent_sandbox_tool_for_review() {
        let _env_lock = ENV_LOCK.lock().unwrap();
        let app = TempApp::new("sandbox-non-idempotent");
        seed_interrupted_tool_run_for_config(
            &app,
            "fib_wasm",
            json!({"n": 10}),
            sandbox_config(false),
            BeatboxConfig::default(),
        );
        let _env = EnvGuard::set("http://127.0.0.1:9");

        resume(app.path(), "run-1", None, BeatboxConfig::default(), |_| {
            Ok(sandbox_config(false))
        })
        .unwrap();

        let journal = Journal::open(app.path()).unwrap();
        assert_eq!(journal.run("run-1").unwrap().status, "needs_review");
        let tool_steps: Vec<_> = journal
            .steps("run-1")
            .unwrap()
            .into_iter()
            .filter(|step| step.kind == "tool_call")
            .collect();
        assert_eq!(tool_steps.len(), 1);
        assert_eq!(tool_steps[0].status, "started");
    }

    #[test]
    fn idempotency_keys_are_stable_per_run_tool_use() {
        assert_eq!(
            tool_idempotency_key("run-1", "toolu_1").as_deref(),
            Some("beater:run-1:tool:toolu_1")
        );
        assert_eq!(tool_idempotency_key("run-1", ""), None);
    }
}
