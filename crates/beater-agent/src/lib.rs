//! Durable agent runtime: tool registry, provider-adapted LLM tool loop,
//! step-lifecycle journal (SQLite).
//!
//! The loop lives in Rust — not in the JS isolate — so it survives hot
//! reloads and every step is journaled before it executes (ARCHITECTURE.md §5).

mod anthropic;
mod cpp_bridge;
mod llm;
mod ownership;
mod registry;
mod resume_contract;
mod runner;
mod trace_export;

pub use beater_journal::{
    Goal, GoalMutation, GoalPatch, GoalRevision, GoalRunBinding, GoalRunGate, GoalRunNeedsReview,
    GoalScope, HistoryWindow, Journal, Milestone, MilestoneRevision, PlaybookIdentity, RunRow,
    StepPartialRow, StepRow,
};
pub use registry::{
    AgentConfig, BeatboxConfig, DEFAULT_BEATBOX_URL, ToolCallContext, ToolDecl, ToolNeedsReview,
    ToolRegistry, browser_session_dir, cleanup_stale_browser_sessions,
};
pub use runner::{
    GoalRunRequest, JournaledToolCall, complete_journaled_tool_call, fail_journaled_tool_call,
    list_runs, resume, run, run_for_goal, start_journaled_tool_call,
};
