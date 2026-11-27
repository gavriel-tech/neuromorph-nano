//! The whole path on the served path (2026-11-12; architecture by Luis Jake Gabriel III; his
//! direction of the same day: everything is built and joined). `serve --whole` makes
//! `Organism::ask` answer through the whole path instead of the forage frame:
//!
//! 1. Drones, eight tentacles, one `g0_tx` each, G1 and G2's gate (`drone_path::stack_run`,
//!    processor only; the drones' intake is kept per code version and taken in again only for
//!    chunks whose content changed).
//! 2. Core Brain: the released Qwen3-Reranker-0.6B scores every piece sent up (card prompt,
//!    shared prompt head); its order leads (measured 2026-11-12: over the 60 CX1 questions it
//!    holds better on unseen questions than the three-judge agreement); then the second look
//!    inside the winning file (`decide::second_look`, base Qwen3-0.6B, joint yes-or-no read).
//! 3. The pack: the Core Brain's order as `chunk` packets, then what memory recalls, within the
//!    token budget; nothing is cut to fit: what does not fit is named in the pack's reason as
//!    queued references.
//!
//! Measured on the development and claim halves of CX1: right code first 18 and 17 of 30
//! (`docs/2026-11-10-neumor-plan-whole-architecture.md`, 2026-11-12). Off unless the flag is
//! given. Encodings live in RAM only.
use crate::drone_path;
use crate::drone_windows::Keep;
use crate::joint_read::Piece;
use nmengn::model::Model;
use nmengn::placement::Device;
use nmsubs::Substrate;
use nmtype::{Category, Fault, FaultKind, Pack, Packet, Provenance, Stance, Status};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.whole", FaultKind::Execution, m)
}

/// Key prefix of the whole path's packets.
pub const KEY: &str = "whole";

/// Pieces a call takes in at most before it answers while the drones' intake is not complete
/// (memory areas, 2026-11-27): its question's likely pieces first, then the material in order.
pub const FIRST_STEP: usize = 2 * crate::drone_windows::STEP_PIECES;

/// The pieces `question` likely needs, at most `cap`, most matching first: chunks whose text or
/// path holds the most of the question's words of four or more letters (stop words left out).
/// Indices into `sub.chunks`; empty when the question names no such word.
#[must_use]
pub fn question_pieces(sub: &Substrate, question: &str, cap: usize) -> Vec<usize> {
    let q = question.to_lowercase();
    let mut terms: Vec<&str> = q.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|w| w.chars().count() >= 4 && !nmlex::STOP.contains(w)).collect();
    terms.sort_unstable();
    terms.dedup();
    if terms.is_empty() || cap == 0 {
        return Vec::new();
    }
    let hits: Vec<Option<usize>> = nmtype::budget::run(sub.chunks.len(), |i| {
        sub.chunks.get(i).map_or(0, |ch| {
            let path = sub.files.get(ch.file).map(|f| f.path.to_lowercase()).unwrap_or_default();
            terms.iter().filter(|t| ch.low.contains(**t) || path.contains(**t)).count()
        })
    });
    let mut scored: Vec<(usize, usize)> = hits.into_iter().enumerate().filter_map(|(i, h)| h.filter(|n| *n > 0).map(|n| (n, i))).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(cap).map(|(_, i)| i).collect()
}
/// Confidence of the Core Brain's first packet, and the step down per place.
const TOP: f32 = 0.9;
const STEP: f32 = 0.05;
const FLOOR: f32 = 0.3;

/// The whole path bound to one organism: the drones' intake per code version and the three
/// models, opened on first use and held.
pub struct WholePath {
    home: PathBuf,
    keep: Mutex<Option<(u64, Keep)>>,
    /// The relevance-key store beside the drones' intake (master plan 3.2, Phase C2,
    /// 2026-11-18): one store whose packets follow the material version (`Keys::intake` drops
    /// packets of pieces gone and encodes new or changed pieces), started on every `Keep::sync`
    /// when `key_scan::intake_on`; the pieces the last question's windows touched go first.
    relevance: Mutex<Option<Arc<crate::key_scan::Keys>>>,
    /// Memory areas (2026-11-27): a background pass is taking the material in, step by step
    /// (`warm_keep`); a call reads what is taken in and does not wait for the rest.
    warming: std::sync::atomic::AtomicBool,
    /// Where the intake stood after its last step, readable while a step holds the intake.
    keep_seen: Mutex<Option<crate::drone_windows::KeepState>>,
    touched: Mutex<Vec<usize>>,
    reranker: Mutex<Option<Arc<Model>>>,
    base: Mutex<Option<Arc<Model>>>,
    pub up: Option<usize>,
    /// Stage 5 WP-C1: this call's pack budget in tokens for G2's room, set by the caller before
    /// each run (`forage::ask_whole`); 0: the `budget.pack_tokens` control, else
    /// `drone_path::PACK_TOKENS`.
    pub room_tokens: std::sync::atomic::AtomicUsize,
    /// Stage 5 WP-C2: this call's refinement allocation (`nbv`, `uniform`, `off`), set by the caller
    /// before each run from the organism's refine mode, which G3's allocation site decides for the
    /// call; empty: the rule (`nbv`).
    pub refine: std::sync::Mutex<String>,
    /// When the Core Brain asks for more context (`nmapi::ask`).
    pub ask: crate::ask::Mode,
    /// The question's time budget in seconds (the session's `max_run_ms`; fallback
    /// `decide::BUDGET_FALLBACK_S`).
    pub budget_s: f64,
    /// Decoy rule (2026-11-13): the pack is withheld when the reranker's best P(yes) over every
    /// candidate is below this cut; `None` serves every pack. Default `WITHHOLD_BELOW`.
    pub withhold_below: Option<f32>,
    /// I11 standing order (2026-11-13): G2 sends up this many more of G1's queue on every
    /// question (0: none, the default until measured).
    pub up_next: usize,
    /// Within-file reach (2026-11-13): the chunks just before and after the first place in its
    /// file are judged by the reranker and added before the order is set (Q1 without asking).
    pub reach: bool,
    /// Use sites (2026-11-13): chunks in other files using an identifier the question shares with
    /// the first place, judged by the reranker and added (`USES_K` at most).
    pub uses: bool,
    /// Fault injection for the reliability run (2026-11-13; `serve --inject-fault <part>`, test
    /// only): `reranker`, `look`, `reach`, `uses` or `ask` fails on every question.
    pub inject: Option<String>,
    /// Tokens of each piece the reranker reads (2026-11-13, speed; `serve --rerank-cap <n>`):
    /// `None` derives the cap from the card as before.
    pub rerank_cap: Option<usize>,
    /// The second look (2026-11-13; `serve --look off` turns it off for every question).
    pub look: bool,
    /// Each tentacle's window composed by its lens (`drone_path::compose_lenses`), `serve --lenses
    /// on`; off (the default until it holds the served sets) the windows fill from the BM25F flags.
    pub lenses: bool,
    /// The mode of a call that names none (`serve --flow`: the one flow); `None` reads the saved
    /// `mode` control.
    pub default_mode: Option<Mode>,
    /// Score every reranker set both batched and one at a time and record the largest difference
    /// and whether the orders agree (`serve --rerank-compare`, measurement only).
    pub rerank_compare: bool,
    /// The second look runs only when the reranker's top two scores are closer than this many
    /// log-odds (`serve --look-gate <x>`); `None` runs it on every question its arrangement allows.
    pub look_gate: Option<f32>,
    /// Pieces of the stack's order the reranker reads in fast mode (`serve --fast-k <n>`).
    pub fast_k: usize,
    /// Tokens the judge reads of each piece (`serve --judge-window <tokens>|off`; `JUDGE_WINDOW`):
    /// `None` reads pieces whole. The pack carries whole pieces either way.
    pub judge_window: Option<usize>,
    /// Tokens the judge reads of each piece in fast mode (`serve --fast-judge-window
    /// <tokens>|off`; `FAST_JUDGE_WINDOW`).
    pub fast_judge_window: Option<usize>,
    /// The second look reads the judge's windows too (`serve --look-window on|off`).
    pub look_window: bool,
    /// Chunks of the first place's file the reach takes (`serve --reach-k <n>`; default 2, the neighbours).
    pub reach_k: usize,
    /// Live anatomy (2026-11-13): the arrangement per question kind, from the `whole_path`
    /// section of `designs/compositions.json` (kind to reach, use sites and second look); a kind
    /// not listed keeps the fixed arrangement above. Empty unless `serve --anatomy` is given.
    pub arrangements: std::collections::BTreeMap<String, Arrangement>,
    /// G2's standing instructions (I9 gains, I10 inhibitions) across questions.
    pub desk: Mutex<nmbrain::instruct::Desk>,
    /// Reflexes G2 installed below the Core Brain (`nmbrain::reflex`), and the files the organism
    /// itself edited last (set before each question; R4 marks their spans self-caused).
    pub reflexes: Mutex<nmbrain::reflex::Set>,
    pub own_files: Mutex<Vec<String>>,
}

/// What one question did on the whole path (reported in the answer's context map).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Run {
    /// Chunks in the Core Brain's final order.
    pub order: Vec<usize>,
    /// Chunks G1 queued (not sent up), passed on as references.
    pub queued: Vec<usize>,
    pub moved_by_look: bool,
    /// For the decision ledger: candidates (chunks) in arrival order with the reranker's P(yes),
    /// and the second look's read set (chunks, winner first) with its yes-or-no scores.
    pub candidates: Vec<usize>,
    pub rerank: Vec<f32>,
    pub look: Vec<usize>,
    pub look_scores: Vec<f32>,
    /// G1's stop reason and rounds, and how many G2 sent up (for the behaviour log).
    pub g1_stop: String,
    pub g1_rounds: u32,
    /// The second look's long order, as its one-line record.
    pub order_line: String,
    /// Chunks memory recalled for this question that G2 sent up beside the stack's own (build
    /// item 2, 2026-11-12: recall enters at G2 and the Core Brain, re-read fresh, labelled memory).
    pub recalled: Vec<usize>,
    /// How many recalled chunks the tentacle windows hold (each is guaranteed a place).
    pub recalled_in_windows: usize,
    pub look_margin: f32,
    pub stack_s: f64,
    pub brain_s: f64,
    /// Asking for more context: the judge's class of the first place (when judged), its
    /// confidence, and each requested chunk with its request (`Q3:<chunk>`, `Q1:<chunk>`).
    pub ask_class: Option<crate::ask::Class>,
    pub ask_conf: Option<f64>,
    pub asked: Vec<String>,
    /// What G2's instructions did on this question, one line each.
    pub g2: Vec<String>,
    /// Weight tensors staged on the copy stream during the stack run (0 when all were held).
    pub staged: usize,
    /// Parts that failed on this question and were left out (2026-11-13, reliability): the answer
    /// went on without them.
    pub faults: Vec<String>,
    /// Whether the reranker's scores are real (false: it failed and arrival order stands in).
    pub rerank_ok: bool,
    /// Seconds per stage of this question, in the order the stages ran (`STAGES`).
    pub stages: Vec<(&'static str, f64)>,
    /// With `rerank_compare`: the largest batched against one-at-a-time score difference over
    /// this question's sets, and whether every set kept the same order.
    pub rerank_check: Option<(f32, bool)>,
    /// Log-odds gap between the reranker's two best scores (infinite with fewer than two).
    pub top_gap: f32,
    /// Whether the second look ran, and the first place before it (`path:first-last`).
    pub look_ran: bool,
    pub pre_look_first: String,
    /// Chunks the dense lane added to the candidates (empty when the lane is off or not ready).
    pub dense_added: Vec<usize>,
    /// Reranker documents this question scored in batches, and those early drop stopped midway.
    pub rerank_docs: usize,
    pub rerank_dropped: usize,
    /// The reranker's document cap in tokens when this question's first set was scored.
    pub rerank_cap: usize,
    /// What the judge read: pieces, how many of them as a window, and tokens (path lines aside).
    pub judge_pieces: usize,
    pub judge_windowed: usize,
    pub judge_tokens: usize,
    /// What the second look read: pieces, tokens over every pass, passes.
    pub look_pieces: usize,
    pub look_tokens: usize,
    pub look_passes: usize,
    /// The one flow: the stack's pack status (`answered`, `low`, `withheld`) and the request's parts
    /// no piece of the answer holds; `None` on the other modes.
    pub flow_status: Option<(String, Vec<String>)>,
    /// The one flow with three states (`NANO_FLOW_STATES=on`): the fitted confidence of the pack
    /// and the two cuts it was read against (withhold, settle); `None` otherwise.
    pub flow_score: Option<(f32, f32, f32)>,
}

/// The second look's gate in log-odds between the reranker's top two: the largest gap at which
/// the look moved a development question's first place to the right lines was 2.91, so a gap of
/// 3.0 or more skips it (it ran on 37 of 40 there and cost no right first).
pub const LOOK_GATE: f32 = 3.0;

/// How much of the whole path one question runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// The stack's first pieces and the dense lane's, judged by the reranker in one batch.
    Fast,
    /// Every part the arrangement allows: reranker, use sites, reach, asking for more, second look.
    #[default]
    Full,
    /// The one flow: what G2 sends up, in its order (G2's set, then G1's order), as the pack; no
    /// MicroModel round, so the reranker and the base model are not opened.
    Flow,
}
impl Mode {
    /// `fast`, `full` or `flow`.
    /// # Errors
    /// Any other word.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fast" => Ok(Self::Fast),
            "full" => Ok(Self::Full),
            "flow" => Ok(Self::Flow),
            x => Err(format!("mode takes fast, full or flow, got {x}")),
        }
    }
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Full => "full",
            Self::Flow => "flow",
        }
    }
}

/// The saved `mode` control (`settings.json` at the Nano's root), read at each call; full when
/// it is absent or unreadable.
#[must_use]
pub fn mode_control(home: &Path) -> Mode {
    std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("mode").ok().and_then(|m| m.string().ok()).and_then(|m| Mode::parse(m).ok()))
        .unwrap_or_default()
}

/// Reads an integer allowance control `key` from `settings.json` at `home`, clamped to `[min, max]`.
#[must_use]
pub fn control_int(home: &Path, key: &str, min: u64, max: u64) -> Option<u64> {
    std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| {
            let val = v.get(key).ok()?;
            val.integer().ok().or_else(|| val.string().ok()?.trim().parse::<u64>().ok())
        })
        .filter(|n| *n >= min && *n <= max)
}

/// Default pack token budget from `budget.pack_tokens` in `settings.json` (`1,000..=16,000`),
/// falling back to `drone_path::PACK_TOKENS` (`4,000`).
#[must_use]
pub fn pack_tokens_control(home: &Path) -> u32 {
    control_int(home, "budget.pack_tokens", 1000, 16_000)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or_else(|| u32::try_from(crate::drone_path::PACK_TOKENS).unwrap_or(4000))
}

/// Tentacle window token budget from `budget.tentacle_tokens` in `settings.json` (`2,000..=32,000`),
/// falling back to `nmsubs::lens::BUDGET` (`12,000`).
#[must_use]
pub fn tentacle_tokens_control(home: &Path) -> usize {
    control_int(home, "budget.tentacle_tokens", 2000, 32_000)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(nmsubs::lens::BUDGET)
}

/// Judge read window from `budget.judge_window` in `settings.json` (`128..=2,048`), falling back to
/// `JUDGE_WINDOW` (`Some(384)`).
#[must_use]
pub fn judge_window_control(home: &Path) -> Option<usize> {
    control_int(home, "budget.judge_window", 128, 2048)
        .and_then(|n| usize::try_from(n).ok())
        .or(JUDGE_WINDOW)
}

/// Fast-mode candidate count from `budget.fast_k` in `settings.json` (`1..=16`), falling back to
/// `FAST_K` (`4`).
#[must_use]
pub fn fast_k_control(home: &Path) -> usize {
    control_int(home, "budget.fast_k", 1, 16)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(FAST_K)
}

/// Reads a string control `key` from `settings.json` at `home`.
#[must_use]
pub fn control_str(home: &Path, key: &str) -> Option<String> {
    std::fs::read(home.join("settings.json"))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get(key).ok().and_then(|s| s.string().ok()).map(str::to_owned))
}

/// Whether `mcp.tool_control` in `settings.json` is set to `"on"` (default `"off"`):
/// when `"off"`, Neuromorph Nano uses default model/server tool settings (ignoring any saved
/// custom `mcp.enabled_tools` subset); when `"on"`, the user's selected tools in
/// `mcp.enabled_tools` govern `tools/list`, and when `mcp.dynamic_tools` is `"off"`, tools not
/// selected are also blocked in `tools/call`.
#[must_use]
pub fn mcp_tool_control(home: &Path) -> bool {
    control_str(home, "mcp.tool_control").is_some_and(|v| v == "on")
}

/// Whether `mcp.tools` in `settings.json` is set to `"lean"` (expose only the 5 core coding tools
/// in `tools/list` while keeping every tool callable in `tools/call`).
#[must_use]
pub fn mcp_tools_lean(home: &Path) -> bool {
    control_str(home, "mcp.tools").is_some_and(|v| v == "lean")
}

/// Effective number of MCP tools to advertise in `tools/list` (1 to every tool, every tool by
/// default, from `mcp.tool_count` and `mcp.tools` in `settings.json`). 2026-11-27: the default was
/// 19, which left `nm_settings`, `nm_cartridges` and the other interface tools out of the list
/// although the server's instructions named them, and Claude Code calls only listed tools.
#[must_use]
pub fn mcp_tool_count_control(home: &Path) -> usize {
    let all = u64::try_from(crate::mcp::PRIORITY_TOOLS.len()).unwrap_or(25);
    let count = control_int(home, "mcp.tool_count", 1, all)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(crate::mcp::PRIORITY_TOOLS.len());
    if mcp_tools_lean(home) { count.min(5) } else { count }
}

/// Explicit per-tool default list from `mcp.enabled_tools` in `settings.json`, or `None` when
/// `"auto"` / unset (falling back to `mcp.tools` + `mcp.tool_count` priority order).
#[must_use]
pub fn mcp_enabled_tools_control(home: &Path) -> Option<Vec<String>> {
    let raw = control_str(home, "mcp.enabled_tools")?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return None;
    }
    let list: Vec<String> = trimmed
        .split(',')
        .map(str::trim)
        .filter(|t| crate::mcp::PRIORITY_TOOLS.contains(t))
        .map(str::to_owned)
        .collect();
    (!list.is_empty()).then_some(list)
}

/// Whether `mcp.dynamic_tools` in `settings.json` is `"on"` (default `"on"`): dynamically
/// activates additional MCP tools into `tools/list` when a question's intent or a direct tool call
/// needs them.
#[must_use]
pub fn mcp_dynamic_tools_control(home: &Path) -> bool {
    !control_str(home, "mcp.dynamic_tools").is_some_and(|v| v == "off")
}

/// Whether `host.claude_lean` in `settings.json` is `"on"` (write `"enableArtifact": false` into
/// `.claude/settings.local.json` during `setup`).
#[must_use]
pub fn claude_lean_control(home: &Path) -> bool {
    control_str(home, "host.claude_lean").is_some_and(|v| v == "on")
}

/// Whether `budget.adaptive_pack` in `settings.json` is on (default `"on"`).
#[must_use]
pub fn adaptive_pack_control(home: &Path) -> bool {
    !control_str(home, "budget.adaptive_pack").is_some_and(|v| v == "off")
}

/// Focused pack token ceiling from `budget.focused_tokens` in `settings.json` (`500..=8,000`),
/// falling back to `1,800`.
#[must_use]
pub fn focused_tokens_control(home: &Path) -> u32 {
    control_int(home, "budget.focused_tokens", 500, 8000)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(1800)
}

/// Mid-layer candidate drop setting from `rerank.early_drop` in `settings.json` (`"off"`,
/// `"balanced"`, `"aggressive"`). Defaults to `"balanced"` (`layer: 18, margin: 4.0, keep: 5`),
/// proven on `dev-large-2` to cut warm median latency by `-410 ms` (`3.301 -> 2.891 s`) with zero
/// accuracy loss (`60/102` first, `20/20` decoys).
#[must_use]
pub fn rerank_early_drop_control(home: &Path) -> Option<nmengn::roles::EarlyDrop> {
    match control_str(home, "rerank.early_drop").as_deref() {
        Some("off") => None,
        Some("aggressive") => Some(nmengn::roles::EarlyDrop { layer: 16, margin: 3.0, keep: 3 }),
        _ => Some(nmengn::roles::EarlyDrop { layer: 18, margin: 4.0, keep: 5 }),
    }
}

/// Whether `writer.idle` in `settings.json` is on (default `"on"` now that `writer-base@3`
/// coexists with all three finding/ranking MicroModels within 2.17 GiB on a 4 GB card).
#[must_use]
pub fn writer_idle_control(home: &Path) -> bool {
    !control_str(home, "writer.idle").is_some_and(|v| v == "off")
}

/// Whether `muscle` (G3 muscle memory) in `settings.json` is on (default `"on"`).
#[must_use]
pub fn muscle_control(home: &Path) -> bool {
    !control_str(home, "muscle").is_some_and(|v| v == "off")
}

/// Whether `recall_cache` in `settings.json` is on (default `"on"`).
#[must_use]
pub fn recall_cache_control(home: &Path) -> bool {
    !control_str(home, "recall_cache").is_some_and(|v| v == "off")
}

/// Whether `learning` in `settings.json` is on (default `"on"`).
#[must_use]
pub fn learning_control(home: &Path) -> bool {
    !control_str(home, "learning").is_some_and(|v| v == "off")
}

/// `residency.idle_release` (2026-11-25): minutes of quiet after which the serving process releases
/// kept window copies unused that long (15 by default); `None` when off.
#[must_use]
pub fn idle_release_minutes(home: &Path) -> Option<u64> {
    // A measurement run may set `NANO_IDLE_RELEASE_MIN` (1 to 240 minutes, or `off`) in place of the
    // control, as the other `NANO_*` switches do.
    if let Ok(v) = std::env::var("NANO_IDLE_RELEASE_MIN") {
        return v.trim().parse::<u64>().ok().filter(|m| (1..=240).contains(m));
    }
    match control_str(home, "residency.idle_release").as_deref() {
        Some("5") => Some(5),
        Some("15") => Some(15),
        Some("30") => Some(30),
        Some("off") => None,
        _ => Some(15),
    }
}

/// Cycle mode from `learning.cycle` in `settings.json` (`"auto"`, `"immediate"`, `"manual"`).
#[must_use]
pub fn learning_cycle_control(home: &Path) -> String {
    match control_str(home, "learning.cycle").as_deref() {
        Some("immediate") => "immediate".to_owned(),
        Some("manual") => "manual".to_owned(),
        _ => "auto".to_owned(),
    }
}

/// Cumulative context packs and tokens served by this process in the current session.
pub static SESSION_PACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SESSION_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Session token quota ceiling from `quota.session_tokens` in `settings.json` (`0` = unlimited).
#[must_use]
pub fn quota_session_tokens_control(home: &Path) -> u64 {
    control_int(home, "quota.session_tokens", 0, 500_000).unwrap_or(0)
}

/// Daily 24-hour token quota ceiling from `quota.daily_tokens` in `settings.json` (`0` = unlimited).
#[must_use]
pub fn quota_daily_tokens_control(home: &Path) -> u64 {
    control_int(home, "quota.daily_tokens", 0, 2_000_000).unwrap_or(0)
}

/// Session pack call quota ceiling from `quota.session_packs` in `settings.json` (`0` = unlimited).
#[must_use]
pub fn quota_session_packs_control(home: &Path) -> u64 {
    control_int(home, "quota.session_packs", 0, 500).unwrap_or(0)
}

/// Action to take when an active quota is reached (`"compact"`, `"fast"`, `"withhold"`).
#[must_use]
pub fn quota_action_control(home: &Path) -> String {
    match control_str(home, "quota.action").as_deref() {
        Some("fast") => "fast".to_owned(),
        Some("withhold") => "withhold".to_owned(),
        _ => "compact".to_owned(),
    }
}

/// Cumulative context pack tokens recorded in `memory/audit.jsonl` over the last 24 hours,
/// floored at the current process's `SESSION_TOKENS`.
#[must_use]
pub fn daily_audit_tokens(home: &Path) -> u64 {
    let sess = SESSION_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
    let Ok(text) = std::fs::read_to_string(home.join("memory").join("audit.jsonl")) else {
        return sess;
    };
    let cutoff = nmtype::now_ms().saturating_sub(86_400_000);
    let mut sum = 0_u64;
    for line in text.lines() {
        if !line.contains("\"nm_context_pack\"") {
            continue;
        }
        let Ok(v) = nmjson::parse(line.as_bytes()) else { continue };
        if v.get("kind").and_then(nmjson::Value::string).ok() != Some("tool")
            || v.get("tool").and_then(nmjson::Value::string).ok() != Some("nm_context_pack")
            || v.get("state").and_then(nmjson::Value::string).ok() == Some("running")
        {
            continue;
        }
        if v.get("time_ms").and_then(nmjson::Value::integer).unwrap_or(0) < cutoff {
            continue;
        }
        let tok = v
            .get("tokens")
            .and_then(nmjson::Value::integer)
            .ok()
            .unwrap_or_else(|| v.get("sources").and_then(nmjson::Value::array).map_or(0, |s| (s.len() as u64) * 200));
        sum = sum.saturating_add(tok);
    }
    sum.max(sess)
}

/// Decision of the user-configured quota gate before a context pack call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuotaGate {
    /// All active quotas are below 80% (or set to `0` = unlimited).
    Normal,
    /// Usage has reached at least 80% of an active token quota; scale pack down to `budget`.
    NearQuota { budget: u32 },
    /// An active quota has reached 100%; apply `action` (`"compact"`, `"fast"`, or `"withhold"`).
    Reached { action: String, budget: u32, reason: String },
}

/// Evaluate user-configured session and daily quotas before serving a context pack.
#[must_use]
pub fn evaluate_quota(home: &Path, requested_tokens: u32) -> QuotaGate {
    let sess_tok_max = quota_session_tokens_control(home);
    let daily_tok_max = quota_daily_tokens_control(home);
    let sess_pack_max = quota_session_packs_control(home);
    if sess_tok_max == 0 && daily_tok_max == 0 && sess_pack_max == 0 {
        return QuotaGate::Normal;
    }
    let sess_tok = SESSION_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
    let sess_packs = SESSION_PACKS.load(std::sync::atomic::Ordering::Relaxed);
    let daily_tok = if daily_tok_max > 0 { daily_audit_tokens(home) } else { 0 };
    let focused = focused_tokens_control(home);
    let compact_budget = (focused / 2).max(500).min(requested_tokens);
    let action = quota_action_control(home);

    if sess_pack_max > 0 && sess_packs >= sess_pack_max {
        return QuotaGate::Reached {
            action,
            budget: compact_budget,
            reason: format!("withheld: session pack quota reached ({sess_packs}/{sess_pack_max} packs; adjust quota.session_packs in Settings)"),
        };
    }
    if sess_tok_max > 0 && sess_tok >= sess_tok_max {
        return QuotaGate::Reached {
            action,
            budget: compact_budget,
            reason: format!("withheld: session token quota reached ({sess_tok}/{sess_tok_max} tokens; adjust quota.session_tokens in Settings)"),
        };
    }
    if daily_tok_max > 0 && daily_tok >= daily_tok_max {
        return QuotaGate::Reached {
            action,
            budget: compact_budget,
            reason: format!("withheld: daily token quota reached ({daily_tok}/{daily_tok_max} tokens; adjust quota.daily_tokens in Settings)"),
        };
    }
    let near_sess = sess_tok_max > 0 && sess_tok.saturating_mul(100) >= sess_tok_max.saturating_mul(80);
    let near_daily = daily_tok_max > 0 && daily_tok.saturating_mul(100) >= daily_tok_max.saturating_mul(80);
    if near_sess || near_daily {
        return QuotaGate::NearQuota { budget: focused.min(requested_tokens) };
    }
    QuotaGate::Normal
}

/// Pieces of the stack's order the reranker reads in fast mode.
pub const FAST_K: usize = 4;
/// Chunks the dense lane adds in fast mode at most.
pub const FAST_DENSE: usize = 2;

/// Chunks the dense lane adds to the candidates per question at most.
pub const DENSE_ADD: usize = 4;

/// Log-odds of a probability, clamped away from 0 and 1.
#[must_use]
pub fn log_odds(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Log-odds gap between the two highest scores of `rr` (infinite with fewer than two).
#[must_use]
pub fn top_gap(rr: &[f32]) -> f32 {
    let o = rank_order(rr);
    match (o.first().and_then(|i| rr.get(*i)), o.get(1).and_then(|i| rr.get(*i))) {
        (Some(a), Some(b)) => log_odds(*a) - log_odds(*b),
        _ => f32::INFINITY,
    }
}

/// Indices of `v` from the highest score down (ties by index).
fn rank_order(v: &[f32]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..v.len()).collect();
    o.sort_by(|a, b| v[*b].total_cmp(&v[*a]).then(a.cmp(b)));
    o
}

/// The stages of the whole path as the bench reports them, in order.
pub const STAGES: [&str; 9] =
    ["index", "plain ranking", "GCX rules", "load", "embedding", "reranker", "second look", "joint read", "assembly"];

/// Seconds spent per stage, summed over the calls of one question.
#[derive(Default)]
struct Clock(std::cell::RefCell<std::collections::BTreeMap<&'static str, f64>>);
impl Clock {
    fn add(&self, stage: &'static str, secs: f64) {
        *self.0.borrow_mut().entry(stage).or_default() += secs;
    }
    fn time<T>(&self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.add(stage, t.elapsed().as_secs_f64());
        out
    }
    fn list(&self) -> Vec<(&'static str, f64)> {
        let m = self.0.borrow();
        STAGES.iter().map(|s| (*s, m.get(s).copied().unwrap_or(0.0))).collect()
    }
}

/// The audit row's steps of one stack run, from the stack's own stage clocks, which run back to
/// back from `at` to the moment the stack returned: the drones' intake (one shared pass over every
/// chunk, so each drone carries the pass's time), the tentacles' windows with their `g0_tx` (one
/// shared pass), G1, then G2's gate (the stack's time after G1).
fn note_stack_steps(at: u64, t: &drone_path::StackRun) {
    let ms = |s: f64| {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "positive milliseconds")]
        let m = (s.max(0.0) * 1000.0).round() as u64;
        m
    };
    let (sense, compose, g1) = (ms(t.sense_s), ms(t.compose_s), ms(t.g1_s));
    let g2 = ms(t.total_s - t.sense_s - t.compose_s - t.g1_s);
    let n = t.windows.len();
    let flagged = t.flags.iter().filter(|f| **f > 0.0).count() as u64;
    let step = |part: String, start: u64, ms: u64, shared: &'static str, counts: Vec<(&'static str, u64)>| {
        crate::tools::note(crate::tools::Step { part, start_ms: Some(start), ms: Some(ms), shared: Some(shared), counts, ..crate::tools::Step::default() });
    };
    for i in 0..n {
        step(format!("nmpod.unit#d{i}"), at, sense, "drones' intake", vec![("chunks", t.flags.len() as u64), ("flagged", flagged)]);
    }
    for (i, w) in t.windows.iter().enumerate() {
        let depth = t.depths.get(i).copied().unwrap_or(0) as u64;
        step(format!("nmpod.unit#t{i}"), at + sense, compose, "windows and g0_tx", vec![("window", w.len() as u64), ("reported", depth)]);
        step(format!("nmbrain.g0#t{i}"), at + sense, compose, "windows and g0_tx", vec![("window", w.len() as u64), ("reported", depth)]);
    }
    // G1 and G2 are placed back from the moment the stack returned, so rounding of the stage
    // clocks never puts G2's end past the stack's.
    let end = nmtype::now_ms().max(at + sense + compose + g1 + g2);
    let g2_at = end.saturating_sub(g2);
    crate::tools::note(crate::tools::Step {
        counts: vec![("handover", t.handover.len() as u64), ("queue", t.queue.len() as u64), ("rounds", u64::from(t.rounds))],
        ..crate::tools::Step { part: "nmbrain.g1#g1".into(), start_ms: Some(g2_at.saturating_sub(g1)), ms: Some(g1), ..crate::tools::Step::default() }
    });
    crate::tools::note(crate::tools::Step {
        counts: vec![("up", t.up.len() as u64), ("instructions", t.g2.len() as u64)],
        ..crate::tools::Step { part: "nmbrain.g2#g2".into(), start_ms: Some(g2_at), ms: Some(g2), ..crate::tools::Step::default() }
    });
}

impl WholePath {
    #[must_use]
    pub fn new(home: &Path) -> Self {
        Self {
            home: home.to_owned(),
            keep: Mutex::new(None),
            relevance: Mutex::new(None),
            warming: std::sync::atomic::AtomicBool::new(false),
            keep_seen: Mutex::new(None),
            touched: Mutex::new(Vec::new()),
            reranker: Mutex::new(None),
            base: Mutex::new(None),
            up: None,
            room_tokens: std::sync::atomic::AtomicUsize::new(0),
            refine: std::sync::Mutex::new(String::new()),
            ask: crate::ask::Mode::default(),
            budget_s: crate::decide::BUDGET_FALLBACK_S,
            withhold_below: Some(WITHHOLD_BELOW),
            up_next: 0,
            reach: true,
            uses: true,
            inject: None,
            rerank_cap: None,
            look: true,
            lenses: false,
            default_mode: None,
            rerank_compare: false,
            look_gate: Some(LOOK_GATE),
            fast_k: FAST_K,
            judge_window: JUDGE_WINDOW,
            fast_judge_window: FAST_JUDGE_WINDOW,
            look_window: LOOK_WINDOW,
            reach_k: 2,
            arrangements: std::collections::BTreeMap::new(),
            desk: Mutex::new(nmbrain::instruct::Desk::default()),
            reflexes: Mutex::new(nmbrain::reflex::Set::default()),
            own_files: Mutex::new(Vec::new()),
        }
    }

    /// Run the whole path for `question` over `sub`.
    /// # Errors
    /// Weight files, tokenizer or engine faults.
    /// `orders`: G2's instructions for this question (`nmbrain::instruct`); standing ones come from `desk`.
    /// `mode`: fast reads the stack's first `fast_k` pieces (and the dense lane's) with the
    /// reranker and nothing more; full runs every part its arrangement allows.
    #[expect(
        clippy::too_many_arguments,
        reason = "the question, memory's chunks and links, the decision settings, G2's instructions and the mode"
    )]
    pub fn run(
        &self,
        sub: &Substrate,
        question: &str,
        recalled: &[usize],
        cocited: &std::collections::BTreeMap<String, Vec<(String, f32)>>,
        threshold: f32,
        judge: Option<&crate::ask::Judge>,
        orders: &[nmbrain::instruct::Instr],
        mode: Mode,
    ) -> Result<(Run, Vec<Piece>), Fault> {
        let fast = mode == Mode::Fast;
        // Background card work (the dense lane's index) pauses while the question is answered.
        let (_question, c) = crate::tools::clocked(|| nmengn::placement::gpu_gate().question());
        // The background intake pass waits while this call runs (its processor work too).
        let _call = crate::key_scan::call_mark();
        crate::tools::note(crate::tools::Step { cells: vec!["card gate".into()], ..crate::tools::step_at("nmengn.placement", c) });
        let t0 = Instant::now();
        let clock = Clock::default();
        let (scored0, dropped0) = (
            nmengn::roles::SCORED.load(std::sync::atomic::Ordering::Relaxed),
            nmengn::roles::DROPPED.load(std::sync::atomic::Ordering::Relaxed),
        );
        let check: std::cell::RefCell<Option<(f32, bool)>> = std::cell::RefCell::new(None);
        // Loading step L in service (2026-11-13): the Core Brain's two models stage their weights on
        // the copy stream while the stack runs on the processor, so the first forward finds them on
        // the card (`run cart l2`: staging hidden behind other work; logits bit-identical).
        // The one flow opens no model.
        let flow = mode == Mode::Flow;
        let reranker_res = if flow {
            Err(bad("the one flow opens no model"))
        } else {
            let (r, c) = crate::tools::clocked(|| self.reranker());
            crate::tools::note(crate::tools::Step { cells: vec!["open".into()], ..crate::tools::step_at("nmengn.model", c) });
            r
        };
        let staging = reranker_res.ok().map(|r| {
            std::thread::spawn(move || {
                if let Some(g) = r.pop.gpu.as_ref()
                    && let Ok(st) = g.gpu.copy_stream()
                {
                    r.stage_parts(&st, crate::joint_read::full_head()).map_or(0, |s| s.tensors)
                } else {
                    0
                }
            })
        });
        let mut kg = self.keep.lock().map_err(|_| bad("keep lock"))?;
        let version_changed = kg.as_ref().is_none_or(|(v, _)| *v != sub.version);
        // Memory areas (2026-11-27): the call does not wait for a whole-folder intake. It takes in
        // the pieces its question likely needs and at most `FIRST_STEP` in all, then reads the
        // material as taken in so far (a piece not taken in reads as its path); the background
        // pass (`warm_keep`) takes in the rest between calls. A small change is taken in whole.
        let waiting = kg.as_ref().is_some_and(|(_, k)| !k.state().complete() && !k.state().full);
        if version_changed || waiting {
            let mut keep = match kg.take() {
                Some((_, k)) => k,
                None => crate::tools::timed_step("nmapi.drone_windows", || Keep::open(&nmtype::models_dir(&self.home).join("qwen3-0.6b/files")))?,
            };
            let first = question_pieces(sub, question, crate::drone_windows::STEP_PIECES);
            let step = crate::tools::timed_step("nmapi.drone_windows", || clock.time("index", || keep.sync_step(sub, &first, FIRST_STEP)));
            self.note_keep(keep.state());
            *kg = Some((sub.version, keep));
            step?;
        }
        let complete = kg.as_ref().is_some_and(|(_, k)| k.state().complete());
        if complete && (version_changed || waiting || (crate::neural::bound_model().is_some() && self.relevance.lock().is_ok_and(|g| g.is_none()))) {
            self.start_intake(sub, kg.as_ref().map(|(_, k)| k));
        }
        let keep = &kg.as_ref().ok_or_else(|| bad("no intake"))?.1;
        let g2in = {
            let now = nmtype::now_ms();
            let mut desk = self.desk.lock().map_err(|_| bad("desk lock"))?;
            desk.expire(now);
            let reflexes = self.reflexes.lock().map(|r| r.live()).unwrap_or_default();
            let own_files = self.own_files.lock().map(|o| o.clone()).unwrap_or_default();
            drone_path::G2In {
                orders: orders.to_vec(),
                gains: desk.gains(nmsubs::lens::NAMES.len(), now),
                inhibited: desk.inhibited(now),
                reflexes,
                own_files,
                signals: None,
                room_tokens: Some(match self.room_tokens.load(std::sync::atomic::Ordering::Relaxed) {
                    0 => usize::try_from(pack_tokens_control(&self.home)).unwrap_or(crate::drone_path::PACK_TOKENS),
                    n => n,
                }),
                refine: self.refine.lock().ok().map(|r| r.clone()).filter(|r| !r.is_empty()),
            }
        };
        let stack_at = nmtype::now_ms();
        let mut t = if flow {
            drone_path::stack_run_flow(sub, keep, question, self.up, recalled, cocited, &g2in)?
        } else {
            drone_path::stack_run_g2(sub, keep, question, self.up, true, self.lenses, recalled, cocited, &g2in)?
        };
        if let Ok(mut tg) = self.touched.lock() {
            *tg = t.windows.iter().flatten().copied().collect();
        }
        clock.add("plain ranking", t.sense_s);
        clock.add("GCX rules", t.compose_s + t.g1_s);
        note_stack_steps(stack_at, &t);
        // H4 (lane M's `Down::install_from_gate`, `NANO_GATE_REFLEX=off`): G2 installs the reflexes its
        // gate calls for (tie R3, low R2 and R5, withheld R1; each for one question, recorded in G1's
        // ledger first) into the organism's reflex set.
        if flow
            && !std::env::var("NANO_GATE_REFLEX").is_ok_and(|v| v == "off")
            && let Ok(mut set) = self.reflexes.lock()
        {
            let own_edits = self.own_files.lock().is_ok_and(|o| !o.is_empty());
            let band = t.g2run.close_lead.1;
            let gate = t.g2run.gate.clone();
            let _ = t.g2run.motor.install_from_gate(&gate, own_edits, band, &mut set, nmtype::now_ms(), 60_000);
        }
        // Count the question for each installed reflex, with the firings it reported.
        if let Ok(mut set) = self.reflexes.lock() {

            let codes: Vec<nmbrain::reflex::Code> = set.installed.iter().map(|r| r.code).collect();
            let fired: Vec<nmbrain::reflex::Code> =
                codes.into_iter().flat_map(|c| t.g2.iter().filter(move |l| l.contains(c.name())).map(move |_| c)).collect();
            set.after_question(&fired);
        }
        drop(kg);
        // The one flow: what G2 sent up, in its order, is the answer; memory's packets join at the
        // pack as on the other modes.
        if flow {
            let stack_s = t0.elapsed().as_secs_f64();
            let mut status = stack_status(&t.pack);
            // Three states from two cuts (`NANO_FLOW_STATES=on`): the pack's fitted confidence
            // decides settled, low or withheld; a pack in which no piece holds a part stays
            // withheld.
            let score = flow_states().then(|| {
                let fit = flow_fit();
                (fit.score(&t.g2run.signals), fit.withhold, fit.settle)
            });
            if let Some((s, lo, hi)) = score
                && status.0 != "withheld"
            {
                // The fit lowers the stack's status, never raises it: a pack the stack reads as low
                // (a part no piece holds) stays low (fit and judge raising lows settled four
                // decoys of the second large code set).
                let fit = if s >= hi { "answered" } else if s < lo { "withheld" } else { "low" };
                if !(fit == "answered" && status.0 == "low") {
                    status.0 = fit.to_owned();
                }
            }
            // W7 with Phase H2 (2026-11-19): the judge's sufficiency read settles the middle state:
            // a low pack the judge reads as answering is answered; an answered pack it reads as not
            // answering is low (C110's missing signal).
            let suff = t.g2run.signals.sufficient.filter(|_| score.is_some());
            // The fit alone does not withhold a pack the stack would serve: withheld from the fit
            // stands only when the judge's sufficiency read also says the set does not answer
            // (large code val13 lost its answers to a fit-only withhold); otherwise it is low.
            if status.0 == "withheld" && stack_status(&t.pack).0 != "withheld" && !suff.is_some_and(|z| z < 0.0) {
                status.0 = "low".to_owned();
            }
            // The judge's read only lowers: an answered pack it reads as not answering is low. It
            // does not raise a low pack to answered (that settled four decoys of the second large
            // code set, 20 of 20 held to 16).
            if let Some(z) = suff
                && z < 0.0
                && status.0 == "answered"
            {
                status.0 = "low".to_owned();
            }
            // A gate with no finite rival is settled only by the judge's sufficiency read; without
            // it the tie serves as low (large2-code dec101 was settled by an infinite lead).
            if t.g2run.unrivalled_tie && status.0 == "answered" {
                status.0 = "low".to_owned();
            }
            // For the fit (`NANO_FLOW_TRACE=<file>`): one line per call, the question, the
            // signals, the stack's status, the status served, the follow-up's pieces and the
            // routed duels (embedder, judge, ties, joint reads), the pieces G2's orders had read,
            // its orders aimed at a meaning cluster, claims refused for a near-duplicate, and the
            // pieces swapped in by G2's room rule.
            if let Some(path) = std::env::var_os("NANO_FLOW_TRACE") {
                use std::io::Write;
                let s = &t.g2run.signals;
                let line = format!(
                    "{}\t{:.4}\t{:.4}\t{:.4}\t{:.0}\t{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{}\n",
                    question.replace(['\t', '\n'], " "),
                    s.held,
                    s.margin,
                    s.lane,
                    s.defines,
                    stack_status(&t.pack).0,
                    status.0,
                    t.g2run.follow.len(),
                    t.g2run.routed,
                    t.g2run.read,
                    t.g2run.cluster_orders,
                    t.g2run.refused_twin,
                    t.g2run.swapped.len()
                );
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = f.write_all(line.as_bytes());
                }
            }
            // G2's set is the answer; after it go only the pieces that hold a part of the request
            // the set leaves open (none when the set holds every part), in G1's order.
            let open_part = |c: usize| {
                t.evidence.as_ref().is_some_and(|ev| ev.parts.iter().enumerate().any(|(k, p)| status.1.contains(&p.1) && ev.part_holds(k, c)))
            };
            let order: Vec<usize> = if flow_fill() {
                t.up.clone()
            } else {
                t.up.iter().copied().filter(|c| t.g2run.set.contains(c) || open_part(*c)).collect()
            };
            // The pack stops where its pieces, in order, hold every part of the request.
            // Roadmap Phase 4 (2026-11-24): a part counts as held only by a piece the judge read as
            // answering (pack 47 to 37 and every 43 to 34 when any holder counted, answer or not);
            // a piece with no judge read holds no part, so the pack runs on past it.
            let order: Vec<usize> = match t.evidence.as_ref().filter(|ev| !ev.parts.is_empty() && flow_cover()) {
                Some(ev) => {
                    let reads = judge_reads(&t.g2run, &t.up);
                    let answers = |c: usize| reads.get(&c).is_some_and(|z| *z > 0.0);
                    cover_order(order, ev.parts.len(), &|k, c| answers(c) && ev.part_holds(k, c))
                }
                None => order,
            };
            // Audit W13 (lane T1, 2026-11-19; `NANO_PACK_SERVE=off` restores): the stack's
            // answer-first pack is served (answer spans grown toward their definitions, what is
            // needed with them, rivals as outline lines), then the open-part pieces it leaves out;
            // a withheld pack serves the filtered order as before.
            let mut pieces = if t.pack_pieces.is_empty() {
                drone_path::pieces(sub, &order, &t.windows)
            } else {
                // The pack's answer spans and what is needed with them first, then the served
                // order's other pieces whole (the open-part holders: on the first card smoke they
                // fell past the token cut behind the rivals' lines), the rivals' outline lines last.
                // Order: the answer spans, the served order's other pieces whole (the set's and
                // the open parts' holders, served before tonight), what is needed with the answer,
                // the rivals' lines (needed pieces ahead of the set's other pieces pushed answers
                // past the token cut on large code: val03, exp18).
                let (full, outlines): (Vec<crate::joint_read::Piece>, Vec<crate::joint_read::Piece>) = t.pack_pieces.iter().cloned().partition(|s| s.end > s.start);
                let (mut p, needed): (Vec<crate::joint_read::Piece>, Vec<crate::joint_read::Piece>) = full.into_iter().partition(|s| s.evidence.tentacle != 1);
                let rest: Vec<crate::joint_read::Piece> = drone_path::pieces(sub, &order, &t.windows)
                    .into_iter()
                    .filter(|x| !p.iter().any(|s| s.path == x.path && s.start <= x.start && x.end <= s.end))
                    .filter(|x| !p.iter().any(|s| s.chunk == x.chunk))
                    .collect();
                let shown: Vec<usize> = rest.iter().map(|x| x.chunk).collect();
                p.extend(rest);
                p.extend(needed.into_iter().filter(|o| !shown.contains(&o.chunk)));
                p.extend(outlines.into_iter().filter(|o| !shown.contains(&o.chunk)));
                p
            };

            // A pack of low confidence serves strips: of each piece, the lines that hold a part of
            // the request with the header of the definition they sit in; the whole piece stays
            // reachable by its place (`NANO_FLOW_STRIPS=on`; off by default).
            // The second look in the served flow (lane T1, 2026-11-19; `NANO_FLOW_LOOK=off`): on a
            // tie at G2's gate the close-call judge did not settle, the reader on the card reads the
            // first pieces of the winner's file together (`decide::second_look`, joint yes-or-no,
            // both orders); first place moves only past its measured margin. Within the call's time.
            if !std::env::var("NANO_FLOW_LOOK").is_ok_and(|v| v == "off")
                && t.g2run.gate == "tie"
                && t.g2run.close_fired == 0
                && pieces.len() >= 2
                && drone_path::warm_reader() == drone_path::JudgeState::Card
                && nmbrain::homeostasis::TIME_BUDGET.lock().is_ok_and(|b| b.allow(nmbrain::homeostasis::Stage::Chain, drone_path::call_budget() - t0.elapsed().as_secs_f64(), &[]))
                && let Ok(base) = crate::neural::model(&self.home)
            {
                let set = crate::tentacle_pass::drones(&self.home);
                let mut idx: Vec<usize> = (0..pieces.len()).collect();
                let tl = Instant::now();
                if let Ok(l) = crate::decide::second_look(&base, &set.heads, set.last_layer(), question, &pieces, &mut idx, crate::decide::LOOK_MARGIN) {
                    if l.switched {
                        let p: Vec<Piece> = idx.iter().filter_map(|i| pieces.get(*i).cloned()).collect();
                        if p.len() == pieces.len() {
                            pieces = p;
                        }
                    }
                    if let Ok(mut b) = nmbrain::homeostasis::TIME_BUDGET.lock() {
                        b.observe(nmbrain::homeostasis::Stage::Chain, tl.elapsed().as_secs_f64(), l.switched);
                    }
                }
            }
            // The short pack when low (`NANO_FLOW_SHORT=off`): a low pack the judge reads as not
            // answering serves only the pieces that hold a part of the request (no filler).
            let suff_no = suff.is_some_and(|z| z < 0.0);
            if status.0 == "low"
                && suff_no
                && !std::env::var("NANO_FLOW_SHORT").is_ok_and(|v| v == "off")
                && let Some(ev) = t.evidence.as_ref()
            {
                let holds = |c: usize| (0..ev.parts.len()).any(|k| ev.part_holds(k, c));
                // The first three pieces always stay (closure c17 lost its answer, first in the
                // stack, to the cut); the rest only when they hold a part.
                let short: Vec<Piece> = pieces.iter().enumerate().filter(|(i, p)| *i < KEEP_WHOLE || holds(p.chunk)).map(|(_, p)| p.clone()).collect();
                if !short.is_empty() {
                    pieces = short;
                }
            }
            // Strips only where the judge read the set as not answering (C110: strips on every low

            // pack lost answers, large code pack 44 to 27).
            if status.0 == "low" && (flow_strips() || suff.is_some_and(|z| z < 0.0))
                && let Some(ev) = t.evidence.as_ref()
            {
                let stems: Vec<&str> = ev.stems.iter().flatten().map(String::as_str).collect();
                // Roadmap Phase 4 (2026-11-24): each strip is served as runs of the file's own lines,
                // each cited by its place (`strip_runs`). The old strip was one text with `...` gaps
                // and a last newline, taken over the chunk and not the piece: it never matched the
                // file's lines from the piece's first line, so the pack showed it as a note with no
                // place and the bench lost every stripped answer (T-061: pack 44 to 27, every range
                // 41 to 22). The runs also keep the lines the lexical head rates highest and the
                // definition lines of the request's names, with their context.
                let terms = term_weights(sub, question);
                // The first three pieces stay whole (closure c17's answer, first in the stack); the
                // rest serve their strips.
                let mut served: Vec<Piece> = Vec::with_capacity(pieces.len());
                for (i, p) in pieces.into_iter().enumerate() {
                    match (i >= KEEP_WHOLE).then(|| strip_runs(sub, &p, &stems, &terms)).flatten() {
                        Some(runs) => served.extend(runs),
                        None => served.push(p),
                    }
                }
                pieces = served;
            }
            // Writer at idle (lane B, 2026-11-19): the served pieces go on the idle writer's queue
            // for summaries, the pieces G1 queued and did not send up are noted first, and the
            // question is kept for idle reasoning. RAM only; nothing when the writer at idle is off.
            if crate::writer_idle::running() {
                let key = |c: &usize| sub.chunks.get(*c).map(|x| x.key.clone());
                let mut served: Vec<String> = pieces.iter().filter_map(|p| key(&p.chunk)).collect();
                // A stripped piece is served as several runs of one chunk: noted once.
                served.dedup();
                let contested: Vec<String> = t.queue.iter().filter(|c| !t.up.contains(c)).filter_map(key).collect();
                crate::writer_idle::served(question, &served, &contested);
            }
            let run = Run {
                order,
                queued: t.queue.iter().copied().filter(|c| !t.up.contains(c)).collect(),
                g1_stop: t.stop.clone(),
                g1_rounds: t.rounds,
                g2: t.g2.clone(),
                stack_s,
                stages: clock.list(),
                flow_status: Some(status),
                flow_score: score,
                ..Run::default()
            };
            return Ok((run, pieces));
        }
        if fast {
            let fk = if self.fast_k == FAST_K { fast_k_control(&nmtype::nano_root()) } else { self.fast_k };
            t.up.truncate(fk.max(1));
        }
        // Memory joins at G2: recalled chunks the stack did not send up go up beside its own, so
        // the Core Brain reads them fresh with the rest; nothing is taken from memory unread.
        let recalled_in_windows = recalled.iter().filter(|c| t.windows.iter().any(|w| w.contains(c))).count();
        let extra: Vec<usize> = recalled.iter().copied().filter(|c| !t.up.contains(c) && *c < sub.chunks.len()).collect();
        t.up.extend(extra.iter().copied());
        // The dense lane adds the chunks nearest the question by meaning that the stack did not
        // send up (candidates the word senses miss); the reranker judges them with the rest.
        let lane = sub.dense.read().ok().and_then(|d| d.clone());
        let dense_added: Vec<usize> = match lane {
            Some(l) => clock
                .time("embedding", || crate::tools::timed_step("nmbrain.slot#embedder", || l.embedding_order(sub, question)))
                .map(|o| o.into_iter().filter(|c| !t.up.contains(c)).take(if fast { FAST_DENSE } else { DENSE_ADD }).collect())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        t.up.extend(dense_added.iter().copied());
        let stack_s = t0.elapsed().as_secs_f64();
        let mut pieces = drone_path::pieces(sub, &t.up, &t.windows);
        let t1 = Instant::now();
        let staged =
            clock.time("load", || crate::tools::timed_step("nmengn.model", || staging.and_then(|h| h.join().ok()).unwrap_or(0)));
        // Live anatomy (2026-11-13): the arrangement for this question's kind, else the fixed one.
        let kind = kind_of(question);
        let arr = if fast {
            Arrangement { reach: false, uses: false, look: false, reach_k: self.reach_k }
        } else {
            self.arrangements.get(kind).copied().unwrap_or(Arrangement {
                reach: self.reach,
                uses: self.uses,
                look: self.look,
                reach_k: self.reach_k,
            })
        };
        // Reliability (2026-11-13): a part that fails is left out and named; the answer goes on.
        let mut faults: Vec<String> = Vec::new();
        let inject = |part: &str| self.inject.as_deref() == Some(part);
        let reranker = clock.time("load", || crate::tools::timed_step("nmengn.model", || self.reranker()))?;
        // What the judge reads of each piece: the whole piece, or its window (`judge_view`).
        let terms = term_weights(sub, question);
        let eff_judge_window = if self.judge_window == JUDGE_WINDOW { judge_window_control(&nmtype::nano_root()) } else { self.judge_window };
        let window = if fast { self.fast_judge_window } else { eff_judge_window };
        let count = |s: &str| reranker.pop.tokenizer.encode(s).map_or(0, |v| v.len());
        let judged = std::cell::Cell::new((0_usize, 0_usize, 0_usize)); // pieces, windowed, tokens
        let view = |p: &Piece| -> Piece {
            let (v, n) = judge_view(p, &terms, window.unwrap_or(usize::MAX), &count);
            let (a, b, c) = judged.get();
            judged.set((a + 1, b + usize::from(v.text.len() < p.text.len()), c + n));
            v
        };
        let doc_of = |p: &Piece| {
            let v = view(p);
            // A document piece carries its heading path to the judge (empty for code).
            let heading = sub
                .chunks
                .get(p.chunk)
                .filter(|c| sub.files.get(c.file).is_some_and(|f| f.path == p.path))
                .map_or("", |c| c.heading.as_str());
            nmsubs::dense::doc_text_with(&v.path, heading, &v.text)
        };
        let docs: Vec<String> = pieces.iter().map(&doc_of).collect();
        let dev = if reranker.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let cap_rule = match self.rerank_cap {
            Some(c) => c,
            None => nmengn::roles::card_doc_cap(&reranker, dev, nmengn::roles::CODE_INSTRUCTION, question)?,
        };
        // Stage 5 WP-C2: the rerank cap is a learnable G3 site. `base` (the cap computed above) is the
        // deterministic rule and the choice whenever G3 is off or its evidence does not beat the
        // rule's margin; a promoted `tight` or `wide` reads a quarter fewer or more pieces.
        let cap_choice = crate::g3::decide_site(nmmusc::priors::Site::RerankCap, "base", false);
        let cap_now = match cap_choice.as_str() {
            "tight" => (cap_rule - cap_rule / 4).max(1),
            "wide" => cap_rule + (cap_rule / 4).max(1),
            _ => cap_rule,
        };
        let _ = crate::g3::note_site(nmmusc::sites::SiteKind::RerankCap, "core-brain", "-", &cap_choice, None);
        let cap_arg = if cap_choice == "base" { self.rerank_cap } else { Some(cap_now) };
        if !nmengn::roles::RERANK_DROP_CLI.load(std::sync::atomic::Ordering::Relaxed)
            && let Ok(mut g) = nmengn::roles::RERANK_DROP.lock()
        {
            *g = rerank_early_drop_control(&nmtype::nano_root());
        }
        let rerank = |docs: &[String], part: &str| -> Result<Vec<f32>, Fault> {
            if inject(part) {
                return Err(bad(format!("{part}: injected fault")));
            }
            let ins = nmengn::roles::CODE_INSTRUCTION;
            let v = clock.time("reranker", || {
                crate::tools::timed_layers("nmbrain.slot#reranker", (0, reranker.pop.dims.layers.saturating_sub(1) as u64), || {
                    nmengn::roles::card_rerank_batch(&reranker, dev, ins, question, docs, cap_arg)
                })
            })?;
            if self.rerank_compare {
                let one = nmengn::roles::card_rerank_set_capped(&reranker, dev, ins, question, docs, cap_arg)?;
                let diff = v.iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                let same = rank_order(&v) == rank_order(&one);
                let mut c = check.borrow_mut();
                *c = Some(c.map_or((diff, same), |(d, s)| (d.max(diff), s && same)));
            }
            Ok(v)
        };
        let (mut rr, rerank_ok) = match rerank(&docs, "reranker") {
            Ok(v) => (v, true),
            Err(e) => {
                // The drones' and G1's order stands in for the reranker's (arrival order).
                faults.push(format!("reranker: {} (arrival order kept)", e.message));
                #[expect(clippy::cast_precision_loss, reason = "piece counts are small")]
                let v: Vec<f32> = (0..pieces.len()).map(|i| 1.0 - i as f32 * 1e-4).collect();
                (v, false)
            }
        };
        // Raw reranker scores (`rr`) go to the ledger, the judge and the withhold rule; `key` orders
        // the pieces and carries the adjustments below (2026-11-13 review: the adjusted values had
        // reached the ledger). The first place is the reranker's own, found once before anything is
        // added, so use sites and reach both work from it.
        let mut key = rr.clone();
        // Material (any-data design 6.5): in a tree that holds code, a document piece orders by its
        // score times its source weight unless the question asks for what documents hold (a
        // definition, a procedure, a summary, a comparison); code pieces keep their score.
        // Documents cut at their own blocks are short and on point, and the judge scored them
        // above the code they describe (any-data stage 1, served path).
        if nmlex::doc_kind(question).is_none() && sub.files.iter().any(nmsubs::repo::FileInfo::is_code) {
            for (k, p) in key.iter_mut().zip(&pieces) {
                if nmtype::is_document(&p.path) {
                    *k *= nmtype::source_weight(&p.path);
                }
            }
        }
        let top0 = (0..pieces.len()).max_by(|a, b| {
            key.get(*a).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&key.get(*b).copied().unwrap_or(f32::NEG_INFINITY)).then(b.cmp(a))
        });
        let first0 = top0.and_then(|i| pieces.get(i).map(|p| p.chunk));
        let cap0 = top0.and_then(|i| rr.get(i).copied()).unwrap_or(0.0);
        // Use sites: a relation is an edge; the first place holds one end, its users the other.
        if arr.uses
            && asks_callers(question)
            && let Some(first) = first0
        {
            // 2026-11-14: callers already among the pieces count too (a caller sent up by the stack
            // was never lifted before); the new ones are judged by the reranker and added.
            let callers = use_sites(sub, question, first, &[first], USES_K);
            let held_idx: Vec<usize> = callers.iter().filter_map(|c| pieces.iter().position(|p| p.chunk == *c)).collect();
            let near: Vec<usize> = callers.iter().copied().filter(|c| !pieces.iter().any(|p| p.chunk == *c)).collect();
            let base_len = pieces.len();
            if !near.is_empty() {
                let more = drone_path::pieces(sub, &near, &t.windows);
                let more_docs: Vec<String> = more.iter().map(&doc_of).collect();
                let more_rr = match rerank(&more_docs, "uses") {
                    Ok(v) => v,
                    Err(e) => {
                        faults.push(format!("use sites: {}", e.message));
                        Vec::new()
                    }
                };
                if more_rr.len() == more.len() {
                    key.extend(more_rr.iter().copied());
                    rr.extend(more_rr);
                    t.up.extend(near);
                    pieces.extend(more);
                }
            }
            // The question asks for the other end of the edge: the caller the reranker judges best,
            // held or new, goes first, just above the definition (in the ordering key only).
            // Callers in test or check code are left out (2026-11-14: c20's wrong pick was a call in
            // the definition file's own check; preferring other files instead lost l20 and m19 on
            // CX2, whose callers sit in the definition's file); all callers only if none remain.
            let all: Vec<usize> = held_idx.into_iter().chain(base_len..pieces.len()).collect();
            let is_test = |i: &usize| {
                pieces
                    .get(*i)
                    .and_then(|p| sub.chunks.get(p.chunk))
                    .is_some_and(|c| c.low.contains("fn check") || c.low.contains("#[test]") || c.low.contains("mod tests"))
            };
            let real: Vec<usize> = all.iter().copied().filter(|i| !is_test(i)).collect();
            let candidates = if real.is_empty() { all } else { real };
            if let Some(best) = candidates.into_iter().max_by(|a, b| {
                rr.get(*a).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&rr.get(*b).copied().unwrap_or(f32::NEG_INFINITY)).then(b.cmp(a))
            }) && let Some(x) = key.get_mut(best)
            {
                *x = x.max(above(cap0));
            }
        }
        // Within-file reach: the neighbours of the reranker's first place in its file, judged and
        // added, ordered strictly below that first place.
        if arr.reach
            && let Some(first) = first0
        {
            let held: Vec<usize> = std::iter::once(first).chain(pieces.iter().map(|p| p.chunk)).collect();
            let near: Vec<usize> = if arr.reach_k <= 2 {
                crate::ask::request(sub, &held, &[], 0).into_iter().map(|(_, c)| c).collect()
            } else {
                file_reach(sub, first, &held, arr.reach_k)
            };
            if !near.is_empty() {
                let more = drone_path::pieces(sub, &near, &t.windows);
                let more_docs: Vec<String> = more.iter().map(&doc_of).collect();
                // A neighbour is read because of the first place; on development it displaced a
                // right first when it could rank above it.
                let more_rr = match rerank(&more_docs, "reach") {
                    Ok(v) => v,
                    Err(e) => {
                        faults.push(format!("reach: {}", e.message));
                        Vec::new()
                    }
                };
                let more = if more_rr.len() == more.len() { more } else { Vec::new() };
                let near: Vec<usize> = if more.is_empty() { Vec::new() } else { near };
                key.extend(more_rr.iter().map(|x| below(*x, cap0)));
                rr.extend(more_rr);
                t.up.extend(near);
                pieces.extend(more);
            }
        }
        // Asking for more context (`nmapi::ask`, 2026-11-12): the judge sorts the first place; while
        // the rule allows, one request cycle (Q3 then Q1) brings new chunks, judged by the reranker
        // and added; context grows, nothing held is dropped.
        let budget_s = self.budget_s;
        let (mut ask_class, mut ask_conf, mut asked, mut cycles, mut improved, mut cycle_s) =
            (None, None, Vec::new(), 0_usize, true, 0.0_f64);
        loop {
            if let Some(j) = judge {
                let c = j.confidence(&rr);
                ask_conf = Some(c);
                ask_class = Some(j.crag.class(c));
            }
            let ask_mode = if fast { crate::ask::Mode::Off } else { self.ask };
            if !crate::ask::another(ask_mode, ask_class, cycles, improved, budget_s - t0.elapsed().as_secs_f64(), cycle_s) {
                break;
            }
            let tc = Instant::now();
            let mut held: Vec<usize> = (0..pieces.len()).collect();
            held.sort_by(|a, b| {
                key.get(*b)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY)
                    .total_cmp(&key.get(*a).copied().unwrap_or(f32::NEG_INFINITY))
                    .then(a.cmp(b))
            });
            let lead = held.first().and_then(|i| key.get(*i).copied()).unwrap_or(0.0);
            let held: Vec<usize> = held.iter().filter_map(|i| pieces.get(*i).map(|p| p.chunk)).collect();
            let queued: Vec<usize> = t.handover.iter().chain(&t.queue).copied().collect();
            let k = if ask_class == Some(crate::ask::Class::Incorrect) || self.ask == crate::ask::Mode::Always {
                crate::ask::Q3_K * 4
            } else {
                crate::ask::Q3_K * 2
            };
            let req = crate::ask::request(sub, &held, &queued, k);
            if req.is_empty() {
                break;
            }
            let chunks: Vec<usize> = req.iter().map(|(_, c)| *c).collect();
            let more = drone_path::pieces(sub, &chunks, &t.windows);
            let more_docs: Vec<String> = more.iter().map(&doc_of).collect();
            let more_rr = match rerank(&more_docs, "ask") {
                Ok(v) => v,
                Err(e) => {
                    faults.push(format!("asking for more context: {}", e.message));
                    break;
                }
            };
            let best = rr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            improved = more_rr.iter().any(|x| *x > best);
            asked.extend(req.iter().map(|(q, c)| format!("{q}:{c}")));
            // Q1 brings the first place's neighbours: ordered below it, as reach's are.
            let tags: Vec<&str> = more.iter().map(|p| req.iter().find(|(_, c)| *c == p.chunk).map_or("Q3", |(q, _)| *q)).collect();
            key.extend(more_rr.iter().zip(&tags).map(|(x, q)| if *q == "Q1" { below(*x, lead) } else { *x }));
            t.up.extend(chunks);
            pieces.extend(more);
            rr.extend(more_rr);
            cycles += 1;
            cycle_s = tc.elapsed().as_secs_f64();
        }
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by(|a, b| {
            key.get(*b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&key.get(*a).copied().unwrap_or(f32::NEG_INFINITY)).then(a.cmp(b))
        });
        let set = crate::tentacle_pass::drones(&self.home);
        // The second look is a long order from G2 to the Core Brain (motor division, 2026-11-12):
        // goal id, lifecycle, feedback, a copy to G1, valid for one minute.
        let now = nmtype::now_ms();
        let mut ord = nmtype::order::Order::new(now, nmtype::order::Verb::JointRead, "nmbrain.g2", "nmbrain.core-brain", now, 60_000);
        ord.accept(nmtype::now_ms())?;
        ord.start()?;
        let gap = if rerank_ok { top_gap(&rr) } else { f32::INFINITY };
        let pre_look_first =
            order.first().and_then(|i| pieces.get(*i)).map(|p| format!("{}:{}-{}", p.path, p.start, p.end)).unwrap_or_default();
        let gated = self.look_gate.is_some_and(|g| gap >= g);
        // Stage 5 WP-C2: the second look is a learnable G3 site. `gated` (run unless the top-two gap
        // reaches the gate) is the deterministic rule and the choice whenever G3 is off or its evidence
        // does not beat the rule's margin; a promoted `always` or `skip` runs it on every call or none.
        let look_choice = if arr.look { crate::g3::decide_site(nmmusc::priors::Site::SecondLook, "gated", false) } else { "skip".to_owned() };
        let look_ran = match look_choice.as_str() {
            "always" => arr.look,
            "skip" => false,
            _ => arr.look && !gated,
        };
        if arr.look {
            let _ = crate::g3::note_site(nmmusc::sites::SiteKind::Look, "core-brain", "-", &look_choice, None);
        }
        let looked = if !look_ran {
            Ok(crate::decide::Look::default())
        } else if inject("look") {
            Err(bad("second look: injected fault"))
        } else {
            let t = Instant::now();
            let look_at = nmtype::now_ms();
            // With `look_window`, the second look reads the judge's windows of its pieces.
            let windowed: Vec<Piece>;
            let read_from: &[Piece] = if self.look_window && self.judge_window.is_some() {
                let w = self.judge_window.unwrap_or(usize::MAX);
                windowed = pieces.iter().map(|p| judge_view(p, &terms, w, &count).0).collect();
                &windowed
            } else {
                &pieces
            };
            let base = clock.time("load", || crate::tools::timed_step("nmengn.model", || self.base()))?;
            let look_span = nmengn::prof::Span::new("look");
            let l = crate::decide::second_look(&base, &set.heads, set.last_layer(), question, read_from, &mut order, threshold);
            drop(look_span);
            let joint = l.as_ref().map_or(0.0, |l| l.seconds);
            if std::env::var_os("NM_JOINT_TRACE").is_some() {
                eprintln!("second look: joint read {joint:.3} s");
            }
            clock.add("joint read", joint);
            clock.add("second look", (t.elapsed().as_secs_f64() - joint).max(0.0));
            // The second look on the base model, and the joint read inside it (its own clock gives
            // the length only).
            crate::tools::note_layers(
                "nmapi.decide",
                Some(look_at),
                Some(u64::try_from(t.elapsed().as_millis()).unwrap_or(u64::MAX)),
                (0, set.last_layer() as u64),
            );
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "positive milliseconds")]
            crate::tools::note_step("nmapi.joint_read", None, Some((joint * 1000.0).round() as u64));
            l
        };
        let look = match looked {
            Ok(l) => {
                ord.progress(&format!("read {} pieces together", l.read.len()))?;
                ord.succeed(if l.switched { "first place moved" } else { "first place kept" })?;
                l
            }
            Err(e) => {
                // The reranker's order stands; the order is aborted and the fault named.
                let _ = ord.abort(&e.message);
                faults.push(format!("second look: {} (reranker order kept)", e.message));
                crate::decide::Look::default()
            }
        };
        let chunks: Vec<usize> = order.iter().filter_map(|i| pieces.get(*i).map(|p| p.chunk)).collect();
        let queued: Vec<usize> = t.handover.iter().chain(&t.queue).copied().filter(|c| !chunks.contains(c)).collect();
        let ordered: Vec<Piece> = order.iter().filter_map(|i| pieces.get(*i).cloned()).collect();
        let chunk_of = |i: &usize| pieces.get(*i).map(|p| p.chunk);
        // The judged set and the reranker's scores, handed to the training store (nothing kept when it is off).
        nmtype::examples::tap(question, pieces.iter().map(|p| (p.path.as_str(), p.start, p.end, p.text.as_str())), &rr, rerank_ok);
        crate::tools::note_run_end(nmtype::now_ms());
        Ok((
            Run {
                order: chunks,
                queued,
                moved_by_look: look.switched,
                candidates: pieces.iter().map(|p| p.chunk).collect(),
                rerank: rr.clone(),
                look: look.read.iter().filter_map(chunk_of).collect(),
                look_scores: look.scores.clone(),
                order_line: ord.line(),
                g1_stop: t.stop.clone(),
                g1_rounds: t.rounds,
                recalled: extra,
                recalled_in_windows,
                look_margin: look.margin,
                stack_s,
                brain_s: t1.elapsed().as_secs_f64(),
                ask_class,
                ask_conf,
                asked,
                g2: t.g2.clone(),
                staged,
                faults,
                rerank_ok,
                stages: clock.list(),
                rerank_check: *check.borrow(),
                top_gap: gap,
                look_ran,
                pre_look_first,
                dense_added,
                rerank_docs: nmengn::roles::SCORED.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(scored0),
                rerank_cap: cap_now,
                rerank_dropped: nmengn::roles::DROPPED.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(dropped0),
                judge_pieces: judged.get().0,
                judge_windowed: judged.get().1,
                judge_tokens: judged.get().2,
                look_pieces: look.read.len(),
                look_tokens: look.tokens,
                look_passes: look.passes,
                flow_status: None,
                flow_score: None,
            },
            ordered,
        ))
    }

    /// Bind the Core Brain's two models and stage their weights on the card now, so the first
    /// question finds them warm (a server calls this at start, beside answering). Returns the
    /// seconds it took and the tensors staged. Safe to call again: held models are not rebound.
    /// # Errors
    /// Weight or engine faults.
    /// Open the drones' intake and take `sub` in now (the one flow at start), so no call pays for
    /// it; the pieces taken in.
    /// # Errors
    /// The reader's weights or the tokenizer.
    ///
    /// Memory areas (2026-11-27): the intake runs in bounded steps (`Keep::sync_step`,
    /// `STEP_PIECES` each), the lock released between steps and the pass waiting while a call
    /// runs, so a call never waits for the whole folder; it stops when every piece is in, when
    /// the `content` memory area is full, or when the material moved on (the next call takes the
    /// newer version in and starts the pass again). One pass runs at a time.
    pub fn warm_keep(&self, sub: &Substrate) -> Result<usize, Fault> {
        use std::sync::atomic::Ordering;
        if self.warming.swap(true, Ordering::AcqRel) {
            return Ok(0);
        }
        struct Done<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _done = Done(&self.warming);
        let mut total = 0_usize;
        loop {
            if crate::key_scan::stopping() {
                break;
            }
            // A call in flight has the processor: the pass yields between its steps.
            crate::key_scan::wait_for_calls();
            let mut kg = self.keep.lock().map_err(|_| bad("keep lock"))?;
            if kg.as_ref().is_some_and(|(v, _)| *v > sub.version) {
                break;
            }
            if kg.as_ref().is_some_and(|(v, k)| *v == sub.version && (k.state().complete() || k.state().full)) {
                break;
            }
            let mut keep = match kg.take() {
                Some((_, k)) => k,
                None => Keep::open(&nmtype::models_dir(&self.home).join("qwen3-0.6b/files"))?,
            };
            let step = keep.sync_step(sub, &[], crate::drone_windows::STEP_PIECES);
            let state = keep.state();
            self.note_keep(state);
            *kg = Some((sub.version, keep));
            total += step?;
            if state.complete() {
                self.start_intake(sub, kg.as_ref().map(|(_, k)| k));
                break;
            }
            if state.full {
                break;
            }
            drop(kg);
        }
        Ok(total)
    }

    /// Whether the drones' intake still waits on pieces it has room for (no pass running): the
    /// caller starts `warm_keep` on its own thread.
    #[must_use]
    pub fn keep_waiting(&self) -> bool {
        !self.warming.load(std::sync::atomic::Ordering::Acquire) && self.keep_state().is_some_and(|s| !s.complete() && !s.full)
    }

    /// Where the drones' intake stands after its last step (`None` before it opened).
    #[must_use]
    pub fn keep_state(&self) -> Option<crate::drone_windows::KeepState> {
        self.keep_seen.lock().ok().and_then(|g| *g)
    }

    fn note_keep(&self, s: crate::drone_windows::KeepState) {
        if let Ok(mut g) = self.keep_seen.lock() {
            *g = Some(s);
        }
    }

    /// Whether a background intake pass runs now.
    #[must_use]
    pub fn warming(&self) -> bool {
        self.warming.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Start the relevance-key intake over `sub` on its own thread (Phase C2), when
    /// `key_scan::intake_on`; a no-op otherwise, while one runs, or after a finished intake of
    /// this material version. The store is made on first use for every drone head of the base
    /// population (default and tentacle sets), its byte limit the derived RAM limit. Returns
    /// whether a pass started. `keep` gives each piece's drone signals.
    pub fn start_intake(&self, sub: &Substrate, keep: Option<&Keep>) -> bool {
        if !crate::key_scan::intake_on() {
            return false;
        }
        let Some(model) = crate::neural::bound_model() else { return false };
        let keys = {
            let Ok(mut g) = self.relevance.lock() else { return false };
            Arc::clone(g.get_or_insert_with(|| {
                let sets = crate::tentacle_pass::drones(&nmtype::nano_root());
                let mut heads = sets.heads.clone();
                for v in sets.tentacles.values() {
                    heads.extend(v.iter().copied());
                }
                heads.sort_unstable();
                heads.dedup();
                let d = model.pop.dims;
                let group = d.heads / d.kv_heads.max(1);
                Arc::new(crate::key_scan::Keys::new_alone(&heads, group, d.head_dim, nmengn::kvstore::derive_ram_limit().bytes))
            }))
        };
        let touched = self.touched.lock().map(|t| t.clone()).unwrap_or_default();
        let order = crate::key_scan::intake_order(sub, &keys, &touched);
        let pieces = crate::key_scan::intake_pieces(sub, keep, &order);
        let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        keys.start_intake(model, dev, pieces, sub.version)
    }

    /// The relevance-key store beside the drones' intake, once made (Phase C2).
    #[must_use]
    pub fn relevance_keys(&self) -> Option<Arc<crate::key_scan::Keys>> {
        self.relevance.lock().ok().and_then(|g| g.clone())
    }

    pub fn warm(&self) -> Result<(f64, usize), Fault> {
        let t = Instant::now();
        // The one flow opens no model at start, save the close-call judge, which loads on the card
        // in the background (W4; nothing waits for it, and without a card it is never loaded).
        if self.default_mode == Some(Mode::Flow) {
            if crate::drone_path::w2_on() {
                let _ = crate::drone_path::warm_close_judge();
            }
            if crate::tentacle_read::on() {
                let _ = crate::drone_path::warm_reader();
            }
            // Lane B: the project outline the writer wrote at idle, for the session's start (RAM).
            if let Some(o) = crate::writer_idle::outline() {
                eprintln!("writer: project outline held ({} characters)", o.len());
            }

            return Ok((0.0, 0));
        }
        let stage = |m: &Arc<Model>| -> Result<usize, Fault> {
            if let Some(g) = m.pop.gpu.as_ref() {
                let st = g.gpu.copy_stream().map_err(bad)?;
                Ok(m.stage_parts(&st, crate::joint_read::full_head())?.tensors)
            } else {
                Ok(0)
            }
        };
        let r = self.reranker()?;
        let mut n = stage(&r)?;
        if self.look {
            let b = self.base()?;
            n += stage(&b)?;
        }
        Ok((t.elapsed().as_secs_f64(), n))
    }

    fn reranker(&self) -> Result<Arc<Model>, Fault> {
        let mut g = self.reranker.lock().map_err(|_| bad("reranker lock"))?;
        if let Some(r) = g.as_ref() {
            return Ok(Arc::clone(r));
        }
        let r = crate::neural::reranker_model(&self.home)?;
        *g = Some(Arc::clone(&r));
        Ok(r)
    }

    fn base(&self) -> Result<Arc<Model>, Fault> {
        let mut g = self.base.lock().map_err(|_| bad("base lock"))?;
        if let Some(b) = g.as_ref() {
            return Ok(Arc::clone(b));
        }
        let b = crate::neural::model(&self.home)?;
        *g = Some(Arc::clone(&b));
        Ok(b)
    }
}

/// Whether the second look reads the judge's windows by default (`serve --look-window on|off`):
/// on; measured on the development set it kept every count (one more right first) and took the
/// joint read from 5.35 to 4.48 s per question (speed pass 2).
pub const LOOK_WINDOW: bool = true;

/// Tokens the judge reads of each piece by default (`serve --judge-window <tokens>|off`; `None`
/// reads every piece whole). 384 is the smallest of 384, 256 and 192 that kept right first, top 3
/// and in pack on the development set with no right location lost from a pack (speed pass 2).
pub const JUDGE_WINDOW: Option<usize> = Some(384);
/// Tokens the judge reads of each piece in fast mode by default: whole pieces. Fast mode reads 4
/// to 6 pieces with no reach, asking or second look behind it; there a window of 384 lowered top 3
/// by one development question (speed pass 2), so fast mode reads whole pieces.
pub const FAST_JUDGE_WINDOW: Option<usize> = None;

/// The question's key terms with their rarity (`Substrate::idf`), for `judge_view`.
#[must_use]
pub fn term_weights(sub: &Substrate, question: &str) -> Vec<(String, f32)> {
    nmlex::key_terms(question).into_iter().map(|t| (t.clone(), sub.idf(&t))).collect()
}

/// What the judge reads of a piece: when the piece holds more than `w` tokens (by `count`, per
/// line with its line break), the run of whole lines of at most `w` tokens that holds the most
/// question-term evidence (each line scores the rarity of the key terms it contains, as the plain
/// ranking weighs them), placed so the evidence sits in its middle; with no evidence, the piece's
/// first lines. `start` and `end` follow the lines kept. A piece of `w` tokens or fewer comes back
/// whole. The second value is the tokens of what is read. Only the judge's reading changes: the
/// pack carries the whole piece.
#[must_use]
pub fn judge_view(p: &Piece, terms: &[(String, f32)], w: usize, count: &dyn Fn(&str) -> usize) -> (Piece, usize) {
    let lines: Vec<&str> = p.text.lines().collect();
    let toks: Vec<usize> = lines.iter().map(|l| count(&format!("{l}\n"))).collect();
    let total: usize = toks.iter().sum();
    if total <= w || lines.len() < 2 {
        return (p.clone(), total);
    }
    let score: Vec<f32> = lines
        .iter()
        .map(|l| {
            let low = l.to_lowercase();
            terms.iter().filter(|(t, _)| low.contains(t.as_str())).map(|(_, x)| *x).sum()
        })
        .collect();
    // Every start with its longest run of whole lines within `w` (at least one line).
    let mut best: Option<(f32, usize, usize, usize)> = None; // score, imbalance, start, end
    let mut j = 0;
    let mut sum = 0;
    for i in 0..lines.len() {
        if j < i {
            j = i;
            sum = 0;
        }
        while j < lines.len() && (j == i || sum + toks[j] <= w) {
            sum += toks[j];
            j += 1;
        }
        let end = j - 1;
        let s: f32 = score[i..=end].iter().sum();
        let hits: Vec<usize> = (i..=end).filter(|k| score[*k] > 0.0).collect();
        let imbalance = match (hits.first(), hits.last()) {
            (Some(a), Some(b)) => (a - i).abs_diff(end - b),
            _ => i,
        };
        if best.is_none_or(|(bs, bi, _, _)| s > bs || (s == bs && imbalance < bi)) {
            best = Some((s, imbalance, i, end));
        }
        sum -= toks[i];
    }
    let (_, _, a, b) = best.unwrap_or((0.0, 0, 0, 0));
    let kept: Vec<&str> = lines.get(a..=b).map(<[&str]>::to_vec).unwrap_or_default();
    let read = toks.get(a..=b).map_or(0, |t| t.iter().sum());
    (Piece { text: kept.join("\n"), start: p.start + a, end: p.start + b, ..p.clone() }, read)
}

/// Distinctive terms used by re-read verification: the question's key terms the tree holds, rarest first.
pub const VERIFY_TERMS: usize = 3;

/// The question's distinctive terms for verification: its key terms that occur in some chunk,
/// rarest first by the number of chunks holding them, at most `VERIFY_TERMS`.
#[must_use]
pub fn verify_terms(sub: &Substrate, question: &str) -> Vec<String> {
    let mut terms: Vec<(usize, String)> = nmlex::key_terms(question)
        .into_iter()
        .map(|t| (sub.chunks.iter().filter(|c| c.low.contains(t.as_str())).count(), t))
        .filter(|(n, _)| *n > 0)
        .collect();
    terms.sort();
    terms.dedup_by(|a, b| a.1 == b.1);
    terms.into_iter().take(VERIFY_TERMS).map(|(_, t)| t).collect()
}

/// Whether a lower-cased text holds the verification terms: at least two of them (all when
/// fewer); with no term nothing is verified.
#[must_use]
pub fn verified_text(terms: &[String], low: &str) -> bool {
    let need = terms.len().min(2);
    need > 0 && terms.iter().filter(|t| low.contains(t.as_str())).count() >= need
}

/// Re-read verification of chunks (the L1 signal, research file Solution 6; 2026-11-12): each
/// chunk is read fresh from the file on disk (not the indexed copy) and counts as verified when it
/// holds at least two of the question's `VERIFY_TERMS` rarest key terms that occur in the tree
/// (all of them when fewer). A chunk whose file or lines are gone is not verified; with no such
/// term in the question nothing is.
#[must_use]
pub fn verify_chunks(sub: &Substrate, question: &str, chunks: &[usize]) -> Vec<bool> {
    let terms = verify_terms(sub, question);
    chunks
        .iter()
        .map(|&c| {
            let Some(ch) = sub.chunks.get(c) else { return false };
            let Some(f) = sub.files.get(ch.file) else { return false };
            let Some(text) = nmsubs::repo::fresh_text(&sub.root, &f.path) else { return false };
            let part: String = text
                .lines()
                .skip(ch.start.saturating_sub(1))
                .take(ch.end.saturating_sub(ch.start) + 1)
                .collect::<Vec<_>>()
                .join("\n")
                .to_lowercase();
            verified_text(&terms, &part)
        })
        .collect()
}

/// Agent use (2026-11-13; research file Solution 6, L1 signal): when a later tool call's arguments
/// (`text`) name a file, or an item defined in a chunk, that one of the last `USE_RECENT` whole-path
/// decisions chose (within `USE_WINDOW_MS`), an agent-use line is written to the ledger for it.
/// Returns how many lines were written.
pub fn note_use(org: &crate::forage::Organism, text: &str, tool: &str) -> usize {
    if !org.memory_on {
        return 0;
    }
    let Ok(dir) = org.memory.lock().map(|m| m.dir.clone()) else { return 0 };
    let path = ledger_path(&org.home, &dir);
    let Ok(ds) = nmlearn::ledger::load(&path) else { return 0 };
    let sub = org.sub();
    let now = nmtype::now_ms();
    let mut written = 0;
    for d in ds.iter().rev().take(USE_RECENT).filter(|d| now.saturating_sub(d.at) <= USE_WINDOW_MS && d.agent_used.is_none()) {
        let (Some(ep), Some(ch)) = (d.episode.as_ref(), sub.chunks.get(d.chosen)) else { continue };
        let file = sub.files.get(ch.file).map_or(String::new(), |f| f.path.replace('\\', "/"));
        let names: Vec<String> = ch.text.lines().filter_map(nmmem::migrate::defined_name).filter(|n| n.len() > 3).collect();
        let hit = (!file.is_empty() && text.replace('\\', "/").contains(&file)) || names.iter().any(|n| text.contains(n.as_str()));
        if hit && nmlearn::ledger::append_use(&path, ep, &format!("{file}:{}", ch.start), tool, now).is_ok() {
            written += 1;
        }
    }
    written
}

/// Decisions agent use is matched against, and how long after them.
pub const USE_RECENT: usize = 8;
pub const USE_WINDOW_MS: u64 = 30 * 60 * 1000;

/// Where the decision ledger of a session lives (`nmlearn::ledger`): `units/decisions.jsonl`
/// for the live memory folder; beside the memory folder of a test session otherwise.
#[must_use]
pub fn ledger_path(home: &Path, memory_dir: &Path) -> PathBuf {
    if memory_dir == home.join("memory") {
        home.join("units").join("decisions.jsonl")
    } else if memory_dir.parent() == Some(nmtype::project::projects_dir(home).as_path()) {
        // A project memory (Nano, 2026-11-15, E1) keeps its ledger inside its own folder.
        memory_dir.join("decisions.jsonl")
    } else {
        let name = memory_dir.file_name().and_then(|n| n.to_str()).unwrap_or("session");
        memory_dir.with_file_name(format!("{name}-decisions.jsonl"))
    }
}

/// Approximate tokens of a packet's text (the CX1 and LC1 count: characters / 4 + 8).
#[must_use]
pub fn tokens_of(text: &str) -> u32 {
    u32::try_from(text.len() / 4 + 8).unwrap_or(u32::MAX)
}

/// The Core Brain's order as `chunk` packets, first place first.
#[must_use]
pub fn packets(sub: &Substrate, ordered: &[Piece], recalled: &[usize]) -> Vec<Packet> {
    let mut out = Vec::new();
    for (k, p) in ordered.iter().enumerate() {
        let src = format!("{}:{}", p.path, p.start);
        #[expect(clippy::cast_precision_loss, reason = "places are small")]
        let conf = (TOP - STEP * k as f32).max(FLOOR);
        out.push(
            Packet::new(
                "chunk",
                &format!("{KEY}{k:02}:{src}"),
                &p.text,
                Provenance::new(
                    &src,
                    if recalled.contains(&p.chunk) { "nmapi.whole#core-brain/recall" } else { "nmapi.whole#core-brain" },
                    sub.version,
                    nmtype::now_ms(),
                ),
            )
            .status(Status::Observed)
            .stance(Stance::Objective)
            .category(Category::Fact)
            .confidence(conf)
            .evidence({
                // Lane B's hook B (`NANO_WRITER_PACK=on`): the writer's summary beside the cited
                // source, never in its place.
                let mut ev = vec![format!("{src} lines {}..{}", p.start, p.end)];
                if crate::writer_idle::pack_summaries()
                    && let Some(s) = sub.chunks.get(p.chunk).and_then(|c| crate::writer_idle::summary(&c.key))
                {
                    ev.push(format!("summary: {s}"));
                }
                ev
            }),
        );
    }
    out
}

/// A question asking for the callers of something (2026-11-13; a rule on its words, as
/// `nmbrain::objectives::classify`): who or which code calls it, its caller, where it is
/// started, invoked or called from.
/// The rule is the request drone's (`nmpod::intake::asks_callers`).
#[must_use]
pub fn asks_callers(question: &str) -> bool {
    nmpod::intake::asks_callers(question)
}

/// Source spans read from the substrate, exactly as stored (2026-11-13, internal communication
/// format: the `nm-msg/1` envelope turns back into the source text through this resolver).
pub struct SubResolver<'a>(pub &'a Substrate);
impl nmtype::message::Resolver for SubResolver<'_> {
    fn span(&self, path: &str, start: u32, end: u32) -> Result<Option<String>, Fault> {
        let Some(c) = self.0.chunks.iter().find(|c| {
            self.0.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == path)
                && c.start == start as usize
                && c.end == end as usize
        }) else {
            return Ok(None);
        };
        Ok(Some(c.text.clone()))
    }
    fn tokens(&self, _: &[u32]) -> Result<Option<String>, Fault> {
        Ok(None)
    }
}

/// G2's hand-over to the Core Brain as an in-house message (2026-11-13): a report from
/// `nmbrain.g2` to `nmbrain.core-brain` whose references are the source spans of the pieces sent
/// up, in order; the pieces travel by reference and read back exactly through `SubResolver`.
/// # Errors
/// The envelope's own refusals.
pub fn handover(sub: &Substrate, chunks: &[usize]) -> Result<nmtype::Envelope, Fault> {
    let refs: Vec<nmtype::Reference> = chunks
        .iter()
        .filter_map(|c| sub.chunks.get(*c).and_then(|ch| sub.files.get(ch.file).map(|f| (f, ch))))
        .map(|(f, ch)| nmtype::Reference::Span {
            path: f.path.replace('\\', "/"),
            start: u32::try_from(ch.start).unwrap_or(u32::MAX),
            end: u32::try_from(ch.end).unwrap_or(u32::MAX),
        })
        .collect();
    nmtype::Envelope::new(
        nmtype::MessageKind::Report,
        "nmbrain.g2",
        "nmbrain.core-brain",
        1.0,
        &format!("hand-over of {} pieces", refs.len()),
        refs,
        Vec::new(),
    )
}

/// The route measure (`run msg route`, 2026-11-13): for each question, G2's hand-over as a
/// message; whether its bare and resolved text forms parse back to the same envelope, whether the
/// resolved spans equal the pieces' text, and the tokens of each form against the pieces' text.
/// # Errors
/// Stack or tokenizer failures.
pub fn route_report(sub: &Substrate, keep: &Keep, tok: &nmtokz::Tokenizer, questions: &[String]) -> Result<String, Fault> {
    let (mut exact, mut parsed, mut n) = (0_usize, 0_usize, 0_usize);
    let (mut t_bare, mut t_full, mut t_text) = (0_usize, 0_usize, 0_usize);
    let count = |s: &str| tok.encode(s).map(|v| v.len()).map_err(|e| bad(e.to_string()));
    for q in questions {
        let t = drone_path::stack_run_g2(
            sub,
            keep,
            q,
            None,
            true,
            true,
            &[],
            &std::collections::BTreeMap::new(),
            &drone_path::G2In::default(),
        )?;
        let env = handover(sub, &t.up)?;
        let bare = env.render_bare();
        let full = env.render(&SubResolver(sub))?;
        parsed += usize::from(nmtype::Envelope::parse(&bare).as_ref() == Ok(&env) && nmtype::Envelope::parse(&full).as_ref() == Ok(&env));
        let texts: Vec<String> = t.up.iter().filter_map(|c| sub.chunks.get(*c).map(|ch| ch.text.clone())).collect();
        let resolved: Vec<String> = env
            .refs()
            .iter()
            .filter_map(|r| match r {
                nmtype::Reference::Span { path, start, end } => {
                    nmtype::message::Resolver::span(&SubResolver(sub), path, *start, *end).ok().flatten()
                }
                _ => None,
            })
            .collect();
        exact += usize::from(resolved == texts);
        t_bare += count(&bare)?;
        t_full += count(&full)?;
        t_text += count(&texts.join("\n"))?;
        n += 1;
    }
    let m = n.max(1);
    Ok(format!(
        "Internal communication, route G2 to the Core Brain (hand-over as nm-msg/1 report), {n} questions\nparse(render) equal to the envelope, bare and resolved: {parsed}/{n}\nresolved spans equal to the pieces' text: {exact}/{n}\ntokens per hand-over: by reference (bare) {}, resolved {}, the pieces' text alone {}\n",
        t_bare / m,
        t_full / m,
        t_text / m
    ))
}

/// The `k` chunks of `first`'s file nearest to it (by line distance), not in `held`
/// (2026-11-13, a wider within-file reach for questions whose answer sits further away in the file).
#[must_use]
pub fn file_reach(sub: &Substrate, first: usize, held: &[usize], k: usize) -> Vec<usize> {
    let Some(fc) = sub.chunks.get(first) else { return Vec::new() };
    let mut near: Vec<(usize, usize)> = sub
        .chunks
        .iter()
        .enumerate()
        .filter(|(i, c)| c.file == fc.file && *i != first && !held.contains(i))
        .map(|(i, c)| (if c.start > fc.end { c.start - fc.end } else { fc.start.saturating_sub(c.end) }, i))
        .collect();
    near.sort_unstable();
    near.into_iter().take(k).map(|(_, i)| i).collect()
}

/// The arrangement of the whole path for one question (live anatomy, 2026-11-13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arrangement {
    pub reach: bool,
    pub uses: bool,
    pub look: bool,
    /// Chunks of the first place's file the reach takes, nearest first (2 = the two neighbours).
    pub reach_k: usize,
}

/// The question's kind by rule (2026-11-13, as `nmbrain::objectives::classify` does for objectives):
/// `relation` (asks for callers), `value` (a default, limit, threshold or count), `explain` (why, or
/// how something works), `name` (where something is defined, or which function or type does
/// something), else `behaviour`.
/// The rule is the request drone's (`nmpod::intake::wording_kind`).
#[must_use]
pub fn kind_of(question: &str) -> &'static str {
    nmpod::intake::wording_kind(question)
}

/// The arrangements of the `whole_path` section of a compositions table.
/// # Errors
/// A malformed section.
pub fn arrangements(table: &nmjson::Value) -> Result<std::collections::BTreeMap<String, Arrangement>, Fault> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(sec) = table.get("whole_path") else { return Ok(out) };
    let rows = sec.get("kinds").and_then(nmjson::Value::object).map_err(|e| bad(format!("whole_path.kinds: {e}")))?;
    for (k, v) in rows {
        let flag = |f: &str| match v.get(f) {
            Ok(nmjson::Value::Bool(b)) => Ok(*b),
            _ => Err(bad(format!("whole_path.kinds.{k}.{f}: a boolean"))),
        };
        let reach_k = v.get("reach_k").and_then(nmjson::Value::integer).ok().and_then(|n| usize::try_from(n).ok()).unwrap_or(2);
        out.insert(k.clone(), Arrangement { reach: flag("reach")?, uses: flag("uses")?, look: flag("look")?, reach_k });
    }
    Ok(out)
}

/// Use sites read per question at most (2026-11-13).
pub const USES_K: usize = 4;

/// Use sites of the first place (2026-11-13): the functions the `first` chunk defines whose name
/// shares a stem (five letters) with a question word, then up to `k` chunks not in `held` that
/// call one (`name(` or `.name(`, not its definition), `.rs` files first, then in tree order.
#[must_use]
pub fn use_sites(sub: &Substrate, question: &str, first: usize, held: &[usize], k: usize) -> Vec<usize> {
    let Some(fc) = sub.chunks.get(first) else { return Vec::new() };
    let words: Vec<String> =
        question.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 5).map(str::to_lowercase).collect();
    let stem = |a: &str, b: &str| a.chars().count() >= 5 && b.chars().count() >= 5 && a.chars().take(5).eq(b.chars().take(5));
    // 2026-11-14: names defined anywhere in the first place's file (its first chunk is often the
    // module's header, with the definition in the next).
    let mut names: Vec<String> = sub
        .chunks
        .iter()
        .filter(|c| c.file == fc.file)
        .flat_map(|c| {
            c.low
                .match_indices("fn ")
                .filter_map(|(i, _)| c.low.get(i + 3..))
                .map(|r| r.chars().take_while(|ch| ch.is_alphanumeric() || *ch == '_').collect::<String>())
                .collect::<Vec<_>>()
        })
        .filter(|n| n.len() >= 3 && n.split('_').any(|part| words.iter().any(|w| stem(part, w))))
        .collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Vec::new();
    }
    // The defining file's module name and crate folder: a caller in another file of the same
    // crate, naming the module, is the likeliest edge; a same-named method elsewhere the least.
    // Outside Rust the module is the file name without its extension, named by an import path
    // (`./module`) or a qualified call (`module.f`), and callers in the same language rank first.
    let path = |c: &nmsubs::Chunk| sub.files.get(c.file).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    let home = path(fc);
    let rust = home.ends_with(".rs");
    let calls = |low: &str, n: &str| {
        let call = format!("{n}(");
        low.match_indices(&call).any(|(i, _)| {
            let before = low.get(..i).unwrap_or("");
            let prev = before.chars().last();
            prev.is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
                && !before.ends_with("fn ")
                && (rust || !(before.ends_with("def ") || before.ends_with("function ")))
        })
    };
    let file_name = home.rsplit('/').next().unwrap_or("");
    let module = if rust { file_name.strip_suffix(".rs").unwrap_or("") } else { file_name.split('.').next().unwrap_or("") }.to_lowercase();
    let ext = file_name.rsplit_once('.').map_or("", |(_, e)| e).to_owned();
    let names_module = |low: &str| {
        if rust {
            low.contains(&format!("{module}::"))
        } else {
            low.contains(&format!("{module}.")) || low.contains(&format!("/{module}\"")) || low.contains(&format!("/{module}'"))
        }
    };
    let krate = |p: &str| p.split("/src/").next().unwrap_or("").to_owned();
    let mut found: Vec<((bool, bool, bool, bool), usize)> = sub
        .chunks
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != first && !held.contains(i))
        .filter(|(_, c)| names.iter().any(|n| calls(&c.low, n)))
        .map(|(i, c)| {
            let p = path(c);
            ((p != home, krate(&p) == krate(&home), !module.is_empty() && names_module(&c.low), p.ends_with(&format!(".{ext}"))), i)
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    found.into_iter().take(k).map(|x| x.1).collect()
}

/// Just above `cap` in the ordering key (2026-11-13).
#[must_use]
pub fn above(cap: f32) -> f32 {
    cap + f32::EPSILON.max(cap.abs() * 1e-6)
}

/// A within-file neighbour's score, kept strictly below the first place's `cap` (2026-11-13).
#[must_use]
pub fn below(x: f32, cap: f32) -> f32 {
    if x < cap { x } else { cap - f32::EPSILON.max(cap.abs() * 1e-6) }
}

/// The decoy cut (2026-11-13), pre-registered from the CX1 design set before CX2 was read: the
/// log-odds midpoint of the highest decoy best score (0.930) and the lowest real best score
/// (0.997) there. On CX2 it withholds 4 of 6 decoys and costs 2 right answers (served answers'
/// precision 0.65 against 0.59). A stated trade, not a free gain.
pub const WITHHOLD_BELOW: f32 = 0.985;
/// Hard withhold cut on the reranker's best P(yes): non-empty packs scoring strictly below this
/// (or with no finite reranker score) are withheld rather than served as low confidence.
pub const HARD_WITHHOLD_BELOW: f32 = 0.05;

/// Mark `pack` low confidence (`floor_met` false) when the best of `rerank` is below `cut`, and
/// withhold it when empty, unscored, or strictly below `cut.min(HARD_WITHHOLD_BELOW)`.
#[must_use]
pub fn withhold_weak(mut pack: Pack, rerank: &[f32], cut: Option<f32>) -> Pack {
    let Some(cut) = cut else { return pack };
    let best = rerank.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if pack.withheld || best >= cut {
        return pack;
    }
    let hard = cut.min(HARD_WITHHOLD_BELOW);
    if pack.packets.is_empty() || !best.is_finite() || best < hard {
        let limit = if !pack.packets.is_empty() && best.is_finite() { hard } else { cut };
        let why = format!("no chunk the reranker judges relevant enough (best P(yes) {:.3} below {limit:.3})", best.max(0.0));
        pack.withheld = true;
        pack.packets.clear();
        pack.tokens = 0;
        pack.floor_met = false;
        pack.reason = format!("withheld: {why}");
    } else {
        let why = format!("no chunk the reranker judges relevant enough (best P(yes) {:.3} below {cut:.3})", best.max(0.0));
        pack.floor_met = false;
        pack.reason = format!("low confidence: {why}; read the cited lines before relying on them");
    }
    pack
}

/// Whether the one flow fills the room after G2's set even when the set holds every part of the
/// request (`NANO_FLOW_FILL=on`, the first form; off by default).
fn flow_fill() -> bool {
    std::env::var("NANO_FLOW_FILL").is_ok_and(|v| v == "on")
}

/// The pieces at the head of a low pack that the short pack and strips never cut (the bench's top
/// three; stated).
const KEEP_WHOLE: usize = 3;

/// Whether a low-confidence flow pack serves strips of its pieces (`NANO_FLOW_STRIPS=on`; off by
/// default).
fn flow_strips() -> bool {
    std::env::var("NANO_FLOW_STRIPS").is_ok_and(|v| v == "on")
}

/// Lines kept either side of a strip's deciding line outside a definition (stated, as the strip's
/// first form).
const STRIP_CONTEXT: usize = 2;

/// A gap of at most this many lines between two kept runs is kept too: one run is cheaper to read
/// and cite than two (stated).
const STRIP_GAP: usize = 2;

/// A line the lexical head rates at least this share of the piece's best line is a deciding line
/// (stated: the best line and any near it).
const STRIP_LEX_SHARE: f32 = 0.5;

/// Roadmap Phase 4 (2026-11-24): the strips of piece `p` as runs of the file's own lines, each a
/// piece cited by its own first and last line (the pack's verbatim check then gives each its place),
/// in file order. Over the piece's own lines (an answer span may reach past its chunk; a rival's
/// outline line reads its chunk's lines), a line is kept when it decides the answer:
/// - it holds a stem of the request (`stems`): with the whole of the smallest definition it sits in
///   (its comment and attribute lines above it), or two lines either side outside any definition;
/// - the lexical head rates it highest (the rarity of the question's key terms it holds, `terms`,
///   as `judge_view` scores lines): at least half the piece's best line, with two lines either side;
/// - it opens a definition whose name holds a stem of the request (the definition line), with its
///   comment lines above it and two lines below.
///
/// Every deciding line also keeps the first line of the definition it sits in (the section's first
/// line for a document), even above the piece. Gaps of two lines or fewer are kept. `None` when no
/// line decides (the piece stays whole) or when every line of the piece is kept.
fn strip_runs(sub: &Substrate, p: &Piece, stems: &[&str], terms: &[(String, f32)]) -> Option<Vec<Piece>> {
    use std::collections::BTreeSet;
    let ch = sub.chunks.get(p.chunk)?;
    let f = sub.files.get(ch.file)?;
    let lines: Vec<&str> = f.text.lines().collect();
    let (lo, hi) = if p.end > p.start { (p.start, p.end) } else { (ch.start, ch.end) };
    let (lo, hi) = (lo.max(1), hi.min(lines.len()));
    if lo >= hi {
        return None;
    }
    let line = |n: usize| lines.get(n.wrapping_sub(1)).copied().unwrap_or("");
    let stem_of = |w: &str| w.split(|x: char| !(x.is_alphanumeric() || x == '_')).filter(|w| !w.is_empty()).flat_map(nmlex::name_parts).any(|q| stems.contains(&nmlex::stem(&q).as_str()));
    let comment = |l: &str| {
        let t = l.trim_start();
        t.starts_with("///") || t.starts_with("//!") || t.starts_with("#[") || t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") || t.starts_with('@')
    };
    // The first comment or attribute line of the run right above line `n`, not above `floor`.
    let doc_top = |n: usize, floor: usize| {
        let mut top = n;
        while top > floor.max(1) && comment(line(top - 1)) {
            top -= 1;
        }
        top
    };
    let defs: Vec<&nmsubs::repo::Symbol> = sub.symbols.iter().filter(|s| s.file == f.path && s.line <= hi && s.end >= lo).collect();
    let inner = |n: usize| defs.iter().filter(|s| s.line <= n && n <= s.end).min_by_key(|s| s.end - s.line).copied();
    // The first line of what line `n` sits in: its definition's for code, its section's for a document.
    let header = |n: usize| if f.is_code() { inner(n).map(|d| d.line) } else { ch.section.map(|(s, _)| s).filter(|s| *s <= n) };
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    let around = |keep: &mut BTreeSet<usize>, n: usize| keep.extend(n.saturating_sub(STRIP_CONTEXT).max(lo)..=(n + STRIP_CONTEXT).min(hi));
    let score = |n: usize| -> f32 {
        let low = line(n).to_lowercase();
        terms.iter().filter(|(t, _)| low.contains(t.as_str())).map(|(_, x)| *x).sum()
    };
    let best = (lo..=hi).map(score).fold(0.0_f32, f32::max);
    for n in lo..=hi {
        let by_stem = stem_of(line(n));
        let by_lex = best > 0.0 && score(n) >= best * STRIP_LEX_SHARE;
        if !by_stem && !by_lex {
            continue;
        }
        if by_stem {
            match inner(n).filter(|_| f.is_code()) {
                Some(d) => keep.extend(doc_top(d.line, lo).max(lo)..=d.end.min(hi)),
                None => around(&mut keep, n),
            }
        }
        if by_lex {
            around(&mut keep, n);
        }
        if let Some(h) = header(n) {
            keep.insert(h);
        }
    }
    // The definition lines of the request's names.
    for d in defs.iter().filter(|d| (lo..=hi).contains(&d.line) && stem_of(&d.name)) {
        keep.extend(doc_top(d.line, lo)..=(d.line + STRIP_CONTEXT).min(d.end).min(hi));
    }
    if keep.is_empty() {
        return None;
    }
    // Short gaps are kept.
    let kept: Vec<usize> = keep.iter().copied().collect();
    for w in kept.windows(2) {
        if let [a, b] = *w
            && b > a + 1
            && b - a - 1 <= STRIP_GAP
        {
            keep.extend(a + 1..b);
        }
    }
    if (lo..=hi).all(|n| keep.contains(&n)) {
        return None;
    }
    // Runs of consecutive lines, each a piece of the file's own lines (no newline after the last).
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for n in keep {
        match runs.last_mut() {
            Some(r) if n == r.1 + 1 => r.1 = n,
            _ => runs.push((n, n)),
        }
    }
    Some(
        runs.into_iter()
            .map(|(a, b)| Piece { start: a, end: b, text: (a..=b).map(line).collect::<Vec<&str>>().join("\n"), ..p.clone() })
            .collect(),
    )
}

/// Whether the one flow's pack status takes three states from two cuts on its fitted confidence:
/// on by default since 2026-11-19 with the judge's sufficiency read (W7; `NANO_FLOW_STATES=off`
/// restores two states). Low packs the judge reads as not answering serve strips.
fn flow_states() -> bool {
    !std::env::var("NANO_FLOW_STATES").is_ok_and(|v| v == "off")
}


/// The flow's confidence fit (research item 4: sufficient context's selective generation, CRAG's
/// two thresholds): a logistic regression over the pack's signals (`drone_path::PackSignals`),
/// and two cuts on its value, withhold below the first, settled at or above the second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowFit {
    pub bias: f32,
    pub weights: [f32; 4],
    pub withhold: f32,
    pub settle: f32,
}

impl FlowFit {
    /// The fitted chance that the pack holds the answer.
    #[must_use]
    pub fn score(&self, s: &crate::drone_path::PackSignals) -> f32 {
        let z = self.bias + self.weights.iter().zip(s.row()).map(|(w, x)| w * x).sum::<f32>();
        1.0 / (1.0 + (-z).exp())
    }
    /// `bias,w_held,w_margin,w_lane,w_defines,withhold,settle`.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
        let [bias, a, b, c, d, withhold, settle] = v.as_slice() else { return None };
        (withhold <= settle).then_some(Self { bias: *bias, weights: [*a, *b, *c, *d], withhold: *withhold, settle: *settle })
    }
}

/// The fit served until the tuning folds give one (provisional, not fitted): the share of parts
/// held carries it; a clear margin, the lane's agreement and a definition site add to it. With
/// every part held and nothing else it reads 0.62 (low); a pack holding no part reads under 0.12.
pub const FLOW_FIT: FlowFit = FlowFit { bias: -2.0, weights: [2.5, 0.5, 1.0, 1.0], withhold: 0.12, settle: 0.7 };

/// The fit in force: `NANO_FLOW_FIT` when it parses, else `FLOW_FIT`.
fn flow_fit() -> FlowFit {
    std::env::var("NANO_FLOW_FIT").ok().and_then(|v| FlowFit::parse(&v)).unwrap_or(FLOW_FIT)
}

/// Whether the one flow's pack stops where its pieces hold every part of the request
/// (`NANO_FLOW_COVER=on`; off by default).
fn flow_cover() -> bool {
    std::env::var("NANO_FLOW_COVER").is_ok_and(|v| v == "on")
}

/// Roadmap Phase 4 (2026-11-24): the judge's reads G2 kept, by piece (label log-odds less the
/// content-free mean; above 0 the piece reads as answering), for the cover stop. The edge judge's
/// reads of each piece in its duels (the last read of a piece stands), then the sufficiency read
/// (`PackSignals::sufficient`), which is of the set's first piece: the first piece sent up (`up`)
/// that is in G2's set. A piece the judge did not read is not in the map.
fn judge_reads(g2: &drone_path::G2Run, up: &[usize]) -> std::collections::BTreeMap<usize, f32> {
    let mut reads: std::collections::BTreeMap<usize, f32> = g2.judge_pairs.iter().filter(|(_, _, z)| z.is_finite()).map(|(c, _, z)| (*c, *z)).collect();
    if let Some(z) = g2.signals.sufficient.or(g2.sufficient).filter(|z| z.is_finite())
        && let Some(lead) = up.iter().copied().find(|c| g2.set.contains(c)).or_else(|| g2.set.first().copied())
    {
        reads.insert(lead, z);
    }
    reads
}

/// The cover stop (`NANO_FLOW_COVER=on`): `order` up to and with the first piece at which every one
/// of the `parts` is held, a part `k` held by piece `c` when `holds(k, c)`; the whole order when the
/// parts are never all held.
fn cover_order(order: Vec<usize>, parts: usize, holds: &dyn Fn(usize, usize) -> bool) -> Vec<usize> {
    let mut held = vec![false; parts];
    let mut out = Vec::new();
    for c in order {
        if held.iter().all(|h| *h) {
            break;
        }
        for (k, h) in held.iter_mut().enumerate() {
            *h |= holds(k, c);
        }
        out.push(c);
    }
    out
}

/// The status of a pack the stack wrote (`drone_path::write_pack`): the word after `status` on its
/// head line, and the parts its `confidence: low; not held:` line names.
fn stack_status(pack: &str) -> (String, Vec<String>) {
    let status = pack
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().skip_while(|w| *w != "status").nth(1))
        .unwrap_or("answered")
        .to_owned();
    let open = pack
        .lines()
        .find_map(|l| l.strip_prefix("confidence: low; not held: "))
        .map(|r| r.split("; ").map(str::to_owned).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    (status, open)
}

/// The one flow's pack carries the stack's status: withheld (no piece holds a part of the request:
/// nothing is served) or low confidence naming the parts not held; any other mode's pack as given.
#[must_use]
pub fn flow_status(mut pack: Pack, run: &Run) -> Pack {
    let Some((status, open)) = run.flow_status.as_ref() else { return pack };
    // The one flow's pack names its own path: what G2 sent up, not the whole path's Core Brain.
    pack.reason = pack.reason.replacen("whole path", "the one flow: G2's set, then the pieces that hold a part it leaves open", 1);
    match status.as_str() {
        "withheld" => {
            pack.packets.clear();
            pack.tokens = 0;
            pack.floor_met = false;
            pack.withheld = true;
            pack.reason = match run.flow_score {
                Some((s, lo, _)) if s < lo => format!("withheld: the pack's confidence {s:.2} is below the withhold cut {lo:.2}"),
                _ => "withheld: no piece holds a part of the request".to_owned(),
            };
        }
        "low" if !pack.withheld => {
            pack.floor_met = false;
            let why = if open.is_empty() {
                run.flow_score.map_or_else(|| "the pack is not settled".to_owned(), |(s, _, hi)| format!("the pack's confidence {s:.2} is below the settled cut {hi:.2}"))
            } else {
                format!("no piece of the answer holds {}", open.join("; "))
            };
            let strips = if run.flow_score.is_some() || flow_strips() { "; each piece shows the lines that hold a part, the whole piece is at its place" } else { "" };
            pack.reason = format!("low confidence: {why}{strips}; read the cited lines before relying on them");
        }
        _ => {}
    }
    pack
}

/// The pack: packets in the given order while they fit `budget` tokens. Nothing is cut inside a
/// packet; what does not fit is named in the reason as references (`path:line`).
#[must_use]
pub fn pack(task: u64, all: Vec<Packet>, budget: u32) -> Pack {
    let (mut used, mut kept, mut left) = (0_u32, Vec::new(), Vec::new());
    for p in all {
        let t = tokens_of(p.text());
        if kept.is_empty() || used + t <= budget {
            used += t;
            kept.push(p);
        } else {
            left.push(p.key().to_owned());
        }
    }
    let reason = if left.is_empty() { "whole path".to_owned() } else { format!("whole path; queued by reference: {}", left.join(", ")) };
    Pack { task, packets: kept, withheld: false, reason, tokens: used, floor_met: true }
}

/// Section check (processor only): the pack keeps order, always holds the first packet, and names
/// what did not fit instead of cutting it.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    // The flow's three states: on by default (W7, 2026-11-19); the fit parses only with its seven
    // values in order of cuts; a pack holding every part reads low under the provisional fit, none
    // withheld.
    let all = crate::drone_path::PackSignals { held: 1.0, ..Default::default() };
    let none = crate::drone_path::PackSignals::default();
    if (std::env::var("NANO_FLOW_STATES").is_err() && !flow_states())

        || FlowFit::parse("1,2,3,4,5,0.2,0.1").is_some()
        || FlowFit::parse("-1,2,0,0,0,0.1,0.9").map(|f| f.weights) != Some([2.0, 0.0, 0.0, 0.0])
        || !(FLOW_FIT.withhold..FLOW_FIT.settle).contains(&FLOW_FIT.score(&all))
        || FLOW_FIT.score(&none) >= FLOW_FIT.withhold
    {
        return Err("nmapi.whole: flow states".into());
    }
    // The judge's window: whole lines around the question's evidence, within the token limit; a
    // short piece whole; no evidence gives the first lines.
    let text: String = (1..=12).map(|i| if i == 8 { "let ring = Ring::new();\n".to_owned() } else { format!("line {i}\n") }).collect();
    let piece = Piece { chunk: 0, path: "a.rs".into(), start: 101, end: 112, text, ..Piece::default() };
    let one_each = |_: &str| 1;
    let terms = vec![("ring".to_owned(), 2.0_f32)];
    let (v, n) = judge_view(&piece, &terms, 5, &one_each);
    if (v.start, v.end, n) != (106, 110, 5) || !v.text.contains("Ring::new") || v.text.lines().count() != 5 {
        return Err(format!("nmapi.whole: judge window {}-{} of {n} tokens: {:?}", v.start, v.end, v.text));
    }
    let (whole, _) = judge_view(&piece, &terms, 12, &one_each);
    let (plain, _) = judge_view(&piece, &[], 3, &one_each);
    if whole != piece || (plain.start, plain.end) != (101, 103) {
        return Err("nmapi.whole: judge window for a short piece or without evidence".into());
    }
    let p = |k: &str, n: usize| Packet::new("chunk", k, &"x".repeat(n), Provenance::new("a:1", "t", 1, 1));
    let pk = pack(7, vec![p("a", 400), p("b", 400), p("c", 4000)], 300);
    if pk.packets.iter().map(|x| x.key().to_owned()).collect::<Vec<_>>() != vec!["a".to_owned(), "b".to_owned()]
        || !pk.reason.contains("c")
        || pk.tokens != 216
    {
        return Err(format!("nmapi.whole: pack {:?} {} {}", pk.packets.iter().map(Packet::key).collect::<Vec<_>>(), pk.reason, pk.tokens));
    }
    let one = pack(7, vec![p("big", 8000)], 10);
    let low = withhold_weak(pack(7, vec![p("a", 10)], 300), &[0.2, 0.9], Some(WITHHOLD_BELOW));
    if low.withheld || low.floor_met || low.packets.len() != 1 {
        return Err("whole.withhold: a pack whose best score is below the cut must be served marked low".into());
    }
    if !withhold_weak(pack(7, Vec::new(), 300), &[0.2], Some(WITHHOLD_BELOW)).withheld {
        return Err("whole.withhold: an empty pack below the cut must be withheld".into());
    }
    let hard = withhold_weak(pack(7, vec![p("a", 10)], 300), &[0.01, 0.04], Some(WITHHOLD_BELOW));
    if !hard.withheld || !hard.packets.is_empty() || hard.tokens != 0 || hard.floor_met {
        return Err("whole.withhold: a pack below HARD_WITHHOLD_BELOW must be withheld".into());
    }
    let unscored = withhold_weak(pack(7, vec![p("a", 10)], 300), &[], Some(WITHHOLD_BELOW));
    if !unscored.withheld || !unscored.packets.is_empty() || unscored.tokens != 0 || unscored.floor_met {
        return Err("whole.withhold: an unscored pack must be withheld".into());
    }
    let borderline = withhold_weak(pack(7, vec![p("a", 10)], 300), &[0.104], Some(WITHHOLD_BELOW));
    if borderline.withheld || borderline.floor_met || borderline.packets.len() != 1 {
        return Err("whole.withhold: a borderline in-repo pack at 0.104 must remain low confidence".into());
    }
    if withhold_weak(pack(7, vec![p("a", 10)], 300), &[0.2, 0.99], Some(WITHHOLD_BELOW)).withheld
        || withhold_weak(pack(7, vec![p("a", 10)], 300), &[0.1], None).withheld
    {
        return Err("whole.withhold: a pack at or above the cut, or with the rule off, must be served".into());
    }
    if kind_of("Which code calls the parser?") != "relation"
        || kind_of("What is the default timeout of a build?") != "value"
        || kind_of("Why does the gate refuse a write?") != "explain"
        || kind_of("Where is fn parse defined?") != "name"
        || kind_of("What happens when the queue is full?") != "behaviour"
    {
        return Err("whole.anatomy: question kinds".into());
    }
    if !asks_callers("Which code calls the parser?")
        || asks_callers("What does the parser return?")
        || asks_callers("Which function recalls episodes?")
    {
        return Err("whole.uses: caller questions".into());
    }
    if above(0.99999) <= 0.99999 || above(1.0) <= 1.0 {
        return Err("whole.uses: a caller must order strictly above the first place".into());
    }
    if below(0.2, 0.9) != 0.2 || below(0.99, 0.9) >= 0.9 || below(1.0, 1.0) >= 1.0 {
        return Err("whole.reach: a neighbour must rank strictly below the first place".into());
    }
    if one.packets.len() != 1 {
        return Err("nmapi.whole: the first packet must always be kept".into());
    }
    if Mode::parse("fast") != Ok(Mode::Fast) || Mode::parse("full") != Ok(Mode::Full) || Mode::parse("flow") != Ok(Mode::Flow) || Mode::parse("quick").is_ok() {
        return Err("whole.mode: fast, full and flow parse, nothing else does".into());
    }
    if (top_gap(&[0.5, 0.9, 0.5]) - log_odds(0.9)).abs() > 1e-4
        || top_gap(&[0.9]).is_finite()
        || rank_order(&[0.2, 0.7, 0.7]) != vec![1, 2, 0]
    {
        return Err("whole.look_gate: top-two gap and order".into());
    }
    // Re-read verification: the chunk holding the question's rare terms is verified, an unrelated
    // one is not; after the file changes on disk, the fresh read decides, not the index.
    let root = nmtype::check_dir("nmapi-whole-verify").map_err(|e| e.to_string())?;
    let w = |name: &str, body: &str| std::fs::write(root.join(name), body).map_err(|e| e.to_string());
    w("switch.rs", "pub fn tide_level_at(hour: u64) -> u64 {\n    hour * 2\n}\n")?;
    w("other.rs", "pub fn tokens_left(n: u64) -> u64 {\n    n - 1\n}\n")?;
    w("third.rs", "pub fn unrelated(n: u64) -> u64 {\n    n + 1\n}\n")?;
    let sub = Substrate::scan(&root, &[], Arc::new(|_: &str| Vec::new()));
    let of = |f: &str| sub.chunks.iter().position(|c| sub.files.get(c.file).is_some_and(|x| x.path.ends_with(f)));
    let (Some(a), Some(b)) = (of("switch.rs"), of("other.rs")) else { return Err("nmapi.whole: verify tree has no chunks".into()) };
    let q = "how is the tide level at an hour computed";
    if verify_chunks(&sub, q, &[a, b]) != vec![true, false] {
        return Err(format!("nmapi.whole: verify gave {:?}", verify_chunks(&sub, q, &[a, b])));
    }
    w("switch.rs", "pub fn renamed(x: u64) -> u64 {\n    x * 2\n}\n")?;
    if verify_chunks(&sub, q, &[a]) != vec![false] {
        return Err("nmapi.whole: verification used the indexed copy, not a fresh read".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_pack(keys: &[&str], budget: u32) -> Pack {
        let ps = keys
            .iter()
            .map(|k| Packet::new("chunk", k, "pub fn sample() {}\n", Provenance::new("a:1", "t", 1, 1)))
            .collect();
        pack(1, ps, budget)
    }

    #[test]
    fn withhold_weak_withholds_critically_below_hard_cut() {
        let p = withhold_weak(mk_pack(&["a", "b"], 4000), &[0.01, 0.04], Some(WITHHOLD_BELOW));
        assert!(p.withheld);
        assert!(p.packets.is_empty());
        assert_eq!(p.tokens, 0);
        assert!(!p.floor_met);
        assert!(p.reason.starts_with("withheld:"));
    }

    #[test]
    fn withhold_weak_withholds_empty_rerank() {
        let p = withhold_weak(mk_pack(&["a"], 4000), &[], Some(WITHHOLD_BELOW));
        assert!(p.withheld);
        assert!(p.packets.is_empty());
        assert_eq!(p.tokens, 0);
        assert!(!p.floor_met);
    }

    #[test]
    fn withhold_weak_preserves_borderline_in_repo_pack() {
        let p = withhold_weak(mk_pack(&["a"], 4000), &[0.016, 0.104], Some(WITHHOLD_BELOW));
        assert!(!p.withheld);
        assert!(!p.floor_met);
        assert_eq!(p.packets.len(), 1);
    }

    #[test]
    fn withhold_weak_respects_custom_low_cut_and_none() {
        let p1 = withhold_weak(mk_pack(&["a"], 4000), &[0.02], Some(0.01));
        assert!(!p1.withheld);
        assert!(p1.floor_met);

        let p2 = withhold_weak(mk_pack(&["a"], 4000), &[0.0], None);
        assert!(!p2.withheld);
        assert!(p2.floor_met);
    }

    #[test]
    fn flow_status_withheld_clears_tokens_and_floor_met() {
        let r = Run { flow_status: Some(("withheld".into(), Vec::new())), ..Run::default() };
        let p = flow_status(mk_pack(&["a"], 4000), &r);
        assert!(p.withheld);
        assert!(p.packets.is_empty());
        assert_eq!(p.tokens, 0);
        assert!(!p.floor_met);
    }

    #[test]
    fn start_intake_does_not_load_unbound_base() {
        let dir = nmtype::check_dir("nmapi-whole-start-intake").unwrap();
        std::fs::write(dir.join("lib.rs"), "pub fn alpha() -> u32 { 1 }\n").unwrap();
        let sub = Substrate::scan(&dir, &[], Arc::new(|_: &str| Vec::new()));
        let bogus_home = dir.join("no-weights-home");
        let w = WholePath::new(&bogus_home);
        if crate::neural::bound_model().is_none() {
            assert!(!w.start_intake(&sub, None));
        }
    }

    /// Roadmap Phase 4 (2026-11-24): a strip is runs of the file's own lines from their cited first
    /// line (so the pack keeps their place), and it keeps the deciding lines: the request's
    /// definition and the line the lexical head rates highest, not the filler between.
    #[test]
    fn strip_runs_keep_deciding_lines_as_file_lines() {
        let dir = nmtype::check_dir("nmapi-whole-strip-runs").unwrap();
        let mut text = String::from("pub fn filler_one(n: u64) -> u64 {\n");
        for (i, v) in ["a", "b", "c", "d", "e", "f"].iter().enumerate() {
            text.push_str(&format!("    let {v} = n + {i};\n"));
        }
        text.push_str("    f\n}\n\npub fn tide_level_at(hour: u64) -> u64 {\n    hour * 2\n}\n\npub fn filler_two(n: u64) -> u64 {\n");
        for (i, v) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().enumerate() {
            text.push_str(&format!("    let {v} = n + {i};\n"));
        }
        text.push_str("    let quillwort = h * 9;\n");
        for (i, v) in ["i", "j", "k", "l", "m"].iter().enumerate() {
            text.push_str(&format!("    let {v} = n + {i};\n"));
        }
        text.push_str("    m\n}\n");
        std::fs::write(dir.join("gauge.rs"), &text).unwrap();
        let sub = Substrate::scan(&dir, &[], Arc::new(|_: &str| Vec::new()));
        let file_lines: Vec<&str> = text.lines().collect();
        let at = |s: &str| file_lines.iter().position(|l| l.contains(s)).unwrap() + 1;
        let (tide, quill, filler) = (at("tide_level_at"), at("quillwort"), at("let c = n + 2"));
        let c = sub.chunks.iter().position(|ch| sub.files.get(ch.file).is_some_and(|f| f.path.ends_with("gauge.rs"))).unwrap();
        let p = Piece { chunk: c, path: "gauge.rs".into(), start: 1, end: file_lines.len(), text: text.clone(), ..Piece::default() };
        let stem = nmlex::stem("tide");
        let runs = strip_runs(&sub, &p, &[stem.as_str()], &[("quillwort".to_owned(), 5.0)]).unwrap();
        assert!(!runs.is_empty());
        for r in &runs {
            let want = file_lines.get(r.start - 1..r.end).unwrap().join("\n");
            assert_eq!(r.text, want, "run {}-{} is not the file's lines", r.start, r.end);
            assert!(!r.text.ends_with('\n') && !r.text.contains("..."));
            assert_eq!(r.chunk, c);
        }
        let shown = |n: usize| runs.iter().any(|r| r.start <= n && n <= r.end);
        assert!(shown(tide), "the definition line of the request's name");
        assert!(shown(tide + 1), "the definition's body");
        assert!(shown(quill), "the line the lexical head rates highest");
        assert!(!shown(filler), "filler is left out");
        // With the symbols read, the deciding line keeps the first line of its definition.
        if sub.symbols.iter().any(|s| s.name == "filler_two") {
            assert!(shown(at("fn filler_two")));
        }
        // A piece with no deciding line stays whole.
        assert!(strip_runs(&sub, &p, &["zzzz"], &[]).is_none());
    }

    /// Roadmap Phase 4 (2026-11-24): the cover stop counts a part as held only by a piece the judge
    /// read as answering; a holder with no read, or read as not answering, does not stop the pack.
    #[test]
    fn cover_counts_only_judged_answer_pieces() {
        let mut g2 = drone_path::G2Run { set: vec![7, 2], judge_pairs: vec![(3, 0.5, 1.2), (4, 0.5, -0.4)], ..drone_path::G2Run::default() };
        g2.signals.sufficient = Some(0.8);
        let up = [2, 1, 3, 4, 5, 7];
        let reads = judge_reads(&g2, &up);
        // The sufficiency read is of the set's first piece sent up (2), not of the set's first (7).
        assert_eq!(reads.get(&2), Some(&0.8));
        assert!(!reads.contains_key(&7) && !reads.contains_key(&1));
        let answers = |c: usize| reads.get(&c).is_some_and(|z| *z > 0.0);
        // Piece 1 holds both parts but has no judge read; 4 holds part 1 but reads as not answering.
        let holds = |k: usize, c: usize| matches!((k, c), (0 | 1, 1) | (0, 2) | (1, 4) | (1, 3));
        let order = up.to_vec();
        assert_eq!(cover_order(order.clone(), 2, &|k, c| holds(k, c)), vec![2, 1]);
        assert_eq!(cover_order(order.clone(), 2, &|k, c| answers(c) && holds(k, c)), vec![2, 1, 3]);
        // No judge read at all: no piece holds a part and the whole order is served.
        let none = judge_reads(&drone_path::G2Run::default(), &up);
        assert_eq!(cover_order(order.clone(), 2, &|k, c| none.contains_key(&c) && holds(k, c)), order);
    }

    #[test]
    fn warm_in_flow_mode_skips_models() {
        let dir = nmtype::check_dir("nmapi-whole-warm-flow").unwrap();
        let mut w = WholePath::new(&dir);
        w.default_mode = Some(Mode::Flow);
        assert_eq!(w.warm().unwrap(), (0.0, 0));
    }
}
