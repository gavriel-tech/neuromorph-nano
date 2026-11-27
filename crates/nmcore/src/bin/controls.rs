//! User controls of the Neuromorph Nano and their audit log (2026-11-15, build plan step 3).
//! Controls live in `settings.json` at the Nano's root, apart from the design files; every change
//! is appended to `memory/audit.jsonl` with its old and new value, and any change can be undone
//! (the undo is itself a logged change). Standard library and nmjson only.
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Bounds and unit of a customizable integer allowance control (`kind: "integer"`).
#[derive(Clone, Copy)]
pub struct IntRange {
    pub min: u64,
    pub max: u64,
    pub step: u64,
    pub unit: &'static str,
}

/// One control: its key, group, the values (or quick-pick presets) it takes, its default, what it
/// does, whether the engine applies it yet (`wired`) and when (`applies`), and optional numeric
/// allowance bounds (`range`).
pub struct Control {
    pub key: &'static str,
    pub group: &'static str,
    pub values: &'static [&'static str],
    pub default: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    pub wired: bool,
    pub applies: &'static str,
    pub range: Option<IntRange>,
}

const WIDTHS: &[&str] = &["default", "16", "8", "4"];

pub const CONTROLS: &[Control] = &[
    Control {
        key: "precision.base",
        group: "Precision",
        values: WIDTHS,
        default: "default",
        label: "Base model precision",
        help: "Qwen3 0.6B, the drones and the judge.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "precision.writer",
        group: "Precision",
        values: WIDTHS,
        default: "default",
        label: "Writer precision",
        help: "Qwen3 1.7B.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "precision.reranker",
        group: "Precision",
        values: WIDTHS,
        default: "default",
        label: "Reranker precision",
        help: "Qwen3 Reranker 0.6B.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "precision.embedder",
        group: "Precision",
        values: WIDTHS,
        default: "default",
        label: "Embedder precision",
        help: "Qwen3 Embedding 0.6B.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "budget.pack_tokens",
        group: "Budgets and allowances",
        values: &["1000", "2000", "4000", "6000", "8000", "12000", "16000"],
        default: "4000",
        label: "Context pack token allowance",
        help: "Default token budget of nm_context_pack when a caller does not specify tokens (1,000 to 16,000 tokens).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 1000, max: 16000, step: 250, unit: "tokens" }),
    },
    Control {
        key: "budget.tentacle_tokens",
        group: "Budgets and allowances",
        values: &["2000", "6000", "12000", "18000", "24000", "32000"],
        default: "12000",
        label: "Tentacle window token allowance",
        help: "Chunk tokens each tentacle window holds when composing its lens territory (2,000 to 32,000 tokens).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 2000, max: 32000, step: 500, unit: "tokens" }),
    },
    Control {
        key: "budget.judge_window",
        group: "Budgets and allowances",
        values: &["128", "256", "384", "512", "768", "1024", "2048"],
        default: "384",
        label: "Judge read window",
        help: "Tokens of each candidate piece the judge reads around the question's evidence in full mode (128 to 2,048 tokens).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 128, max: 2048, step: 64, unit: "tokens" }),
    },
    Control {
        key: "budget.fast_k",
        group: "Budgets and allowances",
        values: &["1", "2", "4", "6", "8", "12", "16"],
        default: "4",
        label: "Fast mode candidate allowance",
        help: "Top candidates from the stack that the reranker judges in fast mode (1 to 16 items).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 1, max: 16, step: 1, unit: "items" }),
    },
    Control {
        key: "budget.max_run_s",
        group: "Budgets and allowances",
        values: &["15", "30", "60", "90", "120", "180"],
        default: "60",
        label: "Run time budget ceiling",
        help: "Maximum wall-clock time allowance in seconds for a single forage or context pack run (5 to 180 seconds).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 5, max: 180, step: 5, unit: "s" }),
    },
    Control {
        key: "budget.gpu_mb_max",
        group: "Budgets and allowances",
        values: &["auto", "2048", "3072", "4096", "8192", "16384", "24576"],
        default: "auto",
        label: "Graphics card memory ceiling",
        help: "Device memory threshold in MiB above which homeostasis narrows neural ordering and suspends idle slots. Auto (default) follows this machine's card: its total memory less the live calls' measured room. A number pins it (1,024 MiB up to 80 GiB).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 1024, max: 81_920, step: 128, unit: "MiB" }),
    },
    Control {
        key: "budget.adaptive_pack",
        group: "Budgets and allowances",
        values: &["on", "off"],
        default: "on",
        label: "Adaptive pack sizing",
        help: "On: when a caller omits an explicit token budget and the top reranked piece dominates with high confidence, the assembled pack is capped to the focused pack token allowance below instead of filling the full pack budget with lower-ranked tail pieces. Off: always fills up to the full context pack token allowance.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "budget.focused_tokens",
        group: "Budgets and allowances",
        values: &["1000", "1500", "1800", "2500", "4000"],
        default: "1800",
        label: "Focused pack token allowance",
        help: "Token ceiling applied when adaptive pack sizing detects a high-confidence focused answer (500 to 8,000 tokens).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 500, max: 8000, step: 250, unit: "tokens" }),
    },
    Control {
        key: "rerank.early_drop",
        group: "Budgets and allowances",
        values: &["off", "balanced", "aggressive"],
        default: "balanced",
        label: "Reranker early candidate drop",
        help: "Mid-layer candidate pruning inside the batched Qwen3 Reranker pass (including close-call judge) when --rerank-drop is not set on the command line. Balanced (default): at layer 18, drops candidates trailing the leader by more than 4.0 log-odds while keeping at least the top 5 through all layers. Aggressive: layer 16, margin 3.0, keep 3. Off: every candidate runs all layers.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "writer.idle",
        group: "Budgets and allowances",
        values: &["on", "off"],
        default: "on",
        label: "Idle writer notes",
        help: "On (default): when the server is quiet and memory allows, Qwen3 1.7B (writer-base@3, ~1,087 MiB alongside the other three MicroModels within 2.17 GiB on a 4 GB card) summarizes served and contested pieces at idle, yielding immediately when a new request arrives or the server exits. Off: disables background writer notes.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "quota.session_tokens",
        group: "Budgets and allowances",
        values: &["0", "25000", "50000", "100000", "250000"],
        default: "0",
        label: "Session context token quota",
        help: "Cumulative token allowance for context packs served in one session (0 for unlimited, up to 500,000 tokens). At 80% of an active quota the Nano automatically scales packs down to the focused allowance; at 100% it applies the quota action below.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 0, max: 500_000, step: 5000, unit: "tokens" }),
    },
    Control {
        key: "quota.daily_tokens",
        group: "Budgets and allowances",
        values: &["0", "100000", "250000", "500000", "1000000"],
        default: "0",
        label: "Daily context token quota",
        help: "Cumulative 24-hour token allowance across served context packs (0 for unlimited, up to 2,000,000 tokens). At 80% of an active quota packs scale down to the focused allowance; at 100% the quota action below applies.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 0, max: 2_000_000, step: 25_000, unit: "tokens" }),
    },
    Control {
        key: "quota.session_packs",
        group: "Budgets and allowances",
        values: &["0", "20", "50", "100", "250"],
        default: "0",
        label: "Session context pack call quota",
        help: "Maximum number of context pack calls served in one session before applying the quota action below (0 for unlimited, up to 500 packs).",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 0, max: 500, step: 5, unit: "packs" }),
    },
    Control {
        key: "quota.action",
        group: "Budgets and allowances",
        values: &["compact", "fast", "withhold"],
        default: "compact",
        label: "When a quota is reached",
        help: "How the Nano responds when a session or daily quota is reached. Compact: scales each pack down to a compact focused budget so top citations still arrive with minimal token cost. Fast: also switches to fast mode (top stack candidates only). Withhold: withholds uncached packs with a quota notice until the quota is raised or reset to 0.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "memory",
        group: "Memory and privacy",
        values: &["on", "off"],
        default: "on",
        label: "Memory",
        help: "Off: nothing recalled and nothing written to memory.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "dense_cache",
        group: "Memory and privacy",
        values: &["on", "off"],
        default: "on",
        label: "Keep the meaning index on disk",
        help: "On: the vectors of the meaning index (one per piece of your material: a content hash and 1,024 numbers, never its text or file names) are kept in the project's memory folder, in dense/vectors.bin, so a restart does not embed everything again. Off: they are neither read nor written and nothing is deleted. With memory off nothing is kept. To clear it: neuromorph-nano dense clear --project <repo>, or POST /api/dense/clear; the next start embeds again.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "recall_cache",
        group: "Memory and privacy",
        values: &["on", "off"],
        default: "on",
        label: "Keep the recall meaning cache on disk",
        help: "On: the episode query vectors, decoy calibration floors and reranker pair scores of recall by meaning (content hashes, vectors and numbers only, never prompt or file text) are kept in the project's memory folder under meaning/, so a restart does not re-run the 28-layer decoy calibration. Off: kept in RAM only for the current process.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "muscle",
        group: "Learning",
        values: &["on", "off"],
        default: "on",
        label: "Muscle memory (G3)",
        help: "On (default): G3 captures repeating positive decision patterns into the separate muscle memory store (never touching base cartridges) and runs bounded idle replays to promote proven behavior packs. Off: disables G3 and muscle memory on serve.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "learning",
        group: "Learning",
        values: &["on", "off"],
        default: "on",
        label: "Continuous learning",
        help: "On: the 3-level Continuous Learning Engine and Supervisor (Level 1 Experiential, Level 2 Knowledge, Level 3 Behavioral) captures outcomes, corrections and preferences, extracts patterns and procedures, adapts retrieval and context allocation, and updates the 6 continual learning cartridges under regression supervision. Off: no continual learning observations or adaptations run.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "learning.cycle",
        group: "Learning",
        values: &["auto", "immediate", "manual"],
        default: "auto",
        label: "Continuous learning cycle",
        help: "Auto: records each judged outcome as an experience as it comes (Cycle A), and runs a learning cycle (Cycle B: patterns, procedures, policies and the preference cartridge, each kept only if it beats the current one on held-out outcomes) once enough verified examples have settled since the last cycle and the serving process has been quiet for a while. Immediate: records experiences as they come and runs a cycle only when you ask for one. Manual: runs a cycle only when you ask for one from the Learning page.",
        wired: true,
        applies: "next call",
        range: None,
    },
    // 2026-11-26 (Luis Jake Gabriel III: audit log retention, asked with the other data controls).
    Control {
        key: "audit.keep_days",
        group: "Memory and privacy",
        values: &["7", "28", "90", "365"],
        default: "28",
        label: "Keep the audit log's tool calls (days of use)",
        help: "How many days of use the audit log keeps its tool-call rows when you trim it (Audit page, Trim the audit log now; or neuromorph-nano keep --apply). Setting changes, stack switches and every other row are always kept, so Undo keeps working. Days of use count only days the Nano ran.",
        wired: true,
        applies: "next trim",
        range: None,
    },
    // 2026-11-25 (roadmap Phase 4.6, the idle KV window release; measured by four soaks, on by
    // default at 15 minutes, NEEDS_HILLCLIMB):
    // kept window copies fill spare card memory and are released under pressure already; this
    // gives them back to the other programs on the machine once the Nano has been idle a while.
    Control {
        key: "residency.idle_release",
        group: "Hardware limits",
        values: &["off", "5", "15", "30"],
        default: "15",
        label: "Give idle card memory back (minutes)",
        help: "Off: kept window copies (the reused parts of earlier reads) stay on the card while it has room, and are released only when something needs the memory. 5, 15 or 30: once the serving process has been idle that many minutes, window copies unused that long and idle scratch buffers are released, so other programs get the memory; the models stay seated. It waits while the idle writer has notes to write. Measured: 40 to 160 MB given back per idle period, with no slower questions. 15 by default.",
        wired: true,
        applies: "live",
        range: None,
    },
    Control {
        key: "mode",
        group: "Claude Code",
        values: &["full", "fast"],
        default: "full",
        label: "Answer mode",
        help: "Full: the whole path (reranker over more candidates, the code around the first place, callers, a second look). Fast: seconds, the reranker on the first few. A call can ask for either; applies when the server runs the whole path.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "mcp.tools",
        group: "Claude Code",
        values: &["full", "lean"],
        default: "full",
        label: "MCP tool exposure profile",
        help: "Full: advertises up to the configured MCP tool count below (default all 25 tools) in tools/list. Lean: caps tools/list to the 5 core coding tools (nm_context_pack, nm_symbol_find, nm_view, nm_deps, nm_feedback) to cut the fixed tool description cost of every turn, while keeping all 25 tools callable by name in tools/call.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "mcp.tool_count",
        group: "Claude Code",
        values: &["1", "5", "8", "12", "19", "25"],
        default: "25",
        label: "MCP advertised tool count",
        help: "Exact number of Neuromorph Nano tools advertised in tools/list in coding priority order (1 for nm_context_pack only, 5 for the core coding set, 8 with repo map and build tools, up to all 25 tools, the default; the last 6 are the interface tools). A client such as Claude Code calls only the tools in tools/list, so a tool left out is reached when dynamic tool activation brings it into the list.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: 1, max: 25, step: 1, unit: "tools" }),
    },
    Control {
        key: "mcp.tool_control",
        group: "Claude Code",
        values: &["off", "on"],
        default: "off",
        label: "MCP tool control",
        help: "Off: uses default model and server tool settings (all 25 tools unless lean mode or a lower tool count is set). On: lets you select individually which of the 25 MCP tools you want to use; when Dynamic tool activation is off, unselected tools are hidden from tools/list and blocked from running.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "mcp.enabled_tools",
        group: "Claude Code",
        values: &["auto"],
        default: "auto",
        label: "Selected MCP tools",
        help: "Active when MCP tool control is On. Comma-separated list of the exact MCP tools you selected to keep ON (for example nm_context_pack,nm_symbol_find,nm_view,nm_build_run). With Dynamic tool activation Off, only these tools are advertised and allowed to run; with Dynamic On, these are your starting tools and additional tools can activate dynamically on demand.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "mcp.dynamic_tools",
        group: "Claude Code",
        values: &["on", "off"],
        default: "on",
        label: "Dynamic MCP tool activation",
        help: "On: dynamically activates additional MCP tools into tools/list (emitting notifications/tools/list_changed) on demand when the task or agent needs them, starting from either your Tool Control selection (when Tool Control is On) or the lean tool count (when Tool Control is Off). Off: keeps tools strictly fixed (the 19 defaults when Tool Control is Off, or strictly your selected tools when Tool Control is On).",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "host.claude_lean",
        group: "Claude Code",
        values: &["off", "on"],
        default: "off",
        label: "Claude Code lean host settings",
        help: "On: setup writes \"enableArtifact\": false into the project's .claude/settings.local.json (or when setup --lean is passed), omitting Claude Code's large built-in Artifact tool description in interactive sessions to save prompt tokens and quota. Off: leaves Claude Code's built-in Artifact setting untouched.",
        wired: true,
        applies: "next setup --write",
        range: None,
    },
    Control {
        key: "air_gap",
        group: "Network",
        values: &["off", "on"],
        default: "off",
        label: "Air-gap mode",
        help: "On: strict local-only lock. Keeps the interface bound to 127.0.0.1 only, turns off local network access and tablet pairing at once, refuses any remote URL or download, and keeps every model weight and cartridge strictly local. Also locked on when NANO_AIR_GAP=on.",
        wired: true,
        applies: "at once",
        range: None,
    },
    Control {
        key: "network",
        group: "Network",
        values: &["off", "view", "control"],
        default: "off",
        label: "Tablets on the local network",
        help: "Off: this computer only. View: paired tablets on your local network can read everything and change nothing. Control: they can also change settings. There is no encryption on the wire: anyone who can listen on your local network can read the traffic, including a tablet's token. Pair a tablet with the six-digit code (neuromorph-nano pair).",
        wired: true,
        applies: "next start of the interface server (switching off applies at once)",
        range: None,
    },
    Control {
        key: "prompts",
        group: "Memory and privacy",
        values: &["keep", "off"],
        default: "keep",
        label: "Keep prompt text",
        help: "Keep: the text of your prompts to Claude Code, seen through its hooks, is kept on this computer in the project's memory folder (prompts.jsonl), with anything that looks like a secret dropped first, for the days set below. Off: only each prompt's length and a hash are kept. Clear it with neuromorph-nano situation --project <repo> --clear-prompts.",
        wired: true,
        applies: "next prompt",
        range: None,
    },
    Control {
        key: "priming",
        group: "Claude Code",
        values: &["off", "quiet", "on"],
        default: "quiet",
        label: "Priming",
        help: "Primes: short notes the Nano adds to Claude Code's context through its hooks, unasked, when they are likely to help: at a prompt, the places that answered a similar verified question before; after a failed build or test, the change that fixed the same failure last time; before an edit, a verified fix in those lines or the callers of a changed signature; after the conversation is compacted, what was in play. Each cites its source, which is read fresh first; the same note is never given twice in a session. Quiet: only from strong evidence. On: also from weaker evidence, at a session start (what the last session left open) and at a stop with changes not built or tested. No count per session: each note is given only when its rule fires, and never twice. Off: none.",
        wired: true,
        applies: "next hook",
        range: None,
    },
    Control {
        key: "reminders",
        group: "Claude Code",
        values: &["on", "off"],
        default: "on",
        label: "Reminders and step nudges",
        help: "On: the reminders you or your coding agent set for a project (Memory page, Guidance; nm_goal; neuromorph-nano guidance) are added to Claude Code's context through its hooks at their triggers, whole and prefixed with who set them, and the next ready step of the step graph is given at prompts and at stops, once per step per session unless it changes. Off: none are delivered; nothing set is removed. Needs the hooks (setup --hooks) and memory on.",
        wired: true,
        applies: "next hook",
        range: None,
    },
    Control {
        key: "loops",
        group: "Claude Code",
        values: &["on", "off"],
        default: "on",
        label: "Loops",
        help: "On: a loop you or your coding agent set keeps Claude Code going at a stop until its condition holds (the last build or test after the last edit passed, a goal closed, every step done, or ended by hand), at most its maximum of iterations (3 by default, never more than 20). This is the one case in which the Nano's hooks block a stop, and only to continue the agent. Off: no loop acts; loops stay as they are. A loop does nothing unless one is set and on.",
        wired: true,
        applies: "next stop",
        range: None,
    },
    Control {
        key: "examples",
        group: "Learning",
        values: &["off", "features", "full"],
        default: "features",
        label: "Keep training examples",
        help: "What the Nano keeps, on this computer in the project's memory folder (training/), for fitting cartridges later from your verified work. Features: for each served context pack, every candidate it judged (place, a hash of its lines, its score from the reranker, whether it was served), what followed (used, verified, refuted, passed over, where Claude Code went instead), the settings, cartridges and hardware in use; preference signals (edits kept or reverted, corrections, stated rules, refusals) without text; timings per hardware piece. No question or passage text. Full: also the question and each judged passage's text, with anything that looks like a secret dropped, so examples stay usable after the code changes. Off: nothing. Nothing is kept with memory off. Clear it with neuromorph-nano examples --project <repo> --clear.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "prompts.days",
        group: "Memory and privacy",
        values: &["1", "7", "30", "90", "365"],
        default: "30",
        label: "Keep prompt text for (days)",
        help: "Kept prompt text older than this is removed at the next read of the situation record, and at a prompt when the last removal was more than an hour ago (1 to 3,650 days).",
        wired: true,
        applies: "next prompt",
        range: Some(IntRange { min: 1, max: 3650, step: 1, unit: "days" }),
    },
    // Roadmap Phase 4.5 (2026-11-24): the storage limit (`nmapi::storage`).
    Control {
        key: "storage.disk_floor_gb",
        group: "Storage",
        values: &["2", "4", "8", "16", "32", "64"],
        default: "4",
        label: "Disk floor (GB)",
        help: "Free space the Nano keeps on the drive holding its folder: below it a job-filter build refuses to start and the storage view warns (1 to 1,000 GB). Prune rebuildable storage to free space.",
        wired: true,
        applies: "next build",
        range: Some(IntRange { min: 1, max: 1000, step: 1, unit: "GB" }),
    },
    // Roadmap Phase 4.5 (2026-11-25): the user's hardware limits, read live by the engine
    // (`nmengn::limits`; values checked by `nmtype::limits::normalize`). They sit beside the
    // homeostasis ceiling `budget.gpu_mb_max` above, which they do not change.
    Control {
        key: "limits.vram_mb",
        group: "Hardware limits",
        values: &["auto", "0", "2048", "3072", "4096", "8192", "16384"],
        default: "auto",
        label: "Graphics card memory limit (MiB)",
        help: "The most card memory the Nano itself keeps: its MicroModels, window copies and working buffers. Under a limit it tiles batches smaller and evicts idle MicroModels (restoring them when needed) to stay within it. Auto (default) uses what the driver reports free. 0 runs every MicroModel on the processor from the next start. Applies live, with no restart (0 at the next start).",
        wired: true,
        applies: "next call (0: next start)",
        range: Some(IntRange { min: nmtype::limits::VRAM_RANGE.0, max: nmtype::limits::VRAM_RANGE.1, step: 128, unit: "MiB" }),
    },
    Control {
        key: "limits.ram_mb",
        group: "Hardware limits",
        values: &["auto", "4096", "8192", "12288", "16384", "24576", "32768"],
        default: "auto",
        label: "System memory limit (MiB)",
        help: "The most host memory the Nano plans for: the kept encoded-chunk cache takes at most half of it, and the idle writer loads only when the limit less the Nano's own memory leaves it room. Auto (default) follows the memory available now. Applies live, with no restart.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: nmtype::limits::RAM_RANGE.0, max: nmtype::limits::RAM_RANGE.1, step: 512, unit: "MiB" }),
    },
    Control {
        key: "limits.cpu_threads",
        group: "Hardware limits",
        values: &["auto", "1", "2", "4", "6", "8", "12", "16"],
        default: "auto",
        label: "Processor threads limit",
        help: "The most processor threads the Nano's work runs on at once (questions and background indexing and writing), so the machine stays responsive. Auto (default) uses every thread; background work still yields to a question. Applies live, at the next stage.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: nmtype::limits::CPU_RANGE.0, max: nmtype::limits::CPU_RANGE.1, step: 1, unit: "threads" }),
    },
    // 2026-11-27 (Luis Jake Gabriel III: "The user should control how much to allocate memory
    // wise to where"): the process budget divided by memory area (`nmengn::limits::area_budget`).
    // The areas set to a size may not exceed the budget together (`set` refuses such a change).
    Control {
        key: "limits.memory.content_mb",
        group: "Hardware limits",
        values: &["auto", "1024", "2048", "4096", "8192"],
        default: "auto",
        label: "Memory for project content (MiB)",
        help: "The project's content in RAM: the files and pieces read, the drones' intake of them, the relevance keys and the dense lane's vectors. The intake, the relevance-key intake and the dense lane work in bounded steps and stop when this area is full; the pieces not reached are found by their paths only until the area is raised. Auto (default) is a quarter of the process budget (15 percent of it stays with the engine's own working memory). Applies live.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: nmtype::limits::AREA_RANGE.0, max: nmtype::limits::AREA_RANGE.1, step: 128, unit: "MiB" }),
    },
    Control {
        key: "limits.memory.caches_mb",
        group: "Hardware limits",
        values: &["auto", "512", "1024", "2048", "4096"],
        default: "auto",
        label: "Memory for kept caches (MiB)",
        help: "The kept caches: the encoded-chunk cache the tentacles re-read. A lower value keeps fewer encodings, so repeated questions encode again and take longer. Auto (default) is 15 percent of the process budget. Applies when the cache is next made.",
        wired: true,
        applies: "next start of the cache",
        range: Some(IntRange { min: nmtype::limits::AREA_RANGE.0, max: nmtype::limits::AREA_RANGE.1, step: 128, unit: "MiB" }),
    },
    Control {
        key: "limits.memory.weights_mb",
        group: "Hardware limits",
        values: &["auto", "2048", "4096", "6144", "8192"],
        default: "auto",
        label: "Memory for MicroModel weights (MiB)",
        help: "The host copies of the MicroModels' weights. A lower value keeps fewer MicroModels' weights in RAM, so a MicroModel not held loads when a call needs it. Auto (default) is 45 percent of the process budget. Applies live.",
        wired: true,
        applies: "next call",
        range: Some(IntRange { min: nmtype::limits::AREA_RANGE.0, max: nmtype::limits::AREA_RANGE.1, step: 128, unit: "MiB" }),
    },
    // 2026-11-25 (directions of Luis Jake Gabriel III: routing is customizable, in the code and the
    // interface). `measured` follows the measured routing record (or the published table).
    Control {
        key: "routing.code",
        group: "Routing",
        values: &["measured", "embedder", "judge"],
        default: "measured",
        label: "Who decides close calls on code",
        help: "Measured (default): the MicroModel the measured routing record names strongest for code (the published strengths table without one). Embedder or judge: that MicroModel decides every close call on code (the judge only where the judge runs). Applies at the next call.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "routing.prose",
        group: "Routing",
        values: &["measured", "embedder", "judge"],
        default: "measured",
        label: "Who decides close calls on prose",
        help: "Measured (default): the MicroModel the measured routing record names strongest for prose and documents. Embedder or judge: that MicroModel decides every close call on prose. Applies at the next call.",
        wired: true,
        applies: "next call",
        range: None,
    },
    Control {
        key: "routing.record",
        group: "Routing",
        values: &["on", "off"],
        default: "on",
        label: "Measured routing record",
        help: "On (default): the routing record measured on the serving base cartridges replaces the published strengths where it measured enough duels; it is set aside by itself when the base cartridges or the lens change. Off: the published table. Applies at the next start.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "routing.ask",
        group: "Routing",
        values: &["judge", "always", "off"],
        default: "judge",
        label: "When the Core Brain asks for more context",
        help: "Judge (default): when the judge finds the pack below its bar. Always: on every question. Off: never. The --ask option of serve wins over this. Applies at the next start.",
        wired: true,
        applies: "next start",
        range: None,
    },
    Control {
        key: "routing.tentacle_judge",
        group: "Routing",
        values: &["default", "heads", "reranker", "rrf", "sum", "place"],
        default: "default",
        label: "How the tentacles judge what they read",
        help: "Default: the served setting. Heads: the drone heads' own reads. Reranker: the judge. Rrf, sum: the two combined by reciprocal rank or by sum. Place: by place in the shortlist. The --tentacle-judge option of serve wins over this. Applies at the next start.",
        wired: true,
        applies: "next start",
        range: None,
    },
    // 2026-11-25 (directions of Luis Jake Gabriel III): what an upgrade does with older base
    // cartridge versions (`library::merge_release`).
    Control {
        key: "library.upgrade",
        group: "Cartridges",
        values: &["keep", "replace"],
        default: "keep",
        label: "When new base cartridges arrive",
        help: "Keep (default): a new base cartridge version is added to the cartridge library beside the older ones and serves by default; choose any version, or remove one, on the Cartridges page. Replace: once the new version is in the cartridge library and usable, the older versions are removed (a version you chose for a MicroModel is never removed). Applies at the next upgrade.",
        wired: true,
        applies: "next upgrade",
        range: None,
    },
];

static WRITE: Mutex<()> = Mutex::new(());

fn settings_path(home: &Path) -> PathBuf {
    home.join("settings.json")
}
fn audit_path(home: &Path) -> PathBuf {
    home.join("memory").join("audit.jsonl")
}
/// Earlier names of controls, read when the current name is not saved: (current, earlier).
const EARLIER: &[(&str, &str)] = &[("priming", "nudges")];

fn find(key: &str) -> Option<&'static Control> {
    let key = EARLIER.iter().find(|e| e.1 == key).map_or(key, |e| e.0);
    CONTROLS.iter().find(|c| c.key == key)
}

fn validate(c: &Control, raw: &str) -> Result<String, String> {
    if c.key == "mcp.enabled_tools" {
        let t = raw.trim();
        if t.eq_ignore_ascii_case("auto") {
            return Ok("auto".to_owned());
        }
        let mut chosen = Vec::new();
        for part in t.split(',') {
            let name = part.trim();
            if name.is_empty() {
                continue;
            }
            if !nmapi::mcp::PRIORITY_TOOLS.contains(&name) {
                return Err(format!("{}: unknown MCP tool '{name}'", c.key));
            }
            if !chosen.contains(&name) {
                chosen.push(name);
            }
        }
        if chosen.is_empty() {
            return Err(format!("{}: choose at least one tool or 'auto'", c.key));
        }
        chosen.sort_by_key(|name| nmapi::mcp::PRIORITY_TOOLS.iter().position(|t| t == name).unwrap_or(usize::MAX));
        return Ok(chosen.join(","));
    }
    // Roadmap Phase 4.5 (2026-11-25): a hardware limit is `auto` or a whole number in its bounds
    // (`nmtype::limits::normalize`, which the engine's reading uses too).
    if let Some(r) = nmtype::limits::normalize(c.key, raw) {
        return r;
    }
    // Roadmap Phase 4.6 (2026-11-24): a ranged control that offers `auto` takes it (the value then
    // derives from this machine, `nmengn::capacity`).
    if c.range.is_some() && c.values.contains(&"auto") && raw.trim().eq_ignore_ascii_case("auto") {
        return Ok("auto".to_owned());
    }
    if let Some(r) = c.range {
        let t = raw.trim();
        let n = t
            .parse::<i128>()
            .map_err(|_| format!("{}: {raw} must be a whole number between {} and {} {}", c.key, r.min, r.max, r.unit))?;
        if n < i128::from(r.min) || n > i128::from(r.max) {
            return Err(format!("{}: {raw} must be between {} and {} {}", c.key, r.min, r.max, r.unit));
        }
        return Ok(n.to_string());
    }
    if c.values.contains(&raw) {
        Ok(raw.to_owned())
    } else {
        Err(format!("{}: {raw} is not one of {}", c.key, c.values.join(", ")))
    }
}

/// The saved values, with defaults for anything not saved. A value saved under a control's earlier
/// name is read when none is saved under its current name.
#[must_use]
pub fn current(home: &Path) -> BTreeMap<String, String> {
    let saved = std::fs::read(settings_path(home)).ok().and_then(|b| nmjson::parse(&b).ok());
    CONTROLS
        .iter()
        .map(|c| {
            let earlier = EARLIER.iter().find(|e| e.0 == c.key).map(|e| e.1);
            let v = saved
                .as_ref()
                .and_then(|s| s.get(c.key).ok().or_else(|| earlier.and_then(|k| s.get(k).ok())))
                .and_then(|v| v.string().ok())
                .and_then(|v| validate(c, v).ok())
                .unwrap_or_else(|| c.default.to_owned());
            (c.key.to_owned(), v)
        })
        .collect()
}

/// The `--precision` text for the saved widths, or `None` when every part is at its default.
#[must_use]
pub fn precision_arg(home: &Path) -> Option<String> {
    let cur = current(home);
    let parts: Vec<String> = ["base", "writer", "reranker", "embedder"]
        .iter()
        .filter_map(|p| cur.get(&format!("precision.{p}")).filter(|w| w.as_str() != "default").map(|w| format!("{p}={w}")))
        .collect();
    (!parts.is_empty()).then(|| parts.join(","))
}

/// Memory switched off in the saved controls.
#[must_use]
pub fn memory_off(home: &Path) -> bool {
    current(home).get("memory").is_some_and(|v| v == "off")
}

/// True when `NANO_AIR_GAP` forces air-gap mode on from the environment.
#[must_use]
pub fn air_gap_env() -> bool {
    std::env::var("NANO_AIR_GAP")
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "on" | "1" | "true" | "yes"))
}

/// True when strict air-gap mode is in force (from `NANO_AIR_GAP=on` or saved `air_gap: on`).
#[must_use]
pub fn air_gap_on(home: &Path) -> bool {
    air_gap_env() || current(home).get("air_gap").is_some_and(|v| v == "on")
}

/// Live verifiable air-gap posture for `GET /api/airgap` and `nm_settings op=airgap`.
#[must_use]
pub fn air_gap_view(home: &Path, lan_listeners: usize) -> Value {
    let env_lock = air_gap_env();
    let cur = current(home);
    let saved_on = cur.get("air_gap").is_some_and(|v| v == "on");
    let active = env_lock || saved_on;
    let source = if env_lock {
        "NANO_AIR_GAP"
    } else if saved_on {
        "settings.json"
    } else {
        "default"
    };
    let net = if active {
        "off"
    } else {
        cur.get("network").map_or("off", String::as_str)
    };
    let effective_lan = if active { 0 } else { lan_listeners };
    let mut o = BTreeMap::new();
    o.insert("air_gap".to_owned(), Value::Bool(active));
    o.insert("source".to_owned(), s(source));
    o.insert("locked_by_env".to_owned(), Value::Bool(env_lock));
    o.insert("network".to_owned(), s(net));
    o.insert("loopback_only".to_owned(), Value::Bool(effective_lan == 0 && (active || net == "off")));
    o.insert("lan_listeners".to_owned(), Value::Number(effective_lan.to_string()));
    // 2026-11-25 (nothing pseudo): the Nano has outbound network code, the downloads of released
    // models and cartridges from Hugging Face (`fetch.rs`), run only when the user asks and refused
    // under the air gap. The former constant `sensory_boundary_enforced` is gone: nothing measured it.
    o.insert("outbound_network_code".to_owned(), Value::Bool(!active));
    o.insert("remote_imports_allowed".to_owned(), Value::Bool(!active));
    o.insert("downloads_running".to_owned(), Value::Bool(crate::fetch::running(home)));
    o.insert("models_local_dir".to_owned(), s(&nmtype::models_dir(home).display().to_string()));
    Value::Object(o)
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}

/// Every control with its current value, for the interface.
#[must_use]
pub fn view(home: &Path) -> Value {
    let cur = current(home);
    Value::Array(
        CONTROLS
            .iter()
            .map(|c| {
                let mut o = BTreeMap::new();
                o.insert("key".to_owned(), s(c.key));
                o.insert("group".to_owned(), s(c.group));
                o.insert("label".to_owned(), s(c.label));
                o.insert("help".to_owned(), s(c.help));
                o.insert("desc".to_owned(), s(c.help));
                o.insert("values".to_owned(), Value::Array(c.values.iter().map(|v| s(v)).collect()));
                o.insert("default".to_owned(), s(c.default));
                let (val, src) = if c.key == "air_gap" && air_gap_env() {
                    ("on", "NANO_AIR_GAP")
                } else {
                    let v = cur.get(c.key).map_or(c.default, String::as_str);
                    (v, if v != c.default { "settings.json" } else { "default" })
                };
                o.insert("value".to_owned(), s(val));
                o.insert("source".to_owned(), s(src));
                o.insert("wired".to_owned(), Value::Bool(c.wired));
                o.insert("applies".to_owned(), s(c.applies));
                if c.key == "mcp.enabled_tools" {
                    o.insert("kind".to_owned(), s("toolset"));
                    o.insert(
                        "all_tools".to_owned(),
                        Value::Array(nmapi::mcp::PRIORITY_TOOLS.iter().map(|t| s(t)).collect()),
                    );
                } else if let Some(r) = c.range {
                    o.insert("kind".to_owned(), s("integer"));
                    o.insert("min".to_owned(), Value::Number(r.min.to_string()));
                    o.insert("max".to_owned(), Value::Number(r.max.to_string()));
                    o.insert("step".to_owned(), Value::Number(r.step.to_string()));
                    o.insert("unit".to_owned(), s(r.unit));
                } else {
                    o.insert("kind".to_owned(), s("enum"));
                }
                Value::Object(o)
            })
            .collect(),
    )
}

fn write_settings(home: &Path, values: &BTreeMap<String, String>) -> Result<(), String> {
    // 2026-11-25: keys that are not controls (`base_cartridges`, written by a rollback or a base
    // version choice) are kept; the file was written again with the controls only, so any change
    // in the interface dropped the base cartridge choice.
    let mut o: BTreeMap<String, Value> = std::fs::read(settings_path(home))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| if let Value::Object(o) = v { Some(o) } else { None })
        .map(|o| o.into_iter().filter(|(k, _)| find(k).is_none()).collect())
        .unwrap_or_default();
    o.extend(values.iter().map(|(k, v)| (k.clone(), s(v))));
    o.insert(
        "note".to_owned(),
        s("User controls of the Neuromorph Nano, written by the interface. Every change is in memory/audit.jsonl."),
    );
    let text = nmjson::pretty::pretty(&Value::Object(o));
    let p = settings_path(home);
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
}

// Roadmap Phase 4.5 (2026-11-24): the audit writer moved to `nmapi::audit`, so the storage
// controls and the MCP tools there write the same rows with the same ids; kept here by name.
pub use nmapi::audit::{audit, audit_row, new_id};

/// Sets one control, logging the change. Returns the audit id, or `None` when the value was
/// already set.
/// # Errors
/// Unknown control, value not allowed, or a write failure.
pub fn set(home: &Path, key: &str, value: &str, undo_of: Option<&str>) -> Result<Option<String>, String> {
    set_by(home, key, value, undo_of, "interface")
}

/// `set`, naming who made the change in the audit row (`interface`, or a paired device's name).
/// # Errors
/// As `set`.
pub fn set_by(home: &Path, key: &str, value: &str, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    if let Some(id) = key.strip_prefix(CARTRIDGE_KEY) {
        return switch_cartridge(home, id, value, undo_of, by);
    }
    // 2026-11-26 (Luis Jake Gabriel III: "plug in and unplug the learned units"): a learned adapter
    // or piece, and a compartment of them.
    if let Some(unit) = key.strip_prefix(crate::learned_units::LEARNED_KEY) {
        return crate::learned_units::switch_unit(home, unit, value, undo_of, by);
    }
    if let Some(c) = key.strip_prefix(crate::learned_units::COMPARTMENT_KEY) {
        return crate::learned_units::switch_compartment(home, c, value, undo_of, by);
    }
    // The maps `settings.json` holds beside the controls (`SAVED_KEYS`), one entry at a time, so
    // every key of the file is set through this one function (the interface's `/api/controls`,
    // `nm_settings` `set`): the base cartridge a MicroModel serves, the model a role runs on, and
    // a base cartridge's compensation.
    if let Some(model) = key.strip_prefix(BASE_KEY) {
        let _g = WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())?;
        return crate::library::choose_base(home, model, value, by).map(Some);
    }
    if let Some(role) = key.strip_prefix(ROLE_KEY) {
        let _g = WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())?;
        return crate::library::choose_role(home, role, value, by).map(|(id, _)| Some(id));
    }
    if let Some(model) = key.strip_prefix(COMPENSATION_KEY) {
        return set_compensation(home, model, value, undo_of, by);
    }
    let c = find(key).ok_or_else(|| format!("unknown control {key}"))?;
    let key = c.key;
    let normalized = validate(c, value)?;
    let _g = WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())?;
    let mut cur = current(home);
    // Memory areas (2026-11-27): the areas set to a size may not exceed the process budget
    // (`limits.ram_mb`, or three eighths of the machine on auto); a change to an area or to the
    // host memory limit that would is refused, with the figures.
    if key == nmtype::limits::RAM_KEY || nmtype::limits::AREA_KEYS.contains(&key) {
        let mut next = cur.clone();
        next.insert(key.to_owned(), normalized.clone());
        let mb = |k: &str| next.get(k).and_then(|v| v.parse::<u64>().ok());
        let areas = nmtype::limits::Areas([mb(nmtype::limits::CONTENT_KEY), mb(nmtype::limits::CACHES_KEY), mb(nmtype::limits::WEIGHTS_KEY)]);
        nmengn::limits::check_areas_now(mb(nmtype::limits::RAM_KEY), &areas).map_err(|e| format!("{key}: {e}"))?;
    }
    if key == "air_gap" && normalized == "off" && air_gap_env() {
        return Err("air_gap is locked on by NANO_AIR_GAP".to_owned());
    }
    if key == "network" && normalized != "off" && (air_gap_env() || cur.get("air_gap").is_some_and(|v| v == "on")) {
        return Err("network cannot be enabled while air_gap is on (turn air_gap off first)".to_owned());
    }
    let from = cur.get(key).cloned().unwrap_or_else(|| c.default.to_owned());
    if from == normalized {
        return Ok(None);
    }
    cur.insert(key.to_owned(), normalized.clone());
    if key == "air_gap" && normalized == "on" {
        cur.insert("network".to_owned(), "off".to_owned());
    }
    write_settings(home, &cur)?;
    let mut fields = vec![
        ("key", s(key)),
        ("label", s(c.label)),
        ("from", s(&from)),
        ("to", s(&normalized)),
        ("by", s(by)),
        ("applies", s(c.applies)),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    audit(home, "setting", fields).map(Some)
}

/// The lock every writer of `settings.json` in this process holds while it reads, changes and
/// writes the file, so two requests cannot drop each other's change.
/// # Errors
/// The lock is poisoned.
pub fn write_guard() -> Result<std::sync::MutexGuard<'static, ()>, String> {
    WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())
}

/// The key prefix of one entry of `base_cartridges` (`base_cartridges:<model id>`; a base cartridge
/// key, `auto` or `open`).
pub const BASE_KEY: &str = "base_cartridges:";
/// The key prefix of one entry of `role_models` (`role_models:<role>`; a model id or `default`).
pub const ROLE_KEY: &str = "role_models:";
/// The key prefix of one entry of `base_compensation` (`base_compensation:<model id>`; `on`, `off`).
pub const COMPENSATION_KEY: &str = "base_compensation:";

/// One key `settings.json` holds that is not a row of `CONTROLS`: what it holds, where each
/// surface sets it, and, for a key nothing sets, why it is read only.
pub struct SavedKey {
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    /// What names an entry of the map (`model` or `role`); empty for a key that is one value.
    pub entry: &'static str,
    /// The route that sets it besides `POST /api/controls` with `<key>:<entry>`.
    pub route: &'static str,
    /// The MCP op that sets it besides `nm_settings` `set` with `<key>:<entry>`.
    pub mcp: &'static str,
    /// Where the interface sets it besides Settings, Controls.
    pub interface: &'static str,
    pub applies: &'static str,
    /// Whether a change of it is undone from the audit log (`POST /api/undo`).
    pub undo: bool,
    pub read_only: Option<&'static str>,
}

/// Every key of `settings.json` beside the controls. A key added to the file is added here, so
/// the Settings page, `/api/controls/saved` and `nm_settings` `saved` show it (`check_parity`).
pub const SAVED_KEYS: &[SavedKey] = &[
    SavedKey {
        key: "base_cartridges",
        label: "Base cartridge served, per MicroModel",
        help: "The base cartridge version each MicroModel serves: a version in the cartridge library, auto (the newest usable version) or open (its upstream open weights).",
        entry: "model",
        route: "POST /api/cartridges/bases",
        mcp: "nm_cartridges base_serve",
        interface: "Cartridges, base cartridge versions",
        applies: "next start",
        undo: true,
        read_only: None,
    },
    SavedKey {
        key: "base_compensation",
        label: "Base cartridge compensation, per MicroModel",
        help: "On: a base cartridge serves with the compensation cartridge its label names (a bias per channel that restores the mean contribution of what the base cartridge left out). Off (default): it serves without it. A base cartridge that names no compensation is not changed by this.",
        entry: "model",
        route: "POST /api/controls",
        mcp: "nm_settings set",
        interface: "Settings, Controls",
        applies: "next start",
        undo: true,
        read_only: None,
    },
    SavedKey {
        key: "role_models",
        label: "Model each role runs on",
        help: "The installed model each role (Reader, Writer, Embedder, Reranker) runs on; default is the model the role was measured on. A model that may not run as the role is refused with the reason.",
        entry: "role",
        route: "POST /api/models/roles",
        mcp: "nm_settings role_set",
        interface: "Models, roles and routing",
        applies: "next start",
        undo: true,
        read_only: None,
    },
    SavedKey {
        key: "note",
        label: "File note",
        help: "One line saying what the file is.",
        entry: "",
        route: "",
        mcp: "",
        interface: "",
        applies: "",
        undo: false,
        read_only: Some("written by the program with every save; it states what the file is and holds no setting"),
    },
];

/// `NANO_BASE_COMP` when it forces the compensation of every MicroModel (`on` or `off`).
fn compensation_env() -> Option<&'static str> {
    match std::env::var("NANO_BASE_COMP").as_deref() {
        Ok("on") => Some("on"),
        Ok("off") => Some("off"),
        _ => None,
    }
}

/// The model ids a `base_compensation` entry may name: the four roles' own models and every
/// installed model.
fn compensation_models(home: &Path) -> Vec<String> {
    let mut v: Vec<String> = nmtype::role_models::ROLES.iter().map(|r| r.1.to_owned()).collect();
    for id in nmtype::role_models::installed(home) {
        if !v.contains(&id) {
            v.push(id);
        }
    }
    v
}

/// The saved `base_compensation` map (model id to `on` or `off`).
fn saved_compensation(home: &Path) -> BTreeMap<String, String> {
    std::fs::read(settings_path(home))
        .ok()
        .and_then(|b| nmjson::parse(&b).ok())
        .and_then(|v| v.get("base_compensation").and_then(Value::object).ok().cloned())
        .map(|m| m.into_iter().filter_map(|(k, x)| x.string().ok().map(|x| (k, x.to_owned()))).collect())
        .unwrap_or_default()
}

/// Sets one MicroModel's `base_compensation` entry (`on` or `off`; off takes the entry out, the
/// default), keeping every other key of the file; logged as a setting under
/// `base_compensation:<model>`, so it is undone like any other.
fn set_compensation(home: &Path, model: &str, value: &str, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let key = format!("{COMPENSATION_KEY}{model}");
    if !matches!(value, "on" | "off") {
        return Err(format!("{key}: {value} is not one of on, off"));
    }
    let models = compensation_models(home);
    if !models.iter().any(|m| m == model) {
        return Err(format!("{key}: {model} is not a MicroModel's model id ({})", models.join(", ")));
    }
    let _g = WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())?;
    let from = if saved_compensation(home).get(model).is_some_and(|v| v == "on") { "on" } else { "off" };
    if from == value {
        return Ok(None);
    }
    let path = settings_path(home);
    let mut v = std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok()).filter(|v| matches!(v, Value::Object(_))).unwrap_or_else(|| Value::Object(BTreeMap::new()));
    if let Value::Object(o) = &mut v {
        let slot = o.entry("base_compensation".to_owned()).or_insert_with(|| Value::Object(BTreeMap::new()));
        if !matches!(slot, Value::Object(_)) {
            *slot = Value::Object(BTreeMap::new());
        }
        if let Value::Object(m) = slot {
            if value == "on" {
                m.insert(model.to_owned(), s("on"));
            } else {
                m.remove(model);
            }
        }
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(&v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut fields = vec![
        ("key", s(&key)),
        ("label", s(&format!("Base cartridge compensation, {model}"))),
        ("from", s(from)),
        ("to", s(value)),
        ("by", s(by)),
        ("applies", s("next start of the server")),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    audit(home, "setting", fields).map(Some)
}

/// `GET /api/controls/saved` and `nm_settings` `saved`: `settings.json` key by key. The controls
/// are counted (`/api/controls` lists them); every other key the file may hold (`SAVED_KEYS`)
/// comes with its entries, each entry's value, default and choices, the key that sets it through
/// `POST /api/controls`, and the other route, MCP op and interface place that set it; a key
/// nothing sets is `read_only` with the reason. `unknown` names keys in the file that no build
/// part reads (kept as they are, never dropped).
#[must_use]
pub fn saved_view(home: &Path) -> Value {
    let o = |pairs: Vec<(&str, Value)>| Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect());
    let text = |v: &Value, k: &str| v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default();
    let choice = |value: &str, ok: bool, why: Option<&str>| o(vec![("value", s(value)), ("ok", Value::Bool(ok)), ("why", why.map_or(Value::Null, s))]);
    let path = settings_path(home);
    let file = std::fs::read(&path).ok().and_then(|b| nmjson::parse(&b).ok());
    let saved: BTreeMap<String, Value> = match &file {
        Some(Value::Object(m)) => m.clone(),
        _ => BTreeMap::new(),
    };
    let in_file = |key: &str, name: &str| saved.get(key).and_then(|m| m.get(name).ok()).is_some();
    let role_title = |model: &str| nmtype::role_models::role_of(model).map(str::to_owned);
    let entries = |k: &SavedKey| -> Vec<Value> {
        match k.key {
            "base_cartridges" => {
                let mut rows: Vec<Value> = crate::library::base_versions(home)
                    .get("models")
                    .and_then(Value::array)
                    .unwrap_or_default()
                    .iter()
                    .map(|m| {
                        let model = text(m, "model");
                        let mut choices = vec![choice("auto", true, None), choice("open", true, None)];
                        for ver in m.get("versions").and_then(Value::array).unwrap_or_default() {
                            let refused = ver.get("refused").and_then(Value::string).ok();
                            let present = matches!(ver.get("present"), Ok(Value::Bool(true)));
                            let why = refused.or(if present { None } else { Some("its file is not in the cartridge library") });
                            choices.push(choice(&text(ver, "key"), why.is_none(), why));
                        }
                        o(vec![
                            ("name", s(&model)),
                            ("role", role_title(&model).map_or(Value::Null, |r| s(&r))),
                            ("value", s(&text(m, "choice"))),
                            ("default", s("auto")),
                            ("saved", Value::Bool(in_file(k.key, &model))),
                            ("in_effect", s(&text(m, "serving"))),
                            ("choices", Value::Array(choices)),
                            ("locked", Value::Null),
                            ("set_key", s(&format!("{BASE_KEY}{model}"))),
                        ])
                    })
                    .collect();
                // A MicroModel with no base cartridge in the cartridge library still has its
                // entry: `auto` or `open`, both serving its open weights until one arrives.
                for (_, model) in nmtype::role_models::ROLES {
                    if rows.iter().any(|r| text(r, "name") == model) {
                        continue;
                    }
                    let value = saved.get(k.key).and_then(|m| m.get(model).and_then(Value::string).ok()).unwrap_or("auto");
                    rows.push(o(vec![
                        ("name", s(model)),
                        ("role", role_title(model).map_or(Value::Null, |r| s(&r))),
                        ("value", s(value)),
                        ("default", s("auto")),
                        ("saved", Value::Bool(in_file(k.key, model))),
                        ("in_effect", s("open weights (no base cartridge of this MicroModel is in the cartridge library)")),
                        ("choices", Value::Array(vec![choice("auto", true, None), choice("open", true, None)])),
                        ("locked", Value::Null),
                        ("set_key", s(&format!("{BASE_KEY}{model}"))),
                    ]));
                }
                rows
            }
            "role_models" => crate::library::roles_view(home)
                .get("roles")
                .and_then(Value::array)
                .unwrap_or_default()
                .iter()
                .map(|r| {
                    let role = text(r, "role");
                    let default = text(r, "default");
                    let mut choices = vec![choice("default", true, None)];
                    for c in r.get("choices").and_then(Value::array).unwrap_or_default() {
                        // The role's own model is the choice `default`.
                        if text(c, "id") == default {
                            continue;
                        }
                        choices.push(choice(&text(c, "id"), matches!(c.get("ok"), Ok(Value::Bool(true))), c.get("why").and_then(Value::string).ok()));
                    }
                    let model = text(r, "model");
                    o(vec![
                        ("name", s(&role)),
                        ("role", s(&role)),
                        ("value", s(if model == default { "default" } else { &model })),
                        ("default", s("default")),
                        ("saved", Value::Bool(in_file(k.key, &role))),
                        ("in_effect", s(&model)),
                        ("choices", Value::Array(choices)),
                        ("locked", r.get("locked").and_then(Value::string).map_or(Value::Null, s)),
                        ("set_key", s(&format!("{ROLE_KEY}{role}"))),
                    ])
                })
                .collect(),
            "base_compensation" => {
                let map = saved_compensation(home);
                let env = compensation_env();
                let mut models = compensation_models(home);
                models.extend(map.keys().filter(|m| !models.contains(m)).cloned().collect::<Vec<_>>());
                models
                    .iter()
                    .map(|model| {
                        let value = if map.get(model).is_some_and(|v| v == "on") { "on" } else { "off" };
                        o(vec![
                            ("name", s(model)),
                            ("role", role_title(model).map_or(Value::Null, |r| s(&r))),
                            ("value", s(value)),
                            ("default", s("off")),
                            ("saved", Value::Bool(map.contains_key(model))),
                            ("in_effect", s(env.unwrap_or(value))),
                            ("choices", Value::Array(vec![choice("off", true, None), choice("on", true, None)])),
                            ("locked", env.map_or(Value::Null, |e| s(&format!("NANO_BASE_COMP={e} is set in the environment and decides it for every MicroModel; the saved value applies when it is not set")))),
                            ("set_key", s(&format!("{COMPENSATION_KEY}{model}"))),
                        ])
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    };
    let keys: Vec<Value> = SAVED_KEYS
        .iter()
        .map(|k| {
            o(vec![
                ("key", s(k.key)),
                ("label", s(k.label)),
                ("help", s(k.help)),
                ("entry", s(k.entry)),
                ("applies", s(k.applies)),
                ("undo", Value::Bool(k.undo)),
                ("read_only", k.read_only.map_or(Value::Null, s)),
                ("in_file", Value::Bool(saved.contains_key(k.key))),
                ("value", if k.entry.is_empty() { saved.get(k.key).cloned().unwrap_or(Value::Null) } else { Value::Null }),
                ("set", if k.read_only.is_some() { Value::Null } else { o(vec![("route", s("POST /api/controls")), ("key", s(&format!("{}:<{}>", k.key, k.entry))), ("mcp", s("nm_settings set"))]) }),
                ("also", if k.read_only.is_some() { Value::Null } else { o(vec![("route", s(k.route)), ("mcp", s(k.mcp)), ("interface", s(k.interface))]) }),
                ("entries", Value::Array(entries(k))),
            ])
        })
        .collect();
    let known = |key: &str| find(key).is_some() || SAVED_KEYS.iter().any(|k| k.key == key);
    let cur = current(home);
    o(vec![
        ("file", s(&path.display().to_string())),
        ("present", Value::Bool(file.is_some())),
        ("controls", Value::Number(CONTROLS.len().to_string())),
        ("controls_saved", Value::Number(CONTROLS.iter().filter(|c| cur.get(c.key).is_some_and(|v| v != c.default)).count().to_string())),
        ("controls_route", s("GET /api/controls")),
        ("keys", Value::Array(keys)),
        ("unknown", Value::Array(saved.keys().filter(|k| !known(k)).map(|k| s(k)).collect())),
    ])
}

/// The key prefix of a cartridge switch in the audit log (`cartridge:<id>`, values `on`, `off`).
pub const CARTRIDGE_KEY: &str = "cartridge:";

/// Switches a learned cartridge on or off: on runs the slot check (the base model's recorded hash
/// equals the one the cartridge fits) and the registry's hash-checked load; off unplugs it. The
/// change is logged as a setting under `cartridge:<id>`, so it is undone like any other.
fn switch_cartridge(home: &Path, id: &str, value: &str, undo_of: Option<&str>, by: &str) -> Result<Option<String>, String> {
    let on = match value {
        "on" => true,
        "off" => false,
        v => return Err(format!("{CARTRIDGE_KEY}{id}: {v} is not one of on, off")),
    };
    let _g = WRITE.lock().map_err(|_| "controls lock poisoned".to_owned())?;
    let mut units = nmlearn::Units::open(home);
    let card = units.get(id).cloned().ok_or_else(|| format!("no cartridge {id}"))?;
    if card.loaded == on {
        return Ok(None);
    }
    let policy = nmlearn::cartridge::Policy::load(home);
    let label = nmlearn::cartridge::from_unit(&card, &units.unit_dir(id), &policy);
    if on {
        let loaded = nmlearn::cartridge::Loaded::population(home, &label.base, label.components.clone())
            .map_err(|e| e.to_fault().to_string())?;
        nmlearn::cartridge::plug(&mut units, &label, &loaded).map_err(|e| e.to_string())?;
    } else {
        nmlearn::cartridge::unplug(&mut units, id).map_err(|e| e.to_string())?;
    }
    let (from, to) = if on { ("off", "on") } else { ("on", "off") };
    let key = format!("{CARTRIDGE_KEY}{id}");
    let mut fields = vec![
        ("key", s(&key)),
        ("label", s(&format!("Cartridge {id}"))),
        ("from", s(from)),
        ("to", s(to)),
        ("by", s(by)),
        ("applies", s("next start of the server")),
    ];
    if let Some(u) = undo_of {
        fields.push(("undo_of", s(u)));
    }
    audit(home, "setting", fields).map(Some)
}

/// Undoes a logged setting change: sets the control back to the value it had before.
/// # Errors
/// The id is not a setting change in the log, or the set fails.
/// `audit.keep_days` (2026-11-26): days of use the audit log's tool rows are kept when trimmed (28 by
/// default).
#[must_use]
pub fn audit_keep_days(home: &Path) -> u64 {
    nmapi::whole::control_str(home, "audit.keep_days").and_then(|v| v.parse::<u64>().ok()).filter(|d| [7, 28, 90, 365].contains(d)).unwrap_or(28)
}

pub fn undo(home: &Path, id: &str) -> Result<Option<String>, String> {
    undo_by(home, id, "interface")
}

/// `undo`, naming who asked for it.
/// # Errors
/// As `undo`.
pub fn undo_by(home: &Path, id: &str, by: &str) -> Result<Option<String>, String> {
    let text = std::fs::read_to_string(audit_path(home)).map_err(|e| format!("audit log: {e}"))?;
    let row = text
        .lines()
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .find(|v| v.get("id").and_then(Value::string).is_ok_and(|x| x == id))
        .ok_or_else(|| format!("no audit row {id}"))?;
    if row.get("kind").and_then(Value::string).ok() == Some("library") {
        return crate::library::undo(home, id, by);
    }
    // 2026-11-25: plugging a job filter in or out, and removing one, can be undone too.
    if row.get("kind").and_then(Value::string).ok() == Some("job-filter") && row.get("action").and_then(Value::string).is_ok_and(|a| a != "undo") {
        return crate::ui::undo_job_filter(home, &row, id, by);
    }
    // 2026-11-26 (Phase B): a parameter map edit (its backup put back), a learning unit taken out of
    // the registry (its card registered again), a memory pinned or unpinned, an objective changed.
    let kind = row.get("kind").and_then(Value::string).unwrap_or("");
    let action = row.get("action").and_then(Value::string).unwrap_or("");
    match (kind, action) {
        ("parameter-map", "save") => return crate::pmaps::undo_save(home, &row, id, by),
        ("parameter-map", "import") => return crate::pmaps::undo_import(home, &row, id, by),
        ("parameter-map", "lineage") => return crate::pmaps::undo_lineage(home, &row, id, by),
        ("parameter-map", "archive" | "restore") => return crate::pmaps::undo_move(home, &row, id, by),
        ("unit", "unregister") => return crate::learned_units::undo_unregister(home, &row, id, by),
        // A parameter edit's new version is set aside (or a new unit retired); its source was never
        // written.
        (crate::cparams::AUDIT_KIND, "edit") => return crate::cparams::undo(home, &row, id, by),
        // A parameter overlay change: what it added comes out, what it removed goes back.
        (crate::overlay::AUDIT_KIND, _) => return crate::overlay::undo(home, &row, id, by),
        ("memory", "pin" | "unpin") => return crate::memory_api::undo(home, &row, id, by),
        ("goal", "add" | "status") => return crate::memory_api::undo_goal(home, &row, id, by),
        // A reminder, loop or step of the agent's guidance put back as it was.
        ("guidance", _) => return crate::guidance::undo(home, &row, id, by),
        // 2026-11-26: a cartridge source added (taken out again) or removed (added again).
        ("cartridge-source", "add" | "remove") => return crate::sources::undo(home, &row, id, by),
        // The user's own backup repositories: added, taken off the list, or a restore.
        ("backup-repo", "add" | "remove" | "restore") => return crate::backups::undo(home, &row, id, by),
        // Distillation's teacher data: an import (its rows out again) or a label (the earlier ones back).
        ("distil", "import" | "label" | "values") => return crate::distil::undo(home, &row, id, by),
        // 2026-11-27: the base cartridge a MicroModel serves, and the model a role runs on, put back
        // as they were before the change (each row keeps its `from`).
        ("library-base", "choose") => return crate::library::undo_base(home, &row, id, by),
        ("role-model", "choose") => return crate::library::undo_role(home, &row, id, by),
        _ => {}
    }
    if row.get("kind").and_then(Value::string).ok() != Some("setting") {
        return Err(format!("{id} is not a setting change"));
    }
    let key = row.get("key").and_then(Value::string).map_err(|e| e.to_string())?;
    let from = row.get("from").and_then(Value::string).map_err(|e| e.to_string())?;
    set_by(home, key, from, Some(id), by)
}

/// Removes the dense index kept on disk for the project whose memory folder is `memory`
/// (`dense/vectors.bin`, `seg-*.bin`, `skin.json`, and any temporary files left by a stop), and
/// logs it in the audit log as a `dense-clear` row. Returns the rows and bytes removed and the
/// audit id. The next start of a server on that project embeds the material again.
/// # Errors
/// The file cannot be removed or the audit log cannot be written.
pub fn dense_clear(home: &Path, memory: &Path, project: &str, by: &str) -> Result<(u64, u64, String), String> {
    let dir = memory.join(nmsubs::dense::store::DIR);
    let rows = nmsubs::dense::store::disk_usage(&dir).map_or(0, |(r, _)| r);
    let bytes = nmsubs::dense::store::clear(&dir)?;
    let id = audit(
        home,
        "dense-clear",
        vec![
            ("project", s(project)),
            ("rows", Value::Number(rows.to_string())),
            ("bytes", Value::Number(bytes.to_string())),
            ("by", s(by)),
            ("label", s("Meaning index on disk cleared")),
        ],
    )?;
    Ok((rows, bytes, id))
}

/// Section check (temporary folder, no network): a cartridge switch passes the base-hash slot
/// check, is logged as a setting and is undone through the same path; a wrong base hash is
/// refused; audit ids written in the same millisecond differ.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let (a, b) = (new_id(7), new_id(7));
    if a == b {
        return Err("controls: two ids in one millisecond are equal".into());
    }
    let home = std::env::temp_dir().join(format!("nano-controls-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&home);
    let _ = std::fs::remove_dir_all(&home);
    r
}
fn check_in(home: &Path) -> Result<(), String> {
    let e = |x: std::io::Error| x.to_string();
    let base = "qwen3-0.6b";
    let sha = nmhash::sha256_bytes(b"base weights");
    std::fs::create_dir_all(home.join("weights").join(base)).map_err(e)?;
    std::fs::write(home.join("weights").join(base).join("part.json"), format!("{{\"weights\":{{\"sha256\":\"{sha}\"}}}}")).map_err(e)?;
    let unit_dir = home.join("units").join("u1");
    std::fs::create_dir_all(&unit_dir).map_err(e)?;
    std::fs::write(unit_dir.join("w.bin"), b"weights").map_err(e)?;
    let mut card = nmlearn::UnitCard::new("u1", base, &sha, "rerank", "nmengn.components#L12.down", 4, 1);
    card.status = "fitted".into();
    card.files = vec![("w.bin".into(), nmhash::sha256_bytes(b"weights"))];
    let mut bad_card = nmlearn::UnitCard::new("u2", base, &"0".repeat(64), "rerank", "nmengn.components#L3.q", 4, 1);
    bad_card.status = "fitted".into();
    let reg = Value::Array(vec![card.to_value(), bad_card.to_value()]).encode().map_err(|x| x.to_string())?;
    std::fs::write(home.join("units").join("registry.json"), reg).map_err(e)?;
    let id = set(home, "cartridge:u1", "on", None)?.ok_or("controls: switching u1 on wrote no row")?;
    if !nmlearn::Units::open(home).get("u1").is_some_and(|c| c.loaded) {
        return Err("controls: u1 is not loaded after switching it on".into());
    }
    undo(home, &id)?;
    if nmlearn::Units::open(home).get("u1").is_some_and(|c| c.loaded) {
        return Err("controls: undo did not switch u1 off".into());
    }
    if set(home, "cartridge:u2", "on", None).is_ok() {
        return Err("controls: a cartridge for other base weights was switched on".into());
    }
    let log = std::fs::read_to_string(audit_path(home)).map_err(e)?;
    if log.lines().count() != 2 || !log.contains("\"key\":\"cartridge:u1\"") || !log.contains("\"undo_of\"") {
        return Err(format!("controls: audit rows of the switch: {log}"));
    }
    // Clearing the dense index on disk: a store with a header naming 3 rows is removed and logged.
    let mem = home.join("memory").join("projects").join("p-00000000");
    std::fs::create_dir_all(mem.join("dense")).map_err(e)?;
    let header = b"dim=2\nrows=3\n";
    let mut file = b"NMDENSE1".to_vec();
    file.extend_from_slice(&u32::try_from(header.len()).map_err(|x| x.to_string())?.to_le_bytes());
    file.extend_from_slice(header);
    std::fs::write(mem.join("dense").join("vectors.bin"), &file).map_err(e)?;
    let (rows, bytes, _) = dense_clear(home, &mem, "p-00000000", "check")?;
    if rows != 3 || bytes != file.len() as u64 || mem.join("dense").join("vectors.bin").exists() {
        return Err(format!("controls: dense clear removed {rows} rows, {bytes} bytes"));
    }
    if !std::fs::read_to_string(audit_path(home)).map_err(e)?.contains("\"kind\":\"dense-clear\"") {
        return Err("controls: dense clear was not logged".into());
    }
    // Customizable integer budget allowances: accept non-preset integers within [min, max], reject
    // out-of-range or non-integer values, and undo back to default.
    let bid = set(home, "budget.pack_tokens", "5250", None)?.ok_or("controls: budget.pack_tokens custom int wrote no row")?;
    if current(home).get("budget.pack_tokens").map(String::as_str) != Some("5250") {
        return Err("controls: budget.pack_tokens did not save custom integer 5250".into());
    }
    if set(home, "budget.pack_tokens", "500", None).is_ok()
        || set(home, "budget.pack_tokens", "99999", None).is_ok()
        || set(home, "budget.pack_tokens", "abc", None).is_ok()
    {
        return Err("controls: budget.pack_tokens accepted out-of-range or non-integer value".into());
    }
    undo(home, &bid)?;
    if current(home).get("budget.pack_tokens").map(String::as_str) != Some("4000") {
        return Err("controls: undo did not restore budget.pack_tokens to 4000".into());
    }
    let fid = set(home, "budget.focused_tokens", "2250", None)?.ok_or("controls: budget.focused_tokens custom int wrote no row")?;
    if current(home).get("budget.focused_tokens").map(String::as_str) != Some("2250") {
        return Err("controls: budget.focused_tokens did not save custom integer 2250".into());
    }
    undo(home, &fid)?;
    let mid = set(home, "mcp.tools", "lean", None)?.ok_or("controls: mcp.tools lean wrote no row")?;
    if current(home).get("mcp.tools").map(String::as_str) != Some("lean") {
        return Err("controls: mcp.tools did not save lean".into());
    }
    undo(home, &mid)?;
    let tcid = set(home, "mcp.tool_count", "8", None)?.ok_or("controls: mcp.tool_count wrote no row")?;
    if current(home).get("mcp.tool_count").map(String::as_str) != Some("8") {
        return Err("controls: mcp.tool_count did not save 8".into());
    }
    undo(home, &tcid)?;
    let qid = set(home, "quota.session_tokens", "75000", None)?.ok_or("controls: quota.session_tokens wrote no row")?;
    if current(home).get("quota.session_tokens").map(String::as_str) != Some("75000") {
        return Err("controls: quota.session_tokens did not save 75000".into());
    }
    undo(home, &qid)?;
    let musc_id = set(home, "muscle", "off", None)?.ok_or("controls: muscle off wrote no row")?;
    if current(home).get("muscle").map(String::as_str) != Some("off") {
        return Err("controls: muscle did not save off".into());
    }
    undo(home, &musc_id)?;
    if current(home).get("muscle").map(String::as_str) != Some("on") {
        return Err("controls: undo did not restore muscle to on".into());
    }
    let ctrl_id = set(home, "mcp.tool_control", "on", None)?.ok_or("controls: mcp.tool_control on wrote no row")?;
    if current(home).get("mcp.tool_control").map(String::as_str) != Some("on") {
        return Err("controls: mcp.tool_control did not save on".into());
    }
    undo(home, &ctrl_id)?;
    if current(home).get("mcp.tool_control").map(String::as_str) != Some("off") {
        return Err("controls: undo did not restore mcp.tool_control to off".into());
    }
    let etid = set(home, "mcp.enabled_tools", "nm_build_run,nm_context_pack", None)?.ok_or("controls: mcp.enabled_tools wrote no row")?;
    if current(home).get("mcp.enabled_tools").map(String::as_str) != Some("nm_context_pack,nm_build_run") {
        return Err("controls: mcp.enabled_tools did not normalize priority order".into());
    }
    if set(home, "mcp.enabled_tools", "nm_not_a_real_tool", None).is_ok() || set(home, "mcp.enabled_tools", "", None).is_ok() {
        return Err("controls: mcp.enabled_tools accepted invalid tool name or empty list".into());
    }
    undo(home, &etid)?;
    if current(home).get("mcp.enabled_tools").map(String::as_str) != Some("auto") {
        return Err("controls: undo did not restore mcp.enabled_tools to auto".into());
    }
    // Roadmap Phase 4.5 (2026-11-25): the hardware limits take auto (any case), 0 for the card
    // and numbers in bounds, refuse the rest, and leave the homeostasis ceiling alone.
    let vid = set(home, "limits.vram_mb", "0", None)?.ok_or("controls: limits.vram_mb 0 wrote no row")?;
    let rid = set(home, "limits.ram_mb", "8192", None)?.ok_or("controls: limits.ram_mb wrote no row")?;
    let tid = set(home, "limits.cpu_threads", "4", None)?.ok_or("controls: limits.cpu_threads wrote no row")?;
    if nmtype::limits::read_at(home) != (nmtype::limits::Limits { vram_mb: Some(0), ram_mb: Some(8192), cpu_threads: Some(4) }) {
        return Err("controls: the engine reads other limits than were saved".into());
    }
    if set(home, "limits.ram_mb", "100", None).is_ok() || set(home, "limits.cpu_threads", "0", None).is_ok() || set(home, "limits.vram_mb", "lots", None).is_ok() {
        return Err("controls: a hardware limit accepted a value out of bounds".into());
    }
    if set(home, "limits.vram_mb", "AUTO", None)?.is_none() || current(home).get("limits.vram_mb").map(String::as_str) != Some("auto") {
        return Err("controls: limits.vram_mb did not take auto".into());
    }
    if current(home).get("budget.gpu_mb_max").map(String::as_str) != Some("auto") {
        return Err("controls: a hardware limit changed the homeostasis ceiling".into());
    }
    // Memory areas (2026-11-27): an area within the budget is taken and read back; areas set
    // past the process budget (limits.ram_mb 8192 here) are refused, and so is a host limit
    // below the areas set.
    let cid = set(home, "limits.memory.content_mb", "2048", None)?.ok_or("controls: limits.memory.content_mb wrote no row")?;
    if nmtype::limits::read_areas_at(home).get("content") != Some(2048) {
        return Err("controls: limits.memory.content_mb 2048 was not read back".into());
    }
    if set(home, "limits.memory.weights_mb", "8192", None).is_ok() {
        return Err("controls: memory areas past the process budget were accepted".into());
    }
    if set(home, "limits.ram_mb", "1024", None).is_ok() {
        return Err("controls: a host memory limit below the memory areas set was accepted".into());
    }
    undo(home, &cid)?;
    for id in [tid, rid, vid] {
        undo(home, &id)?;
    }
    if nmtype::limits::read_at(home) != nmtype::limits::Limits::AUTO {
        return Err("controls: undo did not restore the hardware limits to auto".into());
    }
    // Air-gap mode forces network off, blocks re-enabling network, and reports verifiable proof.
    let _ = set(home, "network", "view", None)?;
    let agid = set(home, "air_gap", "on", None)?.ok_or("controls: air_gap on wrote no row")?;
    if !air_gap_on(home) || current(home).get("network").map(String::as_str) != Some("off") {
        return Err("controls: air_gap on did not force network off".into());
    }
    if set(home, "network", "view", None).is_ok() {
        return Err("controls: network was allowed while air_gap is on".into());
    }
    let agv = air_gap_view(home, 0);
    if !matches!(agv.get("air_gap"), Ok(Value::Bool(true)))
        || !matches!(agv.get("loopback_only"), Ok(Value::Bool(true)))
        || !matches!(agv.get("outbound_network_code"), Ok(Value::Bool(false)))
        || !matches!(agv.get("remote_imports_allowed"), Ok(Value::Bool(false)))
    {
        return Err(format!("controls: air_gap_view unexpected: {agv:?}"));
    }
    undo(home, &agid)?;
    if air_gap_on(home) && !air_gap_env() {
        return Err("controls: undo did not restore air_gap to off".into());
    }
    Ok(())
}

/// Section check `nmcore.parity`: the interface, the API and MCP stay in step.
///
/// - Every control has a valid default, one row in the view, and is set through the route
///   `nm_settings` `set` writes (`POST /api/controls`).
/// - Every key `settings.json` holds beside the controls (`SAVED_KEYS`) is in the saved view
///   (`GET /api/controls/saved`), each settable one through `POST /api/controls` with
///   `<key>:<entry>`; the other keys of the file are kept; a key nothing reads is named `unknown`.
/// - Every route the parity table names (`nmapi::parity::OPS`, `OTHER_TOOL_ROUTES`) is answered
///   by the interface's routes, and every route the interface server's source names is in the
///   table, served by another tool, or listed as not in MCP with its reason.
/// - In a development tree, `docs/configuration.md` names every control and saved key.
/// # Errors
/// One fault line.
pub fn check_parity() -> Result<(), String> {
    let home = std::env::temp_dir().join(format!("nano-parity-check-{}-{}", std::process::id(), nmtype::now_ms()));
    std::fs::create_dir_all(&home).map_err(|e| e.to_string())?;
    let r = check_parity_in(&home);
    let _ = std::fs::remove_dir_all(&home);
    r
}
fn check_parity_in(home: &Path) -> Result<(), String> {
    use nmapi::parity::{NOT_IN_MCP, OPS, OTHER_TOOL_ROUTES};
    let body = |pairs: &[(&str, &str)]| Value::Object(pairs.iter().map(|(k, v)| ((*k).to_owned(), s(v))).collect()).encode().map(String::into_bytes).map_err(|e| e.to_string());
    // Controls: defaults, rows, and the write route MCP borrows.
    let view = view(home);
    let rows = view.array().map_err(|e| e.to_string())?;
    for (i, c) in CONTROLS.iter().enumerate() {
        if CONTROLS.iter().skip(i + 1).any(|d| d.key == c.key) {
            return Err(format!("parity: control {} is defined twice", c.key));
        }
        if validate(c, c.default).as_deref() != Ok(c.default) {
            return Err(format!("parity: control {}: its default {} is not a value it takes", c.key, c.default));
        }
        if rows.iter().filter(|r| r.get("key").and_then(Value::string).is_ok_and(|k| k == c.key)).count() != 1 {
            return Err(format!("parity: control {} has no row in /api/controls", c.key));
        }
        if c.label.is_empty() || c.help.is_empty() || c.group.is_empty() || c.applies.is_empty() {
            return Err(format!("parity: control {} lacks a label, help, group or when it applies", c.key));
        }
        // The air gap cannot be set off while the environment locks it on.
        if c.key == "air_gap" && air_gap_env() {
            continue;
        }
        crate::ui::lent_route(home, "/api/controls", "", Some(&body(&[("key", c.key), ("value", c.default)])?), "check").map_err(|e| format!("parity: control {} is not settable through POST /api/controls: {e}", c.key))?;
    }
    // The saved file: every key beside the controls is shown, the settable ones through one route.
    let model = nmtype::role_models::ROLES[0].1;
    std::fs::write(
        settings_path(home),
        format!("{{\"base_cartridges\": {{\"{model}\": \"open\"}}, \"base_compensation\": {{}}, \"role_models\": {{}}, \"memory\": \"on\", \"note\": \"x\", \"nudges\": \"quiet\", \"left_by_another_build\": 1}}"),
    )
    .map_err(|e| e.to_string())?;
    let saved = saved_view(home);
    let keys = saved.get("keys").and_then(Value::array).map_err(|e| e.to_string())?;
    for k in SAVED_KEYS {
        let row = keys.iter().find(|r| r.get("key").and_then(Value::string).is_ok_and(|x| x == k.key)).ok_or_else(|| format!("parity: saved key {} is not in /api/controls/saved", k.key))?;
        let read_only = row.get("read_only").and_then(Value::string).is_ok();
        if read_only != k.read_only.is_some() || (k.read_only.is_none() && (k.entry.is_empty() || k.route.is_empty() || k.mcp.is_empty() || k.interface.is_empty())) {
            return Err(format!("parity: saved key {} is neither settable on every surface nor read only with a reason", k.key));
        }
        if find(k.key).is_some() {
            return Err(format!("parity: {} is both a control and a saved key", k.key));
        }
    }
    let unknown: Vec<&str> = saved.get("unknown").and_then(Value::array).unwrap_or_default().iter().filter_map(|x| x.string().ok()).collect();
    if unknown != ["left_by_another_build"] {
        return Err(format!("parity: the saved view names these keys unknown: {unknown:?} (a key the file may hold belongs in CONTROLS or SAVED_KEYS)"));
    }
    let comp = format!("{COMPENSATION_KEY}{model}");
    let on = crate::ui::lent_route(home, "/api/controls", "", Some(&body(&[("key", &comp), ("value", "on")])?), "check").map_err(|e| format!("parity: {comp}: {e}"))?;
    let after = std::fs::read(settings_path(home)).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or("parity: settings.json unreadable after a compensation change")?;
    let text_at = |v: &Value, a: &str, b: &str| v.get(a).and_then(|m| m.get(b)).and_then(Value::string).map(str::to_owned).ok();
    if text_at(&after, "base_compensation", model).as_deref() != Some("on") || text_at(&after, "base_cartridges", model).as_deref() != Some("open") || after.get("left_by_another_build").is_err() {
        return Err("parity: setting a compensation entry did not save it, or dropped another key of the file".into());
    }
    let id = on.get("audit_id").and_then(Value::string).map_err(|_| "parity: a compensation change wrote no audit row".to_owned())?;
    undo_by(home, id, "check")?;
    if saved_compensation(home).contains_key(model) {
        return Err("parity: undo did not take the compensation entry out".into());
    }
    if set_by(home, &comp, "maybe", None, "check").is_ok() || set_by(home, &format!("{COMPENSATION_KEY}no-such-model"), "on", None, "check").is_ok() {
        return Err("parity: a compensation entry took a value or a model it must refuse".into());
    }
    // A control write keeps the keys beside the controls.
    set_by(home, "muscle", "off", None, "check")?;
    let after = std::fs::read(settings_path(home)).ok().and_then(|b| nmjson::parse(&b).ok()).ok_or("parity: settings.json unreadable after a control change")?;
    if text_at(&after, "base_cartridges", model).as_deref() != Some("open") || after.get("role_models").is_err() || after.get("base_compensation").is_err() {
        return Err("parity: a control change dropped a key beside the controls".into());
    }
    // Routes, forward: what the table names, the interface answers. A write is sent an empty
    // object in this empty home, so a known route stops at its first missing field.
    let known = |method: &str, path: &str| -> bool {
        // The live state is answered by the server itself, before the route function.
        if path == "/api/live" {
            return true;
        }
        let r = if method == "GET" { crate::ui::lent_route(home, path, "", None, "check") } else { crate::ui::lent_route(home, path, "", Some(b"{}"), "check") };
        !r.is_err_and(|e| e.starts_with("unknown route") || e == "unknown endpoint")
    };
    for (tool, op, method, path) in OPS {
        if !known(method, path) {
            return Err(format!("parity: {tool} {op} names {method} {path}, which the interface does not answer"));
        }
    }
    for (method, path, tool) in OTHER_TOOL_ROUTES {
        if !known(method, path) {
            return Err(format!("parity: {method} {path} ({tool}) is not answered by the interface"));
        }
    }
    // Routes, backward: every route the interface server's source names is in the table, served
    // by another tool, or listed as not in MCP with its reason.
    let source = include_str!("ui.rs");
    let mut at = 0;
    while let Some(i) = source.get(at..).and_then(|rest| rest.find("\"/api/")) {
        let start = at + i + 1;
        let len = source.get(start..).and_then(|rest| rest.find('"')).unwrap_or(0);
        let path = source.get(start..start + len).unwrap_or("");
        at = start + len.max(1);
        // A prefix (`/api/`, `/api/storage/`) is not a route; its routes are in the table.
        if path.ends_with('/') || path.is_empty() {
            continue;
        }
        let listed = OPS.iter().any(|r| r.3 == path) || OTHER_TOOL_ROUTES.iter().any(|r| r.1 == path) || NOT_IN_MCP.iter().any(|r| r.1 == path);
        if !listed {
            return Err(format!("parity: the interface answers {path}, which no MCP op reaches: add its op to nmapi::parity (OPS), or list it in NOT_IN_MCP with the reason"));
        }
    }
    // The documents, where the tree has them.
    if let Ok(doc) = std::fs::read_to_string(nmtype::nano_root().join("docs").join("configuration.md")) {
        let missing: Vec<&str> = CONTROLS.iter().map(|c| c.key).chain(SAVED_KEYS.iter().map(|k| k.key)).filter(|k| !doc.contains(&format!("`{k}`"))).collect();
        if !missing.is_empty() {
            return Err(format!("parity: docs/configuration.md does not name these settings: {}", missing.join(", ")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn interface_api_and_mcp_parity() {
        super::check_parity().unwrap();
    }
}
