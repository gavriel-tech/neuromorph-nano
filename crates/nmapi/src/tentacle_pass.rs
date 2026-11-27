//! The tentacle pass (2026-11-10; designs by Luis Jake Gabriel III,
//! `docs/2026-11-10-neumor-design-e3-tentacles.md` and
//! `docs/2026-11-10-neumor-design-g-layers.md`, build step 1): eight tentacles, each reading
//! its own 12,000-token lens window with the drone heads of the released Qwen3-0.6B base.
//! No chunk sits in two windows and no tentacle sees another's window; each forwards its best
//! chunks, a judge orders them, and the bridge hands them to the Core Brain as `chunk` packets
//! keyed `tentacle:<place>:<file>:<line>` with one window record per tentacle for the GCX context map.
//!
//! The reading method is the one measured in E3.3 (`runs/2026-11-10-E3-3-runbook.md`): chunks
//! are encoded once behind a short anchor and kept in RAM by content key, a window is the
//! encoded chunks in file order, and a read is the question posed after the window with the
//! drone heads' attention mass kept per head and position, beside the same read with no question.
//! Nothing is written to disk and no parameter set is added (released weights as released).
//!
//! Chunk scores (`Score`; derivation in `docs/2026-11-10-neumor-research-reading-statistic.md`).
//! A head's attention over the window is a probability, `a(p) = exp(c(q, p) + b(p)) / Z`, with
//! `c` the content match and `b` what the position and the token draw under any question. The
//! read with no question gives `a0(p) = exp(c0(p) + b(p)) / Z0`. `Mass` (E2, E3.3) sums
//! `a - a0` over a chunk: `b` stays in the score as a factor and the sum grows with the
//! chunk's length. `Density` divides that by the chunk's tokens (length only). `Evidence`
//! takes `ln(sum a / sum a0)` over the chunk: `b` and the length cancel, and what is left is
//! the content match plus `ln(Z0 / Z)`, one constant per window. The anchor is the same text
//! in every window, so its own ratio measures that constant; subtracting it puts every
//! window's scores on one scale with nothing passed between tentacles. `Peak` is the best
//! 32-token block of a chunk by the same measure, for answers that are a few lines of a long
//! chunk.
//!
//! The read is split in two: `scan` (encode, read; kept for the last question) and the judge
//! (shortlist, reranker, bridge), so scores, shortlists and judges are compared on one scan.
//!
//! Kept encoded chunks (2026-11-10, speed fix after G1, `runs/2026-11-10-neumor-note-g1-result.md`):
//! chunks are encoded only through the last layer any drone head reads (`drone_last_layer`,
//! derived from the heads in use) and keep keys and values for those layers only; the RAM limit
//! of the kept chunks is derived from free physical memory when the cache is made
//! (`kvstore::derive_ram_limit`), unless `Settings::cache_bytes` names one. Each scan records
//! bytes per token, the last layer kept, the limit, the free RAM read, the bytes held and the
//! tokens encoded.
use nmbrain::gmap::WindowRecord;
use nmbrain::slot::Slots;
use nmbrain::synthesis::{DENSE_STEP, FLOOR, TENTACLE_KEY, TENTACLE_TOP};
use nmengn::kvstore::{self, ChunkCache, ChunkKv, DroneSet, KeptReads, Probe, RamLimit, Window, encode, encode_through};
use nmengn::model::Model;
use nmengn::model::RunOpts;
use nmengn::placement::Device;
use nmsubs::Substrate;
use nmsubs::lens::{self, BUDGET};
use nmtype::{Category, Fault, FaultKind, Packet, Provenance, Stance, Status};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.tentacle_pass", FaultKind::Execution, m)
}

/// Each tentacle's focus, in `lens::NAMES` order (the E3.3 poses).
pub const FOCUS: [&str; 8] = [
    "the code that does this",
    "the lines that use these words",
    "where these names are defined",
    "the functions that call or use this",
    "what changed recently for this",
    "how this module is put together",
    "the part of the file that answers this",
    "what was found before for this",
];

/// The anchor every chunk is encoded behind when the studied repository is the Neuromorph Nano
/// itself. Fixed model input text: changing it changes the encodings and the section checks of
/// this pass; the lab tool keeps the earlier anchor for reproducing earlier runs.
pub const SELF_ANCHOR: &str =
    "Repository neuromorph-nano: a Rust workspace (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";

/// Who orders the forwarded candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Judge {
    /// The drone heads' calibrated chunk score.
    Heads,
    /// The released reranker's P(yes); falls back to the heads when no such reranker is bound.
    Reranker,
    /// Reciprocal-rank fusion of both (E3.3 rule R1).
    Rrf,
    /// P(yes) plus the head score over the best head score (E3.3 rule R2).
    Sum,
    /// P(yes) inside each place: every tentacle's first choice before any second (E3.3 rule R3).
    WithinPlace,
}
impl Judge {
    /// From a flag value.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "heads" => Self::Heads,
            "reranker" => Self::Reranker,
            "rrf" => Self::Rrf,
            "sum" => Self::Sum,
            "place" => Self::WithinPlace,
            _ => return None,
        })
    }
    /// The flag value.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Heads => "heads",
            Self::Reranker => "reranker",
            Self::Rrf => "rrf",
            Self::Sum => "sum",
            Self::WithinPlace => "place",
        }
    }
}

/// How a tentacle turns its drone heads' attention into a chunk score (module notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// Posed mass minus content-free mass, summed over the chunk (E2, E3.3).
    Mass,
    /// `Mass` per token of the chunk.
    Density,
    /// Log of posed over content-free mass on the chunk, summed over heads, minus the anchor's.
    Evidence,
    /// The chunk's best 32-token block by the same measure.
    Peak,
    /// Tentacle core score (2026-11-10, design `docs/2026-11-10-neumor-design-tentacle-core-v1.md`
    /// core line 3): per drone head, the log-sum-exp of the raw attention scores (before
    /// softmax) over the chunk's positions, averaged over the question tokens, posed minus
    /// content-free at the same positions, summed over the heads. No window-wide normaliser
    /// enters, so a score has the same meaning in every tentacle. Needs a scan with logits
    /// (`Settings::wants_logits`); not in `ALL`, so the runners that walk `ALL` are unchanged.
    Calibrated,
}
impl Score {
    /// Every score, in report order.
    pub const ALL: [Self; 4] = [Self::Mass, Self::Density, Self::Evidence, Self::Peak];
    /// Scores added after `ALL` was fixed by the locked runners (2026-11-10: `Calibrated`).
    pub const EXTRA: [Self; 1] = [Self::Calibrated];
    /// From a flag value.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().chain(Self::EXTRA).find(|x| x.name() == s)
    }
    /// The flag value.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Mass => "mass",
            Self::Density => "density",
            Self::Evidence => "evidence",
            Self::Peak => "peak",
            Self::Calibrated => "calibrated",
        }
    }
}

/// Tokens in a `Score::Peak` block.
pub const BLOCK: usize = 32;
/// Keeps a log ratio finite where a head puts no mass at all.
const EPS: f32 = 1e-9;

/// The pass's settings.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Each tentacle reads with its own focus (else every tentacle reads the plain question).
    pub focused: bool,
    /// Who orders the candidates.
    pub judge: Judge,
    /// The drone heads' chunk score.
    pub score: Score,
    /// Chunks each tentacle forwards.
    pub per_tentacle: usize,
    /// Chunks each tentacle shortlists by its drone score. When this is more than
    /// `per_tentacle` and a reranker judges, the reranker chooses the forwarded chunks from
    /// the shortlist, inside the tentacle (pointwise scores: nothing passes between tentacles).
    pub shortlist: usize,
    /// Add the drone heads' key line spans to each forwarded chunk's evidence.
    pub lines: bool,
    /// RAM for encoded chunks in bytes; 0 (the default since 2026-11-10) derives it from free
    /// physical memory when the cache is made (`kvstore::derive_ram_limit`).
    pub cache_bytes: usize,
    /// Tentacle core first form (2026-11-10, `crate::tentacle_core`): one core per tentacle
    /// reports through its `g0_tx` to the G1 loop, and G1's hand-over replaces the per-tentacle
    /// cut. Off by default; the pass above stays the control and the fallback.
    pub core: bool,
    /// Keys-only scan (2026-11-10, `crate::key_scan`, `--tentacle-scan keys`): the drone heads'
    /// raw scores come from kept drone-key packets and the question path's queries instead of
    /// the exact read. Off by default (the exact scan is the default and the fallback); used
    /// only with the tentacle core or the calibrated score (it gives no attention mass).
    pub keys: bool,
    /// Span read (2026-11-10, test T6, `runs/DX-t6-*.txt`): with the core on, each window is also
    /// read as consecutive spans of about this many tokens from its kept encodings (each span its
    /// own short read behind the anchor, no re-encoding), the best `SPAN_KEEP` chunks of each span
    /// read once more together, and the core's tables come from those short reads. 0 (the
    /// default): the full-window read only (the first form, kept for comparison).
    pub span: usize,
    /// Refinement step 2a (2026-11-10, `crate::core_read`): the selectable components of the
    /// tentacle core's read (second look, token votes, length-matched content-free pose). All
    /// off by default: the first form.
    pub read: crate::core_read::ReadOptions,
    /// Build step T1 (2026-11-10, `crate::drone_windows`, `--tentacle-windows drones`): the
    /// windows are composed from the drones' flags plus relations instead of the lens windows.
    /// `Lens` (the default) keeps today's windows.
    pub windows: crate::drone_windows::WindowMode,
}

/// Chunks of each span that go to the final short read of a span read.
pub const SPAN_KEEP: usize = 2;
impl Default for Settings {
    fn default() -> Self {
        // The E3.3 decisions (`runs/2026-11-10-neumor-note-e3-3-judge.md`): plain poses, the
        // reranker judges, chunks without line spans. Score and shortlist are as E3.3 ran
        // them until G1 decides (`runs/2026-11-10-G1-runbook.md`).
        // 2026-11-10: G1 rule D2 chose shortlist 6 (`runs/2026-11-10-neumor-note-g1-result.md`),
        // but the locked runner `g1.rs` takes its served-phase shortlist from this default, so
        // the default stays 2 until that is settled; the served shortlist is set by the flag.
        // 2026-11-10: `cache_bytes` 0 derives the RAM limit at run time (was a fixed 8 GiB, now
        // only the fallback when free memory cannot be read).
        Self {
            focused: false,
            judge: Judge::Reranker,
            score: Score::Mass,
            per_tentacle: 2,
            shortlist: 2,
            lines: false,
            cache_bytes: 0,
            core: false,
            keys: false,
            span: 0,
            read: crate::core_read::ReadOptions { second_look: false, votes: false, matched_pose: false },
            windows: crate::drone_windows::WindowMode::Lens,
        }
    }
}

/// The shortlist the served path uses (2026-11-10, step P0): G1 rule D2 chose 6
/// (`runs/2026-11-10-neumor-note-g1-result.md`). Only `serve --tentacles` and
/// `trace --tentacles` take it (`from_args`); `Settings::default()` keeps 2 because the locked
/// runner `g1.rs` reads its served-phase shortlist from the default. An explicit
/// `--tentacle-shortlist` still wins.
pub const SERVED_SHORTLIST: usize = 6;

impl Settings {
    /// The served settings: the defaults with the G1 shortlist (`SERVED_SHORTLIST`).
    #[must_use]
    pub fn served() -> Self {
        Self { shortlist: SERVED_SHORTLIST, ..Self::default() }
    }
    /// The scan also records raw attention scores (tentacle core A5): for the calibrated score
    /// and for the tentacle core.
    #[must_use]
    pub fn wants_logits(&self) -> bool {
        self.core || self.score == Score::Calibrated
    }
}

/// One forwarded candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Cand {
    /// The tentacle (lens index) that forwarded it.
    pub tentacle: usize,
    /// Chunk index in the substrate.
    pub chunk: usize,
    /// Drone score of the chunk in its window (`Settings::score`).
    pub score: f32,
    /// Its place in its tentacle (0 best).
    pub place: usize,
    /// Its source lines: (line, calibrated score, tokens).
    pub lines: Vec<(usize, f32, usize)>,
}

/// What one pass produced.
#[derive(Clone, Debug, Default)]
pub struct Pass {
    /// The judged candidates as `chunk` packets, best first.
    pub packets: Vec<Packet>,
    /// One record per tentacle window, for the GCX context map.
    pub windows: Vec<WindowRecord>,
    /// The judged chunk indices, best first (tests compare these with a run record).
    pub order: Vec<usize>,
    /// One line for the trace: device, tokens, times.
    pub note: String,
    /// Refinement step 1: G1's evidence set for the question (forwarded records and the queue),
    /// passed on to the reduce step and G2 under the queue rule; `None` on the first form.
    pub evidence: Option<Arc<Mutex<nmbrain::evidence::EvidenceSet>>>,
}

/// One tentacle's read of its window, kept whole so any score can be taken from it.
#[derive(Debug, Default)]
pub struct WindowScan {
    /// The tentacle (lens index).
    pub tentacle: usize,
    /// Its chunks in window order (file, then line).
    pub place: Vec<usize>,
    /// (first position, tokens) of the anchor, then of every chunk of `place`.
    pub parts: Vec<(usize, usize)>,
    /// Window length in tokens.
    pub len: usize,
    /// Attention mass per question token of every drone head on every position (`heads x len`).
    pub posed: Vec<f32>,
    /// The same for the read with no question.
    pub null: Vec<f32>,
    /// Tentacle core A5 and the calibrated score (2026-11-10): per drone head (rows) and per
    /// part of `parts` (columns, anchor first), the log-sum-exp of the raw attention scores over
    /// the part's positions, averaged over the question tokens; empty when the scan recorded no
    /// logits. `cal_null` is the same for the read with no question.
    pub cal_posed: Vec<f32>,
    pub cal_null: Vec<f32>,
    /// The window itself (chunks by reference), kept only for the tentacle core's re-reads.
    pub window: Option<Arc<Window>>,
    /// Refinement step 1 (2026-11-10): each chunk of `place`'s rank in its lens's assignment
    /// order (0 first; the lens evidence that chose it, kept through the sort into file order),
    /// recorded on the tentacle core path; empty when not recorded (older scans and dumps).
    pub lens_rank: Vec<usize>,
}
impl WindowScan {
    fn heads(&self) -> usize {
        self.posed.len().checked_div(self.len).unwrap_or(0)
    }
    fn sums(&self, h: usize, at: usize, len: usize) -> (f32, f32) {
        let seg = |v: &[f32]| v.get(h * self.len + at..h * self.len + at + len).map_or(0.0, |s| s.iter().sum::<f32>());
        (seg(&self.posed), seg(&self.null))
    }
    /// Posed minus content-free mass on a span, summed over the heads.
    fn mass(&self, at: usize, len: usize) -> f32 {
        (0..self.heads())
            .map(|h| {
                let (a, b) = self.sums(h, at, len);
                a - b
            })
            .sum()
    }
    /// Log of posed over content-free mass on a span, summed over the heads.
    fn log_ratio(&self, at: usize, len: usize) -> f32 {
        (0..self.heads())
            .map(|h| {
                let (a, b) = self.sums(h, at, len);
                ((a.max(0.0) + EPS) / (b.max(0.0) + EPS)).ln()
            })
            .sum()
    }
    /// The calibrated score of every part (anchor first): per head posed minus content-free
    /// log-sum-exp, summed over the heads. Empty without logits.
    #[must_use]
    pub fn calibrated_parts(&self) -> Vec<f32> {
        calibrated(&self.cal_posed, &self.cal_null, self.parts.len())
    }
    /// Drone heads behind the calibrated score (0 without logits).
    #[must_use]
    pub fn cal_heads(&self) -> usize {
        self.cal_posed.len().checked_div(self.parts.len()).unwrap_or(0)
    }
    /// The anchor's log ratio: the window's own constant (module notes).
    #[must_use]
    pub fn anchor(&self) -> f32 {
        self.parts.first().map_or(0.0, |&(at, len)| self.log_ratio(at, len))
    }
    /// The score of every chunk of `place`, in that order.
    #[must_use]
    pub fn scores(&self, score: Score) -> Vec<f32> {
        if score == Score::Calibrated {
            let c = self.calibrated_parts();
            if c.is_empty() {
                return vec![f32::NEG_INFINITY; self.parts.len().saturating_sub(1)];
            }
            return c.into_iter().skip(1).collect();
        }
        let anchor = self.anchor();
        self.parts
            .iter()
            .skip(1)
            .map(|&(at, len)| match score {
                Score::Mass => self.mass(at, len),
                Score::Density => {
                    #[expect(clippy::cast_precision_loss, reason = "chunk token counts are small")]
                    let n = len.max(1) as f32;
                    self.mass(at, len) / n
                }
                Score::Evidence => self.log_ratio(at, len) - anchor,
                Score::Calibrated => f32::NEG_INFINITY,
                Score::Peak => {
                    let (mut best, mut o) = (f32::NEG_INFINITY, 0_usize);
                    loop {
                        let last = o + BLOCK >= len;
                        let start = if last { len.saturating_sub(BLOCK) } else { o };
                        best = best.max(self.log_ratio(at + start, BLOCK.min(len)));
                        if last {
                            break;
                        }
                        o += BLOCK;
                    }
                    best - anchor
                }
            })
            .collect()
    }
    /// (index into `place`, score), best first (ties: window order).
    #[must_use]
    pub fn ranked(&self, score: Score) -> Vec<(usize, f32)> {
        let mut v: Vec<(usize, f32)> = self.scores(score).into_iter().enumerate().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }
    /// Posed minus content-free mass of every token of chunk `k` of `place`, summed over heads.
    fn token_mass(&self, k: usize) -> Vec<f32> {
        let Some(&(at, len)) = self.parts.get(k + 1) else { return Vec::new() };
        (0..len).map(|j| self.mass(at + j, 1)).collect()
    }
}

/// Everything one question's reads produced, before any judge.
#[derive(Debug, Default)]
pub struct Scan {
    /// The question read.
    pub question: String,
    /// Focused poses (else plain).
    pub focused: bool,
    /// The reads recorded raw attention scores (tentacle core A5).
    pub logits: bool,
    /// The substrate version the windows were laid out on.
    pub version: u64,
    /// Chunks in the substrate then.
    pub chunks: usize,
    /// One read per tentacle with a window.
    pub windows: Vec<WindowScan>,
    /// One record per tentacle (what its window held), before the judge fills in its findings.
    pub records: Vec<WindowRecord>,
    /// The device the reads ran on.
    pub device: String,
    /// Tokens the windows hold.
    pub window_tokens: usize,
    /// Tokens encoded for this scan (the rest were held).
    pub enc_tokens: usize,
    /// Encoding time.
    pub enc_ms: f64,
    /// Bytes one kept token holds (keys and values of the layers kept).
    pub bytes_per_token: usize,
    /// The last layer kept per chunk (the highest layer a drone head reads).
    pub last_layer: usize,
    /// Where the drone sets came from (2026-11-10, step P0): `designs/drones.json` or the
    /// fallback constant, with the reason.
    pub drones: String,
    /// The RAM limit of the kept chunks, how it was set, and the free RAM read then.
    pub limit: RamLimit,
    /// Bytes of kept chunks after this scan.
    pub held_bytes: usize,
    /// Chunks kept after this scan.
    pub held_chunks: usize,
    /// Of `enc_tokens`, tokens of chunks this pass had encoded before and then dropped at the
    /// limit (the rest were never encoded in this pass).
    pub enc_again_tokens: usize,
    /// Read time.
    pub read_ms: f64,
    /// Step P2: no-question reads taken from the kept reads (windows whose content did not
    /// change) and reads computed; kept reads dropped because their window changed.
    pub null_kept: usize,
    pub null_read: usize,
    pub null_dropped: usize,
    /// Step P2: seconds the pass waited at the device gate, and for which job.
    pub gate_wait: f64,
    pub gate_for: String,
    /// Step P2: the device budget after the reads (residency manager), and the pool's figures.
    pub budget: String,
    /// The reranker's P(yes) per chunk index, filled as judges ask.
    p: Mutex<BTreeMap<usize, f32>>,
    /// Shared RAM budget (2026-11-10): bytes of encodings the kept windows hold alive (the
    /// tentacle core's re-reads), and bytes alive in the chunk store plus those after the fit.
    pub kept_bytes: usize,
    pub alive_bytes: usize,
    /// Keys-only scan (2026-11-10): what it read and how (empty for an exact scan).
    pub keys: String,
    /// Keys-only scan (2026-11-10): the windows hold outlines of chunks whose full keys and
    /// values were not kept; a tentacle core's re-read fills them through this (encoding only
    /// the chunks it reads). `None` for an exact scan, whose windows hold every encoding.
    pub fill: Option<Fill>,
    /// Refinement step 1 (2026-11-10, item 5): lenses whose empty list took the fallback fill
    /// (`nmsubs::lens::fill_empty`), and the chunks memory recalled for the memory lens.
    pub filled: Vec<usize>,
    pub recalled: usize,
    /// Refinement step 2a (2026-11-10): how the content-free pose was made when the
    /// length-matched pose is on (empty: the first form's plain pose).
    pub pose: String,
    /// Build step T1 (2026-11-10): the windows were composed from drone flags.
    pub drone_windows: bool,
    /// One tentacle system (2026-11-19): the key of the Team's windows read (`TeamWindows::key`;
    /// 0 when the pass laid out its own lens windows).
    pub team: u64,
}

/// Fills chunk encodings a keys-only scan did not keep (2026-11-10): given the parts of a read
/// (kept encodings or outlines without keys and values), returns them with every outline
/// replaced by its full encoding through the drone layers. Called under the device gate.
#[derive(Clone)]
pub struct Fill(pub Arc<FillFn>);
/// The function behind `Fill`.
pub type FillFn = dyn Fn(&[Arc<ChunkKv>]) -> Result<Vec<Arc<ChunkKv>>, Fault> + Send + Sync;
impl std::fmt::Debug for Fill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Fill")
    }
}
impl Scan {
    /// Reranker scores asked so far for this scan (one per chunk).
    #[must_use]
    pub fn reranked(&self) -> usize {
        self.p.lock().map(|p| p.len()).unwrap_or(0)
    }
    /// The reranker's P(yes) per chunk index asked so far for this scan (2026-11-10, scan dump:
    /// the control arm's judge inputs are saved so its rule can be replayed on the processor).
    #[must_use]
    pub fn rerank_scores(&self) -> BTreeMap<usize, f32> {
        self.p.lock().map(|p| p.clone()).unwrap_or_default()
    }
}

struct State {
    /// The last question's scan (a second judge of the same question does not read again).
    last: Option<Arc<Scan>>,
    cache: ChunkCache,
    /// Token count of a chunk's document text, by content key.
    tokens: BTreeMap<String, usize>,
    /// Line of every token relative to the chunk (0 the path line, 1 the chunk's first line).
    rel_lines: BTreeMap<String, Vec<usize>>,
    /// The anchor text, its encoding and the one-part window chunks are encoded behind.
    anchor: Option<(String, Arc<ChunkKv>, Arc<Window>)>,
    /// The RAM limit the cache was made with.
    limit: RamLimit,
    /// No-question reads kept by window content (step P2).
    reads: KeptReads,
    /// Content keys of every chunk this pass has encoded (kept or since dropped).
    ever: std::collections::BTreeSet<String>,
    /// Keys-only scan (2026-11-10): the relevance-key store and its background pass; since
    /// 2026-11-18 (Phase C2) the organism's store beside the drones' intake when one is given
    /// (`TentaclePass::use_keys`), this pass's own otherwise.
    keys: Option<Arc<crate::key_scan::Keys>>,
    /// Refinement step 1: chunks memory recalled for a question (the memory lens's list), set
    /// by the caller before the scan (`TentaclePass::set_recalled`).
    recalled: Option<(String, Vec<usize>)>,
    /// Build step T1: the drone and the tree as it took it in (RAM only, while unchanged).
    drone: Option<crate::drone_windows::Keep>,
    /// One tentacle system (2026-11-19, audit rule 3.3.16): the windows the Team's map gave its
    /// tentacles for a question, with each tentacle's focus (`TentaclePass::set_team_windows`);
    /// when set for the question scanned, they replace the pass's own lens layout.
    team: Option<TeamWindows>,
}

/// The Team's windows for one question (one tentacle system, 2026-11-19): per tentacle its
/// chunks (claimed on the map, no chunk in two windows) and its focus (the aspect phrased for its
/// perspective).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TeamWindows {
    pub question: String,
    pub windows: Vec<Vec<usize>>,
    pub focus: Vec<String>,
}
impl TeamWindows {
    /// A key of the windows and foci (0 for none), so a scan of other windows is not reused.
    #[must_use]
    pub fn key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.windows.hash(&mut h);
        self.focus.hash(&mut h);
        h.finish().max(1)
    }
    /// Tentacle `t`'s focus, or the pass's own lens focus.
    #[must_use]
    pub fn focus_of(team: Option<&Self>, t: usize) -> String {
        team.and_then(|w| w.focus.get(t)).filter(|f| !f.is_empty()).cloned().unwrap_or_else(|| FOCUS.get(t).copied().unwrap_or("the code that does this").to_owned())
    }
}

/// The highest layer any drone head in use reads: chunks are encoded and kept through it
/// (2026-11-10). Derived from the heads, not typed. Since 2026-11-10 (step P0) the heads are
/// the base population's drone sets in `designs/drones.json` (default and tentacle sets), with
/// `DRONE_HEADS_QWEN3_0_6B` as the stated fallback.
#[must_use]
pub fn drone_last_layer() -> usize {
    drones(&nmtype::nano_root()).last_layer()
}

/// The drone sets of the base population (`crate::neural::POPULATION`) under the mini folder
/// `home`: the file's, else the fallback constant (`DroneSet::source` says which and why).
#[must_use]
pub fn drones(home: &Path) -> DroneSet {
    kvstore::drone_set(home, crate::neural::POPULATION).unwrap_or_else(|e| DroneSet {
        population: crate::neural::POPULATION.to_owned(),
        heads: kvstore::DRONE_HEADS_QWEN3_0_6B.to_vec(),
        tentacles: BTreeMap::new(),
        source: format!("fallback constant DRONE_HEADS_QWEN3_0_6B ({e})"),
        from_file: false,
        read_mode: "exact".to_owned(),
        ..DroneSet::default()
    })
}

/// One drone read of a window (the pass's read, 2026-11-10): `text` posed after the window
/// `w`, the run stopping at `stop`, and the attention mass of every pair of `heads` on every
/// window position per question token (`heads x len`, in the order of `heads`).
///
/// # Errors
/// Tokenizer or engine faults, or a drone layer the run did not read.
pub fn drone_read(
    model: &Model,
    dev: Device,
    w: &Arc<Window>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    text: &str,
) -> Result<Vec<f32>, Fault> {
    let q = model.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    let probe = Arc::new(Probe::of_heads(heads));
    model.forward_at(
        dev,
        &q,
        w.len,
        stop,
        &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
    )?;
    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let per = 1.0 / q.len().max(1) as f32;
    let mut out = Vec::with_capacity(heads.len() * w.len);
    for &(l, h) in heads {
        let mass = probe.get(l).ok_or_else(|| bad(format!("layer {l} not read")))?;
        out.extend((0..w.len).map(|p| mass.get(h * w.len + p).copied().unwrap_or(0.0) * per));
    }
    Ok(out)
}

/// Calibrated score per part from per-head, per-part log-sum-exp rows (posed and
/// content-free, `heads x parts`): per part, posed minus content-free summed over the heads.
/// Empty when the rows do not fit `parts`.
#[must_use]
pub fn calibrated(posed: &[f32], null: &[f32], parts: usize) -> Vec<f32> {
    if parts == 0 || posed.is_empty() || posed.len() != null.len() || !posed.len().is_multiple_of(parts) {
        return Vec::new();
    }
    let heads = posed.len() / parts;
    (0..parts)
        .map(|k| {
            (0..heads).map(|h| posed.get(h * parts + k).copied().unwrap_or(0.0) - null.get(h * parts + k).copied().unwrap_or(0.0)).sum()
        })
        .collect()
}

/// Per part `(first position, tokens)` of a window, the log-sum-exp of one head's raw scores
/// over the part's positions, averaged over the question tokens (`logits` is
/// `[tokens, span]`).
#[must_use]
pub fn part_lse(logits: &[f32], tokens: usize, span: usize, parts: &[(usize, usize)]) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let per = 1.0 / tokens.max(1) as f32;
    parts
        .iter()
        .map(|&(at, len)| {
            let mut acc = 0.0_f32;
            for t in 0..tokens {
                let row = logits.get(t * span + at..t * span + at + len).unwrap_or(&[]);
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max.is_finite() {
                    acc += max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
                }
            }
            acc * per
        })
        .collect()
}

/// `drone_read` that also records the raw attention scores (tentacle core A5, 2026-11-10):
/// returns the same mass, bit for bit (`nmengn.kvstore.logits`), and per drone head (rows, in
/// the order of `heads`) and per part of the window (columns, `w.parts` order) the log-sum-exp
/// of the scores over the part's positions, averaged over the question tokens (`part_lse`).
///
/// # Errors
/// As `drone_read`.
pub fn drone_read_logits(
    model: &Model,
    dev: Device,
    w: &Arc<Window>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    text: &str,
) -> Result<(Vec<f32>, Vec<f32>), Fault> {
    drone_read_logits_with(model, dev, w, stop, heads, text, None)
}

/// `drone_read_logits` through a model whose probe reads only attention components whose id holds
/// `filter` (2026-11-10: a tentacle's own assembly, read at its drone tip).
/// # Errors
/// As `drone_read`.
#[expect(clippy::too_many_arguments, reason = "the read of `drone_read_logits` plus the one filter")]
pub fn drone_read_logits_with(
    model: &Model,
    dev: Device,
    w: &Arc<Window>,
    stop: &[(String, String)],
    heads: &[(usize, usize)],
    text: &str,
    filter: Option<&str>,
) -> Result<(Vec<f32>, Vec<f32>), Fault> {
    let _job = nmengn::chanrec::job("integration");
    let q = model.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    let p = Probe::of_heads(heads).with_logits();
    let probe = Arc::new(match filter {
        Some(f) => p.with_node_filter(f),
        None => p,
    });
    model.forward_at(
        dev,
        &q,
        w.len,
        stop,
        &RunOpts { window: Some(Arc::clone(w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
    )?;
    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let per = 1.0 / q.len().max(1) as f32;
    let mut out = Vec::with_capacity(heads.len() * w.len);
    let mut cal = Vec::with_capacity(heads.len() * w.parts.len());
    for &(l, h) in heads {
        let mass = probe.get(l).ok_or_else(|| bad(format!("layer {l} not read")))?;
        out.extend((0..w.len).map(|p| mass.get(h * w.len + p).copied().unwrap_or(0.0) * per));
        let lg = probe.logits(l, h).ok_or_else(|| bad(format!("layer {l} head {h}: no logits recorded")))?;
        cal.extend(part_lse(&lg.logits, lg.lse.len(), w.len, &w.parts));
    }
    Ok((out, cal))
}

/// The read a span read uses: (window, text) to (mass, per-head per-part rows).
pub type SpanReader<'a> = dyn Fn(&Arc<Window>, &str) -> Result<(Vec<f32>, Vec<f32>), Fault> + 'a;

/// A span read of one window (2026-11-10, test T6): `parts` (anchor first) split into consecutive
/// spans of at most `span` tokens (a longer chunk alone); each span read behind the anchor as its
/// own short window, posed and content-free (`read` gives mass and per-head per-part rows); the best
/// `SPAN_KEEP` chunks of each span (by the calibrated score above the span's anchor) read once more
/// together in window order. Returns posed and content-free tables `heads x parts` in the window's
/// layout whose calibrated score above the anchor is: for the winners, their final short read; for
/// the others, their span read moved just below the lowest winner (their order kept), so the core
/// ranks winners first. The anchor column is zero.
/// # Errors
/// Engine faults.
pub fn span_tables(
    read: &SpanReader<'_>,
    model: &Model,
    parts: &[Arc<ChunkKv>],
    heads: usize,
    texts: (&str, &str),
    span: usize,
) -> Result<(Vec<f32>, Vec<f32>), Fault> {
    let Some(anchor) = parts.first() else { return Ok((Vec::new(), Vec::new())) };
    let n = parts.len();
    // Per head, the reading above the window's anchor of the chunks `idx` (indices into `parts`).
    let short = |idx: &[usize]| -> Result<Vec<Vec<f32>>, Fault> {
        let mut p = vec![Arc::clone(anchor)];
        p.extend(idx.iter().filter_map(|i| parts.get(*i).cloned()));
        let w = Arc::new(Window::build(model, &p)?);
        let (_, pc) = read(&w, texts.0)?;
        let (_, nc) = read(&w, texts.1)?;
        w.release_device();
        let m = idx.len() + 1;
        let at = |v: &[f32], h: usize, k: usize| v.get(h * m + k).copied().unwrap_or(0.0);
        Ok((0..heads).map(|h| (1..m).map(|k| (at(&pc, h, k) - at(&nc, h, k)) - (at(&pc, h, 0) - at(&nc, h, 0))).collect()).collect())
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut used = 0;
    for (i, c) in parts.iter().enumerate().skip(1) {
        if !cur.is_empty() && used + c.tokens > span {
            groups.push(std::mem::take(&mut cur));
            used = 0;
        }
        cur.push(i);
        used += c.tokens;
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    let mut value: Vec<Vec<f32>> = vec![vec![0.0; heads]; n];
    let mut winners: Vec<usize> = Vec::new();
    let mut losers: Vec<usize> = Vec::new();
    for g in &groups {
        let rows = short(g)?;
        let total = |k: usize| rows.iter().map(|r| r.get(k).copied().unwrap_or(0.0)).sum::<f32>();
        let mut idx: Vec<usize> = (0..g.len()).collect();
        idx.sort_by(|a, b| total(*b).total_cmp(&total(*a)).then(a.cmp(b)));
        for (rank, k) in idx.into_iter().enumerate() {
            let Some(&c) = g.get(k) else { continue };
            if let Some(v) = value.get_mut(c) {
                *v = rows.iter().map(|r| r.get(k).copied().unwrap_or(0.0)).collect();
            }
            if rank < SPAN_KEEP { winners.push(c) } else { losers.push(c) }
        }
    }
    winners.sort_unstable();
    let rows = short(&winners)?;
    for (k, c) in winners.iter().enumerate() {
        if let Some(v) = value.get_mut(*c) {
            *v = rows.iter().map(|r| r.get(k).copied().unwrap_or(0.0)).collect();
        }
    }
    let sum = |c: usize| value.get(c).map_or(0.0, |v| v.iter().sum::<f32>());
    let low = winners.iter().map(|c| sum(*c)).fold(f32::INFINITY, f32::min);
    let high = losers.iter().map(|c| sum(*c)).fold(f32::NEG_INFINITY, f32::max);
    if low.is_finite() && high.is_finite() && high >= low {
        #[expect(clippy::cast_precision_loss, reason = "a head count")]
        let shift = (high - low + 1.0) / heads.max(1) as f32;
        for c in &losers {
            if let Some(v) = value.get_mut(*c) {
                v.iter_mut().for_each(|x| *x -= shift);
            }
        }
    }
    let mut posed = vec![0.0_f32; heads * n];
    for (c, v) in value.iter().enumerate() {
        for (h, x) in v.iter().enumerate() {
            if let Some(y) = posed.get_mut(h * n + c) {
                *y = *x;
            }
        }
    }
    Ok((posed, vec![0.0; heads * n]))
}

/// The cache limit for these settings: the one named, else derived from free memory now; never
/// above the `caches` memory area's budget (memory areas, 2026-11-27: the encoded-chunk cache is
/// that area's holder `encoded_chunks`).
fn cache_limit(settings: &Settings) -> RamLimit {
    let mut l = if settings.cache_bytes > 0 { RamLimit { bytes: settings.cache_bytes, ..RamLimit::default() } } else { kvstore::derive_ram_limit() };
    if let Some(b) = nmengn::limits::area_budget("caches") {
        l.bytes = l.bytes.min(usize::try_from(b).unwrap_or(usize::MAX));
    }
    l
}

/// One window of a keys-only scan before scoring: tentacle, chunks, parts, record.
type Outlined = (usize, Vec<usize>, Vec<Arc<ChunkKv>>, WindowRecord);

/// What the keys-only scan needs from `TentaclePass::scan` (2026-11-10).
struct KeyCtx<'a> {
    sub: &'a Substrate,
    question: &'a str,
    settings: &'a Settings,
    model: &'a Arc<Model>,
    dev: Device,
    tokens: &'a [usize],
    windows: &'a [Vec<usize>],
    sets: &'a DroneSet,
    last: usize,
    ak: &'a Arc<ChunkKv>,
    anchor: &'a Arc<Window>,
    /// The Team's windows and foci when they gave the windows (2026-11-19).
    team: Option<&'a TeamWindows>,
}

/// Content key to bytes of every chunk encoding the scan's kept windows hold (2026-11-10, the
/// shared RAM budget); outlines hold nothing.
fn pinned_bytes(sub: &Substrate, windows: &[WindowScan]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for w in windows {
        let Some(win) = w.window.as_ref() else { continue };
        for (c, kv) in w.place.iter().zip(win.chunks.iter().skip(1)) {
            if let Some(ch) = sub.chunks.get(*c)
                && kv.bytes() > 0
            {
                out.insert(ch.key.clone(), kv.bytes());
            }
        }
    }
    out
}

/// The pass, kept for the session: encoded chunks stay in RAM between questions.
pub struct TentaclePass {
    home: PathBuf,
    /// Settings (pose, judge, score, shortlist, lines).
    pub settings: Settings,
    state: Arc<Mutex<State>>,
}

/// Token count of every chunk (kept by content key in `known`) and the eight lens windows for
/// `question`, each chunk in one window only. The embedding order comes from the dense lane
/// when it is bound.
fn lay_out(
    known: &mut BTreeMap<String, usize>,
    sub: &Substrate,
    question: &str,
    tok: &dyn Fn(&str) -> Result<Vec<i32>, Fault>,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let (tokens, windows, _) = lay_out_with(known, sub, question, tok, false, &[]);
    (tokens, windows)
}

/// `lay_out` with the refinement step 1 options (item 5): `recalled` gives the memory lens its
/// list (chunks memory chose for this question), and with `fill` every lens left with an empty
/// list takes the fallback fill in lens order (`nmsubs::lens::fill_empty`), so every window
/// carries context. Returns the token counts, the windows and the filled lenses.
fn lay_out_with(
    known: &mut BTreeMap<String, usize>,
    sub: &Substrate,
    question: &str,
    tok: &dyn Fn(&str) -> Result<Vec<i32>, Fault>,
    fill: bool,
    recalled: &[usize],
) -> (Vec<usize>, Vec<Vec<usize>>, Vec<usize>) {
    let tokens = token_counts(known, sub, tok);
    let lane = sub.dense.read().ok().and_then(|g| g.clone());
    let emb = lane.as_ref().and_then(|l| l.embedding_order(sub, question));
    let lists = lens::orders(sub, question, emb.as_deref(), recalled);
    let (lists, filled) = if fill { lens::fill_empty(&lists) } else { (lists, Vec::new()) };
    let windows = lens::assign(&lists, &|c| tokens.get(c).copied().unwrap_or(usize::MAX), BUDGET);
    (tokens, windows, filled)
}

/// Token count of every chunk's document text, kept by content key in `known`.
fn token_counts(known: &mut BTreeMap<String, usize>, sub: &Substrate, tok: &dyn Fn(&str) -> Result<Vec<i32>, Fault>) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(sub.chunks.len());
    for (i, ch) in sub.chunks.iter().enumerate() {
        let n = match known.get(&ch.key) {
            Some(n) => *n,
            None => {
                let n = tok(&doc_of(sub, i)).map_or(usize::MAX, |v| v.len());
                known.insert(ch.key.clone(), n);
                n
            }
        };
        tokens.push(n);
    }
    tokens
}

/// Candidate indices ordered by `key` descending (ties: candidate order).
fn by(n: usize, key: &dyn Fn(usize) -> f32) -> Vec<usize> {
    let mut o: Vec<usize> = (0..n).collect();
    o.sort_by(|a, b| key(*b).total_cmp(&key(*a)).then(a.cmp(b)));
    o
}

/// The judge's order over the candidates (indices into `c`), best first. `p` is the reranker's
/// P(yes) per candidate; without it every judge is the heads.
#[must_use]
pub fn judge_order(c: &[Cand], p: Option<&[f32]>, judge: Judge) -> Vec<usize> {
    let head = |i: usize| c.get(i).map_or(0.0, |x| x.score);
    let Some(p) = p.filter(|p| p.len() == c.len() && judge != Judge::Heads) else {
        return by(c.len(), &head);
    };
    let pv = |i: usize| p.get(i).copied().unwrap_or(0.0);
    let rank_of = |key: &dyn Fn(usize) -> f32| -> Vec<usize> {
        let mut rk = vec![0; c.len()];
        for (pos, i) in by(c.len(), key).into_iter().enumerate() {
            if let Some(r) = rk.get_mut(i) {
                *r = pos;
            }
        }
        rk
    };
    match judge {
        Judge::Heads => by(c.len(), &head),
        Judge::Reranker => by(c.len(), &pv),
        Judge::Rrf => {
            let (rp, rh) = (rank_of(&pv), rank_of(&head));
            #[expect(clippy::cast_precision_loss, reason = "ranks are small")]
            let f =
                |i: usize| 1.0 / (60.0 + rp.get(i).copied().unwrap_or(0) as f32) + 1.0 / (60.0 + rh.get(i).copied().unwrap_or(0) as f32);
            by(c.len(), &f)
        }
        Judge::Sum => {
            let top = c.iter().map(|x| x.score).fold(f32::NEG_INFINITY, f32::max).max(1e-9);
            by(c.len(), &|i| pv(i) + head(i) / top)
        }
        Judge::WithinPlace => {
            #[expect(clippy::cast_precision_loss, reason = "places are small")]
            let f = |i: usize| pv(i) - c.get(i).map_or(0, |x| x.place) as f32 * 10.0;
            by(c.len(), &f)
        }
    }
}

/// The fewest lines holding half the positive drone mass (at least one), as at most two line
/// spans (first, last), best span first. Lines one apart join into one span.
#[must_use]
pub fn key_spans(lines: &[(usize, f32, usize)]) -> Vec<(usize, usize)> {
    let total: f32 = lines.iter().map(|x| x.1.max(0.0)).sum();
    let mut order = lines.to_vec();
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let (mut picked, mut acc) = (Vec::new(), 0.0_f32);
    for (li, s, _) in order {
        if !picked.is_empty() && acc >= total * 0.5 {
            break;
        }
        acc += s.max(0.0);
        picked.push((li, s.max(0.0)));
    }
    picked.sort_by_key(|x| x.0);
    let mut spans: Vec<(usize, usize, f32)> = Vec::new();
    for (li, s) in picked {
        match spans.last_mut() {
            Some(last) if li <= last.1 + 2 => {
                last.1 = li;
                last.2 += s;
            }
            _ => spans.push((li, li, s)),
        }
    }
    spans.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    spans.into_iter().take(2).map(|(a, b, _)| (a, b)).collect()
}

fn path_of(sub: &Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"))
}
fn doc_of(sub: &Substrate, c: usize) -> String {
    sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file).map(|f| nmsubs::dense::doc_text(&f.path, &ch.text))).unwrap_or_default()
}

/// The bridge: the judged candidates as `chunk` packets, best first. A packet has the lane's
/// chunk text and first evidence line, so the Core Brain merges it with the same chunk from
/// another reader; its origin names the tentacle, lens and pose.
#[must_use]
pub fn bridge(sub: &Substrate, cands: &[Cand], order: &[usize], settings: &Settings) -> Vec<Packet> {
    let pose = if settings.focused { "focus" } else { "plain" };
    let mut out = Vec::new();
    for (k, c) in order.iter().filter_map(|i| cands.get(*i)).enumerate() {
        let (Some(ch), Some(f)) = (sub.chunks.get(c.chunk), sub.chunks.get(c.chunk).and_then(|ch| sub.files.get(ch.file))) else {
            continue;
        };
        let src = format!("{}:{}", f.path, ch.start);
        let lens = lens::NAMES.get(c.tentacle).copied().unwrap_or("lens");
        let origin = format!("nmpod.tentacle#t{}/lens:{lens}/pose:{pose}", c.tentacle);
        let mut evidence = vec![format!("{src} lines {}..{}", ch.start, ch.end)];
        if settings.lines {
            for (a, b) in key_spans(&c.lines) {
                if (a, b) != (ch.start, ch.end) {
                    evidence.push(format!("{}:{a} lines {a}..{b}", f.path));
                }
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "places are small")]
        let conf = (TENTACLE_TOP - DENSE_STEP * k as f32).max(FLOOR);
        out.push(
            Packet::new(
                "chunk",
                &format!("{TENTACLE_KEY}{k:02}:{src}"),
                &ch.text,
                Provenance::new(&src, &origin, sub.version, nmtype::now_ms()),
            )
            .status(Status::Inferred)
            .stance(Stance::Objective)
            .category(Category::Fact)
            .confidence(conf)
            .evidence(evidence),
        );
    }
    out
}

/// The bridge on the new path (refinement step 1, item 1): the forwarded spans as `chunk`
/// packets as `bridge` makes them, each carrying its evidence record's relevance
/// (`nmbrain::synthesis::evidence_relevance`, G1's combined value against the best forwarded
/// record's) as its confidence instead of a value from its place. The record itself travels in
/// `Pass::evidence`.
#[must_use]
pub fn bridge_evidence(sub: &Substrate, cands: &[Cand], set: &nmbrain::evidence::EvidenceSet, settings: &Settings) -> Vec<Packet> {
    let order: Vec<usize> = (0..cands.len()).collect();
    bridge(sub, cands, &order, settings)
        .into_iter()
        .map(|p| {
            let rec = p.provenance().source.rsplit_once(':').and_then(|(f, l)| set.at(&f.replace('\\', "/"), l.parse().ok()?));
            match rec {
                Some(r) => {
                    let c = nmbrain::synthesis::evidence_relevance(set, r);
                    p.confidence(c)
                }
                None => p,
            }
        })
        .collect()
}

/// The pass's settings from command-line flags: `--tentacles` switches it on,
/// `--tentacle-judge heads|reranker|rrf|sum|place`, `--tentacle-pose plain|focus`,
/// `--tentacle-score mass|density|evidence|peak` and `--tentacle-shortlist <n>` set the judge,
/// the pose, the chunk score and the shortlist. `Ok(None)` when the flag is absent. The
/// shortlist starts at `SERVED_SHORTLIST` (6, since 2026-11-10), not the default's 2.
///
/// # Errors
/// An unknown value.
pub fn from_args(args: &[String]) -> Result<Option<Settings>, String> {
    // 2026-11-10: `--tentacle-core` switches the pass on with the tentacle core (first form).
    let core = args.iter().any(|a| a == "--tentacle-core");
    if !core && !args.iter().any(|a| a == "--tentacles") {
        return Ok(None);
    }
    let value = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1));
    // 2026-11-10 (step P0): the served path starts from the G1 shortlist 6, not the default 2.
    let mut s = Settings::served();
    s.core = core;
    // Refinement step 1 (2026-11-10): with the core on, G1 runs the Globalized Context Stack rules (evidence
    // combination, queue forward, evidence set to the reduce step and G2);
    // `--g1-rules first` keeps the first form and `--g1-rules evidence` the fitted evidence
    // combination, for comparison.
    if core {
        match value("--g1-rules").map(String::as_str) {
            None | Some("stack") => nmbrain::g1::set_rules(nmbrain::g1::Rules::stack()),
            Some("first") => nmbrain::g1::set_rules(nmbrain::g1::Rules::first_form()),
            Some("evidence") => nmbrain::g1::set_rules(nmbrain::g1::Rules::evidence()),
            Some(x) => return Err(format!("--g1-rules takes stack, evidence or first, got {x}")),
        }
    }
    if let Some(j) = value("--tentacle-judge") {
        s.judge = Judge::parse(j).ok_or_else(|| format!("--tentacle-judge takes heads, reranker, rrf, sum or place, got {j}"))?;
    } else if let Some(j) = crate::whole::control_str(&nmtype::nano_root(), "routing.tentacle_judge").filter(|j| j != "default").and_then(|j| Judge::parse(&j)) {
        // 2026-11-25: without the option, the `routing.tentacle_judge` control (interface, MCP).
        s.judge = j;
    }
    if let Some(p) = value("--tentacle-pose") {
        s.focused = match p.as_str() {
            "focus" => true,
            "plain" => false,
            _ => return Err(format!("--tentacle-pose takes plain or focus, got {p}")),
        };
    }
    if let Some(x) = value("--tentacle-score") {
        s.score = Score::parse(x).ok_or_else(|| format!("--tentacle-score takes mass, density, evidence, peak or calibrated, got {x}"))?;
    }
    if let Some(x) = value("--tentacle-shortlist") {
        s.shortlist = x
            .parse()
            .ok()
            .filter(|n: &usize| *n >= 1)
            .ok_or_else(|| format!("--tentacle-shortlist takes a count of 1 or more, got {x}"))?;
    }
    // 2026-11-10: span read for the core (test T6; off by default).
    if let Some(x) = value("--tentacle-span") {
        s.span = x.parse().ok().filter(|n: &usize| *n >= 1).ok_or_else(|| format!("--tentacle-span takes a token count, got {x}"))?;
    }
    // 2026-11-10 (refinement step 2a): components of the core's read (off by default).
    if let Some(x) = value("--core-read") {
        s.read = crate::core_read::ReadOptions::parse(x)
            .ok_or_else(|| format!("--core-read takes off, all, or a comma list of look, votes, pose; got {x}"))?;
    }
    // 2026-11-10 (build step T1): windows from drone flags plus relations (off by default).
    if let Some(x) = value("--tentacle-windows") {
        s.windows =
            crate::drone_windows::WindowMode::parse(x).ok_or_else(|| format!("--tentacle-windows takes lens or drones, got {x}"))?;
    }
    // 2026-11-10: keys-only scan (first form, off by default).
    if let Some(x) = value("--tentacle-scan") {
        s.keys = match x.as_str() {
            "keys" => true,
            "exact" => false,
            _ => return Err(format!("--tentacle-scan takes exact or keys, got {x}")),
        };
    }
    Ok(Some(s))
}

/// Switch the pass on for an organism; returns what was set.
pub fn enable(org: &crate::forage::Organism, settings: Settings) -> String {
    if let Ok(mut p) = org.tentacle_pass.write() {
        *p = Some(Arc::new(TentaclePass::new(&org.home, settings)));
    }
    format!(
        "tentacle pass on: 8 lens windows of {BUDGET} tokens, pose {}, judge {}, score {}, {} chunks per tentacle from a shortlist of {}, lines {}, tentacle core {}",
        if settings.focused { "focus" } else { "plain" },
        settings.judge.name(),
        settings.score.name(),
        settings.per_tentacle,
        settings.shortlist.max(settings.per_tentacle),
        if settings.lines { "on" } else { "off" },
        if settings.core { "on (G1 loop over the g0_tx reports; the pass is the fallback)" } else { "off" }
    )
}

impl TentaclePass {
    /// A pass over the mini folder `home` (weights, assemblies).
    #[must_use]
    pub fn new(home: &Path, settings: Settings) -> Self {
        Self {
            home: home.to_path_buf(),
            settings,
            // The cache is made with its limit when the first scan sets the anchor (after the
            // base model is in RAM, so the free memory read counts it).
            state: Arc::new(Mutex::new(State {
                last: None,
                cache: ChunkCache::new(0),
                tokens: BTreeMap::new(),
                rel_lines: BTreeMap::new(),
                anchor: None,
                limit: RamLimit::default(),
                reads: KeptReads::default(),
                ever: std::collections::BTreeSet::new(),
                keys: None,
                recalled: None,
                drone: None,
                team: None,
            })),
        }
    }

    /// The same pass under other settings: the encoded chunks and the last scan are shared,
    /// so nothing is encoded or read again.
    #[must_use]
    pub fn retuned(&self, settings: Settings) -> Self {
        Self { home: self.home.clone(), settings, state: Arc::clone(&self.state) }
    }

    /// Refinement step 1 (item 5): the chunks memory recalled for `question`, the memory lens's
    /// list on the tentacle core path (the scan of that question uses them).
    pub fn set_recalled(&self, question: &str, chunks: Vec<usize>) {
        if let Ok(mut st) = self.state.lock() {
            st.recalled = Some((question.to_owned(), chunks));
        }
    }

    /// One tentacle system (2026-11-19, audit rule 3.3.16): the Team's windows for `question`
    /// (per tentacle its claimed chunks and its focus). The next scan of that question reads these
    /// windows, one core per Team tentacle (`tentacle_core::TentacleCore::from_scan`), in place of
    /// the pass's own lens layout; `clear_team_windows` returns to the lens layout.
    pub fn set_team_windows(&self, question: &str, windows: Vec<Vec<usize>>, focus: Vec<String>) {
        if let Ok(mut st) = self.state.lock() {
            st.team = Some(TeamWindows { question: question.to_owned(), windows, focus });
        }
    }
    /// Return to the pass's own lens layout.
    pub fn clear_team_windows(&self) {
        if let Ok(mut st) = self.state.lock() {
            st.team = None;
        }
    }

    /// Use the organism's relevance-key store (beside the drones' intake, Phase C2) for the
    /// keys-only scan instead of this pass's own.
    pub fn use_keys(&self, keys: Arc<crate::key_scan::Keys>) {
        if let Ok(mut st) = self.state.lock() {
            st.keys = Some(keys);
        }
    }

    /// Bytes of encoded chunks held in RAM.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.state.lock().map(|s| s.cache.bytes()).unwrap_or(0)
    }

    /// The windows `question` would get, without encoding or reading: per window its chunks
    /// and their tokens, and the token count of every chunk of the tree (the part test of the
    /// kept chunks compares the windows of successive questions with it).
    ///
    /// # Errors
    /// Model or tokenizer faults.
    pub fn layout(&self, sub: &Substrate, question: &str) -> Result<(Vec<usize>, Vec<Vec<usize>>), Fault> {
        let model = crate::neural::model(&self.home)?;
        let pop = Arc::clone(&model.pop);
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let mut st = self.state.lock().map_err(|_| bad("pass state poisoned"))?;
        Ok(lay_out(&mut st.tokens, sub, question, &tok))
    }

    /// The RAM limit the kept chunks were given (zero until the first scan).
    #[must_use]
    pub fn limit(&self) -> RamLimit {
        self.state.lock().map(|s| s.limit).unwrap_or_default()
    }

    /// Read the repository for one question: the lens windows, one read per tentacle, the judge
    /// and the bridge. The embedding order comes from the dense lane when it is bound.
    ///
    /// # Errors
    /// Model, tokenizer or engine faults; the caller answers without the pass.
    pub fn read(&self, sub: &Substrate, slots: &Slots, question: &str) -> Result<Pass, Fault> {
        self.read_as(sub, slots, question, &self.settings)
    }

    /// The last scan (a test reads the windows' scores from it).
    #[must_use]
    pub fn last_scan(&self) -> Option<Arc<Scan>> {
        self.state.lock().ok().and_then(|s| s.last.clone())
    }

    /// `read` under other settings (pose, score, shortlist, judge). The scan of the last
    /// question is kept, so a second judge of the same question and pose reads nothing again.
    ///
    /// # Errors
    /// As `read`.
    pub fn read_as(&self, sub: &Substrate, slots: &Slots, question: &str, settings: &Settings) -> Result<Pass, Fault> {
        let started = Instant::now();
        let (scan, reused) = self.scan(sub, question, settings)?;
        // Tentacle core first form (2026-11-10): the cores report through their g0_tx to the G1
        // loop and G1's hand-over is forwarded; on a fault the pass below (the control) answers.
        if settings.core {
            match crate::tentacle_core::judge(&self.home, sub, &scan, settings) {
                Ok((mut p, _)) => {
                    p.note = format!("{}; total {:.0} ms", p.note, started.elapsed().as_secs_f64() * 1000.0);
                    return Ok(p);
                }
                Err(e) => {
                    let control = Settings { core: false, ..*settings };
                    let mut p = self.judge(sub, slots, &scan, &control, reused, started)?;
                    p.note = format!("tentacle core fault ({}); the control pass answered: {}", e.message, p.note);
                    return Ok(p);
                }
            }
        }
        self.judge(sub, slots, &scan, settings, reused, started)
    }

    /// Lay out the eight lens windows and read each with the drone heads: once with the
    /// question, once without. Returns the scan and whether it was the kept one.
    ///
    /// # Errors
    /// Model, tokenizer or engine faults.
    #[expect(clippy::too_many_lines, reason = "one scan, in the order it runs")]
    pub fn scan(&self, sub: &Substrate, question: &str, settings: &Settings) -> Result<(Arc<Scan>, bool), Fault> {
        let mut st = self.state.lock().map_err(|_| bad("pass state poisoned"))?;
        let st = &mut *st;
        let want_logits = settings.wants_logits();
        let team: Option<TeamWindows> = st.team.as_ref().filter(|t| t.question == question).cloned();
        let team_key = team.as_ref().map_or(0, TeamWindows::key);
        if let Some(s) = st.last.as_ref().filter(|s| {
            s.question == question
                && s.team == team_key
                && s.focused == settings.focused
                && s.version == sub.version
                && s.chunks == sub.chunks.len()
                && (s.logits || !want_logits)
                && (!settings.core || s.windows.iter().all(|w| w.window.is_some()))
                && s.keys.is_empty() != (settings.keys && want_logits)
                && s.pose.is_empty() != settings.read.matched_pose
                && s.drone_windows == (settings.windows == crate::drone_windows::WindowMode::Drones)
        }) {
            return Ok((Arc::clone(s), true));
        }
        let model = crate::neural::model(&self.home)?;
        let pop = Arc::clone(&model.pop);
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));

        // Token counts of every chunk, kept by content key, and the windows: eight lens orders,
        // each chunk in one window only.
        // Refinement step 1 (item 5): on the tentacle core path the memory lens gets the chunks
        // memory recalled for this question and every empty lens takes the fallback fill.
        let recalled: Vec<usize> =
            st.recalled.as_ref().filter(|r| settings.core && r.0 == question).map(|r| r.1.clone()).unwrap_or_default();
        let (tokens, windows, filled) = if let Some(t) = team.as_ref() {
            // One tentacle system (2026-11-19): the Team's map gave the windows; each is read as
            // one tentacle core's window (chunks past the substrate dropped).
            let tokens = token_counts(&mut st.tokens, sub, &tok);
            let n = sub.chunks.len();
            (tokens, t.windows.iter().map(|w| w.iter().copied().filter(|c| *c < n).collect()).collect(), Vec::new())
        } else if settings.windows == crate::drone_windows::WindowMode::Drones {
            // Build step T1: the windows from the drones' flags plus relations.
            let tokens = token_counts(&mut st.tokens, sub, &tok);
            if st.drone.is_none() {
                st.drone = Some(crate::drone_windows::Keep::open(&nmtype::models_dir(&self.home).join("qwen3-0.6b").join("files"))?);
            }
            let keep = st.drone.as_mut().ok_or_else(|| bad("no drone"))?;
            keep.sync(sub)?;
            let fused = keep.sense(question)?;
            let graph = sub.refs();
            let c = crate::drone_windows::compose(
                &fused,
                &graph.specific,
                &|c| tokens.get(c).copied().unwrap_or(usize::MAX),
                BUDGET,
                lens::NAMES.len(),
            );
            (tokens, c.windows, Vec::new())
        } else {
            lay_out_with(&mut st.tokens, sub, question, &tok, settings.core, &recalled)
        };

        // Step P2: the role models keep their device weights while the shared device budget
        // allows (`nmengn::residency`: room is made when weights or windows need it, idle
        // weights of the cheapest reload per byte first); they were released here before every
        // scan. The scan holds the device gate, so the dense lane's background indexing does
        // not run on the card meanwhile (it waits, and says so).
        let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let gate = (dev == Device::Gpu).then(|| nmengn::placement::gpu_gate().enter("tentacle pass (encode and read)", true));
        let anchor_text = if sub.files.iter().any(|f| f.path.replace('\\', "/") == "crates/nmcore/Cargo.toml") {
            SELF_ANCHOR.to_owned()
        } else {
            format!("Repository of {} files, read in parts.\n", sub.files.len())
        };
        // Keys and values are encoded and kept through the last layer a drone head reads; the
        // drone sets come from `designs/drones.json` (step P0, fallback the constant).
        let drone_sets = drones(&self.home);
        let last = drone_sets.last_layer();
        if st.anchor.as_ref().is_none_or(|a| a.0 != anchor_text) {
            let ak = Arc::new(encode_through(&model, dev, None, &tok(&anchor_text)?, last)?);
            let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)])?);
            st.limit = cache_limit(settings);
            st.cache = ChunkCache::new(st.limit.bytes);
            st.anchor = Some((anchor_text, ak, aw));
        }
        let (ak, anchor) = st.anchor.as_ref().map(|a| (Arc::clone(&a.1), Arc::clone(&a.2))).ok_or_else(|| bad("no anchor"))?;
        // 2026-11-10: keys-only scan (first form), only where its scores are used (the core or
        // the calibrated score); otherwise the exact scan below runs (the fallback).
        if settings.keys && settings.wants_logits() {
            let cx = KeyCtx {
                sub,
                question,
                settings,
                model: &model,
                dev,
                tokens: &tokens,
                windows: &windows,
                sets: &drone_sets,
                last,
                ak: &ak,
                anchor: &anchor,
                team: team.as_ref(),
            };
            let mut scan = self.keys_scan(st, &cx, gate)?;
            scan.team = team_key;
            let scan = Arc::new(scan);
            st.last = Some(Arc::clone(&scan));
            return Ok((scan, false));
        }
        // The run stops at every attention component of the last drone layer (step P0: found in
        // the assembly; the reference assembly gives `L{last}.score`).
        let stop = kvstore::score_targets(&model, last)?;

        let mut scan = Scan {
            question: question.to_owned(),
            focused: settings.focused,
            logits: want_logits,
            version: sub.version,
            chunks: sub.chunks.len(),
            device: dev.name().to_string(),
            bytes_per_token: kvstore::bytes_per_token(&model, ak.layers()),
            drones: drone_sets.source.clone(),
            last_layer: ak.layers().saturating_sub(1),
            limit: st.limit,
            gate_wait: gate.as_ref().map_or(0.0, |g| g.waited),
            gate_for: gate.as_ref().map(|g| g.waited_for.clone()).unwrap_or_default(),
            ..Scan::default()
        };
        scan.filled.clone_from(&filled);
        scan.drone_windows = settings.windows == crate::drone_windows::WindowMode::Drones;
        scan.recalled = recalled.len();
        scan.team = team_key;
        let anchor_key = format!("anchor {}", nmhash::sha256_bytes(st.anchor.as_ref().map_or("", |a| a.0.as_str()).as_bytes()));
        let mut live_reads = std::collections::BTreeSet::new();
        for (t_i, win) in windows.iter().enumerate() {
            let lens_name = lens::NAMES.get(t_i).copied().unwrap_or("lens");
            let focus_s = TeamWindows::focus_of(team.as_ref(), t_i);
            let focus = focus_s.as_str();
            let pose = if settings.focused { "focus" } else { "plain" };
            #[expect(clippy::cast_possible_truncation, reason = "eight tentacles")]
            let mut rec =
                WindowRecord { tentacle: t_i as u8, lens: lens_name.to_owned(), pose: pose.to_owned(), ..WindowRecord::default() };
            if win.is_empty() {
                scan.records.push(rec);
                continue;
            }
            let mut place: Vec<usize> = win.iter().copied().filter(|c| sub.chunks.get(*c).is_some()).collect();
            place.sort_by(|a, b| {
                (path_of(sub, *a), sub.chunks.get(*a).map_or(0, |c| c.start))
                    .cmp(&(path_of(sub, *b), sub.chunks.get(*b).map_or(0, |c| c.start)))
            });
            // Refinement step 1: the lens rank of each chunk survives the sort into file order.
            let lens_rank: Vec<usize> = if settings.core {
                place.iter().map(|c| win.iter().position(|x| x == c).unwrap_or(usize::MAX)).collect()
            } else {
                Vec::new()
            };
            let mut parts = vec![Arc::clone(&ak)];
            let mut part_keys: Vec<&str> = vec![anchor_key.as_str()];
            for &c in &place {
                let Some(ch) = sub.chunks.get(c) else { continue };
                let kv = if let Some(kv) = st.cache.get(&ch.key) {
                    kv
                } else {
                    let ids = tok(&doc_of(sub, c))?;
                    let t = Instant::now();
                    let kv = Arc::new(encode_through(&model, dev, Some(&anchor), &ids, last)?);
                    scan.enc_ms += t.elapsed().as_secs_f64() * 1000.0;
                    scan.enc_tokens += ids.len();
                    if !st.ever.insert(ch.key.clone()) {
                        scan.enc_again_tokens += ids.len();
                    }
                    st.cache.put(&ch.key, Arc::clone(&kv));
                    kv
                };
                parts.push(kv);
                part_keys.push(ch.key.as_str());
            }
            // Step P2: the window carries its content key, so its device copy can be kept between
            // questions (residency manager) and its no-question read kept (`KeptReads`).
            let wkey = kvstore::window_key(&pop.card.id, last, &part_keys);
            let w = Arc::new(Window::build(&model, &parts)?.with_key(wkey.clone()));
            let t = Instant::now();
            // One read: every drone head's mass on every window position, per question token;
            // the tentacle's own set when the file names one, else the default (step P0).
            let heads = drone_sets.for_tentacle(lens_name);
            let read = |text: &str| drone_read(&model, dev, &w, &stop, heads, text);
            let read_cal = |text: &str| drone_read_logits(&model, dev, &w, &stop, heads, text);
            let (posed, null) = if settings.focused {
                (format!("\nQuestion: {question}\nFocus: {focus}"), format!("\nQuestion: N/A\nFocus: {focus}"))
            } else {
                (format!("\nQuestion: {question}"), "\nQuestion: N/A".to_owned())
            };
            // Refinement step 2a (component 3): with the length-matched pose on, the content-free
            // pose takes the posed text's token count (same frame, neutral filler), so the kept
            // no-question read below, keyed by its pose text, is keyed by that length as well.
            let null = if settings.read.matched_pose {
                let suffix = if settings.focused { format!("\nFocus: {focus}") } else { String::new() };
                let (text, n, why) = crate::core_read::matched_null(&|s: &str| tok(s).ok().map(|v| v.len()), &posed, &suffix);
                scan.pose = format!("{why} ({n} tokens)");
                text
            } else {
                null
            };
            // Tentacle core A5: with logits wanted, the same read also gives the raw scores (the
            // mass is bit for bit the same, `nmengn.kvstore.logits`).
            let (a, a_cal) = if want_logits { read_cal(&posed)? } else { (read(&posed)?, Vec::new()) };
            // Step P2: the no-question read depends only on the window (P1: bit-identical across
            // questions), so it is kept by the window's content key with the pose text, the drone
            // set and the stop layer, and computed only when the window changed.
            let nkey = format!("{wkey}\n{null}\n{heads:?}\n{stop:?}");
            live_reads.insert(nkey.clone());
            // With logits wanted, the per-part log-sum-exp of the no-question read is kept beside
            // its mass under its own key.
            let ckey = format!("{nkey}\nlogits per part");
            if want_logits {
                live_reads.insert(ckey.clone());
            }
            let kept_cal = if want_logits { st.reads.get(&ckey) } else { Some(Arc::new(Vec::new())) };
            let (b, b_cal) = match (st.reads.get(&nkey), kept_cal) {
                (Some(v), Some(c)) => {
                    scan.null_kept += 1;
                    ((*v).clone(), (*c).clone())
                }
                _ => {
                    let (v, c) = if want_logits { read_cal(&null)? } else { (read(&null)?, Vec::new()) };
                    scan.null_read += 1;
                    st.reads.put(&nkey, Arc::new(v.clone()));
                    if want_logits {
                        st.reads.put(&ckey, Arc::new(c.clone()));
                    }
                    (v, c)
                }
            };
            scan.read_ms += t.elapsed().as_secs_f64() * 1000.0;
            // Roadmap Phase 0 item 5 (WP-C5) probe (`NANO_HEAD_PROBE=<file>`, a measurement): every
            // head of every layer the window was encoded through, read with the posed and the
            // content-free text, its calibrated mass per window part (posed less content-free, per
            // question token), one JSON line per window with the lens and the parts' places. The
            // per-lens head sets are chosen from these lines on the fit fold and checked on the
            // check fold (`designs/drones.json` `tentacles`).
            if let Some(probe_file) = std::env::var_os("NANO_HEAD_PROBE") {
                let all: Vec<(usize, usize)> = (0..=last).flat_map(|l| (0..pop.dims.heads).map(move |h| (l, h))).collect();
                if let (Ok(pa), Ok(pb)) = (drone_read(&model, dev, &w, &stop, &all, &posed), drone_read(&model, dev, &w, &stop, &all, &null)) {
                    let mut line = format!(
                        "{{\"question\":\"{}\",\"lens\":\"{lens_name}\",\"parts\":[{}],\"heads\":[{}],\"mass\":[",
                        question.replace('\\', "\\\\").replace('"', "\\\""),
                        place.iter().map(|c| format!("\"{}:{}-{}\"", path_of(sub, *c), sub.chunks.get(*c).map_or(0, |x| x.start), sub.chunks.get(*c).map_or(0, |x| x.end))).collect::<Vec<_>>().join(","),
                        all.iter().map(|(l, h)| format!("[{l},{h}]")).collect::<Vec<_>>().join(",")
                    );
                    for (k, _) in all.iter().enumerate() {
                        let row: Vec<String> = w
                            .parts
                            .iter()
                            .skip(1)
                            .map(|&(start, n)| {
                                let s: f32 = (start..start + n).map(|p| pa.get(k * w.len + p).copied().unwrap_or(0.0) - pb.get(k * w.len + p).copied().unwrap_or(0.0)).sum();
                                format!("{s:.5}")
                            })
                            .collect();
                        line.push_str(&format!("{}[{}]", if k == 0 { "" } else { "," }, row.join(",")));
                    }
                    line.push_str("]}\n");
                    let _ = std::fs::OpenOptions::new().create(true).append(true).open(&probe_file).and_then(|mut h| std::io::Write::write_all(&mut h, line.as_bytes()));
                }
            }

            let mut files: Vec<String> = place.iter().map(|c| path_of(sub, *c)).collect();
            files.sort();
            files.dedup();
            let held: usize = place.iter().map(|c| tokens.get(*c).copied().unwrap_or(0)).sum();
            scan.window_tokens += held;
            rec.chunks.clone_from(&place);
            rec.tokens = held;
            rec.level = 2;
            rec.files = files;
            scan.records.push(rec);
            // The core keeps the window in RAM for its re-reads; its own device copies are dropped here
            // so eight kept windows never pin the card (the residency manager keeps a copy only
            // while its budget allows).
            let keep = settings.core.then(|| {
                w.release_device();
                Arc::clone(&w)
            });
            // 2026-11-10 (test T6): the core's tables from span reads over the kept encodings.
            let (a_cal, b_cal) = if settings.core && settings.span > 0 && !a_cal.is_empty() {
                let t = Instant::now();
                let r = span_tables(
                    &|sw: &Arc<Window>, text: &str| drone_read_logits(&model, dev, sw, &stop, heads, text),
                    &model,
                    &w.chunks,
                    heads.len(),
                    (&posed, &null),
                    settings.span,
                )?;
                scan.read_ms += t.elapsed().as_secs_f64() * 1000.0;
                r
            } else {
                (a_cal, b_cal)
            };
            scan.windows.push(WindowScan {
                tentacle: t_i,
                place,
                parts: w.parts.clone(),
                len: w.len,
                posed: a,
                null: b,
                cal_posed: a_cal,
                cal_null: b_cal,
                window: keep,
                lens_rank,
            });
        }
        // 2026-11-10 (shared RAM budget): the windows the core keeps hold their encodings alive,
        // so those bytes count against the kept chunks' limit; the least recently used chunks
        // not in a kept window are dropped until everything alive fits.
        if settings.core {
            let pinned = pinned_bytes(sub, &scan.windows);
            scan.kept_bytes = pinned.values().sum();
            // Drone-key packets held by a keys-only scan of this session share the same limit.
            let kb = st.keys.as_ref().map_or(0, |k| k.held().0);
            scan.alive_bytes = st.cache.fit_with(&pinned, st.limit.bytes.saturating_sub(kb)).1 + kb;
        }
        // Step P2: the base model keeps its device weights; the judge's model makes room in the
        // shared budget if it needs it (it was released here before every judge). Kept reads of
        // windows this question did not read are dropped (their content changed).
        drop(gate);
        scan.null_dropped = st.reads.retain(&live_reads);
        if let Some(o) = pop.gpu.as_ref() {
            let ps = o.gpu.pool_stats();
            scan.budget = format!(
                "{}; pool: {} MB idle in {} buffers (limit {} MB), {} reused, {} new, {} device allocations, {} frees",
                nmengn::residency::get().report(),
                ps.idle_bytes >> 20,
                ps.idle_buffers,
                ps.idle_limit >> 20,
                ps.hits,
                ps.misses,
                ps.device_allocs,
                ps.device_frees
            );
        }
        scan.held_bytes = st.cache.bytes();
        scan.held_chunks = st.cache.len();
        let scan = Arc::new(scan);
        st.last = Some(Arc::clone(&scan));
        Ok((scan, false))
    }

    /// The keys-only scan (2026-11-10, first form; `crate::key_scan`): per window, the drone-key
    /// packets of its chunks (from the store; a chunk the background pass has not reached yet is
    /// encoded now, its packet kept and its full encoding put in the chunk store), the question
    /// path for the posed and the content-free text (on the device, one short run each, behind the
    /// anchor at the window's end), and the raw-score table scored on the processor, one thread
    /// per tentacle. The windows are outlines: chunks whose full encodings are not held are
    /// filled by a tentacle core's re-read through `Scan::fill`. The store and the chunk store
    /// share one RAM limit (the one derived for the kept chunks): packets first, the chunk store
    /// gets the rest, kept windows counted.
    #[expect(clippy::too_many_lines, reason = "one scan, in the order it runs")]
    fn keys_scan(&self, st: &mut State, cx: &KeyCtx<'_>, gate: Option<nmengn::placement::GateGuard>) -> Result<Scan, Fault> {
        let model = cx.model;
        let pop = Arc::clone(&model.pop);
        let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let d = pop.dims;
        let group = d.heads / d.kv_heads.max(1);
        // Every head any tentacle reads, so one packet serves every tentacle's set.
        let mut all_heads: Vec<(usize, usize)> = cx.sets.heads.clone();
        for v in cx.sets.tentacles.values() {
            all_heads.extend(v.iter().copied());
        }
        all_heads.sort_unstable();
        all_heads.dedup();
        let limit = st.limit.bytes;
        let keys = Arc::clone(st.keys.get_or_insert_with(|| Arc::new(crate::key_scan::Keys::new(&all_heads, group, d.head_dim, limit))));
        // The background pass over the whole tree (no-op while one runs or after a finished pass
        // over this substrate version).
        let docs: Vec<(String, String)> = cx.sub.chunks.iter().enumerate().map(|(i, c)| (c.key.clone(), doc_of(cx.sub, i))).collect();
        // An organism store (pieces encoded alone) is filled by the organism's intake pass, not
        // by the anchored pass.
        let started_bg = !keys.alone && keys.start(Arc::clone(model), cx.dev, Arc::clone(cx.anchor), docs, cx.sub.version);
        let mut scan = Scan {
            question: cx.question.to_owned(),
            focused: cx.settings.focused,
            logits: true,
            version: cx.sub.version,
            chunks: cx.sub.chunks.len(),
            device: cx.dev.name().to_string(),
            bytes_per_token: keys.bytes_per_token,
            drone_windows: cx.settings.windows == crate::drone_windows::WindowMode::Drones,
            drones: cx.sets.source.clone(),
            last_layer: cx.last,
            limit: st.limit,
            gate_wait: gate.as_ref().map_or(0.0, |g| g.waited),
            gate_for: gate.as_ref().map(|g| g.waited_for.clone()).unwrap_or_default(),
            ..Scan::default()
        };
        let anchor_packet = Arc::new(nmengn::drone::RelevanceKeys::from_chunk(cx.ak, &keys.pairs, d.kv_heads, d.head_dim)?);
        let (mut from_store, mut from_cache) = (0_usize, 0_usize);
        let mut qpath_ms = 0.0;
        let mut jobs = Vec::new();
        let mut outlines: Vec<Outlined> = Vec::new();
        let mut names: BTreeMap<usize, (String, String)> = BTreeMap::new();
        for (t_i, win) in cx.windows.iter().enumerate() {
            let lens_name = lens::NAMES.get(t_i).copied().unwrap_or("lens");
            let focus_s = TeamWindows::focus_of(cx.team, t_i);
            let focus = focus_s.as_str();
            #[expect(clippy::cast_possible_truncation, reason = "eight tentacles")]
            let mut rec = WindowRecord {
                tentacle: t_i as u8,
                lens: lens_name.to_owned(),
                pose: if cx.settings.focused { "focus" } else { "plain" }.to_owned(),
                ..WindowRecord::default()
            };
            if win.is_empty() {
                scan.records.push(rec);
                continue;
            }
            let mut place: Vec<usize> = win.iter().copied().filter(|c| cx.sub.chunks.get(*c).is_some()).collect();
            place.sort_by(|a, b| {
                (path_of(cx.sub, *a), cx.sub.chunks.get(*a).map_or(0, |c| c.start))
                    .cmp(&(path_of(cx.sub, *b), cx.sub.chunks.get(*b).map_or(0, |c| c.start)))
            });
            let mut packets = vec![Arc::clone(&anchor_packet)];
            let mut parts: Vec<Arc<ChunkKv>> = vec![Arc::clone(cx.ak)];
            for &c in &place {
                let Some(ch) = cx.sub.chunks.get(c) else { continue };
                let full = st.cache.get(&ch.key);
                let p = if let Some(p) = keys.get(&ch.key) {
                    from_store += 1;
                    p
                } else if let Some(kv) = full.as_ref().filter(|_| !keys.alone) {
                    from_cache += 1;
                    let p = Arc::new(nmengn::drone::RelevanceKeys::from_chunk(kv, &keys.pairs, d.kv_heads, d.head_dim)?);
                    keys.put(&ch.key, Arc::clone(&p));
                    p
                } else {
                    let ids = tok(&doc_of(cx.sub, c))?;
                    let t = Instant::now();
                    let (p, kv) = keys.encode(model, cx.dev, cx.anchor, &ids)?;
                    let p = if keys.alone { keys.encode_alone(model, cx.dev, &ids)? } else { p };
                    scan.enc_ms += t.elapsed().as_secs_f64() * 1000.0;
                    scan.enc_tokens += ids.len();
                    if !st.ever.insert(ch.key.clone()) {
                        scan.enc_again_tokens += ids.len();
                    }
                    keys.put(&ch.key, Arc::clone(&p));
                    st.cache.put(&ch.key, Arc::clone(&kv));
                    p
                };
                let part = st
                    .cache
                    .get(&ch.key)
                    .unwrap_or_else(|| Arc::new(ChunkKv { tokens: p.tokens, anchor: cx.ak.tokens, ..ChunkKv::default() }));
                if part.k_pre.is_empty() {
                    names.insert(crate::scan_dump::addr(&part), (ch.key.clone(), doc_of(cx.sub, c)));
                }
                packets.push(p);
                parts.push(part);
            }
            let outline = Window::outline(&parts);
            let heads = cx.sets.for_tentacle(lens_name);
            let (posed, null) = if cx.settings.focused {
                (format!("\nQuestion: {}\nFocus: {focus}", cx.question), format!("\nQuestion: N/A\nFocus: {focus}"))
            } else {
                (format!("\nQuestion: {}", cx.question), "\nQuestion: N/A".to_owned())
            };
            let t = Instant::now();
            let qp = nmengn::drone::question_path(model, cx.dev, cx.anchor, &tok(&posed)?, outline.len, heads)?;
            let qn = nmengn::drone::question_path(model, cx.dev, cx.anchor, &tok(&null)?, outline.len, heads)?;
            qpath_ms += t.elapsed().as_secs_f64() * 1000.0;
            let held: usize = place.iter().map(|c| cx.tokens.get(*c).copied().unwrap_or(0)).sum();
            scan.window_tokens += held;
            let mut files: Vec<String> = place.iter().map(|c| path_of(cx.sub, *c)).collect();
            files.sort();
            files.dedup();
            rec.chunks.clone_from(&place);
            rec.tokens = held;
            rec.level = 2;
            rec.files = files;
            jobs.push(crate::key_scan::Job { packets, parts: outline.parts.clone(), len: outline.len, posed: qp, null: qn });
            outlines.push((t_i, place, parts, rec));
        }
        drop(gate);
        let (rows, score_ms) = crate::key_scan::score_windows(&jobs, d.head_dim, group, d.theta);
        scan.read_ms = qpath_ms + score_ms;
        for ((t_i, place, parts, rec), (job, r)) in outlines.into_iter().zip(jobs.iter().zip(rows)) {
            let (cal_posed, cal_null) = r?;
            let window = cx.settings.core.then(|| Arc::new(Window::outline(&parts)));
            scan.records.push(rec);
            scan.windows.push(WindowScan {
                tentacle: t_i,
                place,
                parts: job.parts.clone(),
                len: job.len,
                posed: Vec::new(),
                null: Vec::new(),
                cal_posed,
                cal_null,
                window,
                lens_rank: Vec::new(),
            });
        }
        scan.records.sort_by_key(|r| r.tentacle);
        // Shared budget: packets first, the chunk store the rest, kept windows counted.
        let (kb, kn, _) = keys.held();
        let pinned = if cx.settings.core { pinned_bytes(cx.sub, &scan.windows) } else { BTreeMap::new() };
        scan.kept_bytes = pinned.values().sum();
        st.cache.set_cap(limit.saturating_sub(kb));
        scan.alive_bytes = st.cache.fit_with(&pinned, limit.saturating_sub(kb)).1 + kb;
        scan.held_bytes = st.cache.bytes() + kb;
        scan.held_chunks = st.cache.len();
        let bg = keys.status();
        scan.keys = format!(
            "keys-only scan (first form): {} bytes per token over {} (layer, key-value head) pairs; packets from the store {from_store}, cut from held encodings {from_cache}, encoded now {} tokens in {:.0} ms; question path {qpath_ms:.0} ms (two short runs per window on {}), scoring {score_ms:.0} ms (one thread per tentacle); store {} MB in {kn} packets, limit {} MB shared with the chunk store ({} MB); background pass {} (pass {}, {} of {} chunks encoded, {} held before, {} tokens, {:.0} s, waited {:.1} s at the gate{})",
            keys.bytes_per_token,
            keys.pairs.len(),
            scan.enc_tokens,
            scan.enc_ms,
            cx.dev.name(),
            kb >> 20,
            limit >> 20,
            st.cache.bytes() >> 20,
            if started_bg {
                "started"
            } else if bg.running {
                "running"
            } else {
                "idle"
            },
            bg.passes,
            bg.encoded,
            bg.chunks,
            bg.held,
            bg.tokens,
            bg.wall_s,
            bg.waited_s,
            if bg.faults.is_empty() { String::new() } else { format!("; faults {}", bg.faults.join("; ")) }
        );
        if cx.settings.core {
            let state = Arc::downgrade(&self.state);
            let model = Arc::clone(model);
            let (dev, anchor, last) = (cx.dev, Arc::clone(cx.anchor), cx.last);
            let names = Arc::new(names);
            scan.fill = Some(Fill(Arc::new(move |parts: &[Arc<ChunkKv>]| {
                let mut out = Vec::with_capacity(parts.len());
                for p in parts {
                    if !p.k_pre.is_empty() || p.tokens == 0 {
                        out.push(Arc::clone(p));
                        continue;
                    }
                    let (key, doc) = names.get(&crate::scan_dump::addr(p)).ok_or_else(|| bad("an outlined chunk with no name"))?;
                    let st = state.upgrade().ok_or_else(|| bad("the pass is gone"))?;
                    let mut st = st.lock().map_err(|_| bad("pass state poisoned"))?;
                    let kv = if let Some(kv) = st.cache.get(key) {
                        kv
                    } else {
                        let ids = model.pop.tokenizer.encode(doc).map_err(|e| bad(e.to_string()))?;
                        let kv = Arc::new(encode_through(&model, dev, Some(&anchor), &ids, last)?);
                        st.cache.put(key, Arc::clone(&kv));
                        kv
                    };
                    out.push(kv);
                }
                Ok(out)
            })));
        }
        Ok(scan)
    }

    /// Source lines of chunk `k` of a window with their drone mass: (line, mass, tokens).
    fn lines_of(&self, sub: &Substrate, w: &WindowScan, k: usize) -> Result<Vec<(usize, f32, usize)>, Fault> {
        let Some((c, ch)) = w.place.get(k).and_then(|c| Some((*c, sub.chunks.get(*c)?))) else { return Ok(Vec::new()) };
        let mut st = self.state.lock().map_err(|_| bad("pass state poisoned"))?;
        if !st.rel_lines.contains_key(&ch.key) {
            // Source line of every token: the path line is 0, then the chunk's lines.
            let pop = Arc::clone(&crate::neural::model(&self.home)?.pop);
            let ids = pop.tokenizer.encode(&doc_of(sub, c)).map_err(|e| bad(e.to_string()))?;
            let (mut li, mut v) = (0_usize, Vec::new());
            for id in ids {
                v.push(li);
                li += pop.tokenizer.decode(&[id]).matches('\n').count();
            }
            st.rel_lines.insert(ch.key.clone(), v);
        }
        let mass = w.token_mass(k);
        let mut per: BTreeMap<usize, (f32, usize)> = BTreeMap::new();
        for (li, m) in st.rel_lines.get(&ch.key).map_or(&[][..], Vec::as_slice).iter().zip(&mass) {
            let e = per.entry(*li).or_default();
            e.0 += m;
            e.1 += 1;
        }
        Ok(per.into_iter().filter(|(li, _)| *li > 0).map(|(li, (sc, n))| (ch.start + li - 1, sc, n)).collect())
    }

    /// The judge over a scan: each tentacle shortlists by its drone score, the reranker (when
    /// one judges and the shortlist is longer than what is forwarded) chooses inside the
    /// tentacle, the forwarded chunks are ordered, and the bridge makes the packets.
    #[expect(clippy::too_many_arguments, reason = "the scan, its settings and the two facts for the note")]
    fn judge(
        &self,
        sub: &Substrate,
        slots: &Slots,
        scan: &Scan,
        settings: &Settings,
        reused: bool,
        started: Instant,
    ) -> Result<Pass, Fault> {
        let judge_started = Instant::now();
        let per = settings.per_tentacle.max(1);
        let short = if settings.judge == Judge::Heads { per } else { settings.shortlist.max(per) };
        let lists: Vec<Vec<(usize, f32)>> =
            scan.windows.iter().map(|w| w.ranked(settings.score).into_iter().take(short).collect()).collect();

        // The reranker's P(yes) for every shortlisted chunk, asked once per chunk and question.
        let mut p_all: Option<BTreeMap<usize, f32>> = None;
        if settings.judge != Judge::Heads {
            let mut known = scan.p.lock().map_err(|_| bad("scan scores poisoned"))?;
            let listed: Vec<usize> =
                scan.windows.iter().zip(&lists).flat_map(|(w, l)| l.iter().filter_map(|(k, _)| w.place.get(*k).copied())).collect();
            let need: Vec<usize> = listed.iter().copied().filter(|c| !known.contains_key(c)).collect();
            if !need.is_empty() {
                let docs: Vec<String> = need.iter().map(|c| doc_of(sub, *c)).collect();
                if let Some(v) = slots.rerank_set_neural(&scan.question, &docs) {
                    known.extend(need.iter().copied().zip(v));
                }
            }
            if !listed.is_empty() && listed.iter().all(|c| known.contains_key(c)) {
                p_all = Some(known.clone());
            }
        }

        let mut cands = Vec::new();
        let mut records = scan.records.clone();
        for (w, list) in scan.windows.iter().zip(&lists) {
            let mut chosen = list.clone();
            if let Some(p) = p_all.as_ref().filter(|_| short > per) {
                let pv = |k: usize| w.place.get(k).and_then(|c| p.get(c)).copied().unwrap_or(0.0);
                chosen.sort_by(|a, b| pv(b.0).total_cmp(&pv(a.0)));
            }
            chosen.truncate(per);
            let mut found = 0;
            for (place, &(k, score)) in chosen.iter().enumerate() {
                let Some(c) = w.place.get(k).copied() else { continue };
                let lines = if settings.lines { self.lines_of(sub, w, k)? } else { Vec::new() };
                cands.push(Cand { tentacle: w.tentacle, chunk: c, score, place, lines });
                found += 1;
            }
            if let Some(rec) = records.iter_mut().find(|r| usize::from(r.tentacle) == w.tentacle) {
                rec.best = list.first().map_or(0.0, |x| x.1);
                rec.no_information = rec.best <= 0.0;
                rec.findings = found;
            }
        }
        let p: Option<Vec<f32>> = p_all.as_ref().map(|m| cands.iter().map(|c| m.get(&c.chunk).copied().unwrap_or(0.0)).collect());
        // Background indexing waits at the device gate since the last note (step P2).
        let bg = nmengn::placement::gpu_gate().take_background_waits();
        let judged_by = if p.is_some() { settings.judge.name() } else { "heads" };
        let order = judge_order(&cands, p.as_deref(), settings.judge);
        let packets = bridge(sub, &cands, &order, settings);
        let note = format!(
            "tentacle pass on {}: {} windows, {} tokens held, score {}, shortlist {short}, {} candidates, judge {judged_by}; {}, reads {:.0} ms, judge {:.0} ms, total {:.0} ms; encoded chunks in RAM {} MB; kept chunks: {} bytes per token, layers 0..={} kept, limit {} MB ({}), free RAM read {} MB, {} bytes held in {} chunks, {} tokens encoded this question ({} of them encoded before and dropped at the limit); drone sets: {}; no-question reads kept {} computed {} dropped {}; device gate: {}; dense lane indexing waited at the gate {} times ({:.1} s); {}",
            scan.device,
            scan.windows.len(),
            scan.window_tokens,
            settings.score.name(),
            cands.len(),
            if reused {
                "scan kept from the same question".to_owned()
            } else {
                format!("encoded {} new tokens in {:.0} ms", scan.enc_tokens, scan.enc_ms)
            },
            scan.read_ms,
            judge_started.elapsed().as_secs_f64() * 1000.0,
            started.elapsed().as_secs_f64() * 1000.0,
            self.held_bytes() >> 20,
            scan.bytes_per_token,
            scan.last_layer,
            scan.limit.bytes >> 20,
            if scan.limit.fallback {
                "fallback, memory not read"
            } else if scan.limit.free == 0 {
                "named in the settings"
            } else {
                "half of the free RAM"
            },
            scan.limit.free >> 20,
            scan.held_bytes,
            scan.held_chunks,
            scan.enc_tokens,
            scan.enc_again_tokens,
            scan.drones,
            scan.null_kept,
            scan.null_read,
            scan.null_dropped,
            if scan.gate_for.is_empty() {
                "entered without waiting".to_owned()
            } else {
                format!("waited {:.1} s for {}", scan.gate_wait, scan.gate_for)
            },
            bg.0,
            bg.1,
            if scan.budget.is_empty() { "no device budget (CPU)" } else { scan.budget.as_str() },
        );
        Ok(Pass {
            packets,
            windows: records,
            order: order.iter().filter_map(|i| cands.get(*i).map(|c| c.chunk)).collect(),
            note,
            evidence: None,
        })
    }
}

/// A short Rust text for the layer check (written for it; not from any test).
const CHECK_CHUNK: &str = "crates/x/src/cache.rs\npub struct Held { bytes: usize, cap: usize }\nimpl Held {\n    pub fn room(&self) -> usize { self.cap.saturating_sub(self.bytes) }\n}\n";

/// Section check of the kept encoded chunks (2026-11-10), on the released base model: a chunk
/// encoded through the drone heads' last layer only (`encode_through`) holds keys and values
/// equal, bit for bit, to the same layers of the full encoding (`encode`), its bytes are the
/// full bytes' share of the layers kept, and the drone heads read the same attention mass on a
/// window of short encodings as on a window of full ones. The engine computes the same nodes
/// on the same inputs either way, so equality is exact; the largest difference is reported.
///
/// # Errors
/// One fault line.
pub fn check_layers() -> Result<(), String> {
    let home = nmtype::nano_root();
    let model = crate::neural::model(&home).map_err(|e| e.to_string())?;
    let pop = Arc::clone(&model.pop);
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let layers = pop.dims.layers;
    let last = drone_last_layer();
    if last + 1 >= layers {
        return Err(format!("the drone heads read layer {last} of {layers}: nothing to leave out"));
    }
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let (anchor_ids, chunk_ids, q) =
        (tok(SELF_ANCHOR)?, tok(CHECK_CHUNK)?, tok("\nQuestion: where is the free room of the held bytes computed")?);
    let stop = kvstore::score_targets(&model, last).map_err(|e| e.to_string())?;
    let heads = drones(&home).heads;
    // (anchor, chunk, drone mass per head pair) from short or full encodings.
    type Read = (Arc<ChunkKv>, Arc<ChunkKv>, Vec<f32>);
    let run = |short: bool| -> Result<Read, String> {
        let enc = |a: Option<&Arc<Window>>, ids: &[i32]| {
            if short { encode_through(&model, dev, a, ids, last) } else { encode(&model, dev, a, ids) }
        };
        let ak = Arc::new(enc(None, &anchor_ids).map_err(|e| e.to_string())?);
        let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
        let ck = Arc::new(enc(Some(&aw), &chunk_ids).map_err(|e| e.to_string())?);
        let w = Arc::new(Window::build(&model, &[Arc::clone(&ak), Arc::clone(&ck)]).map_err(|e| e.to_string())?);
        let probe = Arc::new(Probe::of_heads(&heads));
        model
            .forward_at(
                dev,
                &q,
                w.len,
                &stop,
                &RunOpts { window: Some(Arc::clone(&w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() },
            )
            .map_err(|e| e.to_string())?;
        let mut mass = Vec::new();
        for &(l, h) in &heads {
            let m = probe.get(l).ok_or_else(|| format!("layer {l} not read"))?;
            mass.extend((0..w.len).map(|p| m.get(h * w.len + p).copied().unwrap_or(0.0)));
        }
        Ok((ak, ck, mass))
    };
    let full = run(false);
    let short = run(true);
    pop.release_device();
    let ((fa, fc, fm), (sa, sc, sm)) = (full?, short?);
    if sc.layers() != last + 1 || sa.layers() != last + 1 || fc.layers() != layers {
        return Err(format!(
            "layers kept: short {} and {}, full {}; want {} and {layers}",
            sa.layers(),
            sc.layers(),
            fc.layers(),
            last + 1
        ));
    }
    for l in 0..=last {
        if sc.k_pre.get(l) != fc.k_pre.get(l)
            || sc.v.get(l) != fc.v.get(l)
            || sa.k_pre.get(l) != fa.k_pre.get(l)
            || sa.v.get(l) != fa.v.get(l)
        {
            return Err(format!("layer {l}: short keys or values differ from the full encoding"));
        }
    }
    if sc.bytes() * layers != fc.bytes() * (last + 1) || sc.bytes() != kvstore::bytes_per_token(&model, last + 1) * sc.tokens {
        return Err(format!("bytes: short {}, full {}, {} tokens", sc.bytes(), fc.bytes(), sc.tokens));
    }
    let worst = fm.iter().zip(&sm).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    if fm.len() != sm.len() || !fm.iter().any(|x| *x > 0.0) || worst != 0.0 {
        return Err(format!("drone readings differ: {} and {} values, largest difference {worst}", fm.len(), sm.len()));
    }
    Ok(())
}

/// Section check: the judge's rules, the line spans and the bridge, on a small in-memory
/// substrate (no model).
///
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let cand = |tentacle: usize, chunk: usize, score: f32, place: usize| Cand { tentacle, chunk, score, place, lines: Vec::new() };
    let c = vec![cand(0, 10, 0.9, 0), cand(0, 11, 0.2, 1), cand(1, 12, 0.5, 0), cand(1, 13, 0.4, 1)];
    let p = [0.1_f32, 0.95, 0.6, 0.7];
    let o = |j: Judge, p: Option<&[f32]>| judge_order(&c, p, j);
    if o(Judge::Heads, Some(&p)) != [0, 2, 3, 1] || o(Judge::Reranker, None) != [0, 2, 3, 1] {
        return Err("heads order wrong, or a judge without reranker scores is not the heads".into());
    }
    if o(Judge::Reranker, Some(&p)) != [1, 3, 2, 0] {
        return Err("reranker order wrong".into());
    }
    if o(Judge::WithinPlace, Some(&p)) != [2, 0, 1, 3] {
        return Err("within-place order wrong: first choices come before second choices".into());
    }
    if o(Judge::Sum, Some(&p)) != [1, 2, 3, 0] {
        return Err(format!("sum order wrong: {:?}", o(Judge::Sum, Some(&p))));
    }
    if o(Judge::Rrf, Some(&p)).len() != 4 || o(Judge::Reranker, Some(&p[..2])) != [0, 2, 3, 1] {
        return Err("fusion lost a candidate, or short reranker scores were used".into());
    }
    // Lines: half the positive mass, joined when one apart, at most two spans, best first.
    let lines = [(5, 0.1, 3), (6, 4.0, 3), (7, -1.0, 2), (8, 3.0, 4), (20, 2.5, 3), (30, 0.2, 1)];
    if key_spans(&lines) != [(6, 8)] {
        return Err(format!("key spans wrong: {:?}", key_spans(&lines)));
    }
    if !key_spans(&[]).is_empty() || key_spans(&[(3, -1.0, 2)]) != [(3, 3)] {
        return Err("key spans: empty stays empty, one line is kept".into());
    }
    // Scores, one head: chunk Y sits where the head looks under any question (large
    // content-free mass), chunk X is where the question moved it. Mass and density follow the
    // position; evidence and peak follow the question, measured from the anchor.
    let w = WindowScan {
        tentacle: 0,
        place: vec![7, 9],
        parts: vec![(0, 2), (2, 3), (5, 3)],
        len: 8,
        posed: vec![0.1, 0.1, 0.05, 0.05, 0.05, 0.25, 0.25, 0.25],
        null: vec![0.1, 0.1, 0.01, 0.01, 0.01, 0.2, 0.2, 0.2],
        ..WindowScan::default()
    };
    let first = |s: Score| w.ranked(s).first().map(|x| x.0);
    if first(Score::Mass) != Some(1)
        || first(Score::Density) != Some(1)
        || first(Score::Evidence) != Some(0)
        || first(Score::Peak) != Some(0)
    {
        return Err(format!(
            "scores: mass {:?}, evidence {:?}, peak {:?}",
            w.scores(Score::Mass),
            w.scores(Score::Evidence),
            w.scores(Score::Peak)
        ));
    }
    let ev = w.scores(Score::Evidence);
    if !ev.first().is_some_and(|x| (x - 5.0_f32.ln()).abs() < 1e-3) {
        return Err(format!("evidence is not the log ratio measured from the anchor: {ev:?}"));
    }
    if Score::ALL.iter().any(|s| Score::parse(s.name()) != Some(*s)) || Score::parse("other").is_some() {
        return Err("score names do not parse back".into());
    }
    // The bridge: keys by judged place, the lane's first evidence line, the tentacle's origin.
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let sub = nmsubs::from_files(&[("a/x.rs", "fn steady() {}\n"), ("a/y.rs", "fn other() {}\n")], embed);
    let at = |p: &str| (0..sub.chunks.len()).find(|i| path_of(&sub, *i) == p);
    let (Some(x), Some(y)) = (at("a/x.rs"), at("a/y.rs")) else {
        return Err("the check substrate has no chunks".into());
    };
    let mut cs = vec![cand(0, x, 0.3, 0), cand(2, y, 0.8, 0)];
    if let (Some(first), Some(ch)) = (cs.get_mut(1), sub.chunks.get(y)) {
        first.lines = vec![(ch.start, 2.0, 4)];
    }
    let order = judge_order(&cs, None, Judge::Heads);
    let packets = bridge(&sub, &cs, &order, &Settings::default());
    let keys: Vec<&str> = packets.iter().map(Packet::key).collect();
    let (Some(chx), Some(chy)) = (sub.chunks.get(x), sub.chunks.get(y)) else { return Err("chunks gone".into()) };
    if keys != [format!("tentacle:00:a/y.rs:{}", chy.start), format!("tentacle:01:a/x.rs:{}", chx.start)] {
        return Err(format!("bridge keys wrong: {keys:?}"));
    }
    let Some(first) = packets.first() else { return Err("no packets".into()) };
    if first.get_evidence().first() != Some(&format!("a/y.rs:{} lines {}..{}", chy.start, chy.start, chy.end)) || first.kind() != "chunk" {
        return Err("bridge packet: first evidence line or kind wrong".into());
    }
    if !first.provenance().origin.contains("nmpod.tentacle#t2/lens:names/pose:plain") {
        return Err(format!("bridge origin wrong: {}", first.provenance().origin));
    }
    Ok(())
}

/// Section check of step P0 item 4 (2026-11-10): the drone reader sets are data. The default
/// set `designs/drones.json` gives for the base population equals `DRONE_HEADS_QWEN3_0_6B`
/// pair for pair and is read from the file (not the fallback); `drone_last_layer()` is the
/// constant's last layer; the pass's own read (`drone_read`) over a small window gives the
/// same posed and content-free mass, bit for bit, with the file's set as with the constant,
/// and so the same chunk scores under every `Score`; malformed sets are refused by the
/// validator `check designs` uses. Values: `tests/section-checks/nmapi-drones/report.txt`.
///
/// # Errors
/// One fault line.
pub fn check_drones() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.tentacle_pass.drones (step P0 item 4)\n");
    let r = drones_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-drones").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn drones_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let constant = kvstore::DRONE_HEADS_QWEN3_0_6B.to_vec();
    let set = drones(&home);
    let _ = writeln!(rep, "set read: {} ({} default pairs, {} tentacle sets)", set.source, set.heads.len(), set.tentacles.len());
    if !set.from_file || set.heads != constant {
        return Err(format!("the default set is not the file's or differs from the constant: {}", set.source));
    }
    let want_last = Probe::of_heads(&constant).last_layer().unwrap_or(0);
    let _ = writeln!(rep, "drone_last_layer() {} (constant {want_last})", drone_last_layer());
    if drone_last_layer() != want_last {
        return Err(format!("drone_last_layer() {} differs from the constant's {want_last}", drone_last_layer()));
    }
    // The validator refuses malformed sets.
    let bad_sets = [
        ("a pair outside the layers", r#"{"populations":{"qwen3-0.6b":{"default":[[28,0]]}}}"#),
        ("a pair outside the heads", r#"{"populations":{"qwen3-0.6b":{"default":[[3,16]]}}}"#),
        ("a pair twice", r#"{"populations":{"qwen3-0.6b":{"default":[[3,1],[3,1]]}}}"#),
        ("an empty default", r#"{"populations":{"qwen3-0.6b":{"default":[]}}}"#),
        ("an unknown population", r#"{"populations":{"no-such-weights":{"default":[[1,1]]}}}"#),
        ("a malformed tentacle set", r#"{"populations":{"qwen3-0.6b":{"default":[[1,1]],"tentacles":{"names":[[1]]}}}}"#),
    ];
    for (what, text) in bad_sets {
        let v = nmjson::parse(text.as_bytes()).map_err(|e| e.to_string())?;
        match kvstore::drone_sets(&home, &v) {
            Ok(_) => return Err(format!("the validator accepted {what}")),
            Err(e) => {
                let _ = writeln!(rep, "refused {what}: {e}");
            }
        }
    }
    // The pass's read with the file's set and with the constant.
    let model = crate::neural::model(&home).map_err(|e| e.to_string())?;
    let pop = Arc::clone(&model.pop);
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let last = set.last_layer();
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| e.to_string());
    type Reads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<(usize, usize)>, usize);
    let run = || -> Result<Reads, String> {
        let ak = Arc::new(encode_through(&model, dev, None, &tok(SELF_ANCHOR)?, last).map_err(|e| e.to_string())?);
        let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
        let ck = Arc::new(encode_through(&model, dev, Some(&aw), &tok(CHECK_CHUNK)?, last).map_err(|e| e.to_string())?);
        let w = Arc::new(Window::build(&model, &[ak, ck]).map_err(|e| e.to_string())?);
        let stop = kvstore::score_targets(&model, last).map_err(|e| e.to_string())?;
        let (posed, null) = ("\nQuestion: where is the free room of the held bytes computed", "\nQuestion: N/A");
        let read = |heads: &[(usize, usize)], text: &str| drone_read(&model, dev, &w, &stop, heads, text).map_err(|e| e.to_string());
        Ok((read(&set.heads, posed)?, read(&set.heads, null)?, read(&constant, posed)?, read(&constant, null)?, w.parts.clone(), w.len))
    };
    let got = run();
    pop.release_device();
    let (fp, fnul, cp, cnul, parts, len) = got?;
    let same = fp.iter().map(|x| x.to_bits()).eq(cp.iter().map(|x| x.to_bits()))
        && fnul.iter().map(|x| x.to_bits()).eq(cnul.iter().map(|x| x.to_bits()));
    let _ = writeln!(
        rep,
        "{} read over {len} positions: {} posed and {} content-free values with the file's set; bit for bit equal to the constant's: {same}",
        dev.name(),
        fp.len(),
        fnul.len()
    );
    if !same || fp.iter().all(|x| *x == 0.0) {
        return Err("the read with the file's set differs from the read with the constant".into());
    }
    let scan = |posed: Vec<f32>, null: Vec<f32>| WindowScan {
        tentacle: 0,
        place: vec![0],
        parts: parts.clone(),
        len,
        posed,
        null,
        ..WindowScan::default()
    };
    let (a, b) = (scan(fp, fnul), scan(cp, cnul));
    for s in Score::ALL {
        let (x, y) = (a.scores(s), b.scores(s));
        let _ = writeln!(rep, "score {}: file {x:?}, constant {y:?}", s.name());
        if !x.iter().map(|v| v.to_bits()).eq(y.iter().map(|v| v.to_bits())) {
            return Err(format!("score {} differs between the file's set and the constant", s.name()));
        }
    }
    Ok(())
}

/// Section check of step P0 item 5 (2026-11-10): the served path (`from_args`, used by
/// `serve` and `trace`) gives shortlist 6, an explicit `--tentacle-shortlist` still wins, the
/// pass stays off without `--tentacles`, every other served setting is the default's, and
/// `Settings::default()` (what the locked `g1.rs` reads) still says 2.
///
/// # Errors
/// One fault line.
pub fn check_served() -> Result<(), String> {
    let args = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<String>>();
    let d = Settings::default();
    if d.shortlist != 2 {
        return Err(format!("the default shortlist is {}, want 2 (the locked g1.rs reads it)", d.shortlist));
    }
    let served = from_args(&args(&["serve", "--root", ".", "--tentacles"]))?.ok_or("--tentacles gave no settings")?;
    if served.shortlist != SERVED_SHORTLIST || SERVED_SHORTLIST != 6 {
        return Err(format!("served shortlist {}, want 6", served.shortlist));
    }
    let same_rest = Settings { shortlist: d.shortlist, ..served };
    if format!("{same_rest:?}") != format!("{d:?}") {
        return Err(format!("served settings differ from the default beyond the shortlist: {served:?}"));
    }
    let named = from_args(&args(&["trace", "--tentacles", "--tentacle-shortlist", "3", "q"]))?.ok_or("no settings")?;
    if named.shortlist != 3 {
        return Err(format!("an explicit --tentacle-shortlist 3 gave {}", named.shortlist));
    }
    if from_args(&args(&["serve", "--root", "."]))?.is_some() {
        return Err("the pass is on without --tentacles".into());
    }
    Ok(())
}

/// Section check of the calibrated score (tentacle core item 2, 2026-11-10): (a) `part_lse` and
/// `calibrated` on hand values (the log-sum-exp of known scores, posed minus content-free summed
/// over heads); (b) on the released base (GPU when open) over a small tree, a scan with the
/// calibrated score holds per-part rows for every window, and its posed and content-free mass
/// equal, bit for bit, those of a scan without logits (so `mass` and the other scores are
/// unchanged); (c) one chunk read in two different windows (other companions, other positions):
/// its calibrated score and its mass score in both, reported (a measurement, not a gate).
/// Values: `tests/section-checks/nmapi-calibrated/report.txt`.
/// # Errors
/// A difference, or an engine fault.
pub fn check_calibrated() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.tentacle_pass.calibrated (tentacle core item 2)\n");
    let r = calibrated_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-calibrated").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn calibrated_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    // (a) Two tokens over a span of 4 with parts (0, 1) and (1, 3).
    let lg = [0.0_f32, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0];
    let got = part_lse(&lg, 2, 4, &[(0, 1), (1, 3)]);
    let want1 = (3.0_f32.ln() + 1.0 + 3.0_f32.ln()) / 2.0;
    if got.len() != 2
        || (got.first().copied().unwrap_or(9.0) - 1.0).abs() > 1e-6
        || (got.get(1).copied().unwrap_or(9.0) - want1).abs() > 1e-6
    {
        return Err(format!("part_lse on hand values: {got:?}, want [1.0, {want1}]"));
    }
    let c = calibrated(&[1.0, 2.0, 0.5, 0.5], &[0.5, 1.0, 0.0, 1.0], 2);
    if c != [1.0, 0.5] || !calibrated(&[1.0], &[1.0, 2.0], 2).is_empty() {
        return Err(format!("calibrated on hand values: {c:?}, want [1.0, 0.5]"));
    }
    let _ = writeln!(rep, "(a) part_lse {got:?}; calibrated [1.0, 0.5]; mismatched rows give none");
    // (b) Scans with and without logits on one tree.
    let home = nmtype::nano_root();
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let files = [
        (
            "a/ring.rs",
            "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n",
        ),
        (
            "a/clock.rs",
            "pub fn tick(now: u64, step: u64) -> u64 { now.wrapping_add(step) }\npub fn late(now: u64, due: u64) -> bool { now > due }\n",
        ),
        (
            "a/limit.rs",
            "pub struct Limit { bytes: usize }\nimpl Limit {\n    pub fn room(&self, used: usize) -> usize { self.bytes.saturating_sub(used) }\n}\n",
        ),
    ];
    let sub = nmsubs::from_files(&files, Arc::clone(&embed));
    let q = "where is the free room of a limit computed?";
    let plain = Settings::default();
    let cal = Settings { score: Score::Calibrated, ..plain };
    let (s0, _) = TentaclePass::new(&home, plain).scan(&sub, q, &plain).map_err(|e| e.to_string())?;
    let (s1, _) = TentaclePass::new(&home, cal).scan(&sub, q, &cal).map_err(|e| e.to_string())?;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let same = s0.windows.len() == s1.windows.len()
        && s0
            .windows
            .iter()
            .zip(&s1.windows)
            .all(|(a, b)| a.place == b.place && bits(&a.posed) == bits(&b.posed) && bits(&a.null) == bits(&b.null));
    let rows = s1.windows.iter().all(|w| w.cal_heads() > 0 && w.scores(Score::Calibrated).iter().all(|x| x.is_finite()));
    let none = s0.windows.iter().all(|w| w.cal_posed.is_empty() && w.scores(Score::Calibrated).iter().all(|x| *x == f32::NEG_INFINITY));
    for w in &s1.windows {
        let _ = writeln!(
            rep,
            "(b) {} window {} ({} chunks, {} heads): mass {:?}; calibrated {:?}",
            s1.device,
            w.tentacle,
            w.place.len(),
            w.cal_heads(),
            w.scores(Score::Mass),
            w.scores(Score::Calibrated)
        );
    }
    let _ = writeln!(
        rep,
        "(b) posed and content-free mass with logits equal those without, bit for bit: {same}; every window holds calibrated rows: {rows}; a scan without logits has none: {none}"
    );
    if !same || !rows || !none || s1.windows.is_empty() {
        return Err("the calibrated scan changed the mass, or its rows are missing".into());
    }
    // (c) One chunk in two windows.
    let model = crate::neural::model(&home).map_err(|e| e.to_string())?;
    let pop = Arc::clone(&model.pop);
    let dev = if pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let set = drones(&home);
    let last = set.last_layer();
    let tok = |s: &str| pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let ak = Arc::new(encode_through(&model, dev, None, &tok(SELF_ANCHOR)?, last).map_err(|e| e.to_string())?);
    let aw = Arc::new(Window::build(&model, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
    let enc = |t: &str| -> Result<Arc<ChunkKv>, String> {
        Ok(Arc::new(encode_through(&model, dev, Some(&aw), &tok(t)?, last).map_err(|e| e.to_string())?))
    };
    let docs: Vec<String> = files.iter().map(|(p, t)| nmsubs::dense::doc_text(p, t)).collect();
    let (x, y, z) = (
        enc(docs.get(2).map_or("", String::as_str))?,
        enc(docs.first().map_or("", String::as_str))?,
        enc(docs.get(1).map_or("", String::as_str))?,
    );
    let stop = kvstore::score_targets(&model, last).map_err(|e| e.to_string())?;
    let read = |parts: Vec<Arc<ChunkKv>>, k: usize| -> Result<(f32, f32), String> {
        let w = Arc::new(Window::build(&model, &parts).map_err(|e| e.to_string())?);
        let (pm, pc) = drone_read_logits(&model, dev, &w, &stop, &set.heads, &format!("\nQuestion: {q}")).map_err(|e| e.to_string())?;
        let (nm, nc) = drone_read_logits(&model, dev, &w, &stop, &set.heads, "\nQuestion: N/A").map_err(|e| e.to_string())?;
        let ws =
            WindowScan { parts: w.parts.clone(), len: w.len, posed: pm, null: nm, cal_posed: pc, cal_null: nc, ..WindowScan::default() };
        Ok((ws.scores(Score::Calibrated).get(k).copied().unwrap_or(f32::NAN), ws.scores(Score::Mass).get(k).copied().unwrap_or(f32::NAN)))
    };
    let (ca, ma) = read(vec![Arc::clone(&ak), Arc::clone(&x), Arc::clone(&y)], 0)?;
    let (cb, mb) = read(vec![Arc::clone(&ak), Arc::clone(&y), Arc::clone(&z), Arc::clone(&x)], 2)?;
    let _ = writeln!(
        rep,
        "(c) {}: the limit chunk first in a window of 2 chunks, then last in a window of 3: calibrated {ca:.4} and {cb:.4}; mass {ma:.4} and {mb:.4}",
        dev.name()
    );
    if !(ca.is_finite() && cb.is_finite()) {
        return Err("calibrated score of the moved chunk is not finite".into());
    }
    nmengn::residency::get().release_windows();
    Ok(())
}

/// Section check of step P2 fix 2 (2026-11-10), the kept no-question reads: (a) `KeptReads`
/// keeps, finds and drops entries by key; (b) on the released base (GPU when open), a pass reads
/// a small tree for two questions: the second question takes the no-question read of every
/// window whose content did not change from the kept reads, and those values equal, bit for bit,
/// the no-question reads a fresh pass computes for the same question; (c) after one file of the
/// tree changes, the kept read of the changed window is dropped and computed again.
/// # Errors
/// A difference, or an engine fault.
pub fn check_kept_reads() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.tentacle_pass.kept_reads (step P2 fix 2)\n");
    let r = kept_reads_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-kept-reads").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn kept_reads_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let mut k = KeptReads::default();
    k.put("a", Arc::new(vec![1.0]));
    k.put("b", Arc::new(vec![2.0]));
    let live: std::collections::BTreeSet<String> = std::iter::once("b".to_owned()).collect();
    let found = k.get("a").is_some();
    let dropped = k.retain(&live);
    if !found || dropped != 1 || k.get("a").is_some() || k.get("b").is_none() || k.len() != 1 || k.bytes() != 4 {
        return Err("KeptReads: keep, find or drop by key wrong".into());
    }
    let _ = writeln!(rep, "(a) KeptReads: kept 2, found by key, retain dropped {dropped} not in use, counts {:?}", k.counts());
    let home = nmtype::nano_root();
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let files = [
        (
            "a/ring.rs",
            "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n",
        ),
        (
            "a/clock.rs",
            "pub fn tick(now: u64, step: u64) -> u64 { now.wrapping_add(step) }\npub fn late(now: u64, due: u64) -> bool { now > due }\n",
        ),
        (
            "a/limit.rs",
            "pub struct Limit { bytes: usize }\nimpl Limit {\n    pub fn room(&self, used: usize) -> usize { self.bytes.saturating_sub(used) }\n}\n",
        ),
    ];
    let sub = nmsubs::from_files(&files, Arc::clone(&embed));
    let settings = Settings::default();
    let pass = TentaclePass::new(&home, settings);
    let (q1, q2, q3) = ("where is it kept?", "where is it held?", "where is it stored?");
    let (s1, _) = pass.scan(&sub, q1, &settings).map_err(|e| e.to_string())?;
    let (s2, _) = pass.scan(&sub, q2, &settings).map_err(|e| e.to_string())?;
    let fresh = TentaclePass::new(&home, settings);
    let (s3, _) = fresh.scan(&sub, q2, &settings).map_err(|e| e.to_string())?;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let same_windows =
        s1.windows.iter().map(|w| (w.tentacle, w.place.clone())).eq(s2.windows.iter().map(|w| (w.tentacle, w.place.clone())));
    let equal = s2.windows.len() == s3.windows.len()
        && s2
            .windows
            .iter()
            .zip(&s3.windows)
            .all(|(a, b)| a.place == b.place && bits(&a.null) == bits(&b.null) && bits(&a.posed) == bits(&b.posed));
    let _ = writeln!(
        rep,
        "(b) {} on a tree of {} chunks: question 1 computed {} no-question reads (kept {}); question 2 (windows unchanged {same_windows}) kept {} computed {}; a fresh pass on question 2 computed {}; kept values equal the fresh pass's bit for bit {equal}",
        s1.device,
        sub.chunks.len(),
        s1.null_read,
        s1.null_kept,
        s2.null_kept,
        s2.null_read,
        s3.null_read
    );
    if s1.null_read == 0 || s2.null_kept == 0 || !equal || (same_windows && s2.null_read != 0) {
        return Err("kept no-question reads not taken, or they differ from a fresh pass".into());
    }
    let mut changed = files;
    changed[1].1 =
        "pub fn tick(now: u64, step: u64) -> u64 { now.saturating_add(step) }\npub fn late(now: u64, due: u64) -> bool { now >= due }\n";
    let sub2 = nmsubs::from_files(&changed, embed);
    let (s4, _) = pass.scan(&sub2, q3, &settings).map_err(|e| e.to_string())?;
    let _ = writeln!(
        rep,
        "(c) after one file changed: kept {} computed {} dropped {} (the changed window's read)",
        s4.null_kept, s4.null_read, s4.null_dropped
    );
    if let Some(o) = crate::neural::model(&home).map_err(|e| e.to_string())?.pop.gpu.as_ref() {
        let _ = writeln!(rep, "{}; pool reused {} buffers", nmengn::residency::get().report(), o.gpu.pool_stats().hits);
    }
    nmengn::residency::get().release_windows();
    if s4.null_read == 0 || s4.null_dropped == 0 {
        return Err("a changed window's kept read was not dropped and computed again".into());
    }
    Ok(())
}

/// Section check of the keys-only scan wired into the pass (2026-11-10, first form), on the
/// released base (GPU when open) over a three-file tree: (a) the exact scan with the core on is
/// unchanged by the option being present (same windows and the same table bits as a pass made
/// without it); (b) with `keys` on, every window holds a calibrated table of the same shape, the
/// background pass finishes over the tree, and a second question takes every packet from the store
/// (no encoding); (c) the tentacle core runs on the keys-only scan through the G1 loop, its re-reads
/// fill outlined chunks on demand, the isolation gate reads 0 and nothing is refused; (d) the kept
/// windows are counted in the shared budget (`kept_bytes`, `alive_bytes`); (e) keys-only against
/// exact rank correlation of the chunk scores per window is reported (a measurement, not a gate).
/// # Errors
/// A difference, or an engine fault.
pub fn check_keys() -> Result<(), String> {
    use std::fmt::Write as _;
    let mut rep = String::from("nmapi.tentacle_pass.keys (keys-only scan, first form)\n");
    let r = keys_check(&mut rep);
    let _ = writeln!(rep, "result: {}", r.as_ref().map_or_else(|e| format!("FAIL {e}"), |()| "ok".to_owned()));
    let dir = nmtype::check_dir("nmapi-keys").map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.txt"), rep).map_err(|e| e.to_string())?;
    r
}
fn keys_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let embed: nmsubs::Embed = Arc::new(|_: &str| vec![0.0; 4]);
    let files = [
        (
            "a/ring.rs",
            "pub struct Ring { items: Vec<u32>, head: usize }\nimpl Ring {\n    pub fn push(&mut self, x: u32) { let n = self.items.len(); self.items[self.head % n] = x; self.head += 1; }\n}\n",
        ),
        (
            "a/clock.rs",
            "pub fn tick(now: u64, step: u64) -> u64 { now.wrapping_add(step) }\npub fn late(now: u64, due: u64) -> bool { now > due }\n",
        ),
        (
            "a/limit.rs",
            "pub struct Limit { bytes: usize }\nimpl Limit {\n    pub fn room(&self, used: usize) -> usize { self.bytes.saturating_sub(used) }\n}\n",
        ),
    ];
    let sub = nmsubs::from_files(&files, embed);
    let q = "where is the free room of a limit computed?";
    let exact = Settings { core: true, ..Settings::served() };
    let keyed = Settings { keys: true, ..exact };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // (a) The exact path.
    let (e1, _) = TentaclePass::new(&home, exact).scan(&sub, q, &exact).map_err(|e| e.to_string())?;
    let (e2, _) = TentaclePass::new(&home, exact).scan(&sub, q, &exact).map_err(|e| e.to_string())?;
    let same = e1.windows.len() == e2.windows.len()
        && e1
            .windows
            .iter()
            .zip(&e2.windows)
            .all(|(a, b)| a.place == b.place && bits(&a.cal_posed) == bits(&b.cal_posed) && bits(&a.posed) == bits(&b.posed))
        && e1.keys.is_empty();
    let _ = writeln!(
        rep,
        "(a) exact scan with the core: repeat identical and no keys note: {same}; kept windows hold {} bytes, alive {} bytes of the {} byte limit",
        e1.kept_bytes, e1.alive_bytes, e1.limit.bytes
    );
    if !same || e1.kept_bytes == 0 || e1.alive_bytes < e1.kept_bytes {
        return Err("the exact scan changed, or the kept windows are not counted".into());
    }
    // (b) Keys-only.
    let pass = TentaclePass::new(&home, keyed);
    let (k1, _) = pass.scan(&sub, q, &keyed).map_err(|e| e.to_string())?;
    let shapes = k1.windows.len() == e1.windows.len()
        && k1.windows.iter().zip(&e1.windows).all(|(a, b)| {
            a.place == b.place
                && a.parts == b.parts
                && a.cal_posed.len() == b.cal_posed.len()
                && a.cal_null.len() == b.cal_null.len()
                && a.scores(Score::Calibrated).iter().all(|x| x.is_finite())
        });
    let keys = pass.state.lock().map_err(|_| "state")?.keys.clone().ok_or("no key store")?;
    let t = Instant::now();
    while keys.status().running && t.elapsed().as_secs() < 120 {
        #[expect(clippy::disallowed_methods, reason = "a check waiting for a background thread; no async runtime here")]
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let bg = keys.status();
    let (k2, _) = pass.scan(&sub, "how does a ring push an item?", &keyed).map_err(|e| e.to_string())?;
    let _ = writeln!(rep, "(b) {}; tables of the exact shape, finite: {shapes}; background pass: {:?}", k1.keys, bg);
    let _ = writeln!(rep, "(b) second question: {} tokens encoded; {}", k2.enc_tokens, k2.keys);
    if !shapes || bg.running || !bg.faults.is_empty() || bg.passes == 0 || k2.enc_tokens != 0 {
        return Err("keys-only tables, background pass or the store's reuse".into());
    }
    // (c) The core on the keys-only scan.
    let (p, cr) = crate::tentacle_core::judge(&home, &sub, &k1, &keyed).map_err(|e| e.to_string())?;
    let _ = writeln!(
        rep,
        "(c) core on the keys-only scan: forwarded {} spans, rounds {}, device reads {}, faults {:?}, gate {}, refused {}",
        p.order.len(),
        cr.outcome.handover.rounds,
        cr.reads,
        cr.faults,
        cr.outcome.handover.gate_direct,
        cr.refused
    );
    if p.order.is_empty() || !cr.faults.is_empty() || cr.outcome.handover.gate_direct != 0 || cr.refused != 0 {
        return Err("the core on the keys-only scan forwarded nothing, faulted, or broke isolation".into());
    }
    // (e) Agreement on this tree.
    for (a, b) in k1.windows.iter().zip(&e1.windows) {
        let rho = crate::key_scan::spearman(&a.scores(Score::Calibrated), &b.scores(Score::Calibrated));
        let _ = writeln!(
            rep,
            "(e) window {}: keys {:?} exact {:?} rank correlation {rho:?}",
            a.tentacle,
            a.scores(Score::Calibrated),
            b.scores(Score::Calibrated)
        );
    }
    nmengn::residency::get().release_windows();
    Ok(())
}
