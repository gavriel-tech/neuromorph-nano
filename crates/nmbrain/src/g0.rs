//! G0, the per-tentacle level of the Globalized Context Stack (step P6, first form, 2026-11-10; design by
//! Luis Jake Gabriel III, `docs/2026-11-10-neumor-guide-how-it-works.md` sections 2.4 and 3).
//! Each tentacle keeps its own G0 instance, `g0_t1` to `g0_t8` (generally `g0_tx`; the count
//! follows the tentacles in use). A G0 holds, in RAM:
//! - references to the encoded chunks of the tentacle's window, by content hash (the encodings
//!   themselves stay in the tentacle pass's chunk store; nothing is copied here);
//! - the drones' readings as a compact quantized table (one signed byte per drone and chunk,
//!   one scale per drone);
//! - its current focus, its report (best spans as references with calibrated confidence, a
//!   no-information flag with the baseline it was judged against, what it would read next);
//! - a small inbox for instructions from above. Only G1 may write to it: an envelope from any
//!   other sender is refused and counted (the isolation gate G1 reads, which must stay at 0).
//!
//! `TentacleSide` is the boundary G1 calls: report, re-read under a new focus, answer a specific
//! question about one span. The first form here (`G0` itself implements it) works over data
//! structures only, fed from what the tentacle pass already produces (the window record, the
//! window's chunks and its per-statistic chunk scores), so it runs and is tested on the
//! processor. The neural tentacle core plugs in later by implementing the same trait.
use crate::evidence::SpanFacts;
use crate::gmap::WindowRecord;
use std::collections::{BTreeMap, VecDeque};

/// The only sender allowed to write into a G0 inbox.
pub const G1_ID: &str = "g1";

/// The G0 identity of tentacle `t` (0-based lens index): `g0_t{t+1}`.
#[must_use]
pub fn g0_id(t: u8) -> String {
    format!("g0_t{}", u16::from(t) + 1)
}

/// A reference to one held chunk: where it sits and its content hash. Never an encoding copy.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanRef {
    pub file: String,
    pub start: usize,
    pub end: usize,
    /// Content hash of the chunk (the substrate chunk key, also the encoded-chunk store key).
    pub key: String,
}
impl SpanRef {
    /// `file:start-end`.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{}:{}-{}", self.file, self.start, self.end)
    }
    /// Same file and overlapping lines.
    #[must_use]
    pub fn overlaps(&self, o: &Self) -> bool {
        self.file == o.file && self.start <= o.end && o.start <= self.end
    }
    /// True when line `l` of `file` lies inside the span.
    #[must_use]
    pub fn contains(&self, file: &str, l: usize) -> bool {
        self.file == file && self.start <= l && l <= self.end
    }
}

/// One chunk of the window: its reference, its substrate chunk index and its tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldChunk {
    pub span: SpanRef,
    pub chunk: usize,
    pub tokens: usize,
}

/// The drones' readings of the window, quantized: `cells[d * chunks + c]` is drone `d` on held
/// chunk `c` as a signed byte, `scales[d]` turns it back (value = cell * scale).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DroneTable {
    pub drones: usize,
    pub chunks: usize,
    pub scales: Vec<f32>,
    pub cells: Vec<i8>,
}
impl DroneTable {
    /// Quantize one row per drone (one value per held chunk). Rows shorter than the longest are
    /// read as zero beyond their end. The scale per drone is its largest magnitude over 127.
    #[must_use]
    pub fn quantize(rows: &[Vec<f32>]) -> Self {
        let chunks = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut scales = Vec::with_capacity(rows.len());
        let mut cells = Vec::with_capacity(rows.len() * chunks);
        for r in rows {
            let peak = r.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales.push(scale);
            for c in 0..chunks {
                let v = r.get(c).copied().unwrap_or(0.0) / scale;
                cells.push(v.round().clamp(-127.0, 127.0) as i8);
            }
        }
        Self { drones: rows.len(), chunks, scales, cells }
    }
    /// Drone `d` on chunk `c`, dequantized.
    #[must_use]
    pub fn get(&self, d: usize, c: usize) -> f32 {
        let cell = self.cells.get(d * self.chunks + c).copied().unwrap_or(0);
        f32::from(cell) * self.scales.get(d).copied().unwrap_or(0.0)
    }
    /// The sum over drones on chunk `c` (the window's drone score of that chunk).
    #[must_use]
    pub fn chunk_sum(&self, c: usize) -> f32 {
        (0..self.drones).map(|d| self.get(d, c)).sum()
    }
    /// Bytes the table holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.cells.len() + self.scales.len() * 4
    }
}

/// What a tentacle reads for: its lens and pose, the statistic it ranks by, and optional files
/// and terms G1 aims it at.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Focus {
    pub lens: String,
    pub pose: String,
    /// Score statistic name (`mass`, `density`, `evidence`, `peak` today); empty lets the
    /// tentacle choose its next unused one.
    pub statistic: String,
    pub files: Vec<String>,
    pub terms: Vec<String>,
}

/// One reported span with the tentacle's own score, calibrated confidence and place.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanReport {
    pub span: SpanRef,
    pub score: f32,
    /// Calibrated: 0.5 at the baseline, towards 1 the further above it (scale from the window).
    pub confidence: f32,
    /// Place in this report (0 best).
    pub place: usize,
}

/// A structured report from one `g0_tx` to G1.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub from: String,
    pub tentacle: u8,
    pub best: Vec<SpanReport>,
    /// True when no held chunk scored above the baseline: an explicit "nothing here".
    pub no_information: bool,
    pub best_score: f32,
    /// The no-information baseline the scores were judged against.
    pub baseline: f32,
    /// Confidence of the report as a whole (the best span's, 0 with no information).
    pub confidence: f32,
    /// What the tentacle would read next (held spans below the reported ones).
    pub next: Vec<SpanRef>,
    pub focus: Focus,
    /// Reads made so far (1 after the first report).
    pub reads: u32,
    /// Evidence of every held span by content key (2026-11-10, refinement step 1): its tokens,
    /// window rank and margin, every score with the window's content-free baseline, its mass
    /// rank, lens and lens rank. Empty from a tentacle that gives none.
    pub facts: BTreeMap<String, SpanFacts>,
    /// Refinement step 2a (2026-11-10): the components of the tentacle core's read that were on
    /// (empty: the first form), and for a second look its shortlist size and how it was set.
    pub read: String,
    /// Reflexes that fired on this read (`crate::reflex`), one line each.
    pub reflexes: Vec<String>,
    /// Spans marked self-caused by R4 (the organism's own edit), by content key.
    pub self_caused: Vec<String>,
    /// The tentacle's spread: the median absolute deviation of its shortlist's scores (the mean
    /// absolute deviation where that is zero, 1 where they do not spread).
    pub spread: f32,
    /// Its confidence as the tentacle core defines it: the first span's margin over the second,
    /// in units of the spread.
    pub margin: f32,
    /// The request's aspects no held piece holds: no information on them in this window.
    pub absent: Vec<String>,
    /// The epistemic tag of each reported span (`defines`, `uses`, `mentions`, `tests`,
    /// `documents`), in the order of `best`; empty without the drones' evidence.
    pub tags: Vec<String>,
    /// What it would look at next beyond its window: pieces related to its best piece that it
    /// does not hold (a request to G1, by reference).
    pub wants: Vec<SpanRef>,
}

/// What the drones' calibrated evidence says about each held piece, for the report: the
/// request's aspects, the aspects each held piece holds, each held piece's epistemic tag, and
/// the pieces related to each held piece that the window does not hold.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readout {
    pub aspects: Vec<String>,
    /// Per held piece: accepted by the drones for some aspect (above the content-free band).
    pub accepted: Vec<bool>,
    pub covers: Vec<Vec<usize>>,
    pub tags: Vec<String>,
    pub related: Vec<Vec<SpanRef>>,
    /// Per held piece: the independent families that accept it for some aspect (its quorum).
    pub quorum: Vec<usize>,
}

/// An instruction from above.
#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    Reread(Focus),
    Ask { span: SpanRef, about: String },
}

/// An addressed instruction: the in-house message envelope (`nmtype::message`, kind
/// `instruction`, sender, target, confidence, the reason as its description, the span as a
/// reference) plus the typed instruction it carries and the round. Behavioural traffic only.
#[derive(Clone, Debug, PartialEq)]
pub struct Order {
    pub msg: nmtype::Envelope,
    pub round: u32,
    pub instruction: Instruction,
}
impl Order {
    /// Make an order from `from` to `to` with the reason `why` (one line).
    /// # Errors
    /// The envelope's own refusals (empty ids, a multi-line reason, a bad span).
    pub fn new(from: &str, to: &str, round: u32, confidence: f32, why: &str, instruction: Instruction) -> Result<Self, String> {
        let refs = match &instruction {
            Instruction::Ask { span, .. } => vec![nmtype::Reference::Span {
                path: span.file.clone(),
                start: u32::try_from(span.start.max(1)).unwrap_or(u32::MAX),
                end: u32::try_from(span.end.max(span.start).max(1)).unwrap_or(u32::MAX),
            }],
            Instruction::Reread(_) => Vec::new(),
        };
        let msg = nmtype::Envelope::new(nmtype::MessageKind::Instruction, from, to, confidence, why, refs, Vec::new())
            .map_err(|f| f.to_string())?;
        Ok(Self { msg, round, instruction })
    }
}

/// A tentacle's answer about one span.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub from: String,
    pub span: SpanRef,
    /// The span is in this tentacle's window.
    pub held: bool,
    pub score: f32,
    pub confidence: f32,
    /// Its rank among the window's chunks under the current statistic.
    pub rank: Option<usize>,
    pub no_information: bool,
}

/// What processing the inbox produced.
#[derive(Clone, Debug, PartialEq)]
pub enum Reply {
    Report(Box<Report>),
    Answer(Answer),
}

/// The inbox of one `g0_tx`: envelopes from G1 only, bounded; refused senders are counted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inbox {
    pub queue: VecDeque<Order>,
    /// Envelopes refused because the sender was not G1 (the isolation gate).
    pub refused_direct: usize,
    /// Envelopes refused because the inbox was full.
    pub refused_full: usize,
    /// Bound: derived by the owner (first form: held chunks plus one re-read).
    pub bound: usize,
}

/// The calls G1 makes on one tentacle. The neural tentacle core implements the first four; the
/// inbox calls have default bodies over `inbox_mut`.
pub trait TentacleSide: Send {
    fn id(&self) -> &str;
    /// References of every chunk the window holds.
    fn held(&self) -> Vec<SpanRef>;
    /// Re-reads with a different statistic still available.
    fn options(&self) -> usize;
    /// The current report (a fresh read on the first call).
    fn report(&mut self) -> Report;
    /// Read the window again under a new focus and report.
    fn reread(&mut self, focus: &Focus) -> Report;
    /// Answer a specific question about one span.
    fn answer(&mut self, span: &SpanRef, about: &str) -> Answer;
    fn inbox_mut(&mut self) -> &mut Inbox;
    fn inbox(&self) -> &Inbox;
    /// Put an envelope in the inbox: refused (and counted) unless it comes from G1 and is
    /// addressed to this tentacle.
    fn deliver(&mut self, env: Order) -> Result<(), String> {
        let me = self.id().to_owned();
        let inbox = self.inbox_mut();
        if env.msg.sender() != G1_ID {
            inbox.refused_direct += 1;
            return Err(format!("{me}: refused an envelope from {} (only G1 may send to a g0_tx)", env.msg.sender()));
        }
        if env.msg.target() != me {
            return Err(format!("{me}: envelope addressed to {}", env.msg.target()));
        }
        if inbox.queue.len() >= inbox.bound.max(1) {
            inbox.refused_full += 1;
            return Err(format!("{me}: inbox full at {}", inbox.queue.len()));
        }
        inbox.queue.push_back(env);
        Ok(())
    }
    /// F21 (Luis 2026-11-18 22:41 UTC; lane M's `motor::Down::send_direct`): an envelope from G2
    /// about this tentacle's own focus, admitted only with G1's acknowledgement of that envelope
    /// (`ledger.verify`), so the copy to G1 cannot be skipped; any other sender, verb or a missing
    /// acknowledgement is refused and counted as a direct refusal.
    fn deliver_direct(&mut self, env: Order, m: &nmtype::message::Motor, ack: &crate::g1::Ack, ledger: &crate::g1::Ledger) -> Result<(), String> {
        let me = self.id().to_owned();
        let ok = env.msg.sender() == crate::motor::G2
            && m.verb() == nmtype::message::Verb::Focus
            && m.receiver() == me
            && matches!(env.instruction, Instruction::Reread(_))
            && ledger.verify(ack, m);
        let inbox = self.inbox_mut();
        if !ok {
            inbox.refused_direct += 1;
            return Err(format!("{me}: refused a direct envelope from {} (only G2's own-focus order with G1's acknowledgement)", env.msg.sender()));
        }
        if inbox.queue.len() >= inbox.bound.max(1) {
            inbox.refused_full += 1;
            return Err(format!("{me}: inbox full at {}", inbox.queue.len()));
        }
        inbox.queue.push_back(env);
        Ok(())
    }
    /// Drain the inbox, carrying out each instruction in order.
    fn process(&mut self) -> Vec<Reply> {
        let jobs: Vec<Order> = self.inbox_mut().queue.drain(..).collect();
        jobs.into_iter()
            .map(|e| match e.instruction {
                Instruction::Reread(f) => Reply::Report(Box::new(self.reread(&f))),
                Instruction::Ask { span, about } => Reply::Answer(self.answer(&span, &about)),
            })
            .collect()
    }
}

/// The G0 state of one tentacle, and its first-form `TentacleSide` over the window's data.
#[derive(Clone, Debug)]
pub struct G0 {
    pub id: String,
    pub tentacle: u8,
    pub held: Vec<HeldChunk>,
    pub readings: DroneTable,
    pub focus: Focus,
    pub report: Report,
    pub inbox: Inbox,
    /// Spans reported per read (derived from the tentacle pass's shortlist; fallback 2).
    pub depth: usize,
    /// Chunk scores the window's read already holds, by statistic, one per held chunk.
    scores: Vec<(String, Vec<f32>)>,
    baseline: f32,
    used: Vec<String>,
    reads: u32,
    /// The common scale: one confidence per held chunk, computed the same way for every tentacle
    /// from whole-tree statistics (on the whole path a score's share of the tree's strongest for
    /// this question, the theoretical min-max of Bruch et al. 2023, https://arxiv.org/abs/2210.11934),
    /// so G1 compares the reports without the tentacles exchanging anything, whatever statistic
    /// each tentacle reads by. Empty: confidences stand inside the tentacle's own window.
    common: Vec<f32>,
    /// Reflexes G2 installed (`crate::reflex`), the definition companion of each held span (R2:
    /// span key to the held index of a chunk defining an item it uses) and the keys of spans in
    /// files the organism itself just edited (R4).
    pub reflexes: Vec<crate::reflex::Installed>,
    pub companions: BTreeMap<String, usize>,
    pub self_edits: std::collections::BTreeSet<String>,
    /// The drones' readout of the held pieces (empty without calibrated evidence).
    pub readout: Readout,
    /// The report gate shared by all tentacles, on the common scale: every held piece at or
    /// above it is reported, beside the window's own shortlist, so what a tentacle reports does
    /// not depend on how large its window is. `None`: the window's shortlist alone.
    pub floor: Option<f32>,
    /// The tentacle core's report for this round (lane DR's `CoreReport::to_g0`; the one tentacle
    /// system): when present, the first read reports the core's best spans in the core's order
    /// (each with this g0's common-scale confidence where it holds one), its no-information flag,
    /// spread, margin, tags, wants and reasons; the evidence facts stay this g0's.
    pub core: Option<Report>,
}

/// Fallback spans per report when the window record names none.
pub const DEPTH_FALLBACK: usize = 2;

fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl G0 {
    /// A G0 from the tentacle pass's results for one window: its record, its held chunks in
    /// window order, the chunk scores per statistic (each one value per held chunk; the first is
    /// the statistic the pass ranked by), and the drones' rows (one value per held chunk each).
    /// `depth` is the pass's shortlist; 0 uses the record's findings, then the fallback.
    #[must_use]
    pub fn from_window(
        rec: &WindowRecord,
        held: Vec<HeldChunk>,
        scores: Vec<(String, Vec<f32>)>,
        drone_rows: &[Vec<f32>],
        depth: usize,
    ) -> Self {
        let depth = if depth > 0 {
            depth
        } else if rec.findings > 0 {
            rec.findings
        } else {
            DEPTH_FALLBACK
        };
        let statistic = scores.first().map(|s| s.0.clone()).unwrap_or_default();
        let focus = Focus { lens: rec.lens.clone(), pose: rec.pose.clone(), statistic, ..Focus::default() };
        let bound = held.len() + 1;
        let id = g0_id(rec.tentacle);
        Self {
            report: Report { from: id.clone(), tentacle: rec.tentacle, baseline: rec.baseline, ..Report::default() },
            id,
            tentacle: rec.tentacle,
            held,
            readings: DroneTable::quantize(drone_rows),
            focus,
            inbox: Inbox { bound, ..Inbox::default() },
            depth,
            scores,
            baseline: rec.baseline,
            used: Vec::new(),
            reads: 0,
            common: Vec::new(),
            reflexes: Vec::new(),
            companions: BTreeMap::new(),
            self_edits: std::collections::BTreeSet::new(),
            readout: Readout::default(),
            floor: None,
            core: None,
        }
    }
    /// The same g0 with the tentacle core's report for its first read (see `core`).
    #[must_use]
    pub fn with_core(mut self, report: Report) -> Self {
        self.core = Some(report);
        self
    }
    /// The report gate shared by all tentacles (on the common scale; see `floor`).
    #[must_use]
    pub fn with_floor(mut self, floor: Option<f32>) -> Self {
        self.floor = floor.filter(|f| f.is_finite());
        self
    }
    /// The drones' readout of the held pieces (one entry per held piece in each list).
    #[must_use]
    pub fn with_readout(mut self, readout: Readout) -> Self {
        self.readout = readout;
        self
    }
    /// The spread of `v` (the scores of the shortlist): the median absolute deviation, the mean
    /// absolute deviation where that is zero, 1 where they do not spread.
    fn shortlist_spread(v: &[f32]) -> f32 {
        let mut f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if f.len() < 2 {
            return 1.0;
        }
        f.sort_by(f32::total_cmp);
        let mid = |w: &[f32]| {
            let n = w.len();
            if n % 2 == 1 { w.get(n / 2).copied().unwrap_or(0.0) } else { (w.get(n / 2 - 1).copied().unwrap_or(0.0) + w.get(n / 2).copied().unwrap_or(0.0)) / 2.0 }
        };
        let m = mid(&f);
        let mut dev: Vec<f32> = f.iter().map(|x| (x - m).abs()).collect();
        dev.sort_by(f32::total_cmp);
        let mad = mid(&dev);
        let mean = dev.iter().sum::<f32>() / dev.len() as f32;
        if mad > 0.0 {
            mad
        } else if mean > 0.0 {
            mean
        } else {
            1.0
        }
    }
    /// Install G2's reflexes with what they need (R2 companions, R4 self-edited keys).
    #[must_use]
    pub fn with_reflexes(
        mut self,
        reflexes: Vec<crate::reflex::Installed>,
        companions: BTreeMap<String, usize>,
        self_edits: std::collections::BTreeSet<String>,
    ) -> Self {
        self.reflexes = reflexes.into_iter().filter(|r| r.code.tentacle()).collect();
        self.companions = companions;
        self.self_edits = self_edits;
        self
    }
    fn installed(&self, code: crate::reflex::Code) -> Option<&crate::reflex::Installed> {
        self.reflexes.iter().find(|r| r.code == code && r.live())
    }
    /// Put this segment's confidences on the common scale: `shares` holds one value in 0 to 1 per
    /// held chunk, in window order (ignored unless one per held chunk).
    #[must_use]
    pub fn on_common_scale(mut self, shares: Vec<f32>) -> Self {
        if shares.len() == self.held.len() {
            self.common = shares;
        }
        self
    }
    /// Confidence of held chunk `i` with score `s`, given the window's own spread `sd`.
    fn confidence(&self, i: usize, s: f32, sd: f32) -> f32 {
        match self.common.get(i) {
            Some(c) => c.clamp(0.0, 1.0),
            None if self.common.is_empty() => logistic((s - self.baseline) / sd),
            None => 0.0,
        }
    }
    /// Scores under `statistic` (the drone table's sums when no score set has that name).
    fn scored(&self, statistic: &str) -> Vec<f32> {
        match self.scores.iter().find(|(n, _)| n == statistic) {
            Some((_, v)) => (0..self.held.len()).map(|i| v.get(i).copied().unwrap_or(f32::NEG_INFINITY)).collect(),
            None => (0..self.held.len()).map(|i| self.readings.chunk_sum(i)).collect(),
        }
    }
    /// The calibration scale of one score set: its standard deviation over the window (derived;
    /// 1.0 only when every score is equal).
    fn spread(v: &[f32]) -> f32 {
        let f: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
        if f.len() < 2 {
            return 1.0;
        }
        let n = f.len() as f32;
        let mean = f.iter().sum::<f32>() / n;
        let sd = (f.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
        if sd > 0.0 { sd } else { 1.0 }
    }
    /// Held-chunk order under the current focus: chunks in the focus files first, then the rest,
    /// each by score (ties: window order).
    fn ranked(&self, v: &[f32]) -> Vec<usize> {
        let aimed = |i: usize| self.held.get(i).is_some_and(|h| self.focus.files.contains(&h.span.file));
        let mut idx: Vec<usize> = (0..self.held.len()).collect();
        idx.sort_by(|&a, &b| {
            aimed(b)
                .cmp(&aimed(a))
                .then(v.get(b).copied().unwrap_or(f32::NEG_INFINITY).total_cmp(&v.get(a).copied().unwrap_or(f32::NEG_INFINITY)))
                .then(a.cmp(&b))
        });
        idx
    }
    fn read(&mut self) -> Report {
        self.reads += 1;
        if !self.used.contains(&self.focus.statistic) {
            self.used.push(self.focus.statistic.clone());
        }
        use crate::reflex::Code;
        let mut fired: Vec<String> = Vec::new();
        let mut groups: Vec<String> = Vec::new();
        let mut v = self.scored(&self.focus.statistic);
        let mut sd = Self::spread(&v);
        let mut order = self.ranked(&v);
        let margin = |v: &[f32], order: &[usize]| match (order.first().and_then(|i| v.get(*i)), order.get(1).and_then(|i| v.get(*i))) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Some(a - b),
            _ => None,
        };
        // R3 tie (listed before R6 in their group: the cheaper settles it when it fires).
        if let Some(r) = self.installed(Code::R3Tie).cloned()
            && margin(&v, &order).is_some_and(|m| m < r.threshold * sd)
            && !groups.contains(&r.group)
        {
            let top: Vec<usize> = order.iter().take(2).copied().collect();
            let mut readings: Vec<Vec<f32>> = self
                .scores
                .iter()
                .map(|(_, x)| (0..self.held.len()).map(|i| x.get(i).copied().unwrap_or(f32::NEG_INFINITY)).collect())
                .collect();
            readings.extend((0..self.readings.drones).map(|d| (0..self.held.len()).map(|i| self.readings.get(d, i)).collect()));
            let ranks = crate::reflex::mean_ranks(&readings, &top);
            if let ([a, b], [ra, rb]) = (top.as_slice(), ranks.as_slice())
                && rb < ra
            {
                order.swap(0, 1);
                fired.push(format!(
                    "{}: first two within {:.2} spreads; order averaging over {} readings put {} first",
                    Code::R3Tie.name(),
                    r.threshold,
                    readings.len(),
                    self.held.get(*b).map_or(String::new(), |h| h.span.label())
                ));
                let _ = a;
            } else {
                fired.push(format!("{}: first two within {:.2} spreads; order averaging kept the first", Code::R3Tie.name(), r.threshold));
            }
            groups.push(r.group.clone());
        }
        // R6 short read: one more read under the next unused statistic, without waiting for G1.
        if let Some(r) = self.installed(Code::R6ShortRead).cloned()
            && !groups.contains(&r.group)
            && margin(&v, &order).is_some_and(|m| m < r.threshold * sd)
            && let Some(next) = self.scores.iter().map(|(n, _)| n.clone()).find(|n| !self.used.contains(n))
        {
            self.used.push(next.clone());
            self.focus.statistic = next.clone();
            self.reads += 1;
            v = self.scored(&next);
            sd = Self::spread(&v);
            order = self.ranked(&v);
            fired.push(format!("{}: top margin within {:.2} spreads; read again under {next}", Code::R6ShortRead.name(), r.threshold));
            groups.push(r.group.clone());
        }
        // R4 self-edit: spans of the organism's own just-edited files are not reported as new.
        let mut self_caused: Vec<String> = Vec::new();
        if self.installed(Code::R4SelfEdit).is_some() && !self.self_edits.is_empty() {
            let (own, rest): (Vec<usize>, Vec<usize>) =
                order.iter().partition(|i| self.held.get(**i).is_some_and(|h| self.self_edits.contains(&h.span.key)));
            if !own.is_empty() {
                self_caused = own.iter().filter_map(|i| self.held.get(*i).map(|h| h.span.key.clone())).collect();
                fired.push(format!("{}: {} spans of the organism's own edit marked self-caused", Code::R4SelfEdit.name(), own.len()));
                order = rest.into_iter().chain(own).collect();
            }
        }
        let best_score = v.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
        // With the drones' readout, no information means the drones accepted no held piece for any
        // aspect of the request (none above the content-free band); else nothing scored above the
        // baseline.
        let ro = &self.readout;
        let no_information = if ro.aspects.is_empty() {
            best_score.partial_cmp(&self.baseline) != Some(std::cmp::Ordering::Greater)
        } else {
            !ro.accepted.iter().any(|a| *a)
        };
        if no_information && self.installed(Code::R1NoInformation).is_some() {
            fired.push(format!("{}: nothing above the baseline {:.3}", Code::R1NoInformation.name(), self.baseline));
        }
        let reportable: Vec<usize> =
            order.iter().copied().filter(|i| self.held.get(*i).is_none_or(|h| !self_caused.contains(&h.span.key))).collect();
        let common_gate = |i: usize| self.floor.is_some_and(|f| self.common.get(i).is_some_and(|c| *c >= f));
        let mut best: Vec<SpanReport> = reportable
            .iter()
            .enumerate()
            .filter(|(rank, i)| *rank < self.depth || common_gate(**i))
            .map(|(_, i)| i)
            .enumerate()
            .filter_map(|(place, &i)| {
                let h = self.held.get(i)?;
                let s = v.get(i).copied().unwrap_or(f32::NEG_INFINITY);
                Some(SpanReport { span: h.span.clone(), score: s, confidence: self.confidence(i, s, sd), place })
            })
            .collect();
        // R2 definition companion: the definition a reported span uses comes up beside it.
        if self.installed(Code::R2Definition).is_some() {
            let mut k = 0;
            while k < best.len() {
                let comp = best.get(k).and_then(|b| self.companions.get(&b.span.key)).copied();
                if let Some(ci) = comp
                    && let Some(h) = self.held.get(ci)
                    && !best.iter().any(|b| b.span.key == h.span.key)
                    && !self_caused.contains(&h.span.key)
                {
                    let s = v.get(ci).copied().unwrap_or(f32::NEG_INFINITY);
                    fired.push(format!(
                        "{}: {} brought up beside {}",
                        Code::R2Definition.name(),
                        h.span.label(),
                        best.get(k).map_or(String::new(), |b| b.span.label())
                    ));
                    best.insert(k + 1, SpanReport { span: h.span.clone(), score: s, confidence: self.confidence(ci, s, sd), place: k + 1 });
                    k += 1;
                }
                k += 1;
            }
            for (p, b) in best.iter_mut().enumerate() {
                b.place = p;
            }
        }
        let shown: Vec<String> = best.iter().map(|b| b.span.key.clone()).collect();
        let next: Vec<SpanRef> = reportable
            .iter()
            .filter_map(|&i| self.held.get(i))
            .filter(|h| !shown.contains(&h.span.key))
            .take(self.depth)
            .map(|h| h.span.clone())
            .chain(self_caused.iter().filter_map(|k| self.held.iter().find(|h| h.span.key == *k).map(|h| h.span.clone())))
            .collect();
        // R5 overflow: the rest goes on by reference, never dropped (logged when installed).
        if self.installed(Code::R5Overflow).is_some() && reportable.len() > best.len() {
            fired.push(format!(
                "{}: {} spans beyond the report go on by reference",
                Code::R5Overflow.name(),
                reportable.len() - best.len()
            ));
        }
        let confidence = if no_information { 0.0 } else { best.first().map_or(0.0, |b| b.confidence) };
        let shortlist: Vec<f32> = best.iter().map(|b| b.score).collect();
        let spread = Self::shortlist_spread(&shortlist);
        let margin = match (shortlist.first(), shortlist.get(1)) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => (a - b) / spread,
            _ => 0.0,
        };
        let held_index = |b: &SpanReport| self.held.iter().position(|h| h.span.key == b.span.key);
        let absent: Vec<String> =
            ro.aspects.iter().enumerate().filter(|(a, _)| !ro.covers.iter().any(|c| c.contains(a))).map(|(_, t)| t.clone()).collect();
        let tags: Vec<String> = if ro.tags.is_empty() {
            Vec::new()
        } else {
            best.iter().map(|b| held_index(b).and_then(|i| ro.tags.get(i)).cloned().unwrap_or_default()).collect()
        };
        let wants: Vec<SpanRef> = best.first().and_then(held_index).and_then(|i| ro.related.get(i)).cloned().unwrap_or_default();
        let mut facts = BTreeMap::new();
        for (rank, &i) in order.iter().enumerate() {
            let Some(h) = self.held.get(i) else { continue };
            let sv = v.get(i).copied().unwrap_or(f32::NEG_INFINITY);
            let below = order.get(rank + 1).and_then(|j| v.get(*j)).copied();
            let scores = self.scores.iter().map(|(n, x)| (n.clone(), x.get(i).copied().unwrap_or(f32::NEG_INFINITY))).collect();
            facts.insert(
                h.span.key.clone(),
                SpanFacts {
                    tokens: h.tokens,
                    window_rank: rank,
                    margin: below.map_or(0.0, |b| sv - b),
                    scores,
                    baseline: self.baseline,
                    null_spread: None,
                    mass_rank: rank,
                    lens: self.focus.lens.clone(),
                    lens_rank: None,
                    read: String::new(),
                    quorum: ro.quorum.get(i).copied().unwrap_or(0),
                    aspects: ro.covers.get(i).cloned().unwrap_or_default(),
                },
            );
        }
        self.report = Report {
            from: self.id.clone(),
            tentacle: self.tentacle,
            best,
            no_information,
            best_score: if best_score.is_finite() { best_score } else { 0.0 },
            baseline: self.baseline,
            confidence,
            next,
            focus: self.focus.clone(),
            reads: self.reads,
            facts,
            read: String::new(),
            reflexes: fired,
            self_caused,
            spread,
            margin,
            absent,
            tags,
            wants,
        };
        if self.reads == 1
            && let Some(c) = self.core.as_ref()
            && !c.best.is_empty()
        {
            let own = std::mem::take(&mut self.report.best);
            let mut best: Vec<SpanReport> = c
                .best
                .iter()
                .map(|s| own.iter().find(|o| o.span.key == s.span.key).cloned().unwrap_or_else(|| s.clone()))
                .collect();
            for (i, b) in best.iter_mut().enumerate() {
                b.place = i;
            }
            self.report.confidence = best.first().map_or(0.0, |b| b.confidence);
            self.report.best_score = best.first().map_or(0.0, |b| b.score);
            self.report.best = best;
            self.report.no_information = c.no_information;
            self.report.spread = c.spread;
            self.report.margin = c.margin;
            if c.tags.len() == self.report.best.len() {
                self.report.tags.clone_from(&c.tags);
            } else {
                self.report.tags.clear();
            }
            for w in &c.wants {
                if !self.report.wants.iter().any(|x| x.key == w.key) {
                    self.report.wants.push(w.clone());
                }
            }
            self.report.read.clone_from(&c.read);
        }
        self.report.clone()
    }

    /// Approximate bytes this G0 holds in RAM (references, table, report, inbox).
    #[must_use]
    pub fn bytes(&self) -> usize {
        let span = |s: &SpanRef| s.file.len() + s.key.len() + 2 * std::mem::size_of::<usize>();
        let held: usize = self.held.iter().map(|h| span(&h.span) + 2 * std::mem::size_of::<usize>()).sum();
        let scores: usize = self.scores.iter().map(|(n, v)| n.len() + v.len() * 4).sum();
        let report: usize =
            self.report.best.iter().map(|b| span(&b.span) + 12).sum::<usize>() + self.report.next.iter().map(span).sum::<usize>();
        held + self.readings.bytes() + scores + report + std::mem::size_of::<Self>()
    }
}

impl TentacleSide for G0 {
    fn id(&self) -> &str {
        &self.id
    }
    fn held(&self) -> Vec<SpanRef> {
        self.held.iter().map(|h| h.span.clone()).collect()
    }
    fn options(&self) -> usize {
        self.scores.iter().filter(|(n, _)| !self.used.contains(n)).count()
    }
    fn report(&mut self) -> Report {
        if self.reads == 0 { self.read() } else { self.report.clone() }
    }
    fn reread(&mut self, focus: &Focus) -> Report {
        let statistic = if !focus.statistic.is_empty() && self.scores.iter().any(|(n, _)| *n == focus.statistic) {
            focus.statistic.clone()
        } else {
            self.scores.iter().map(|(n, _)| n.clone()).find(|n| !self.used.contains(n)).unwrap_or_else(|| self.focus.statistic.clone())
        };
        self.focus = Focus {
            lens: self.focus.lens.clone(),
            pose: self.focus.pose.clone(),
            statistic,
            files: focus.files.clone(),
            terms: focus.terms.clone(),
        };
        self.read()
    }
    fn answer(&mut self, span: &SpanRef, _about: &str) -> Answer {
        let Some(i) = self.held.iter().position(|h| h.span.key == span.key || h.span.overlaps(span)) else {
            return Answer {
                from: self.id.clone(),
                span: span.clone(),
                held: false,
                score: 0.0,
                confidence: 0.0,
                rank: None,
                no_information: true,
            };
        };
        // A focused read of one span: its calibrated confidence under every statistic the
        // window holds, averaged (the current statistic alone when the window holds one).
        let mut confs = Vec::new();
        for (name, _) in &self.scores {
            let v = self.scored(name);
            let sd = Self::spread(&v);
            if let Some(s) = v.get(i).copied().filter(|x| x.is_finite()) {
                confs.push(self.confidence(i, s, sd));
            }
        }
        let v = self.scored(&self.focus.statistic);
        if confs.is_empty() {
            let sd = Self::spread(&v);
            confs.push(self.confidence(i, v.get(i).copied().unwrap_or(0.0), sd));
        }
        let confidence = confs.iter().sum::<f32>() / confs.len() as f32;
        let score = v.get(i).copied().unwrap_or(0.0);
        let rank = self.ranked(&v).iter().position(|&k| k == i);
        Answer {
            from: self.id.clone(),
            span: self.held.get(i).map_or_else(|| span.clone(), |h| h.span.clone()),
            held: true,
            score,
            confidence,
            rank,
            no_information: score.partial_cmp(&self.baseline) != Some(std::cmp::Ordering::Greater),
        }
    }
    fn inbox_mut(&mut self) -> &mut Inbox {
        &mut self.inbox
    }
    fn inbox(&self) -> &Inbox {
        &self.inbox
    }
}

/// A first-form G0 for checks: one chunk per named file (lines 1 to 40, key `h{t}-{i}`), one
/// score set per (name, scores) pair, baseline 0.
#[must_use]
pub fn scripted(t: u8, files: &[&str], sets: &[(&str, &[f32])]) -> G0 {
    let held: Vec<HeldChunk> = files
        .iter()
        .enumerate()
        .map(|(i, f)| HeldChunk {
            span: SpanRef { file: (*f).to_owned(), start: 1, end: 40, key: format!("h{t}-{i}") },
            chunk: usize::from(t) * 100 + i,
            tokens: 100,
        })
        .collect();
    let rec = WindowRecord {
        tentacle: t,
        lens: "meaning".into(),
        pose: "focused".into(),
        chunks: held.iter().map(|h| h.chunk).collect(),
        tokens: held.len() * 100,
        level: 2,
        ..WindowRecord::default()
    };
    let scores: Vec<(String, Vec<f32>)> = sets.iter().map(|(n, v)| ((*n).to_owned(), v.to_vec())).collect();
    let rows: Vec<Vec<f32>> = scores.first().map(|s| vec![s.1.clone()]).unwrap_or_default();
    G0::from_window(&rec, held, scores, &rows, DEPTH_FALLBACK)
}

/// Self-check of the `g0` sub-component: identity, quantized readings, a report with calibrated
/// confidence, an explicit no-information report against the baseline, a re-read under the
/// next statistic, an answer about a held and an unheld span, and the inbox refusing any sender
/// but G1.
pub(crate) fn check() -> Result<(), String> {
    let mut a = scripted(0, &["a/x.rs", "a/y.rs", "a/z.rs"], &[("mass", &[0.9, 0.2, -0.1]), ("evidence", &[0.1, 1.2, -0.3])]);
    if a.id != "g0_t1" || g0_id(7) != "g0_t8" {
        return Err(format!("nmbrain.g0: identity {}", a.id));
    }
    if (a.readings.get(0, 0) - 0.9).abs() > 0.01 || a.readings.bytes() != 3 + 4 {
        return Err(format!("nmbrain.g0: quantized readings {:?}", a.readings));
    }
    let r = a.report();
    if r.no_information
        || r.best.len() != 2
        || r.best.first().map(|b| b.span.file.as_str()) != Some("a/x.rs")
        || r.confidence.partial_cmp(&0.5) != Some(std::cmp::Ordering::Greater)
        || r.next.len() != 1
    {
        return Err(format!("nmbrain.g0: first report {r:?}"));
    }
    let r2 = a.reread(&Focus::default());
    if r2.focus.statistic != "evidence"
        || r2.best.first().map(|b| b.span.file.as_str()) != Some("a/y.rs")
        || a.options() != 0
        || r2.reads != 2
    {
        return Err(format!("nmbrain.g0: re-read {r2:?}"));
    }
    let held = a.held.get(1).map(|h| h.span.clone()).ok_or("nmbrain.g0: no held span")?;
    let ans = a.answer(&held, "where");
    let away = a.answer(&SpanRef { file: "b/q.rs".into(), start: 1, end: 2, key: "zz".into() }, "where");
    if !ans.held || ans.rank != Some(0) || away.held || !away.no_information {
        return Err(format!("nmbrain.g0: answers {ans:?} {away:?}"));
    }
    let mut empty = scripted(1, &["b/x.rs", "b/y.rs"], &[("mass", &[-0.2, -0.4])]);
    let e = empty.report();
    if !e.no_information || e.confidence != 0.0 || e.baseline != 0.0 {
        return Err(format!("nmbrain.g0: no-information report {e:?}"));
    }
    let direct = Order::new("g0_t1", "g0_t2", 1, 0.5, "direct", Instruction::Reread(Focus::default()))?;
    if empty.deliver(direct).is_ok() || empty.inbox().refused_direct != 1 {
        return Err("nmbrain.g0: inbox took an envelope from another g0_tx".into());
    }
    let from_g1 = Order::new(G1_ID, "g0_t2", 1, 0.5, "low confidence", Instruction::Reread(Focus::default()))?;
    empty.deliver(from_g1)?;
    if empty.process().len() != 1 || !empty.inbox().queue.is_empty() {
        return Err("nmbrain.g0: inbox not processed".into());
    }
    Ok(())
}

/// Section check of the tentacle reflexes (`crate::reflex`, 2026-11-13): R3 settles a near tie by
/// order averaging; R6 reads once more under the next statistic on a narrow margin (and R3, in
/// its group, wins when both are installed); R2 brings a held definition up beside the span using
/// it; R4 moves a self-edited span out of the report to `next`; R1 and R5 are logged; nothing
/// fires without installation.
pub(crate) fn check_reflexes() -> Result<(), String> {
    use crate::reflex::Installed;
    let files = ["a/x.rs", "a/y.rs", "a/z.rs", "a/w.rs"];
    // x and y nearly tied on mass (also the drone row); evidence, density and peak prefer y.
    let sets: [(&str, &[f32]); 4] = [
        ("mass", &[1.0, 0.98, 0.1, 0.0]),
        ("evidence", &[0.1, 1.5, 0.2, 0.0]),
        ("density", &[0.2, 0.9, 0.1, 0.0]),
        ("peak", &[0.3, 0.7, 0.1, 0.0]),
    ];
    let plain = scripted(0, &files, &sets).report();
    if !plain.reflexes.is_empty() || plain.best.first().map(|b| b.span.file.as_str()) != Some("a/x.rs") {
        return Err(format!("nmbrain.g0.reflexes: plain {plain:?}"));
    }
    let r = |t: &str| Installed::parse(t).map_err(|e| e.to_string());
    let mut g = scripted(0, &files, &sets).with_reflexes(vec![r("R3")?], BTreeMap::new(), std::collections::BTreeSet::new());
    let tie = g.report();
    if tie.best.first().map(|b| b.span.file.as_str()) != Some("a/y.rs") || !tie.reflexes.iter().any(|l| l.contains("R3")) {
        return Err(format!("nmbrain.g0.reflexes: R3 {tie:?}"));
    }
    let mut g = scripted(0, &files, &sets).with_reflexes(vec![r("R6")?], BTreeMap::new(), std::collections::BTreeSet::new());
    let short = g.report();
    if short.focus.statistic != "evidence" || short.reads != 2 || short.best.first().map(|b| b.span.file.as_str()) != Some("a/y.rs") {
        return Err(format!("nmbrain.g0.reflexes: R6 {short:?}"));
    }
    let mut g = scripted(0, &files, &sets).with_reflexes(vec![r("R3")?, r("R6")?], BTreeMap::new(), std::collections::BTreeSet::new());
    let both = g.report();
    if both.reflexes.iter().any(|l| l.contains("R6")) || both.reads != 1 {
        return Err(format!("nmbrain.g0.reflexes: exclusion group {both:?}"));
    }
    // R2: x uses an item defined in w (held index 3); R4: z is self-edited; R1 and R5 logged.
    let mut comp = BTreeMap::new();
    comp.insert("h0-0".to_owned(), 3_usize);
    let mut own = std::collections::BTreeSet::new();
    own.insert("h0-1".to_owned());
    let mut g = scripted(0, &files, &sets).with_reflexes(vec![r("R2")?, r("R4")?, r("R5")?], comp, own);
    let rep = g.report();
    let order: Vec<&str> = rep.best.iter().map(|b| b.span.file.as_str()).collect();
    if order.first() != Some(&"a/x.rs")
        || order.get(1) != Some(&"a/w.rs")
        || order.contains(&"a/y.rs")
        || rep.self_caused != vec!["h0-1".to_owned()]
        || !rep.next.iter().any(|s| s.key == "h0-1")
    {
        return Err(format!("nmbrain.g0.reflexes: R2 R4 R5 {rep:?}"));
    }
    let mut over = scripted(0, &files, &sets).with_reflexes(vec![r("R5")?], BTreeMap::new(), std::collections::BTreeSet::new());
    if !over.report().reflexes.iter().any(|l| l.contains("R5")) {
        return Err("nmbrain.g0.reflexes: R5 not logged".into());
    }
    let mut empty =
        scripted(1, &["b/x.rs"], &[("mass", &[-0.2])]).with_reflexes(vec![r("R1")?], BTreeMap::new(), std::collections::BTreeSet::new());
    if !empty.report().reflexes.iter().any(|l| l.contains("R1")) {
        return Err("nmbrain.g0.reflexes: R1 not logged".into());
    }
    Ok(())
}

// ---- Phase E1 (lane T1, 2026-11-19): comparable evidence across tentacles ----

/// The window's nuisance basis for one held piece (tentacle master plan 3.4): its place in the
/// window, tokens to the window's end, log length, newline share, and the content-free read
/// (0 when none was taken).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Nuisance {
    pub place: f32,
    pub to_end: f32,
    pub log_len: f32,
    pub newlines: f32,
    pub content_free: f32,
}

impl Nuisance {
    /// The basis of every piece of a window, in window order, from each piece's tokens and its
    /// newline share (content-free read 0).
    #[must_use]
    pub fn of_window_full(tokens: &[usize], newlines: &[f32]) -> Vec<Self> {
        Self::full(tokens, newlines)
    }
    /// The basis served (G0-R on the TC1 dump, lane T1 2026-11-19): the last-position indicator
    /// alone in `place` (1 for the window's last piece). The full basis (place, tokens to the end,
    /// log length, newline share) took first in window from 10 to 6 of 29 and log length alone
    /// costs answers (longer pieces carry them); the indicator removes the last-position penalty
    /// (-16.9 to 0.0) and keeps first in window at 10 of 29.
    #[must_use]
    pub fn of_window(tokens: &[usize], newlines: &[f32]) -> Vec<Self> {
        let k = tokens.len().saturating_sub(1);
        let _ = newlines;
        (0..tokens.len()).map(|i| Self { place: if i == k { 1.0 } else { 0.0 }, ..Self::default() }).collect()
    }
    fn full(tokens: &[usize], newlines: &[f32]) -> Vec<Self> {

        let total: usize = tokens.iter().sum();
        let mut before = 0_usize;
        tokens
            .iter()
            .enumerate()
            .map(|(i, t)| {
                before += t;
                #[expect(clippy::cast_precision_loss, reason = "window places and token counts")]
                let n = Self {
                    place: i as f32,
                    to_end: total.saturating_sub(before) as f32,
                    log_len: ((*t).max(1) as f32).ln(),
                    newlines: newlines.get(i).copied().unwrap_or(0.0),
                    content_free: 0.0,
                };
                n
            })
            .collect()
    }
    fn row(&self) -> [f32; 6] {
        [1.0, self.place, self.to_end, self.log_len, self.newlines, self.content_free]
    }
}

/// A score set residualised on the window's nuisance basis by a robust (Huber) linear fit
/// (iteratively reweighted least squares, the Huber constant 1.345 times the residuals' median
/// absolute deviation; the LMS negative image of the electric fish): what the basis predicts is
/// removed. Values that are not finite stay as they are and take no part in the fit; with fewer
/// finite values than the basis has columns plus two, the scores come back unchanged.
#[must_use]
pub fn residualise(scores: &[f32], basis: &[Nuisance]) -> Vec<f32> {
    const K: usize = 6;
    let idx: Vec<usize> = (0..scores.len().min(basis.len())).filter(|i| scores.get(*i).is_some_and(|x| x.is_finite())).collect();
    if idx.len() < K + 2 {
        return scores.to_vec();
    }
    // Standardise the columns (but the intercept) so the normal equations are well scaled.
    let rows: Vec<[f32; K]> = idx.iter().filter_map(|i| basis.get(*i).map(Nuisance::row)).collect();
    let mut mean = [0.0_f64; K];
    let mut sd = [1.0_f64; K];
    #[expect(clippy::cast_precision_loss, reason = "a window's count")]
    let n = rows.len() as f64;
    for j in 1..K {
        let m = rows.iter().map(|r| f64::from(r[j])).sum::<f64>() / n;
        let v = rows.iter().map(|r| (f64::from(r[j]) - m).powi(2)).sum::<f64>() / n;
        mean[j] = m;
        sd[j] = if v > 1e-12 { v.sqrt() } else { 0.0 };
    }
    let x: Vec<[f64; K]> = rows
        .iter()
        .map(|r| {
            let mut o = [0.0_f64; K];
            o[0] = 1.0;
            for j in 1..K {
                o[j] = if sd[j] > 0.0 { (f64::from(r[j]) - mean[j]) / sd[j] } else { 0.0 };
            }
            o
        })
        .collect();
    let y: Vec<f64> = idx.iter().map(|i| f64::from(scores.get(*i).copied().unwrap_or(0.0))).collect();
    let mut w = vec![1.0_f64; y.len()];
    let mut beta = [0.0_f64; K];
    for _ in 0..10 {
        // Weighted normal equations with a small ridge, solved by Gaussian elimination.
        let mut a = [[0.0_f64; K]; K];
        let mut b = [0.0_f64; K];
        for ((xi, yi), wi) in x.iter().zip(&y).zip(&w) {
            for r in 0..K {
                b[r] += wi * xi[r] * yi;
                for c in 0..K {
                    a[r][c] += wi * xi[r] * xi[c];
                }
            }
        }
        for (r, row) in a.iter_mut().enumerate() {
            row[r] += 1e-6;
        }
        let Some(sol) = solve(a, b) else { return scores.to_vec() };
        beta = sol;
        let res: Vec<f64> = x.iter().zip(&y).map(|(xi, yi)| yi - xi.iter().zip(&beta).map(|(p, q)| p * q).sum::<f64>()).collect();
        let mut abs: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let mad = abs.get(abs.len() / 2).copied().unwrap_or(0.0).max(1e-9);
        let c = 1.345 * mad / 0.6745;
        w = res.iter().map(|r| if r.abs() <= c { 1.0 } else { c / r.abs() }).collect();
    }
    let mut out = scores.to_vec();
    for (k, i) in idx.iter().enumerate() {
        if let (Some(o), Some(xi), Some(yi)) = (out.get_mut(*i), x.get(k), y.get(k)) {
            // The intercept stays: only what the nuisance columns predict is removed.
            let pred: f64 = xi.iter().zip(&beta).skip(1).map(|(p, q)| p * q).sum();
            #[expect(clippy::cast_possible_truncation, reason = "back to the score's precision")]
            let v = (yi - pred) as f32;
            *o = v;
        }
    }
    out
}

fn solve<const K: usize>(mut a: [[f64; K]; K], mut b: [f64; K]) -> Option<[f64; K]> {
    for col in 0..K {
        let piv = (col..K).max_by(|p, q| a[*p][col].abs().total_cmp(&a[*q][col].abs()))?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in 0..K {
            if r != col {
                let f = a[r][col] / a[col][col];
                let piv_row = a[col];
                for (x, &p) in a[r][col..K].iter_mut().zip(&piv_row[col..K]) {
                    *x -= f * p;
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0_f64; K];
    for i in 0..K {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// The shared scale every tentacle reports on (tentacle master plan 3.4, until fitted labels are
/// allowed): reciprocal rank fusion's form of a residualised score set, 1 / (60 + rank) with rank
/// from 1 (Cormack et al. 2009, k 60); values that are not finite stay so.
#[must_use]
pub fn shared_scale(residual: &[f32]) -> Vec<f32> {
    const RRF_K: f32 = 60.0;
    let mut idx: Vec<usize> = (0..residual.len()).filter(|i| residual.get(*i).is_some_and(|x| x.is_finite())).collect();
    idx.sort_by(|a, b| residual.get(*b).copied().unwrap_or(0.0).total_cmp(&residual.get(*a).copied().unwrap_or(0.0)).then(a.cmp(b)));
    let mut out: Vec<f32> = residual.iter().map(|x| if x.is_finite() { 0.0 } else { *x }).collect();
    for (r, i) in idx.iter().enumerate() {
        if let Some(o) = out.get_mut(*i) {
            #[expect(clippy::cast_precision_loss, reason = "a window's ranks")]
            let v = 1.0 / (RRF_K + r as f32 + 1.0);
            *o = v;
        }
    }
    out
}
