//! The evidence record of the Globalized Context Stack (2026-11-10, refinement step 1, items 1 to 3; architecture by
//! Luis Jake Gabriel III, `docs/2026-11-10-neumor-guide-how-it-works.md` section 3 and the dated
//! additions of 2026-11-10 08:04 UTC (context is never reduced) and 08:12 UTC (encodings never
//! on disk); analysis `docs/2026-11-10-neumor-analysis-c-routing-and-globalized-context-stack.md` sections 9.1 option
//! 2, 9.4 and 8.3).
//!
//! What travels through every hand-over from the tentacle core to the Core Brain on the new path
//! (`--tentacle-core`): for each span its references (path, lines, content hash, the key of its
//! kept encoding in the chunk store), every score it has with its window's content-free baseline,
//! its rank in its window and its margin over the next, which tentacles and lenses found it, its
//! relations to the other spans of the set, and why it was forwarded or queued. Before this the
//! bridge passed only each chunk's place, and every number was dropped there (analysis 9.1, H5).
//!
//! Nothing is discarded. What G1 does not forward in a round stays in the queue (references and
//! evidence only, in RAM), ordered by its evidence, offered in further rounds and released on
//! request from above: NEXT (the next spans by evidence), EXPAND (the spans related by the
//! reference graph to a named span) and NEIGHBOURS (the spans next to a named span in its file).
//! Spans a later filter would have dropped (the G2 word-overlap filter, the reducer's relevance
//! floor or its token budget) come back here with a low-priority mark.
//!
//! Class: sensory (`nmtype::DataClass::Sensory`); the instructions that release spans are
//! behavioural and are logged as such, apart from the records.
use crate::g0::SpanRef;
use nmjson::Value;
use nmtype::class::{Classed, DataClass};

/// What a tentacle core knows about one held span when it reports (all from its window's read;
/// nothing encoded is copied).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpanFacts {
    /// Tokens of the span as encoded in its window.
    pub tokens: usize,
    /// Rank in its window by the reported score (0 best).
    pub window_rank: usize,
    /// Reported score minus the next lower one in the window (the window's last: 0).
    pub margin: f32,
    /// Every score kind the read gave: (kind, value). `calibrated` is above the anchor.
    pub scores: Vec<(String, f32)>,
    /// The window's content-free baseline: the anchor's calibrated score (the reported scores
    /// are relative to it), and the spread of the content-free read over the window's parts.
    pub baseline: f32,
    pub null_spread: Option<f32>,
    /// Rank in its window by the attention-mass score (0 best).
    pub mass_rank: usize,
    /// The lens that chose the window and the span's rank in that lens's window (assignment
    /// order); `None` when the scan did not record it.
    pub lens: String,
    pub lens_rank: Option<usize>,
    /// Refinement step 2a (2026-11-10): the components of the tentacle core's read that were on
    /// for this span's window (second look, token votes, length-matched content-free pose);
    /// empty for the first form. With a second look the scores hold `second_look` (unscaled, for
    /// the spans it read) beside the full read's `calibrated` and `mass`.
    pub read: String,
    /// The drones' readout of the span: the independent families that accept it for some aspect
    /// of the request (its quorum) and the aspects it holds; 0 and empty without the drones'
    /// calibrated evidence.
    pub quorum: usize,
    pub aspects: Vec<usize>,
}
impl SpanFacts {
    /// A named score.
    #[must_use]
    pub fn score(&self, kind: &str) -> Option<f32> {
        self.scores.iter().find(|(k, _)| k == kind).map(|(_, v)| *v)
    }
    /// Margin against the content-free read: the score above the baseline in units of the
    /// content-free spread, through the logistic (0.5 at the baseline).
    #[must_use]
    pub fn null_margin(&self) -> f32 {
        let s = self.score("calibrated").unwrap_or(0.0);
        match self.null_spread.filter(|x| x.is_finite() && *x > 0.0) {
            Some(t) if s.is_finite() => 1.0 / (1.0 + (-s / t).exp()),
            _ => 0.5,
        }
    }
}

/// Where a record stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Sent upward in a round.
    Forwarded,
    /// Held in the queue, offered in further rounds.
    #[default]
    Queued,
    /// Held in the queue with a low-priority mark: a filter would have dropped it.
    Low,
}
impl Priority {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Forwarded => "forwarded",
            Self::Queued => "queued",
            Self::Low => "low",
        }
    }
}

/// One span's evidence record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpanEvidence {
    pub span: SpanRef,
    /// References to kept encodings: the chunk store key of the span's encoding (`kv:<content
    /// key>`) while its tentacle holds it. References only, never a copy.
    pub kept: Vec<String>,
    pub facts: SpanFacts,
    /// Tentacles (`g0_tx`) and lenses that found it.
    pub found_by: Vec<String>,
    pub lenses: Vec<String>,
    /// Other tentacles that reported spans in the same file.
    pub agreement: usize,
    /// Whole-code-base layer support (0 to 1) and reference-graph rank (0 to 1), this round.
    pub support: f32,
    pub structure: f32,
    /// The core's confidence and G1's combined value under the rule in service.
    pub confidence: f32,
    pub combined: f32,
    /// Relations to other spans of the set: (content key, relation).
    pub relations: Vec<(String, String)>,
    /// Why it was forwarded, queued or marked low.
    pub reasons: Vec<String>,
    pub priority: Priority,
    /// Round in which it went up (0 while queued).
    pub round: u32,
}
impl SpanEvidence {
    /// One line for a trace: label, priority, combined, scores.
    #[must_use]
    pub fn line(&self) -> String {
        let sc: Vec<String> = self.facts.scores.iter().map(|(k, v)| format!("{k} {v:.3}")).collect();
        format!(
            "{} [{}] combined {:.3}; window rank {} margin {:.3}; {}; by {}",
            self.span.label(),
            self.priority.name(),
            self.combined,
            self.facts.window_rank,
            self.facts.margin,
            sc.join(", "),
            self.found_by.join("+")
        )
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let s = |t: &str| Value::String(t.to_owned());
        let f = |x: f32| Value::Number(if x.is_finite() { format!("{x:.4}") } else { "0".into() });
        let n = |x: usize| Value::Number(x.to_string());
        let obj = |v: Vec<(&str, Value)>| Value::Object(v.into_iter().map(|(k, x)| (k.to_owned(), x)).collect());
        let mut fields = vec![
            ("span", s(&self.span.label())),
            ("key", s(&self.span.key)),
            ("kept", Value::Array(self.kept.iter().map(|k| s(k)).collect())),
            ("tokens", n(self.facts.tokens)),
            ("scores", Value::Object(self.facts.scores.iter().map(|(k, v)| (k.clone(), f(*v))).collect())),
            ("baseline", f(self.facts.baseline)),
            ("null_spread", self.facts.null_spread.map_or(Value::Null, f)),
            ("window_rank", n(self.facts.window_rank)),
            ("margin", f(self.facts.margin)),
            ("mass_rank", n(self.facts.mass_rank)),
            ("lens", s(&self.facts.lens)),
            ("lens_rank", self.facts.lens_rank.map_or(Value::Null, n)),
            ("found_by", Value::Array(self.found_by.iter().map(|k| s(k)).collect())),
            ("agreement", n(self.agreement)),
            ("support", f(self.support)),
            ("structure", f(self.structure)),
            ("confidence", f(self.confidence)),
            ("combined", f(self.combined)),
            ("relations", Value::Array(self.relations.iter().map(|(k, r)| s(&format!("{r} {k}"))).collect())),
            ("reasons", Value::Array(self.reasons.iter().map(|k| s(k)).collect())),
            ("priority", s(self.priority.name())),
            ("round", n(self.round as usize)),
            ("class", s("sensory")),
        ];
        // Refinement step 2a: named only when a read component was on (the first form's record
        // is unchanged).
        if !self.facts.read.is_empty() {
            fields.push(("read", s(&self.facts.read)));
        }
        obj(fields)
    }
}

/// The route capacity of one round, in tokens, and how it was set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Capacity {
    pub tokens: usize,
    pub why: String,
}
/// Stated fallback of the round capacity when the caller names no receiving room: the default
/// token budget of a context pack (`nmapi::tools`, `nm_context_pack` default 4,000 tokens), the
/// room the forwarded spans are read into.
pub const ROOM_FALLBACK: usize = 4000;
impl Capacity {
    /// The rule: the capacity of a round is the token room of the stage that receives it (the
    /// question's pack budget when the caller names one, else `ROOM_FALLBACK`); spans go up by
    /// evidence order while their own tokens fit (a span larger than what is left is skipped and
    /// smaller ones after it still fit), at least one per round. The count per round therefore
    /// follows the measured span sizes; nothing is a fixed count.
    #[must_use]
    pub fn derive(room: Option<usize>) -> Self {
        match room.filter(|r| *r > 0) {
            Some(r) => Self {
                tokens: r,
                why: format!(
                    "capacity {r} tokens per round = the receiving room named by the caller; spans admitted by evidence while their tokens fit"
                ),
            },
            None => Self {
                tokens: ROOM_FALLBACK,
                why: format!(
                    "capacity {ROOM_FALLBACK} tokens per round = the default context pack room (stated fallback, no room named); spans admitted by evidence while their tokens fit"
                ),
            },
        }
    }
    /// Pick from `order` (indices into `tokens`, best first) what fits: (picked, tokens used).
    #[must_use]
    pub fn fit(&self, order: &[usize], tokens: &dyn Fn(usize) -> usize) -> (Vec<usize>, usize) {
        let mut used = 0;
        let mut out = Vec::new();
        for &i in order {
            let t = tokens(i).max(1);
            if out.is_empty() || used + t <= self.tokens {
                used += t;
                out.push(i);
            }
        }
        (out, used)
    }
}

/// The reference graph as G1 sees it (plain calls by content key; built by the caller from the
/// substrate, `nmsubs::refgraph`, so this crate keeps its dependencies).
pub trait Structure: Send + Sync {
    /// Personalised rank of the spans `ask` (content keys) from `seeds` (content key, weight).
    fn rank(&self, seeds: &[(String, f32)], ask: &[String]) -> Vec<f32>;
    /// Spans related to one span: (content key, relation name, weight), heaviest first.
    fn related(&self, key: &str) -> Vec<(String, String, f32)>;
    /// One line: what the graph holds.
    fn describe(&self) -> String;
}

/// A request from above on the queue (behavioural; what it releases is sensory).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    /// NEXT(k): the next `k` queued spans by evidence (`k` 0: one round's capacity).
    Next(usize),
    /// EXPAND(span): the queued spans the reference graph relates to the span (uses, used by).
    Expand(String),
    /// NEIGHBOURS(span): the queued spans next to the span in its file (adjacent lines).
    Neighbours(String),
}
impl Request {
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::Next(0) => "NEXT(one round's capacity)".into(),
            Self::Next(k) => format!("NEXT({k})"),
            Self::Expand(k) => format!("EXPAND({})", short(k)),
            Self::Neighbours(k) => format!("NEIGHBOURS({})", short(k)),
        }
    }
}
fn short(k: &str) -> String {
    k.chars().take(12).collect()
}

/// The evidence set of one question: what went up (round by round) and the queue.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvidenceSet {
    pub forwarded: Vec<SpanEvidence>,
    /// Ordered: priority (queued before low), then combined, best first.
    pub queue: Vec<SpanEvidence>,
    pub capacity: Capacity,
    /// Rounds offered so far (the G1 hand-over is round 1).
    pub rounds: u32,
    /// The rule G1 combined by (text).
    pub rule: String,
    /// Requests served: (request, spans released).
    pub log: Vec<(String, usize)>,
}
impl Classed for EvidenceSet {
    fn data_class(&self) -> DataClass {
        DataClass::Sensory
    }
}
impl EvidenceSet {
    /// Order the queue: queued before low, then combined (best first), then label.
    pub fn sort_queue(&mut self) {
        self.queue.sort_by(|a, b| a.priority.cmp(&b.priority).then(b.combined.total_cmp(&a.combined)).then(a.span.cmp(&b.span)));
    }
    /// Every record (forwarded first, then the queue).
    pub fn all(&self) -> impl Iterator<Item = &SpanEvidence> {
        self.forwarded.iter().chain(&self.queue)
    }
    /// The record of a content key or label anywhere in the set.
    #[must_use]
    pub fn find(&self, key: &str) -> Option<&SpanEvidence> {
        self.all().find(|e| e.span.key == key || e.span.label() == key)
    }
    /// The forwarded record whose span holds `path:line` (for parts that carry only a citation).
    #[must_use]
    pub fn at(&self, path: &str, line: usize) -> Option<&SpanEvidence> {
        self.all().find(|e| e.span.contains(path, line))
    }
    /// Tokens of the spans held (forwarded and queued).
    #[must_use]
    pub fn tokens(&self) -> (usize, usize) {
        (self.forwarded.iter().map(|e| e.facts.tokens).sum(), self.queue.iter().map(|e| e.facts.tokens).sum())
    }
    fn release(&mut self, idx: &[usize], why: &str) -> Vec<SpanEvidence> {
        let round = self.rounds + 1;
        let mut take: Vec<usize> = idx.to_vec();
        take.sort_unstable();
        take.dedup();
        let mut out = Vec::new();
        for i in take.into_iter().rev() {
            if i < self.queue.len() {
                let mut e = self.queue.remove(i);
                e.priority = Priority::Forwarded;
                e.round = round;
                e.reasons.push(why.to_owned());
                out.push(e);
            }
        }
        out.reverse();
        if !out.is_empty() {
            self.rounds = round;
            self.forwarded.extend(out.iter().cloned());
        }
        out
    }
    /// Serve a request from above; returns the released records (now forwarded).
    pub fn request(&mut self, r: &Request, structure: Option<&dyn Structure>) -> Vec<SpanEvidence> {
        self.sort_queue();
        let out = match r {
            Request::Next(k) => {
                // NEXT offers the queued spans; a span marked low stays until a request names it.
                let order: Vec<usize> =
                    (0..self.queue.len()).filter(|i| self.queue.get(*i).is_some_and(|e| e.priority != Priority::Low)).collect();
                let pick = if *k == 0 {
                    let q = &self.queue;
                    self.capacity.fit(&order, &|i| q.get(i).map_or(1, |e| e.facts.tokens)).0
                } else {
                    order.into_iter().take(*k).collect()
                };
                self.release(&pick, &format!("released by {}", r.text()))
            }
            Request::Expand(key) => {
                let rel: Vec<(String, String, f32)> = structure.map(|s| s.related(key)).unwrap_or_default();
                let pick: Vec<usize> =
                    self.queue.iter().enumerate().filter(|(_, e)| rel.iter().any(|(k, _, _)| *k == e.span.key)).map(|(i, _)| i).collect();
                self.release(&pick, &format!("released by {} (reference graph)", r.text()))
            }
            Request::Neighbours(key) => {
                let Some(at) = self.find(key).map(|e| e.span.clone()) else { return Vec::new() };
                let pick: Vec<usize> = self
                    .queue
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| {
                        e.span.file == at.file && e.span.key != at.key && (e.span.end + 1 >= at.start && e.span.start <= at.end + 1)
                    })
                    .map(|(i, _)| i)
                    .collect();
                self.release(&pick, &format!("released by {} (adjacent lines)", r.text()))
            }
        };
        self.log.push((r.text(), out.len()));
        out
    }
    /// Move forwarded records matching `keep_out` back to the queue with the low mark (a filter
    /// would have dropped them), and add `extra` (records from outside the set) as low.
    pub fn mark_low(&mut self, keep_out: &dyn Fn(&SpanEvidence) -> bool, why: &str, extra: Vec<SpanEvidence>) -> usize {
        let mut moved = 0;
        let mut kept = Vec::new();
        for mut e in std::mem::take(&mut self.forwarded) {
            if keep_out(&e) {
                e.priority = Priority::Low;
                e.reasons.push(why.to_owned());
                self.queue.push(e);
                moved += 1;
            } else {
                kept.push(e);
            }
        }
        self.forwarded = kept;
        for mut e in extra {
            if self.find(&e.span.key).is_none() {
                e.priority = Priority::Low;
                e.reasons.push(why.to_owned());
                self.queue.push(e);
                moved += 1;
            }
        }
        self.sort_queue();
        moved
    }
    #[must_use]
    pub fn to_value(&self) -> Value {
        let n = |x: usize| Value::Number(x.to_string());
        let (ft, qt) = self.tokens();
        Value::Object(
            [
                ("class", Value::String("sensory".into())),
                ("rule", Value::String(self.rule.clone())),
                ("capacity_tokens", n(self.capacity.tokens)),
                ("capacity", Value::String(self.capacity.why.clone())),
                ("rounds", n(self.rounds as usize)),
                ("forwarded_tokens", n(ft)),
                ("queued_tokens", n(qt)),
                ("forwarded", Value::Array(self.forwarded.iter().map(SpanEvidence::to_value).collect())),
                ("queue", Value::Array(self.queue.iter().map(SpanEvidence::to_value).collect())),
                ("requests", Value::Array(self.log.iter().map(|(r, k)| Value::String(format!("{r}: {k} released"))).collect())),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
}

/// Self-check of `evidence`: the capacity fits by tokens with skip-and-continue and at least one
/// span; NEXT releases by evidence order and moves records to forwarded with the next round;
/// NEIGHBOURS releases only adjacent spans of the same file; EXPAND follows the graph; a span
/// marked low leaves the forwarded list and is held (nothing is lost: the count of records is
/// constant); the set is sensory.
pub(crate) fn check() -> Result<(), String> {
    let rec = |file: &str, a: usize, b: usize, key: &str, combined: f32, tokens: usize| SpanEvidence {
        span: SpanRef { file: file.into(), start: a, end: b, key: key.into() },
        combined,
        facts: SpanFacts { tokens, ..SpanFacts::default() },
        ..SpanEvidence::default()
    };
    let cap = Capacity::derive(Some(1000));
    let toks = [600_usize, 500, 300, 2000];
    let (pick, used) = cap.fit(&[0, 1, 2, 3], &|i| toks.get(i).copied().unwrap_or(0));
    let (one, _) = Capacity::derive(Some(10)).fit(&[3, 0], &|i| toks.get(i).copied().unwrap_or(0));
    if pick != vec![0, 2] || used != 900 || one != vec![3] || Capacity::derive(None).tokens != ROOM_FALLBACK {
        return Err(format!("nmbrain.evidence: capacity picked {pick:?} ({used} tokens), small room {one:?}"));
    }
    let mut set = EvidenceSet { capacity: cap, rounds: 1, ..EvidenceSet::default() };
    set.forwarded.push(rec("a.rs", 1, 10, "k1", 0.9, 400));
    set.queue = vec![
        rec("a.rs", 11, 20, "k2", 0.5, 400),
        rec("b.rs", 1, 10, "k3", 0.7, 400),
        rec("a.rs", 40, 50, "k4", 0.6, 400),
        rec("c.rs", 1, 9, "k5", 0.2, 400),
    ];
    let before = set.all().count();
    let next = set.request(&Request::Next(1), None);
    if next.len() != 1
        || next.first().map(|e| e.span.key.as_str()) != Some("k3")
        || set.rounds != 2
        || set.forwarded.last().map(|e| e.round) != Some(2)
    {
        return Err(format!("nmbrain.evidence: NEXT released {:?}", next.iter().map(|e| &e.span.key).collect::<Vec<_>>()));
    }
    let nb = set.request(&Request::Neighbours("k1".into()), None);
    if nb.iter().map(|e| e.span.key.as_str()).collect::<Vec<_>>() != vec!["k2"] {
        return Err(format!("nmbrain.evidence: NEIGHBOURS released {:?}", nb.iter().map(|e| &e.span.key).collect::<Vec<_>>()));
    }
    struct G;
    impl Structure for G {
        fn rank(&self, _: &[(String, f32)], ask: &[String]) -> Vec<f32> {
            vec![0.0; ask.len()]
        }
        fn related(&self, key: &str) -> Vec<(String, String, f32)> {
            if key == "k1" { vec![("k5".into(), "uses".into(), 1.0)] } else { Vec::new() }
        }
        fn describe(&self) -> String {
            "test".into()
        }
    }
    let ex = set.request(&Request::Expand("k1".into()), Some(&G));
    if ex.iter().map(|e| e.span.key.as_str()).collect::<Vec<_>>() != vec!["k5"] {
        return Err("nmbrain.evidence: EXPAND did not follow the graph".into());
    }
    let moved = set.mark_low(&|e| e.span.key == "k2", "no word shared with the question", vec![rec("d.rs", 1, 2, "x1", 0.1, 50)]);
    let after = set.all().count();
    if moved != 2
        || after != before + 1
        || set.queue.last().map(|e| e.priority) != Some(Priority::Low)
        || set.forwarded.iter().any(|e| e.span.key == "k2")
        || set.data_class() != DataClass::Sensory
    {
        return Err(format!("nmbrain.evidence: low mark moved {moved}, records {before} -> {after}"));
    }
    if set.log.len() != 3 || set.to_value().get("queue").is_err() {
        return Err("nmbrain.evidence: request log or record value missing".into());
    }
    Ok(())
}
