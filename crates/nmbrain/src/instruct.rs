//! G2's instructions downward, I1 to I11 (2026-11-12; analysis C, section 8.3, table of
//! instructions; directions of Luis Jake Gabriel III of 2026-11-10: G2 decides and hands over,
//! G2 may want a specific packet, joint reasoning over several `g0_tx`, tentacles never exchange).
//!
//! Each instruction is typed and travels in the in-house envelope (`nmtype::Envelope`, kind
//! instruction, behavioural traffic only) from `nmbrain.g2` to the part that carries it out:
//! - G1 relays I1 FOCUS, I2 REREAD and I3 ANSWER to the tentacles (only G1 may send to a `g0_tx`),
//!   applies I9 GAIN in its combination, and answers I7 EXPAND from the structural layer;
//! - G2's own gate carries out I4 FORWARD, I5 FORWARD_ALL, I10 INHIBIT and I11 NEXT;
//! - I6 JOINT names a joint read for the Core Brain (spans of several `g0_tx` by reference, within
//!   a token capacity); I8 RESHAPE_WINDOW gives seeds for the next window formation.
//!   Standing instructions (I9, I10) live on the `Desk` with a lifetime: a gain fades back to 1 over
//!   it, an inhibition ends with it.
//!
//! Spans are addressed by chunk number on the whole path (the substrate's chunk, the same key the
//! tentacles hold).
use nmtype::{Envelope, MessageKind};

/// G2's id as sender.
pub const G2_ID: &str = "nmbrain.g2";
/// G1's id.
pub const G1_ID: &str = "nmbrain.g1";
/// The Core Brain's id.
pub const CORE_BRAIN_ID: &str = "nmbrain.core-brain";

/// How a re-read is made (I2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadMode {
    /// A short window around the spans (today's re-read).
    Short,
    /// Averaged over `n` orders of the pieces.
    Averaged(u8),
    /// With a temperature derived from the length read.
    Tempered,
}

/// Which relation I7 follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Callers,
    Callees,
    Definitions,
    Imports,
    Any,
}

/// What I10 holds back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Chunk(usize),
    File(String),
}

/// One instruction of G2.
#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    /// I1: order held spans with the named files first and re-read with a focus pose.
    Focus { files: Vec<String>, terms: Vec<String>, gain: f32 },
    /// I2: re-read these spans in the given mode.
    Reread { chunks: Vec<usize>, mode: ReadMode },
    /// I3: a point read of one span.
    Answer { chunk: usize, about: String },
    /// I4: these spans go up regardless of rank.
    Forward { chunks: Vec<usize> },
    /// I5: a tentacle's whole held list to the queue, by reference.
    ForwardAll { tentacle: usize },
    /// I6: one joint read over spans of several `g0_tx`, within `capacity` tokens.
    Joint { chunks: Vec<usize>, capacity: usize },
    /// I7: the items related to a span join the queue (held), or are scheduled (not held).
    Expand { chunk: usize, relation: Relation },
    /// I8: the next window formation for a tentacle uses these seeds.
    Reshape { tentacle: usize, seeds: Vec<usize>, budget: usize },
    /// I9: a tentacle's gain in G1's combination, fading to 1 over `ttl_ms`.
    Gain { tentacle: usize, g: f32, ttl_ms: u64 },
    /// I10: not forwarded while it lasts; still held.
    Inhibit { target: Target, reason: String, ttl_ms: u64 },
    /// I11: the next `k` spans of G1's queue go up.
    Next { k: usize },
}

impl Instr {
    /// Its number in analysis C's table.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Focus { .. } => "I1",
            Self::Reread { .. } => "I2",
            Self::Answer { .. } => "I3",
            Self::Forward { .. } => "I4",
            Self::ForwardAll { .. } => "I5",
            Self::Joint { .. } => "I6",
            Self::Expand { .. } => "I7",
            Self::Reshape { .. } => "I8",
            Self::Gain { .. } => "I9",
            Self::Inhibit { .. } => "I10",
            Self::Next { .. } => "I11",
        }
    }
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Focus { .. } => "FOCUS",
            Self::Reread { .. } => "REREAD",
            Self::Answer { .. } => "ANSWER",
            Self::Forward { .. } => "FORWARD",
            Self::ForwardAll { .. } => "FORWARD_ALL",
            Self::Joint { .. } => "JOINT",
            Self::Expand { .. } => "EXPAND",
            Self::Reshape { .. } => "RESHAPE_WINDOW",
            Self::Gain { .. } => "GAIN",
            Self::Inhibit { .. } => "INHIBIT",
            Self::Next { .. } => "NEXT",
        }
    }
    /// Its verb in the one vocabulary of the motor nervous system (`nm-msg/2`).
    #[must_use]
    pub const fn verb(&self) -> nmtype::message::Verb {
        use nmtype::message::Verb;
        match self {
            Self::Focus { .. } => Verb::Focus,
            Self::Reread { .. } => Verb::Reread,
            Self::Answer { .. } => Verb::Answer,
            Self::Forward { .. } => Verb::Forward,
            Self::ForwardAll { .. } => Verb::ForwardAll,
            Self::Joint { .. } => Verb::Joint,
            Self::Expand { .. } => Verb::Expand,
            Self::Reshape { .. } => Verb::Reshape,
            Self::Gain { .. } => Verb::Gain,
            Self::Inhibit { .. } => Verb::Inhibit,
            Self::Next { .. } => Verb::Next,
        }
    }
    /// The part that carries it out.
    #[must_use]
    pub const fn target(&self) -> &'static str {
        match self {
            Self::Focus { .. }
            | Self::Reread { .. }
            | Self::Answer { .. }
            | Self::Expand { .. }
            | Self::Reshape { .. }
            | Self::Gain { .. } => G1_ID,
            Self::Joint { .. } => CORE_BRAIN_ID,
            Self::Forward { .. } | Self::ForwardAll { .. } | Self::Inhibit { .. } | Self::Next { .. } => G2_ID,
        }
    }
    /// Refusals: an empty list, a gain not above 0, a capacity or `k` of 0, an empty reason.
    /// # Errors
    /// The first rule broken.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |m: &str| Err(format!("{} {}: {m}", self.code(), self.name()));
        match self {
            Self::Focus { files, terms, gain } if (files.is_empty() && terms.is_empty()) || *gain <= 0.0 => {
                bad("needs files or terms and a gain above 0")
            }
            Self::Reread { chunks, .. } | Self::Forward { chunks } if chunks.is_empty() => bad("no spans"),
            Self::Joint { chunks, capacity } if chunks.len() < 2 || *capacity == 0 => bad("a joint read needs two spans and a capacity"),
            Self::Reshape { seeds, budget, .. } if seeds.is_empty() || *budget == 0 => bad("needs seeds and a budget"),
            Self::Gain { g, ttl_ms, .. } if !g.is_finite() || *g <= 0.0 || *ttl_ms == 0 => bad("needs a gain above 0 and a lifetime"),
            Self::Inhibit { reason, .. } if reason.trim().is_empty() => bad("needs a reason"),
            Self::Next { k } if *k == 0 => bad("k is 0"),
            _ => Ok(()),
        }
    }
    /// The instruction in its envelope, with a one-line reason.
    /// # Errors
    /// Validation, or the envelope's own refusals.
    pub fn envelope(&self, confidence: f32, why: &str) -> Result<Envelope, String> {
        self.validate()?;
        Envelope::new(
            MessageKind::Instruction,
            G2_ID,
            self.target(),
            confidence,
            &format!("{} {}: {why}", self.code(), self.name()),
            Vec::new(),
            Vec::new(),
        )
        .map_err(|f| f.to_string())
    }
    /// Parse the short form used by tests and tools: `I11:4`, `I9:3:1.5:60000`, `I5:2`,
    /// `I10:chunk:12`, `I10:file:src/a.rs`, `I4:5,9`, `I7:12:any`, `I1:files:a.rs,b.rs`,
    /// `I1:terms:x,y`, `I3:12`, `I2:5,9`, `I6:5,9:4000`, `I8:2:5,9:12000`.
    /// # Errors
    /// An unknown or malformed instruction.
    pub fn parse(t: &str) -> Result<Self, String> {
        let p: Vec<&str> = t.split(':').collect();
        let us = |s: Option<&&str>| s.and_then(|x| x.parse::<usize>().ok()).ok_or_else(|| format!("{t}: a number expected"));
        let list = |s: Option<&&str>| -> Result<Vec<usize>, String> {
            s.ok_or_else(|| format!("{t}: a list expected"))?
                .split(',')
                .map(|x| x.parse::<usize>().map_err(|e| format!("{t}: {e}")))
                .collect()
        };
        let words =
            |s: Option<&&str>| s.map(|x| x.split(',').filter(|w| !w.is_empty()).map(str::to_owned).collect::<Vec<_>>()).unwrap_or_default();
        let i = match p.first().copied() {
            Some("I1") => match p.get(1).copied() {
                Some("files") => Self::Focus { files: words(p.get(2)), terms: Vec::new(), gain: 1.5 },
                Some("terms") => Self::Focus { files: Vec::new(), terms: words(p.get(2)), gain: 1.5 },
                _ => return Err(format!("{t}: I1:files:<list> or I1:terms:<list>")),
            },
            Some("I2") => Self::Reread { chunks: list(p.get(1))?, mode: ReadMode::Short },
            Some("I3") => Self::Answer { chunk: us(p.get(1))?, about: String::new() },
            Some("I4") => Self::Forward { chunks: list(p.get(1))? },
            Some("I5") => Self::ForwardAll { tentacle: us(p.get(1))? },
            Some("I6") => Self::Joint { chunks: list(p.get(1))?, capacity: us(p.get(2))? },
            Some("I7") => Self::Expand {
                chunk: us(p.get(1))?,
                relation: match p.get(2).copied().unwrap_or("any") {
                    "callers" => Relation::Callers,
                    "callees" => Relation::Callees,
                    "definitions" => Relation::Definitions,
                    "imports" => Relation::Imports,
                    _ => Relation::Any,
                },
            },
            Some("I8") => Self::Reshape { tentacle: us(p.get(1))?, seeds: list(p.get(2))?, budget: us(p.get(3))? },
            Some("I9") => Self::Gain {
                tentacle: us(p.get(1))?,
                g: p.get(2).and_then(|x| x.parse().ok()).ok_or_else(|| format!("{t}: gain"))?,
                ttl_ms: p.get(3).and_then(|x| x.parse().ok()).unwrap_or(60_000),
            },
            Some("I10") => Self::Inhibit {
                target: match p.get(1).copied() {
                    Some("chunk") => Target::Chunk(us(p.get(2))?),
                    Some("file") => Target::File(p.get(2..).map(|x| x.join(":")).unwrap_or_default()),
                    _ => return Err(format!("{t}: I10:chunk:<n> or I10:file:<path>")),
                },
                reason: "named in the request".into(),
                ttl_ms: 0,
            },
            Some("I11") => Self::Next { k: us(p.get(1))? },
            _ => return Err(format!("{t}: unknown instruction (I1 to I11)")),
        };
        i.validate()?;
        Ok(i)
    }
}

/// I9's bounds (master design 3.6; tentacle master plan 3.8): a gain from above is held within
/// 0.25 to 4, also when several multiply on one tentacle.
pub const GAIN_MIN: f32 = 0.25;
pub const GAIN_MAX: f32 = 4.0;

/// Standing instructions (I9 gains, I10 inhibitions) with the time each was issued.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Desk {
    pub standing: Vec<(Instr, u64)>,
    /// G2 vetoes every muscle-memory pack step (MO8; `serve --packs off`).
    pub veto_packs: bool,
}

impl Desk {
    /// Issue an instruction at `now` (ms); only I9 and I10 stand, the rest are for one question.
    /// A lifetime of 0 means this question only, and does not stand either.
    /// # Errors
    /// Validation.
    pub fn issue(&mut self, i: Instr, now: u64) -> Result<bool, String> {
        i.validate()?;
        let stands = matches!(&i, Instr::Gain { ttl_ms, .. } | Instr::Inhibit { ttl_ms, .. } if *ttl_ms > 0);
        if stands {
            self.standing.push((i, now));
        }
        Ok(stands)
    }
    /// Drop what has run out at `now`.
    pub fn expire(&mut self, now: u64) {
        self.standing.retain(|(i, at)| match i {
            Instr::Gain { ttl_ms, .. } | Instr::Inhibit { ttl_ms, .. } => now.saturating_sub(*at) < *ttl_ms,
            _ => false,
        });
    }
    /// G2's gain per tentacle at `now` (`n` tentacles): each standing gain fades linearly to 1 over
    /// its lifetime; several on one tentacle multiply.
    #[must_use]
    pub fn gains(&self, n: usize, now: u64) -> Vec<f32> {
        let mut g = vec![1.0_f32; n];
        for (i, at) in &self.standing {
            if let Instr::Gain { tentacle, g: x, ttl_ms } = i
                && let Some(v) = g.get_mut(*tentacle)
            {
                #[expect(clippy::cast_precision_loss, reason = "milliseconds")]
                let left = (1.0 - now.saturating_sub(*at) as f32 / *ttl_ms as f32).clamp(0.0, 1.0);
                *v *= 1.0 + (x.clamp(GAIN_MIN, GAIN_MAX) - 1.0) * left;
            }
        }
        g.into_iter().map(|v| v.clamp(GAIN_MIN, GAIN_MAX)).collect()
    }
    /// What is inhibited at `now`.
    #[must_use]
    pub fn inhibited(&self, now: u64) -> Vec<Target> {
        self.standing
            .iter()
            .filter_map(|(i, at)| match i {
                Instr::Inhibit { target, ttl_ms, .. } if now.saturating_sub(*at) < *ttl_ms => Some(target.clone()),
                _ => None,
            })
            .collect()
    }
}

/// What G2's gate starts from on one question.
pub struct Gate<'a> {
    /// G1's hand-over, then its queue, in its order.
    pub handover: &'a [usize],
    pub queue: &'a [usize],
    /// Each tentacle's held chunks.
    pub windows: &'a [Vec<usize>],
    /// The receiving room in tokens: what goes up by the capacity gate fits in it.
    pub room: usize,
    pub file_of: &'a dyn Fn(usize) -> String,
    pub tokens: &'a dyn Fn(usize) -> usize,
    /// Related chunks of a chunk by the structural layer.
    pub related: &'a dyn Fn(usize, Relation) -> Vec<usize>,
}

/// What G2's gate did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Routed {
    /// What goes up to the Core Brain, in order.
    pub sent: Vec<usize>,
    /// G1's queue after the instructions (held, by reference).
    pub queue: Vec<usize>,
    /// The joint read named by I6 (empty when none).
    pub joint: Vec<usize>,
    /// Chunks scheduled for the next window formation (I7 on items no window holds, I8 seeds).
    pub scheduled: Vec<usize>,
    /// One line per instruction: what it did.
    pub done: Vec<String>,
}

/// G2's gate with its instructions for this question (`orders`) and what stands (`inhibited`).
/// Order of work: I5 and I7 grow the queue; I11 raises the capacity; I10 holds back (still held);
/// I4 goes first regardless of rank and of inhibition (an explicit request wins); then the rest of
/// the hand-over and queue in G1's order, each piece that fits the room in tokens (I11 releases
/// its count of pieces beyond the room). I6 names its joint read within its
/// capacity. Instructions for G1 (I1, I2, I3, I8, I9) are reported as relayed.
#[must_use]
pub fn route(g: &Gate, orders: &[Instr], inhibited: &[Target]) -> Routed {
    let mut out = Routed { queue: g.queue.to_vec(), ..Routed::default() };
    let held = |c: usize| g.windows.iter().any(|w| w.contains(&c));
    let mut extra = 0_usize;
    let mut forward: Vec<usize> = Vec::new();
    let mut inhibit: Vec<Target> = inhibited.to_vec();
    for i in orders {
        let line = match i {
            Instr::ForwardAll { tentacle } => {
                let add: Vec<usize> = g
                    .windows
                    .get(*tentacle)
                    .map(|w| w.iter().copied().filter(|c| !g.handover.contains(c) && !out.queue.contains(c)).collect())
                    .unwrap_or_default();
                let n = add.len();
                out.queue.extend(add);
                format!("{} chunks of tentacle {} queued", n, tentacle + 1)
            }
            Instr::Expand { chunk, relation } => {
                let rel = (g.related)(*chunk, *relation);
                let (mut q, mut s) = (0, 0);
                let at = out.queue.iter().position(|c| c == chunk).map_or(0, |p| p + 1);
                let fresh: Vec<usize> =
                    rel.into_iter().filter(|c| c != chunk && !g.handover.contains(c) && !out.queue.contains(c)).collect();
                for (k, c) in fresh.into_iter().enumerate() {
                    if held(c) {
                        out.queue.insert((at + k).min(out.queue.len()), c);
                        q += 1;
                    } else if !out.scheduled.contains(&c) {
                        out.scheduled.push(c);
                        s += 1;
                    }
                }
                format!("{q} related chunks of {chunk} queued, {s} scheduled for the next windows")
            }
            Instr::Next { k } => {
                extra += k;
                format!("{k} more released beyond the room, {extra} in all")
            }
            Instr::Inhibit { target, .. } => {
                inhibit.push(target.clone());
                format!("{target:?} held back this question")
            }
            Instr::Forward { chunks } => {
                for c in chunks {
                    if !forward.contains(c) {
                        forward.push(*c);
                    }
                }
                format!("{} chunks sent up regardless of rank", chunks.len())
            }
            Instr::Joint { chunks, capacity } => {
                let mut used = 0;
                out.joint = chunks
                    .iter()
                    .copied()
                    .take_while(|c| {
                        used += (g.tokens)(*c);
                        used <= *capacity
                    })
                    .collect();
                format!("joint read of {} spans within {capacity} tokens for the Core Brain", out.joint.len())
            }
            Instr::Reshape { tentacle, seeds, .. } => {
                for c in seeds {
                    if !out.scheduled.contains(c) {
                        out.scheduled.push(*c);
                    }
                }
                format!("next window formation of tentacle {} seeded with {} chunks", tentacle + 1, seeds.len())
            }
            Instr::Focus { .. } | Instr::Reread { .. } | Instr::Answer { .. } | Instr::Gain { .. } => "relayed to G1".to_owned(),
        };
        out.done.push(format!("{} {}: {line}", i.code(), i.name()));
    }
    let blocked = |c: usize| {
        inhibit.iter().any(|t| match t {
            Target::Chunk(x) => *x == c,
            Target::File(f) => (g.file_of)(c) == *f,
        })
    };
    let mut used: usize = forward.iter().map(|c| (g.tokens)(*c)).sum();
    let mut sent: Vec<usize> = forward;
    for c in g.handover.iter().chain(&out.queue).copied() {
        if sent.contains(&c) || blocked(c) {
            continue;
        }
        let t = (g.tokens)(c);
        if used + t <= g.room {
            used += t;
            sent.push(c);
        } else if extra > 0 {
            extra -= 1;
            used += t;
            sent.push(c);
        }
    }
    out.sent = sent;
    out
}

/// Section check: every instruction parses from its short form, validates, and goes in an envelope
/// to its part; gains fade to 1 and inhibitions end with their lifetime; the gate: I11 raises the
/// count, I10 holds back a chunk or a file (I4 overrides it), I4 goes first, I5 queues a window,
/// I7 queues held relations and schedules others, I6 stops at its capacity.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    for (t, code) in [
        ("I1:files:a.rs", "I1"),
        ("I2:5,9", "I2"),
        ("I3:12", "I3"),
        ("I4:5,9", "I4"),
        ("I5:2", "I5"),
        ("I6:5,9:4000", "I6"),
        ("I7:12:any", "I7"),
        ("I8:2:5,9:12000", "I8"),
        ("I9:3:1.5:60000", "I9"),
        ("I10:file:src/a.rs", "I10"),
        ("I11:4", "I11"),
    ] {
        let i = Instr::parse(t)?;
        let env = i.envelope(0.5, "check")?;
        if i.code() != code || env.target() != i.target() || env.sender() != G2_ID {
            return Err(format!("nmbrain.instruct: {t} gave {i:?}"));
        }
    }
    if Instr::parse("I11:0").is_ok() || Instr::parse("I6:5:100").is_ok() || Instr::parse("I12:1").is_ok() {
        return Err("nmbrain.instruct: a bad instruction parsed".into());
    }
    let mut d = Desk::default();
    d.issue(Instr::Gain { tentacle: 1, g: 2.0, ttl_ms: 1000 }, 0)?;
    d.issue(Instr::Inhibit { target: Target::Chunk(7), reason: "own edit".into(), ttl_ms: 500 }, 0)?;
    let g = d.gains(3, 500);
    if (g[1] - 1.5).abs() > 1e-6 || (g[0] - 1.0).abs() > 1e-6 || d.inhibited(400) != vec![Target::Chunk(7)] || !d.inhibited(600).is_empty()
    {
        return Err(format!("nmbrain.instruct: desk {g:?}"));
    }
    let mut big = Desk::default();
    big.issue(Instr::Gain { tentacle: 0, g: 99.0, ttl_ms: 1000 }, 0)?;
    big.issue(Instr::Gain { tentacle: 0, g: 3.0, ttl_ms: 1000 }, 0)?;
    big.issue(Instr::Gain { tentacle: 1, g: 0.01, ttl_ms: 1000 }, 0)?;
    let b = big.gains(2, 0);
    if (b[0] - GAIN_MAX).abs() > 1e-6 || (b[1] - GAIN_MIN).abs() > 1e-6 {
        return Err(format!("nmbrain.instruct: gain bounds {b:?}"));
    }
    d.expire(1000);
    if !d.standing.is_empty() || d.issue(Instr::Next { k: 2 }, 0) != Ok(false) {
        return Err("nmbrain.instruct: desk expiry or a one-question instruction stood".into());
    }
    let windows = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8]];
    let file_of = |c: usize| if c <= 3 { "a.rs".to_owned() } else { "b.rs".to_owned() };
    let tokens = |_c: usize| 1000;
    let related = |c: usize, _r: Relation| if c == 1 { vec![2, 9] } else { Vec::new() };
    let gate = Gate { handover: &[1, 4], queue: &[6], windows: &windows, room: 2000, file_of: &file_of, tokens: &tokens, related: &related };
    if route(&gate, &[], &[]).sent != vec![1, 4] {
        return Err("nmbrain.instruct: the plain gate".into());
    }
    let r = route(&gate, &[Instr::Next { k: 1 }], &[]);
    if r.sent != vec![1, 4, 6] {
        return Err(format!("nmbrain.instruct: I11 {:?}", r.sent));
    }
    let r = route(&gate, &[Instr::Inhibit { target: Target::File("a.rs".into()), reason: "x".into(), ttl_ms: 0 }], &[]);
    if r.sent != vec![4, 6] {
        return Err(format!("nmbrain.instruct: I10 {:?}", r.sent));
    }
    let r = route(&gate, &[Instr::Forward { chunks: vec![8] }], &[Target::Chunk(8)]);
    if r.sent.first() != Some(&8) {
        return Err(format!("nmbrain.instruct: I4 {:?}", r.sent));
    }
    let r = route(
        &gate,
        &[
            Instr::ForwardAll { tentacle: 2 },
            Instr::Expand { chunk: 1, relation: Relation::Any },
            Instr::Next { k: 10 },
            Instr::Joint { chunks: vec![1, 4, 6], capacity: 2500 },
        ],
        &[],
    );
    if r.queue != vec![2, 6, 7, 8] || r.scheduled != vec![9] || r.joint != vec![1, 4] || r.sent != vec![1, 4, 2, 6, 7, 8] {
        return Err(format!("nmbrain.instruct: I5 I7 I6 {r:?}"));
    }
    Ok(())
}
