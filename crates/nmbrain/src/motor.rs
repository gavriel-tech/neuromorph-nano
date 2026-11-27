//! The motor nervous system: orders down, reports up, on separate typed paths (master design
//! 3.6; tentacle master plan 3.8, Phases G1-b and H4; protocol pass Solution 3, Motor, and
//! Solution 4; analysis H E.3). Completed by lane M, 2026-11-19.
//!
//! - One vocabulary: every order is an `nm-msg/2` envelope (`nmtype::message::Motor`), every
//!   instruction of G2 (I1 to I11) maps to one of its verbs (`crate::instruct::Instr::verb`).
//! - The copy to G1 cannot be skipped: G1's order ledger (`crate::g1::Ledger`, kept by G1's
//!   coordination layer) records an envelope first and returns an `Ack` only it can make; every
//!   delivery takes that `Ack` and checks it against the ledger before a segment admits the order.
//! - Route (fault F21, settled by Luis 2026-11-18 22:41 UTC): an order about one tentacle's own
//!   focus (I1 FOCUS addressed to one `g0_tx`) goes direct to that `g0_tx`, admitted only after
//!   G1's ledger recorded it (`Down::send_direct`); orders that coordinate several tentacles go
//!   through G1 only (`Down::send`, G1 relays). `route_of` names the route of every verb.
//! - A long order (focus, re-window, joint read) carries a goal id and a state
//!   (`nmtype::order::Order`).
//! - Gains (I9) bounded 0.25 to 4 with an expiry (`nmtype::order::Gains`); reflexes R1 to R5
//!   installed from above by a controlled-reflex envelope recorded in G1's ledger
//!   (`Down::install_reflex`, `crate::reflex::Set::install_envelope`), nothing installed unless
//!   named.
//! - Efference copy (master design 3.6 rule 12): every order sent registers what it expects back
//!   (`Efference`), and an own edit registers its predicted content hash, used up once matched.
//! - Upward, a tentacle's reply is behavioural (`Behavioural<Reply>`); the evidence it names is
//!   sensory and travels apart, by reference (`Sensory<SpanRef>`).
//! - Separation by construction: `Sensory<T>` and `Behavioural<T>` (`nmtype::class`), one checked
//!   conversion with a reason; the topology: the allowed edges by class; any edge outside them, or
//!   carrying the other class, is refused; G2 to a `g0_tx` only for a direct FOCUS.
//! - I8 RESHAPE_WINDOW: `reshape` builds a full window seeded by the ordered piece.
//!
//! The served flow is unchanged until lane T1 sends these orders: `Down::send`, `deliver`,
//! `topology` and `Down.ledger` keep their behaviour; `Down::default()` holds its own ledger until
//! the flow hands it G1's (`Down::with_ledger(g1.ledger_handle())`).
use crate::g0::{Answer, Focus, G1_ID, Instruction, Order, Report, SpanRef, TentacleSide};
pub use crate::g1::{Ack, Ledger, LedgerBook};
use nmtype::class::{Behavioural, DataClass, Sensory};
use nmtype::message::{Arg, MessageKind, Motor, Verb};
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;

/// G2's id on the motor path.
pub const G2: &str = "g2";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmbrain.motor", FaultKind::Policy, m)
}

/// Whether a unit id names a tentacle's `g0_tx`.
#[must_use]
pub fn is_g0(id: &str) -> bool {
    id.starts_with("g0_")
}

/// The route of an order (F21).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Direct to one `g0_tx` with the enforced copy to G1 (one tentacle's own focus).
    Direct,
    /// Through G1 only (coordinating orders; G1 relays or applies).
    ViaG1,
    /// Carried out by G2's own gate (FORWARD, FORWARD_ALL, INHIBIT, NEXT), copy to G1 kept.
    G2Gate,
    /// To the Core Brain (JOINT), copy to G1 kept.
    CoreBrain,
}

/// The route of a verb to a receiver: FOCUS addressed to one `g0_tx` is direct; REREAD, ANSWER,
/// EXPAND, RESHAPE_WINDOW, GAIN (and a FOCUS not addressed to one `g0_tx`) go through G1;
/// FORWARD, FORWARD_ALL, INHIBIT and NEXT are G2's gate; JOINT is the Core Brain's.
#[must_use]
pub fn route_of(verb: Verb, receiver: &str) -> Route {
    match verb {
        Verb::Focus if is_g0(receiver) => Route::Direct,
        Verb::Forward | Verb::ForwardAll | Verb::Inhibit | Verb::Next => Route::G2Gate,
        Verb::Joint => Route::CoreBrain,
        _ => Route::ViaG1,
    }
}

/// The efference copy: what each order sent expects back, and own edits by predicted content hash
/// (a timed prediction, used up once matched).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Efference {
    /// Order id to (receiver, verb) for orders whose reply has not come.
    pub expected: BTreeMap<u64, (String, Verb)>,
    /// Replies matched to an order sent (self-caused), and replies with no order (from outside).
    pub matched: usize,
    pub unexpected: usize,
    /// Own edits: path to the predicted content hash after the edit.
    pub predicted: BTreeMap<String, String>,
}
impl Efference {
    /// Register an order sent.
    pub fn sent(&mut self, m: &Motor) {
        self.expected.insert(m.id(), (m.receiver().to_owned(), m.verb()));
    }
    /// A reply to order `id` from `from`: true when it answers an order sent to `from` (used up).
    pub fn reply(&mut self, id: u64, from: &str) -> bool {
        if self.expected.get(&id).is_some_and(|(r, _)| r == from || !is_g0(r)) {
            self.expected.remove(&id);
            self.matched += 1;
            true
        } else {
            self.unexpected += 1;
            false
        }
    }
    /// Register an own edit: the content hash the file will have after it.
    pub fn predict_edit(&mut self, path: &str, sha: &str) {
        self.predicted.insert(path.to_owned(), sha.to_owned());
    }
    /// A change observed: true when it is the organism's own (the prediction is used up).
    pub fn observe(&mut self, path: &str, sha: &str) -> bool {
        if self.predicted.get(path).is_some_and(|p| p == sha) {
            self.predicted.remove(path);
            true
        } else {
            false
        }
    }
}

/// The path down: a handle to G1's ledger, the long orders' goals, refusals at the segments,
/// G2's gains in force and the efference copy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Down {
    /// G1's ledger (a shared handle; `Down::default()` has its own until given G1's).
    pub ledger: Ledger,
    pub goals: Vec<nmtype::order::Order>,
    /// Envelopes a segment refused because G1's ledger had not acknowledged them (must stay 0
    /// on every served path).
    pub refused: usize,
    /// Every envelope sent, in order.
    pub sent: Vec<Motor>,
    /// Orders sent direct to a `g0_tx` (F21).
    pub direct: usize,
    /// G2's gains per tentacle (I9), bounded 0.25 to 4, with expiry.
    pub gains: nmtype::order::Gains,
    pub efference: Efference,
    next: u64,
}
impl Down {
    /// The path down on G1's ledger (`crate::g1::G1::ledger_handle`).
    #[must_use]
    pub fn with_ledger(ledger: Ledger) -> Self {
        Self { ledger, ..Self::default() }
    }
    /// The next envelope id.
    pub fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }
    /// Deliver an envelope to a segment: refused unless G1's ledger acknowledged it; G1 relays it
    /// as its own order (the route through G1).
    /// # Errors
    /// Not acknowledged, or the segment's own refusal.
    pub fn deliver(&mut self, m: &Motor, segment: &mut dyn TentacleSide, ins: Instruction) -> Result<(), Fault> {
        if !self.ledger.acknowledged(&m.digest()) {
            self.refused += 1;
            return Err(bad(format!("{}: order {} refused, G1 has no copy", segment.id(), m.id())));
        }
        let why = format!("nm-msg/2 {} {}: {}", m.id(), m.verb().name(), m.description());
        let order = Order::new(G1_ID, segment.id(), 0, m.confidence(), &why, ins).map_err(bad)?;
        segment.deliver(order).map_err(bad)
    }
    /// The admission check by token: the `Ack` is this envelope's and G1's ledger holds it.
    fn admit(&mut self, ack: &Ack, m: &Motor, at: &str) -> Result<(), Fault> {
        if self.ledger.verify(ack, m) {
            Ok(())
        } else {
            self.refused += 1;
            Err(bad(format!("{at}: order {} refused, G1's ledger has not acknowledged it", m.id())))
        }
    }
    fn goal_of(&mut self, m: &Motor, r: &Result<(), Fault>, now_ms: u64) -> Result<(), Fault> {
        if let Some(g) = m.goal_id() {
            let verb = match m.verb() {
                Verb::Reshape => nmtype::order::Verb::Rewindow,
                Verb::Joint => nmtype::order::Verb::JointRead,
                _ => nmtype::order::Verb::Focus,
            };
            let mut goal = nmtype::order::Order::new(g, verb, m.sender(), m.receiver(), now_ms, m.expires_ms().saturating_sub(now_ms));
            match r {
                Ok(()) => {
                    goal.accept(now_ms)?;
                    goal.start()?;
                }
                Err(f) => goal.reject(&f.to_string())?,
            }
            self.goals.push(goal);
        }
        Ok(())
    }
    /// Send an order through G1: G1's ledger records it first, then G1 relays it to the segment.
    /// A long order (with a goal id) gets its goal, accepted and executing until its reply comes.
    /// # Errors
    /// As `Ledger::record` and `deliver`.
    pub fn send(&mut self, m: Motor, segment: &mut dyn TentacleSide, ins: Instruction, now_ms: u64) -> Result<(), Fault> {
        let ack = self.ledger.record(&m)?;
        let r = self.admit(&ack, &m, segment.id()).and_then(|()| self.deliver(&m, segment, ins));
        self.goal_of(&m, &r, now_ms)?;
        self.efference.sent(&m);
        self.sent.push(m);
        r
    }
    /// F21: an order about one tentacle's own focus, direct to its `g0_tx`. G1's ledger records
    /// the envelope first (the enforced copy); the tentacle admits it only with G1's `Ack`; it
    /// re-reads under the focus and its report comes up on the behavioural channel. Refused: any
    /// verb whose route is not direct (coordinating orders go through G1 only), or an envelope
    /// addressed to another tentacle.
    /// # Errors
    /// The route, the addressee, the ledger's or the admission's refusal.
    pub fn send_direct(&mut self, m: Motor, segment: &mut dyn TentacleSide, focus: &Focus, now_ms: u64) -> Result<Behavioural<Report>, Fault> {
        if route_of(m.verb(), m.receiver()) != Route::Direct {
            return Err(bad(format!("order {} ({}): not one tentacle's own focus; it goes through G1", m.id(), m.verb().name())));
        }
        if m.receiver() != segment.id() {
            return Err(bad(format!("order {} for {} offered to {}", m.id(), m.receiver(), segment.id())));
        }
        let ack = self.ledger.record(&m)?;
        let r = self.admit(&ack, &m, segment.id());
        self.goal_of(&m, &r, now_ms)?;
        self.efference.sent(&m);
        self.sent.push(m);
        r?;
        self.direct += 1;
        Ok(Behavioural::new(segment.reread(focus)))
    }
    /// I9 GAIN through G1: the envelope (argument `tentacle` a word, `gain` a scale) is recorded in
    /// G1's ledger, then the gain stands for that tentacle until the envelope's expiry, bounded
    /// 0.25 to 4.
    /// # Errors
    /// Another verb, a missing argument, or the ledger's refusal.
    pub fn gain(&mut self, m: Motor, now_ms: u64) -> Result<f32, Fault> {
        if m.verb() != Verb::Gain || route_of(m.verb(), m.receiver()) != Route::ViaG1 {
            return Err(bad(format!("order {}: not a gain", m.id())));
        }
        let word = |n: &str| m.args().iter().find_map(|(k, a)| if let (true, Arg::Word(w)) = (k == n, a) { Some(w.clone()) } else { None });
        let scale = m.args().iter().find_map(|(k, a)| if let (true, Arg::Scale(x)) = (k == "gain", a) { Some(*x) } else { None });
        let (Some(t), Some(g)) = (word("tentacle"), scale) else {
            return Err(bad(format!("order {}: a gain needs tentacle and gain", m.id())));
        };
        if !g.is_finite() || g <= 0.0 {
            return Err(bad(format!("order {}: gain {g}", m.id())));
        }
        let ack = self.ledger.record(&m)?;
        self.admit(&ack, &m, G1_ID)?;
        self.gains.set(&t, g, m.sender(), now_ms, m.expires_ms().saturating_sub(now_ms));
        self.sent.push(m);
        Ok(self.gains.get(&t, now_ms))
    }
    /// G2's gain per tentacle at `now_ms`, for G1's combination (`crate::g1::G1::with_g2`).
    #[must_use]
    pub fn gain_vector(&self, ids: &[&str], now_ms: u64) -> Vec<f32> {
        ids.iter().map(|t| self.gains.get(t, now_ms)).collect()
    }
    /// Install a reflex from above: the controlled-reflex envelope is recorded in G1's ledger
    /// first, then the reflex goes into `set` with its threshold and expiry.
    /// # Errors
    /// The ledger's refusal or the set's (`crate::reflex::Set::install_envelope`).
    pub fn install_reflex(&mut self, m: Motor, set: &mut crate::reflex::Set, now_ms: u64) -> Result<crate::reflex::Code, Fault> {
        let ack = self.ledger.record(&m)?;
        let at = m.receiver().to_owned();
        self.admit(&ack, &m, &at)?;
        let code = set.install_envelope(&m, now_ms).map_err(bad)?;
        self.sent.push(m);
        Ok(code)
    }
    /// Reflexes from a served decision (master design 3.6 rule 9; lane M 2026-11-19): G2 names the
    /// reflexes its gate calls for and installs each through `install_reflex` (a controlled-reflex
    /// envelope recorded in G1's ledger first), lasting this question and `ttl_ms`. Nothing else is
    /// installed. Returns the codes installed.
    /// # Errors
    /// The first refusal of `install_reflex`.
    #[expect(clippy::too_many_arguments, reason = "gate, edit flag, tie threshold, reflex set, clock and expiry are independent inputs")]
    pub fn install_from_gate(&mut self, gate: &str, own_edits: bool, tie_threshold: f32, set: &mut crate::reflex::Set, now_ms: u64, ttl_ms: u64) -> Result<Vec<crate::reflex::Code>, Fault> {
        let mut out = Vec::new();
        for (short, why) in reflexes_for(gate, own_edits, tie_threshold) {
            let id = self.id();
            let m = Motor::new(id, None, MessageKind::Reflex, Verb::Gain, G2, G1_ID, 0.5, &why, now_ms.saturating_add(ttl_ms))?.arg("reflex", Arg::Word(short))?;
            out.push(self.install_reflex(m, set, now_ms)?);
        }
        Ok(out)
    }
    /// I8 from G2: the RESHAPE_WINDOW envelope (receiver the tentacle) is recorded in G1's ledger
    /// first (G1 forms windows, so the order goes through G1), then the tentacle's next window is
    /// built full and seeded by the ordered piece (`reshape`).
    /// # Errors
    /// Another verb, or the ledger's refusal.
    #[expect(clippy::too_many_arguments, reason = "envelope, seeds, neighbours, budget, token counter and taken predicate are independent inputs")]
    pub fn reshape_order(&mut self, m: Motor, seeds: &[usize], around: &[usize], budget: usize, tokens: &dyn Fn(usize) -> usize, taken: &dyn Fn(usize) -> bool) -> Result<Vec<usize>, Fault> {
        if m.verb() != Verb::Reshape {
            return Err(bad(format!("order {}: not a re-window", m.id())));
        }
        let ack = self.ledger.record(&m)?;
        self.admit(&ack, &m, G1_ID)?;
        self.efference.sent(&m);
        self.sent.push(m);
        Ok(reshape(seeds, around, budget, tokens, taken))
    }
    /// Close a long order with its result.
    /// # Errors
    /// No such goal executing.
    pub fn close(&mut self, goal: u64, ok: bool, result: &str) -> Result<(), Fault> {
        let g = self.goals.iter_mut().find(|g| g.goal == goal).ok_or_else(|| bad(format!("no goal {goal}")))?;
        if ok { g.succeed(result) } else { g.abort(result) }
    }
}

/// The path up: the tentacles' replies (behavioural) and the evidence they name (sensory, by
/// reference), on two channels.
#[derive(Debug, Default)]
pub struct Up {
    /// (order id, the reply).
    pub reports: Vec<(u64, Behavioural<Answer>)>,
    pub evidence: Vec<Sensory<SpanRef>>,
}
impl Up {
    /// One reply to order `id`: the report on the behavioural channel, the span it names on the
    /// sensory channel.
    pub fn reply(&mut self, id: u64, a: Answer) {
        self.evidence.push(Sensory::new(a.span.clone()));
        self.reports.push((id, Behavioural::new(a)));
    }
    /// The same, matched against the efference copy: true when the reply answers an order sent.
    pub fn reply_seen(&mut self, eff: &mut Efference, id: u64, a: Answer) -> bool {
        let seen = eff.reply(id, &a.from);
        self.reply(id, a);
        seen
    }
}

/// The reflexes a gate calls for, as short forms with a reason (nothing unless the gate names it):
/// a tie, R3 (order averaging over every reading held) at `tie_threshold` spreads; low, R2 (the
/// definition beside a reported use) and R5 (the rest queued by reference); withheld or absent, R1
/// (no information logged); own edits this question, R4 (self-caused spans marked). Each lasts
/// one question. Settled calls for none.
#[must_use]
pub fn reflexes_for(gate: &str, own_edits: bool, tie_threshold: f32) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let t = if tie_threshold.is_finite() && tie_threshold > 0.0 { tie_threshold } else { crate::reflex::MARGIN_FALLBACK };
    match nmbrain_gate(gate) {
        "tie" => out.push((format!("R3:{t}:1"), "gate tie: settle by mean rank".into())),
        "low" => {
            out.push(("R2:1".into(), "gate low: definitions beside their uses".into()));
            out.push(("R5:1".into(), "gate low: the rest by reference".into()));
        }
        "withheld" => out.push(("R1:1".into(), "gate withheld: log no information".into())),
        _ => {}
    }
    if own_edits {
        out.push(("R4:1".into(), "own edits this question: mark self-caused".into()));
    }
    out
}
fn nmbrain_gate(gate: &str) -> &'static str {
    match crate::chain::GateStatus::of(gate) {
        crate::chain::GateStatus::Tie => "tie",
        crate::chain::GateStatus::Low => "low",
        crate::chain::GateStatus::Withheld => "withheld",
        crate::chain::GateStatus::Settled => "settled",
    }
}

/// I8 RESHAPE_WINDOW: the next window of a tentacle, full and seeded by the ordered piece. The
/// seeds come first, then the pieces in `around` (the caller's order around the seeds: their
/// relations, then their neighbours in the file), each piece no other window holds (`taken`) that
/// fits the budget in tokens; a piece that does not fit is passed over so the window still fills.
#[must_use]
pub fn reshape(seeds: &[usize], around: &[usize], budget: usize, tokens: &dyn Fn(usize) -> usize, taken: &dyn Fn(usize) -> bool) -> Vec<usize> {
    let mut w: Vec<usize> = Vec::new();
    let mut used = 0_usize;
    for &c in seeds.iter().chain(around) {
        if w.contains(&c) || taken(c) {
            continue;
        }
        let t = tokens(c);
        if used + t <= budget {
            used += t;
            w.push(c);
        }
    }
    w
}

/// The allowed edges of the Globalized Context Stack by class: (from, to, class). Orders and
/// reports are behavioural; records of the material are sensory. No edge joins two `g0_tx`; G2
/// reaches a `g0_tx` only by the direct focus edge (`DIRECT`), checked with its verb.
pub const EDGES: &[(&str, &str, DataClass)] = &[
    ("g2", "g1", DataClass::Behavioural),
    ("g1", "g0", DataClass::Behavioural),
    ("g0", "g1", DataClass::Behavioural),
    ("g0", "g1", DataClass::Sensory),
    ("g1", "g2", DataClass::Behavioural),
    ("g1", "g2", DataClass::Sensory),
    ("g2", "core-brain", DataClass::Behavioural),
    ("g2", "core-brain", DataClass::Sensory),
];

/// F21's direct edge: G2 to one `g0_tx`, behavioural, FOCUS only.
pub const DIRECT: (&str, &str, DataClass, Verb) = ("g2", "g0", DataClass::Behavioural, Verb::Focus);

/// The topology check: every edge used, with the class of the item it carried, must be allowed.
/// # Errors
/// The first edge outside `EDGES` or carrying a class its channel does not take.
pub fn topology(used: &[(&str, &str, DataClass)]) -> Result<(), Fault> {
    for (from, to, class) in used {
        if !EDGES.iter().any(|(f, t, c)| f == from && t == to && c == class) {
            return Err(bad(format!("{}: edge {from} -> {to} carrying {} is not allowed", nmtype::class::WRONG_CLASS, class.name())));
        }
    }
    Ok(())
}

/// The topology check with the verb of each order edge: as `topology`, and G2 to a `g0_tx` is
/// allowed only for a behavioural FOCUS (F21's direct route).
/// # Errors
/// The first edge refused.
pub fn topology_routed(used: &[(&str, &str, DataClass, Option<Verb>)]) -> Result<(), Fault> {
    for (from, to, class, verb) in used {
        let direct = (*from, *to, *class) == (DIRECT.0, DIRECT.1, DIRECT.2) && *verb == Some(DIRECT.3);
        if !direct {
            topology(&[(*from, *to, *class)])?;
        }
    }
    Ok(())
}

fn s(f: Fault) -> String {
    f.to_string()
}

#[expect(clippy::too_many_arguments, reason = "the envelope's own fields, for the checks")]
fn envelope(id: u64, kind: MessageKind, verb: Verb, from: &str, to: &str, why: &str, expires: u64) -> Result<Motor, String> {
    Motor::new(id, None, kind, verb, from, to, 0.5, why, expires).map_err(s)
}

/// MO1: one message enum. Every instruction I1 to I11 maps to one verb, its envelope reads back
/// equal, and an envelope with the copy to G1 off does not read.
fn mo1() -> Result<(), String> {
    use crate::instruct::Instr;
    for t in ["I1:files:a.rs", "I2:5,9", "I3:12", "I4:5,9", "I5:2", "I6:5,9:4000", "I7:12:any", "I8:2:5,9:12000", "I9:3:1.5:60000", "I10:file:src/a.rs", "I11:4"] {
        let i = Instr::parse(t)?;
        let m = envelope(1, MessageKind::Instruction, i.verb(), G2, "g0_t1", i.name(), 10)?;
        let m = m.arg("pieces", Arg::Pieces(vec!["k1".into(), "k2".into()])).map_err(s)?;
        if Motor::parse(&m.render()).map_err(s)? != m {
            return Err(format!("MO1: {t} did not read back"));
        }
    }
    if Motor::parse(&envelope(2, MessageKind::Instruction, Verb::Focus, G2, "g0_t1", "x", 9)?.render().replace("copy-to-g1 true", "copy-to-g1 false")).is_ok() {
        return Err("MO1: an envelope with the copy off was read".into());
    }
    Ok(())
}

/// MO2: class boundaries. Replies go up on two typed channels; the one conversion needs a reason;
/// a lateral edge, G2 to a `g0_tx` without the direct FOCUS, and a sensory item on an order channel
/// are refused; the direct FOCUS edge is allowed.
fn mo2() -> Result<(), String> {
    topology(&[("g2", "g1", DataClass::Behavioural), ("g1", "g0", DataClass::Behavioural), ("g0", "g1", DataClass::Sensory)]).map_err(s)?;
    for e in [("g0", "g0", DataClass::Behavioural), ("g2", "g0", DataClass::Behavioural), ("g2", "g1", DataClass::Sensory)] {
        if topology(&[e]).is_ok() {
            return Err(format!("MO2: edge {e:?} allowed"));
        }
    }
    topology_routed(&[("g2", "g0", DataClass::Behavioural, Some(Verb::Focus))]).map_err(s)?;
    for e in [
        ("g2", "g0", DataClass::Behavioural, Some(Verb::Reshape)),
        ("g2", "g0", DataClass::Sensory, Some(Verb::Focus)),
        ("g2", "g0", DataClass::Behavioural, None),
        ("g0", "g0", DataClass::Behavioural, Some(Verb::Focus)),
    ] {
        if topology_routed(&[e]).is_ok() {
            return Err(format!("MO2: routed edge {e:?} allowed"));
        }
    }
    if Behavioural::new(1_u8).into_sensory("").is_ok() || Behavioural::new(1_u8).into_sensory("own edit read back").is_err() {
        return Err("MO2: the checked conversion".into());
    }
    Ok(())
}

/// MO3: own-edit tracking. A predicted content hash is matched once and used up; another hash is
/// new input; an order's reply is matched to the order sent (efference copy) and a reply with no
/// order is not.
fn mo3() -> Result<(), String> {
    let mut e = Efference::default();
    e.predict_edit("src/a.rs", "h1");
    if e.observe("src/a.rs", "h2") || !e.observe("src/a.rs", "h1") || e.observe("src/a.rs", "h1") {
        return Err("MO3: own-edit prediction".into());
    }
    e.sent(&envelope(4, MessageKind::Instruction, Verb::Answer, G2, "g0_t1", "x", 9)?);
    if e.reply(4, "g0_t2") || !e.reply(4, "g0_t1") || e.reply(4, "g0_t1") || e.matched != 1 || e.unexpected != 2 {
        return Err(format!("MO3: efference {e:?}"));
    }
    Ok(())
}

/// MO4: the order route. Through G1: an envelope G1 has not recorded is refused at the segment and
/// the same envelope recorded is admitted; a long order runs accepted, executing, succeeded.
/// Direct (F21): a FOCUS to one `g0_tx` is recorded in G1's ledger first and admitted with its
/// `Ack`; a coordinating verb or another tentacle's address is refused on the direct route; the
/// ledger is G1's (`G1::ledger_handle`), so the copy is in G1.
fn mo4() -> Result<(), String> {
    let mut seg = crate::g0::scripted(1, &["a.rs", "b.rs"], &[("s", &[2.0, 1.0]), ("t", &[1.0, 2.0])]);
    let span = seg.held.first().map(|h| h.span.clone()).ok_or("MO4: no held span")?;
    let g1 = crate::g1::G1::default();
    let mut down = Down::with_ledger(g1.ledger_handle());
    let id = down.id();
    let m = envelope(id, MessageKind::Instruction, Verb::Answer, G2, &seg.id, "answer about a.rs", 10)?;
    if down.deliver(&m, &mut seg, Instruction::Ask { span: span.clone(), about: "q".into() }).is_ok() || down.refused != 1 {
        return Err("MO4: an order G1 had not recorded was admitted".into());
    }
    down.send(m.goal(7), &mut seg, Instruction::Ask { span, about: "q".into() }, 0).map_err(s)?;
    let mut up = Up::default();
    for r in crate::g0::TentacleSide::process(&mut seg) {
        if let crate::g0::Reply::Answer(a) = r {
            up.reply_seen(&mut down.efference, id, a);
        }
    }
    down.close(7, true, "answered").map_err(s)?;
    if up.reports.len() != 1 || up.evidence.len() != 1 || down.goals.first().map(|g| g.state) != Some(nmtype::order::State::Succeeded) || g1.ledger.len() != 1 || down.efference.matched != 1 {
        return Err(format!("MO4: the order's way {:?} {:?}", down.goals, g1.ledger.lines()));
    }
    let id = down.id();
    let f = envelope(id, MessageKind::Instruction, Verb::Focus, G2, &seg.id, "own focus on b.rs", 10)?.goal(8);
    let focus = Focus { statistic: "t".into(), files: vec!["b.rs".into()], ..Focus::default() };
    let rep = down.send_direct(f, &mut seg, &focus, 0).map_err(s)?;
    if down.direct != 1 || g1.ledger.len() != 2 || rep.get().focus.files != vec!["b.rs".to_owned()] || seg.inbox.refused_direct != 0 {
        return Err(format!("MO4: direct focus {} {:?}", down.direct, g1.ledger.lines()));
    }
    let id = down.id();
    let coord = envelope(id, MessageKind::Instruction, Verb::Reshape, G2, &seg.id, "re-window", 10)?;
    let other = envelope(id + 1, MessageKind::Instruction, Verb::Focus, G2, "g0_t9", "someone else", 10)?;
    if down.send_direct(coord, &mut seg, &focus, 0).is_ok() || down.send_direct(other, &mut seg, &focus, 0).is_ok() || down.direct != 1 || g1.ledger.len() != 2 {
        return Err("MO4: a coordinating or misaddressed order went direct".into());
    }
    if route_of(Verb::Focus, "g1") != Route::ViaG1 || route_of(Verb::Joint, "core-brain") != Route::CoreBrain || route_of(Verb::Forward, "g1") != Route::G2Gate {
        return Err("MO4: routes".into());
    }
    Ok(())
}

/// MO5: logged decision events on the motor path: every order sent leaves one line in G1's ledger
/// naming its verb, sender and receiver.
fn mo5() -> Result<(), String> {
    let led = Ledger::default();
    let mut down = Down::with_ledger(led.clone());
    let mut seg = crate::g0::scripted(2, &["a.rs"], &[("s", &[1.0])]);
    let span = seg.held.first().map(|h| h.span.clone()).ok_or("MO5: no span")?;
    for k in 0..3 {
        let id = down.id();
        let m = envelope(id, MessageKind::Instruction, Verb::Answer, G2, &seg.id, &format!("ask {k}"), 10)?;
        down.send(m, &mut seg, Instruction::Ask { span: span.clone(), about: "q".into() }, 0).map_err(s)?;
        crate::g0::TentacleSide::process(&mut seg);
    }
    let lines = led.lines();
    let want = format!("answer g2 -> {}", seg.id);
    if lines.len() != down.sent.len() || lines.iter().any(|l| !l.contains(&want)) {
        return Err(format!("MO5: ledger lines {lines:?}"));
    }
    Ok(())
}

/// MO6 and MO7 on the motor path: G2's gains (the gain estimate's channel) are bounded 0.25 to 4,
/// expire, and travel through G1's ledger; a gain order without its arguments is refused.
fn mo7() -> Result<(), String> {
    let mut down = Down::default();
    let id = down.id();
    let m = envelope(id, MessageKind::Instruction, Verb::Gain, G2, "g1", "lift tentacle 3", 1_000)?
        .arg("tentacle", Arg::Word("g0_t3".into()))
        .map_err(s)?
        .arg("gain", Arg::Scale(9.0))
        .map_err(s)?;
    let g = down.gain(m, 0).map_err(s)?;
    let v = down.gain_vector(&["g0_t1", "g0_t3"], 500);
    if (g - 4.0).abs() > 1e-6 || v != vec![1.0, 4.0] || down.gain_vector(&["g0_t3"], 1_001) != vec![1.0] || down.ledger.len() != 1 {
        return Err(format!("MO7: gain {g} {v:?}"));
    }
    let id = down.id();
    if down.gain(envelope(id, MessageKind::Instruction, Verb::Gain, G2, "g1", "no args", 10)?, 0).is_ok() {
        return Err("MO7: a gain without arguments".into());
    }
    let low = envelope(down.id(), MessageKind::Instruction, Verb::Gain, G2, "g1", "lower", 1_000)?
        .arg("tentacle", Arg::Word("g0_t1".into()))
        .map_err(s)?
        .arg("gain", Arg::Scale(0.01))
        .map_err(s)?;
    if (down.gain(low, 0).map_err(s)? - 0.25).abs() > 1e-6 {
        return Err("MO7: lower bound".into());
    }
    Ok(())
}

/// MO8 on the motor path: reflexes act only as installed from above. A controlled-reflex
/// envelope from G2 installs the named reflex with its threshold and expiry, recorded in G1's
/// ledger; one from a tentacle is refused; nothing is installed unless named.
fn mo8() -> Result<(), String> {
    let mut down = Down::default();
    let mut set = crate::reflex::Set::default();
    let id = down.id();
    let m = envelope(id, MessageKind::Reflex, Verb::Gain, G2, "g0_t1", "install R3", 2_000)?.arg("reflex", Arg::Word("R3:0.3".into())).map_err(s)?;
    let code = down.install_reflex(m, &mut set, 1_000).map_err(s)?;
    let r = set.installed.first().ok_or("MO8: nothing installed")?;
    if code != crate::reflex::Code::R3Tie || set.installed.len() != 1 || (r.threshold - 0.3).abs() > 1e-6 || r.expires_ms != 2_000 || down.ledger.len() != 1 {
        return Err(format!("MO8: install {set:?}"));
    }
    let id = down.id();
    let from_below = envelope(id, MessageKind::Reflex, Verb::Gain, "g0_t1", "g0_t1", "self install", 2_000)?.arg("reflex", Arg::Word("R4".into())).map_err(s)?;
    if down.install_reflex(from_below, &mut set, 1_000).is_ok() || set.installed.len() != 1 {
        return Err("MO8: a reflex installed from below".into());
    }
    set.expire(2_001);
    if !set.installed.is_empty() || !crate::reflex::Set::default().installed.is_empty() {
        return Err("MO8: expiry or a default install".into());
    }
    let codes = down.install_from_gate("tie: two within the band", true, 0.4, &mut set, 0, 1_000).map_err(s)?;
    if codes != vec![crate::reflex::Code::R3Tie, crate::reflex::Code::R4SelfEdit] || !down.install_from_gate("settled", false, 0.4, &mut set, 0, 1_000).map_err(s)?.is_empty() {
        return Err(format!("MO8: reflexes from the gate {codes:?}"));
    }
    Ok(())
}

/// ISO on the motor path: I4 FORWARD overrides inhibition; I8 builds a full window seeded by the
/// ordered piece holding no piece another window holds; a direct order never reaches a second
/// tentacle and no tentacle's inbox counts a refused direct envelope.
fn iso() -> Result<(), String> {
    use crate::instruct::{Gate, Instr, Relation, Target, route};
    let windows = vec![vec![1, 2], vec![3, 4]];
    let file_of = |c: usize| format!("f{c}.rs");
    let tokens = |_c: usize| 100;
    let related = |_c: usize, _r: Relation| Vec::new();
    let gate = Gate { handover: &[1, 3], queue: &[], windows: &windows, room: 1_000, file_of: &file_of, tokens: &tokens, related: &related };
    let r = route(&gate, &[Instr::Forward { chunks: vec![4] }], &[Target::Chunk(4)]);
    if r.sent.first() != Some(&4) {
        return Err(format!("ISO: I4 did not override inhibition {:?}", r.sent));
    }
    let toks = |c: usize| if c == 7 { 1_000 } else { 300 };
    let taken = |c: usize| c == 5;
    let w = reshape(&[9], &[5, 7, 6, 8, 10], 1_200, &toks, &taken);
    if w != vec![9, 6, 8, 10] || w.iter().map(|c| toks(*c)).sum::<usize>() != 1_200 {
        return Err(format!("ISO: I8 window {w:?}"));
    }
    let mut a = crate::g0::scripted(1, &["a.rs"], &[("s", &[1.0])]);
    let b = crate::g0::scripted(2, &["b.rs"], &[("s", &[1.0])]);
    let mut down = Down::default();
    let id = down.id();
    let m = envelope(id, MessageKind::Instruction, Verb::Focus, G2, &b.id, "b's focus", 10)?;
    if down.send_direct(m, &mut a, &Focus::default(), 0).is_ok() || a.inbox.refused_direct != 0 || b.inbox.refused_direct != 0 || a.inbox.queue.len() + b.inbox.queue.len() != 0 {
        return Err("ISO: a direct order reached another tentacle".into());
    }
    Ok(())
}

/// Section check (tests MO1 to MO8 on the motor path, and ISO's motor part).
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    for (name, f) in [("MO1", mo1 as fn() -> Result<(), String>), ("MO2", mo2), ("MO3", mo3), ("MO4", mo4), ("MO5", mo5), ("MO7", mo7), ("MO8", mo8), ("ISO", iso)] {
        f().map_err(|e| format!("nmbrain.motor {name}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mo_and_iso() {
        assert_eq!(check(), Ok(()));
    }

    /// The served flow's calls as `drone_path` makes them today (record, then push the envelope;
    /// `send` through G1; `topology` on the 3-field edges) behave as before lane M: one ledger
    /// line per recorded envelope, nothing refused, G2 to a `g0_tx` still refused, no direct order.
    #[test]
    fn served_calls_unchanged() {
        let mut d = Down::default();
        let id = d.id();
        let m = Motor::new(id, None, MessageKind::Instruction, Verb::Reshape, G2, "g0_t1", 0.5, "read pieces never seen", 10).unwrap();
        assert!(d.ledger.record(&m).is_ok());
        d.sent.push(m.goal(id));
        let mut seg = crate::g0::scripted(1, &["a.rs"], &[("s", &[1.0])]);
        let span = seg.held[0].span.clone();
        let id = d.id();
        let m = Motor::new(id, None, MessageKind::Instruction, Verb::Answer, G2, &seg.id, 0.5, "chosen by G2's selection", 10).unwrap();
        d.send(m, &mut seg, Instruction::Ask { span, about: "q".into() }, 0).unwrap();
        assert_eq!((d.ledger.len(), d.refused, d.sent.len(), d.direct), (2, 0, 2, 0));
        assert!(topology(&[("g2", "g0", DataClass::Behavioural)]).is_err());
        assert!(topology(&[("g2", "g1", DataClass::Behavioural), ("g1", "g2", DataClass::Sensory), ("g2", "core-brain", DataClass::Sensory)]).is_ok());
        assert_eq!(seg.inbox.queue.len(), 1);
    }

    #[test]
    fn ledger_is_g1s() {
        let g1 = crate::g1::G1::default();
        let mut d = Down::with_ledger(g1.ledger_handle());
        let id = d.id();
        let m = Motor::new(id, None, MessageKind::Instruction, Verb::Forward, G2, "g1", 0.5, "up", 10).unwrap();
        let ack = g1.record_order(&m).unwrap();
        assert!(d.ledger.verify(&ack, &m));
        let other = Motor::new(id + 1, None, MessageKind::Instruction, Verb::Forward, G2, "g1", 0.5, "up", 10).unwrap();
        assert!(!d.ledger.verify(&ack, &other));
        assert_eq!(g1.ledger.len(), 1);
    }
}
