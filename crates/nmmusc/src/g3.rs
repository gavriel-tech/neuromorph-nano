//! G3, the pattern level of the Globalized Context Stack, as its own level in RAM.
//!
//! G3 reads the configuration records of the calls (behaviour only: the request's kind, the area,
//! the choice made at each named decision site, who made it, and the verified outcome once the
//! verdict is joined) and learns which behaviour recurs and gains. One cycle:
//! 1. Withdrawal: every call a pack decided whose outcome is known is replayed with the rule's
//!    choice; the paired difference is the pack's observed gain, recorded in the store, whose
//!    change rule withdraws a pack that stopped gaining (`withdraw::should_withdraw_change`).
//! 2. Candidates: recurring chunks of the records' symbol sequences under the first-order Markov
//!    chance model (`miner::mine`); a candidate is a request context (kind, area) with the choice
//!    made at one site.
//! 3. Negative selection: a candidate whose recurrence rests on self-caused or test-derived records
//!    is refused (its service occurrences alone must still reach its derived recurrence threshold).
//! 4. Successive halving by paired replay: for each context and site, the other choices are the
//!    arms; each round every surviving arm is replayed on twice as many of the context's
//!    selection records, scored by the paired difference against the outcome as served, and the
//!    better half stays. The one left is a provisional bias in RAM. It acts as a prior only once
//!    its gain holds on held-out records (2026-11-18: biases chosen on four selection records moved
//!    served calls to fewer answer ranges on a tuning set).
//! 5. Promotion: a bias becomes a pack when its paired gain holds on the context's held-out
//!    records (records it was not selected on), `gain > z * se`. How often counts as "a lot" is
//!    thus derived from the repeat count against the measured gain and its spread: the needed
//!    repeats grow as `(z * sd / gain)^2`. A bias whose held-out gain is not positive is dropped.
//!
//! Selection and held-out records are fixed per record by a bit of the SHA-256 of its id, so a
//! record never moves between the two as records accumulate. Contexts are taken specific first
//! (kind and area, then area alone, then kind alone, then none); a wider context is judged only on
//! the records no narrower winner already covers, so one gain is learned once.
//!
//! Packs act downward as priors at their sites (`priors::Control`): the rule keeps control unless
//! the pack's evidence exceeds the rule's margin; G2 may veto. A withdrawn pack is never loaded and
//! never written again.
use crate::gain::GainRecord;
use crate::log::{BehaviourLog, Trace};
use crate::pack::Pack;
use crate::priors::{Control, Prior, Site, site_of};
use crate::store::MuscleStore;
use crate::symbol::{BehaviouralSymbol, Message, MessageKind};
use crate::{Fault, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Where a configuration record came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// A call served to a caller.
    Service,
    /// A call the organism caused itself (its own prime, a call after its own edit).
    SelfCaused,
    /// A call made by a check or a test run.
    Test,
}

/// One call's configuration record: behaviour only, never the request's words or the material.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub id: String,
    /// The request's kind (its wording kind).
    pub kind: String,
    /// The area of the material the call worked in (the project, or a part of it).
    pub area: String,
    /// The choice made at each named site, in the order taken.
    pub choices: Vec<(Site, String)>,
    /// The pack that decided a site (its fingerprint), if any.
    pub by_pack: Option<String>,
    pub origin: Origin,
    /// Verified outcome in [0, 1], higher is better; `None` until the verdict is joined.
    pub outcome: Option<f64>,
    /// What the flow did on the call, as behaviour (2026-11-18): the parts that ran (`ran:<part>`)
    /// and the pack's status (`status:<served|withheld>`, `floor:<met|unmet>`); mined with the
    /// choices, never a site G3 sets.
    pub observed: Vec<String>,
}

impl Record {
    /// The record's steps as behavioural messages.
    #[must_use]
    pub fn steps(&self) -> Vec<Message> {
        let mut out = vec![
            Message::behavioural(MessageKind::Request, "g2", &format!("kind:{}", self.kind)),
            Message::behavioural(MessageKind::Request, "g1", &format!("area:{}", self.area)),
        ];
        out.extend(self.choices.iter().map(|(s, c)| Message::behavioural(MessageKind::Instruction, "g1", &s.action(c))));
        out.extend(self.observed.iter().map(|d| Message::behavioural(MessageKind::Report, "g2", d)));
        out
    }
    /// The choice made at `site`.
    #[must_use]
    pub fn choice(&self, site: Site) -> Option<&str> {
        self.choices.iter().find(|(s, _)| *s == site).map(|(_, c)| c.as_str())
    }
}

/// A request context: a kind and an area, each possibly open (`None` matches any).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Context {
    pub kind: Option<String>,
    pub area: Option<String>,
}
impl Context {
    #[must_use]
    pub fn matches(&self, kind: &str, area: &str) -> bool {
        self.kind.as_deref().is_none_or(|k| k == kind) && self.area.as_deref().is_none_or(|a| a == area)
    }
    /// Narrower contexts first: both named, then the area alone, the kind alone, none.
    fn width(&self) -> u8 {
        match (&self.kind, &self.area) {
            (Some(_), Some(_)) => 0,
            (None, Some(_)) => 1,
            (Some(_), None) => 2,
            (None, None) => 3,
        }
    }
    fn covers(&self, other: &Self) -> bool {
        (self.kind.is_none() || self.kind == other.kind) && (self.area.is_none() || self.area == other.area)
    }
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{} requests in {}",
            self.kind.as_deref().unwrap_or("all"),
            self.area.as_deref().map_or_else(|| "every area".to_owned(), |a| format!("area {a}"))
        )
    }
}

/// A learned behaviour: in this context, at this site, this choice in place of the served one.
#[derive(Clone, Debug, PartialEq)]
pub struct Bias {
    pub context: Context,
    pub site: Site,
    pub choice: String,
    /// The choice it replaces (the rule's, as served on the records it was learned from).
    pub served: String,
    /// Paired gain on the selection records, and on the held-out records once there are two.
    pub selection: GainRecord,
    pub held_out: Option<GainRecord>,
    /// The pack it became, when promoted.
    pub pack: Option<String>,
}
impl Bias {
    /// Evidence in standard errors: held-out when known, else selection.
    #[must_use]
    pub fn evidence(&self) -> f64 {
        evidence_of(self.held_out.as_ref().unwrap_or(&self.selection))
    }
}

/// Gain over its standard error; a positive gain with no spread counts as unbounded evidence.
#[must_use]
pub fn evidence_of(g: &GainRecord) -> f64 {
    if g.se > 0.0 {
        g.diff / g.se
    } else if g.diff > 0.0 {
        f64::MAX
    } else {
        0.0
    }
}

/// A reloaded pack: its context and its priors.
#[derive(Clone, Debug, PartialEq)]
struct Live {
    fp: String,
    context: Context,
    priors: Vec<Prior>,
    /// The choice the pack replaced (from its description).
    served: Option<String>,
}

/// What one cycle did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cycle {
    /// Candidates mined, and those refused by negative selection.
    pub candidates: usize,
    pub refused_self: Vec<String>,
    /// Groups whose best arm did not gain (on selection or held-out records).
    pub refused_gain: Vec<String>,
    pub provisional: Vec<String>,
    /// Packs promoted: (fingerprint, description).
    pub promoted: Vec<(String, String)>,
    pub withdrawn: Vec<String>,
    pub replays: usize,
}

/// The replay hook: run the call of a record again with `choice` at `site` and return its
/// verified outcome (`None` when it cannot be run or verified).
pub type Replay<'a> = dyn FnMut(&Record, Site, &str) -> Option<f64> + 'a;

/// G3 in RAM.
#[derive(Debug, Default)]
pub struct G3 {
    records: Vec<Record>,
    log: BehaviourLog,
    /// Replayed outcomes: (record id, site, choice).
    replays: BTreeMap<(String, Site, String), f64>,
    biases: Vec<Bias>,
    live: Vec<Live>,
    /// Calls a pack decided whose gain was already recorded.
    accounted: BTreeSet<String>,
    pub base_hash: String,
    pub gate_z: f64,
    pub withdraw_z: f64,
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = if xs.len() > 1 { xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, (v / n).sqrt())
}

/// The paired gain record of `pairs` (outcome with the choice, outcome as served).
#[must_use]
pub fn paired(pairs: &[(f64, f64)]) -> GainRecord {
    let d: Vec<f64> = pairs.iter().map(|(w, s)| w - s).collect();
    let (diff, se) = mean_se(&d);
    let n = pairs.len() as u64;
    let (wm, _) = mean_se(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let (sm, _) = mean_se(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    GainRecord { with_n: n, with_mean: wm, without_n: n, without_mean: sm, diff, se }
}

/// Whether a record is held out from selection: one bit of the SHA-256 of its id.
#[must_use]
pub fn held_out(id: &str) -> bool {
    nmhash::sha256_bytes(id.as_bytes()).bytes().next().is_some_and(|b| u32::from(b).is_multiple_of(2))
}

impl G3 {
    /// An empty G3 for the base weights `base_hash`, with the gates of `designs/muscle.json`.
    #[must_use]
    pub fn new(base_hash: &str) -> Self {
        let p = crate::Policy::load();
        Self { base_hash: base_hash.to_owned(), gate_z: p.gate_z, withdraw_z: p.withdraw_z, ..Self::default() }
    }

    /// Log one more behavioural step of a record already observed.
    /// # Errors
    /// `nmmusc-sensory-refused` for a sensory item.
    pub fn log_step(&mut self, id: &str, item: &impl BehaviouralSymbol) -> Result<u32> {
        self.log.record(id, item)
    }

    /// Take in one call's configuration record.
    /// # Errors
    /// `nmmusc-duplicate` for an id already observed; a sensory step.
    pub fn observe(&mut self, r: Record) -> Result<()> {
        if self.records.iter().any(|x| x.id == r.id) {
            return Err(Fault::new("nmmusc-duplicate", format!("record {} already observed", r.id)));
        }
        for m in r.steps() {
            self.log.record(&r.id, &m)?;
        }
        if let Some(o) = r.outcome {
            self.log.set_outcome(&r.id, o)?;
        }
        self.records.push(r);
        Ok(())
    }

    /// Join a verified outcome to a record.
    /// # Errors
    /// `nmmusc-unknown-trace`; a non-finite outcome.
    pub fn verdict(&mut self, id: &str, outcome: f64) -> Result<()> {
        self.log.set_outcome(id, outcome)?;
        if let Some(r) = self.records.iter_mut().find(|r| r.id == id) {
            r.outcome = Some(outcome);
        }
        Ok(())
    }

    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.records
    }
    #[must_use]
    pub fn biases(&self) -> &[Bias] {
        &self.biases
    }
    /// Fingerprints of the packs acting now.
    #[must_use]
    pub fn live_packs(&self) -> Vec<String> {
        self.live.iter().map(|l| l.fp.clone()).collect()
    }

    /// Load every pack of `store` valid for this base and not withdrawn as priors.
    /// # Errors
    /// An unreadable index.
    pub fn reload(&mut self, store: &MuscleStore) -> Result<usize> {
        self.live.clear();
        for row in store.index()? {
            if row.withdrawn || row.base_hash != self.base_hash {
                continue;
            }
            let Ok(p) = store.load(&row.fingerprint, &self.base_hash) else { continue };
            if let Some(l) = live_of(&p, 0.0) {
                self.live.push(l);
            }
        }
        Ok(self.live.len())
    }

    /// The priors for a call of `kind` in `area`: live packs, and provisional biases whose gain holds
    /// on held-out records, whose context matches, strongest evidence first (the first prior at a
    /// site is the one arbitrated).
    #[must_use]
    pub fn control(&self, kind: &str, area: &str) -> Control {
        let mut pr: Vec<Prior> = self.live.iter().filter(|l| l.context.matches(kind, area)).flat_map(|l| l.priors.clone()).collect();
        let held = |b: &Bias| b.held_out.is_some_and(|g| g.promotable(self.gate_z));
        pr.extend(self.biases.iter().filter(|b| b.pack.is_none() && held(b) && b.context.matches(kind, area)).map(|b| Prior {
            site: b.site,
            alternative: b.choice.clone(),
            evidence: b.evidence(),
            pack: format!("bias:{}", b.context.describe()),
            step: 0,
        }));
        pr.sort_by(|a, b| b.evidence.total_cmp(&a.evidence));
        Control::new(pr)
    }

    /// Drop the replays held for record `id` (its outcome was judged again on another measure, so
    /// its replays are scored again).
    pub fn forget_replays(&mut self, id: &str) {
        self.replays.retain(|(r, _, _), _| r != id);
    }

    /// Whether a replay of record `id` with `choice` at `site` is held.
    #[must_use]
    pub fn has_replay(&self, id: &str, site: Site, choice: &str) -> bool {
        self.replays.contains_key(&(id.to_owned(), site, choice.to_owned()))
    }
    /// Hold the verified outcome of a replay run outside a cycle.
    pub fn note_replay(&mut self, id: &str, site: Site, choice: &str, outcome: f64) {
        if outcome.is_finite() {
            self.replays.insert((id.to_owned(), site, choice.to_owned()), outcome);
        }
    }
    /// Per verified service call not decided by a pack: each choice at `site` with its verified
    /// outcome (as served, and as replayed).
    #[must_use]
    pub fn outcomes(&self, site: Site) -> Vec<BTreeMap<String, f64>> {
        self.records
            .iter()
            .filter(|r| r.origin == Origin::Service && r.by_pack.is_none())
            .filter_map(|r| {
                let mut m: BTreeMap<String, f64> =
                    self.replays.iter().filter(|((id, s, _), _)| *id == r.id && *s == site).map(|((_, _, c), v)| (c.clone(), *v)).collect();
                m.insert(r.choice(site)?.to_owned(), r.outcome?);
                Some(m)
            })
            .collect()
    }

    fn replayed(&mut self, r: &Record, site: Site, choice: &str, replay: &mut Replay<'_>, n: &mut usize) -> Option<f64> {
        if r.choice(site) == Some(choice) {
            return r.outcome;
        }
        let key = (r.id.clone(), site, choice.to_owned());
        if let Some(v) = self.replays.get(&key) {
            return Some(*v);
        }
        let v = replay(r, site, choice)?;
        *n += 1;
        self.replays.insert(key, v);
        Some(v)
    }

    /// The decoded steps of a chunk: one (context, site, choice) per site it names. A chunk that
    /// names several sites (on the flow, every call has an allocation and a chain) is evidence
    /// about each of them, each judged on its own by paired replay at that site.
    fn decode(&self, symbols: &[u32]) -> Vec<(Context, Site, String)> {
        let mut ctx = Context { kind: None, area: None };
        let mut at: Vec<(Site, String)> = Vec::new();
        for &c in symbols {
            let Some(e) = self.log.codebook.entry(c) else { return Vec::new() };
            let d = &e.descriptor;
            let action = d.rsplit(' ').next().unwrap_or(d);
            if let Some(k) = action.strip_prefix("kind:") {
                ctx.kind = Some(k.to_owned());
            } else if let Some(a) = action.strip_prefix("area:") {
                ctx.area = Some(a.to_owned());
            } else if let Some(s) = site_of(action) {
                if at.iter().any(|(x, _)| *x == s.0) {
                    return Vec::new();
                }
                at.push(s);
            }
        }
        at.into_iter().map(|(s, c)| (ctx.clone(), s, c)).collect()
    }

    /// One learning cycle (module note). `alternatives` gives the choices a site offers.
    /// # Errors
    /// Store faults while recording a gain or writing a pack.
    pub fn cycle(&mut self, store: Option<&MuscleStore>, alternatives: &dyn Fn(Site) -> Vec<String>, replay: &mut Replay<'_>) -> Result<Cycle> {
        let mut out = Cycle::default();
        let mut n = 0usize;
        self.account(store, replay, &mut n, &mut out)?;
        let traces: Vec<Trace> = self.log.traces().iter().filter(|t| t.outcome.is_some()).cloned().collect();
        let origin: BTreeMap<&str, Origin> = self.records.iter().map(|r| (r.id.as_str(), r.origin)).collect();
        let mined = crate::miner::mine(&traces);
        out.candidates = mined.candidates.len();
        let mut groups: BTreeMap<(Context, Site, String), ()> = BTreeMap::new();
        for c in &mined.candidates {
            let service = c.occurrences.iter().filter(|o| traces.get(o.trace).and_then(|t| origin.get(t.id.as_str())) == Some(&Origin::Service)).count();
            for (ctx, site, served) in self.decode(&c.symbols) {
                if (service as f64) < c.threshold {
                    out.refused_self.push(format!("{} at {}: {} of {} occurrences in service", ctx.describe(), site.name(), service, c.count()));
                    continue;
                }
                groups.insert((ctx, site, served), ());
            }
        }
        let mut keys: Vec<(Context, Site, String)> = groups.into_keys().collect();
        keys.sort_by_key(|k| (k.0.width(), k.0.clone(), k.1, k.2.clone()));
        for (ctx, site, served) in keys {
            self.group(store, &ctx, site, &served, alternatives, replay, &mut n, &mut out)?;
        }
        out.replays = n;
        Ok(out)
    }

    /// Withdrawal accounting (step 1).
    fn account(&mut self, store: Option<&MuscleStore>, replay: &mut Replay<'_>, n: &mut usize, out: &mut Cycle) -> Result<()> {
        let Some(st) = store else { return Ok(()) };
        let due: Vec<Record> =
            self.records.iter().filter(|r| r.by_pack.is_some() && r.outcome.is_some() && !self.accounted.contains(&r.id)).cloned().collect();
        for r in due {
            let (Some(fp), Some(o)) = (r.by_pack.clone(), r.outcome) else { continue };
            let Some(l) = self.live.iter().find(|l| l.fp == fp) else { continue };
            let (Some(p), Some(rule)) = (l.priors.first(), l.served.clone()) else { continue };
            let site = p.site;
            let Some(base) = self.replayed(&r, site, &rule, replay, n) else { continue };
            self.accounted.insert(r.id.clone());
            let u = st.record_gain(&fp, &self.base_hash, o - base, self.withdraw_z)?;
            if u.withdrawn {
                self.live.retain(|l| l.fp != fp);
                self.biases.retain(|b| b.pack.as_deref() != Some(fp.as_str()));
                out.withdrawn.push(fp);
            }
        }
        Ok(())
    }

    /// Successive halving, held-out gate and promotion for one context and site (steps 4, 5).
    #[expect(clippy::too_many_arguments, reason = "the group, the store, the site's choices, the replay hook and the report")]
    fn group(
        &mut self,
        store: Option<&MuscleStore>,
        ctx: &Context,
        site: Site,
        served: &str,
        alternatives: &dyn Fn(Site) -> Vec<String>,
        replay: &mut Replay<'_>,
        n: &mut usize,
        out: &mut Cycle,
    ) -> Result<()> {
        let name = format!("{} at {} ({served} as served)", ctx.describe(), site.name());
        // Already learned here: a live pack of this context and site.
        if self.live.iter().any(|l| l.context == *ctx && l.priors.iter().any(|p| p.site == site)) {
            return Ok(());
        }
        // Narrower winners already cover some records; this context is judged on the rest.
        let narrower: Vec<Context> = self
            .biases
            .iter()
            .filter(|b| b.site == site && b.context != *ctx && ctx.covers(&b.context))
            .map(|b| b.context.clone())
            .chain(self.live.iter().filter(|l| l.context != *ctx && ctx.covers(&l.context)).map(|l| l.context.clone()))
            .collect();
        let recs: Vec<Record> = self
            .records
            .iter()
            .filter(|r| {
                r.origin == Origin::Service
                    && r.outcome.is_some()
                    && r.by_pack.is_none()
                    && r.choice(site) == Some(served)
                    && ctx.matches(&r.kind, &r.area)
                    && !narrower.iter().any(|w| w.matches(&r.kind, &r.area))
            })
            .cloned()
            .collect();
        if recs.is_empty() {
            return Ok(());
        }
        let (held, sel): (Vec<Record>, Vec<Record>) = recs.into_iter().partition(|r| held_out(&r.id));
        let mut arms: Vec<String> = alternatives(site).into_iter().filter(|c| c != served).collect();
        let mut scores: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut budget = 2usize;
        loop {
            for arm in &arms {
                let mut pairs = Vec::new();
                for r in sel.iter().take(budget) {
                    if let (Some(w), Some(s)) = (self.replayed(r, site, arm, replay, n), r.outcome) {
                        pairs.push((w, s));
                    }
                }
                scores.insert(arm.clone(), pairs);
            }
            let mean = |a: &String| scores.get(a).map_or(f64::MIN, |p| if p.len() < 2 { f64::MIN } else { paired(p).diff });
            arms.sort_by(|a, b| mean(b).total_cmp(&mean(a)).then(a.cmp(b)));
            if arms.len() <= 1 || budget >= sel.len() {
                break;
            }
            arms.truncate(arms.len().div_ceil(2));
            budget *= 2;
        }
        let Some(best) = arms.first().cloned() else { return Ok(()) };
        let sel_pairs = scores.get(&best).cloned().unwrap_or_default();
        if sel_pairs.len() < 2 || !paired(&sel_pairs).promotable(self.gate_z) {
            out.refused_gain.push(format!("{name}: no arm gains on {} selection records", sel_pairs.len()));
            self.biases.retain(|b| !(b.context == *ctx && b.site == site));
            return Ok(());
        }
        let mut hp = Vec::new();
        for r in &held {
            if let (Some(w), Some(s)) = (self.replayed(r, site, &best, replay, n), r.outcome) {
                hp.push((w, s));
            }
        }
        let held_g = (hp.len() >= 2).then(|| paired(&hp));
        let mut bias = Bias { context: ctx.clone(), site, choice: best.clone(), served: served.to_owned(), selection: paired(&sel_pairs), held_out: held_g, pack: None };
        self.biases.retain(|b| !(b.context == *ctx && b.site == site));
        match held_g {
            Some(g) if g.diff <= 0.0 => {
                out.refused_gain.push(format!("{name}: {best} gains on selection, not on {} held-out records", hp.len()));
            }
            Some(g) if g.promotable(self.gate_z) => {
                let fp = self.promote(store, &bias)?;
                if let Some((fp, d)) = fp {
                    bias.pack = Some(fp.clone());
                    out.promoted.push((fp, d));
                } else {
                    out.provisional.push(format!("{name}: {best} (held-out gain holds; no store to write)"));
                }
                self.biases.push(bias);
            }
            _ => {
                out.provisional.push(format!("{name}: {best}, selection gain {:.3} on {}", bias.selection.diff, sel_pairs.len()));
                self.biases.push(bias);
            }
        }
        Ok(())
    }

    /// Write a bias as a pack; `None` without a store or when that pack was withdrawn before.
    fn promote(&mut self, store: Option<&MuscleStore>, b: &Bias) -> Result<Option<(String, String)>> {
        let Some(st) = store else { return Ok(None) };
        let mut steps = Vec::new();
        if let Some(k) = &b.context.kind {
            steps.push(Message::behavioural(MessageKind::Request, "g2", &format!("kind:{k}")));
        }
        if let Some(a) = &b.context.area {
            steps.push(Message::behavioural(MessageKind::Request, "g1", &format!("area:{a}")));
        }
        steps.push(Message::behavioural(MessageKind::Instruction, "g1", &b.site.action(&b.choice)));
        let mut chunk = Vec::new();
        for m in &steps {
            chunk.push(self.log.codebook.code(m)?);
        }
        let g = b.held_out.unwrap_or(b.selection);
        let desc = format!(
            "{}: {} {} in place of {} (paired gain {:.3}, se {:.3}, {} held-out replays)",
            b.context.describe(),
            b.site.name(),
            b.choice,
            b.served,
            g.diff,
            g.se,
            g.with_n
        );
        let pack = Pack::build(&chunk, &self.log.codebook, &desc, &self.base_hash, g)?;
        let fp = pack.header.fingerprint.clone();
        if st.index()?.iter().any(|r| r.fingerprint == fp && r.withdrawn) {
            return Ok(None);
        }
        st.write(&pack)?;
        if let Some(l) = live_of(&pack, 0.0) {
            self.live.push(l);
        }
        Ok(Some((fp, desc)))
    }
}

fn live_of(p: &Pack, opposed: f64) -> Option<Live> {
    let steps: Vec<String> = p.steps().into_iter().map(|(_, d)| d).collect();
    let mut ctx = Context { kind: None, area: None };
    for d in &steps {
        let a = d.rsplit(' ').next().unwrap_or(d);
        if let Some(k) = a.strip_prefix("kind:") {
            ctx.kind = Some(k.to_owned());
        } else if let Some(x) = a.strip_prefix("area:") {
            ctx.area = Some(x.to_owned());
        }
    }
    let served = p.header.description.rsplit_once(" in place of ").and_then(|(_, r)| r.split_whitespace().next()).map(str::to_owned);
    let ev = evidence_of(&p.header.promotion) - opposed.max(0.0);
    let priors = crate::priors::priors(&p.header.fingerprint, &steps, ev);
    (!priors.is_empty()).then(|| Live { fp: p.header.fingerprint.clone(), context: ctx, priors, served })
}

/// The sites' choices used by the checks.
fn three(_: Site) -> Vec<String> {
    ["nbv", "uniform", "off"].iter().map(|s| (*s).to_owned()).collect()
}

/// Scripted service: `n` calls of each kind in one area, served with `nbv`; outcomes from
/// `score(kind, choice, i)`.
fn scripted(g: &mut G3, kinds: &[&str], n: usize, origin: Origin, score: &dyn Fn(&str, &str, usize) -> f64) -> Result<()> {
    for k in kinds {
        for i in 0..n {
            g.observe(Record {
                id: format!("{k}-{i}-{origin:?}"),
                kind: (*k).to_owned(),
                area: "p".into(),
                choices: vec![(Site::Allocation, "nbv".into())],
                by_pack: None,
                origin,
                outcome: Some(score(k, "nbv", i)),
                observed: Vec::new(),
            })?;
        }
    }
    Ok(())
}

/// A small deterministic spread for scripted outcomes.
fn jitter(i: usize, salt: usize) -> f64 {
    ((i * 37 + salt * 11) % 13) as f64 / 13.0 * 0.06 - 0.03
}

/// G3-T (specification 4.6): a good pattern is promoted; a neutral one and a self-confounded one
/// are refused; after drift the pack is withdrawn in fewer uses than the whole-history margin rule (by the
/// change rule, with no more false withdrawals).
/// # Errors
/// One fault line.
pub fn check_g3t() -> std::result::Result<(), String> {
    let e = |f: Fault| f.to_string();
    let dir = crate::scratch_dir("g3t").map_err(e)?;
    let st = MuscleStore::open(&dir).map_err(e)?;
    // Name requests gain from `uniform`; explain requests gain from nothing (neutral).
    let score = |k: &str, c: &str, i: usize| if k == "name" && c == "uniform" { 0.85 + jitter(i, 1) } else { 0.5 + jitter(i, 0) };
    let mut g = G3::new("base-a");
    scripted(&mut g, &["name", "explain"], 12, Origin::Service, &score).map_err(e)?;
    // A self-confounded pattern: `value` requests seen only in the organism's own calls, where
    // `off` scores well because the organism caused the outcome.
    let selfish = |k: &str, c: &str, i: usize| if k == "value" && c == "off" { 0.95 + jitter(i, 2) } else { 0.4 + jitter(i, 0) };
    scripted(&mut g, &["value"], 12, Origin::SelfCaused, &selfish).map_err(e)?;
    let mut replay = |r: &Record, _: Site, c: &str| {
        let i: usize = r.id.split('-').nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
        Some(if r.origin == Origin::Service { score(&r.kind, c, i) } else { selfish(&r.kind, c, i) })
    };
    let cy = g.cycle(Some(&st), &three, &mut replay).map_err(e)?;
    let promoted: Vec<&String> = cy.promoted.iter().map(|p| &p.1).collect();
    if cy.promoted.len() != 1 || !promoted.iter().all(|d| d.starts_with("name requests in area p") && d.contains(" uniform in place of nbv")) {
        return Err(format!("nmmusc.g3t: promoted {promoted:?}, refused {:?} {:?}", cy.refused_gain, cy.refused_self));
    }
    if !cy.refused_gain.iter().any(|r| r.starts_with("explain requests")) {
        return Err(format!("nmmusc.g3t: the neutral pattern was not refused: {:?}", cy.refused_gain));
    }
    if !cy.refused_self.iter().any(|r| r.starts_with("value requests")) || g.biases.iter().any(|b| b.context.kind.as_deref() == Some("value")) {
        return Err(format!("nmmusc.g3t: the self-confounded pattern was not refused: {:?}", cy.refused_self));
    }
    // The pack decides the next name request, and only that.
    let mut c = g.control("name", "p");
    let none = |_: &Prior| false;
    let alts = ["nbv", "uniform", "off"];
    if c.decide(Site::Allocation, "nbv", &alts, &|_| 0.0, &none, &none) != "uniform"
        || g.control("explain", "p").decide(Site::Allocation, "nbv", &alts, &|_| 0.0, &none, &none) != "nbv"
    {
        return Err(format!("nmmusc.g3t: the pack did not act as a prior: {:?}", c.log));
    }
    // Drift: the pack's gain falls to zero; the recent rule withdraws it sooner than the
    // whole-history margin rule, with no more false withdrawals.
    let fp = cy.promoted.first().map(|p| p.0.clone()).unwrap_or_default();
    let reference = GainRecord { with_n: 20, with_mean: 0.65, without_n: 180, without_mean: 0.5, diff: 0.15, se: 0.03 };
    let [change, recent, whole] = drift_uses(&reference, 0.1, g.withdraw_z);
    if let Ok(mut r) = REPORT.lock() {
        *r = format!(
            "promoted {promoted:?}
provisional {:?}
refused (gain) {:?}
refused (self) {:?}
candidates {}, replays {}
drift (false withdrawals of 60, median uses after drift): change rule {change:?}, recent rule {recent:?}, whole-history rule {whole:?}
",
            cy.provisional, cy.refused_gain, cy.refused_self, cy.candidates, cy.replays
        );
    }
    if change.0 > whole.0 || change.1 >= whole.1 {
        return Err(format!("nmmusc.g3t: drift: change rule false {} delay {}, whole-history false {} delay {}", change.0, change.1, whole.0, whole.1));
    }
    // Withdrawal in service: calls the pack decides now lose; G3 withdraws it and never loads it.
    for i in 0..40 {
        let id = format!("drift-{i}");
        g.observe(Record {
            id: id.clone(),
            kind: "name".into(),
            area: "p".into(),
            choices: vec![(Site::Allocation, "uniform".into())],
            by_pack: Some(fp.clone()),
            origin: Origin::Service,
            outcome: Some(0.5 + jitter(i, 1)),
            observed: Vec::new(),
        })
        .map_err(e)?;
        let cy = g.cycle(Some(&st), &three, &mut |_: &Record, _: Site, _: &str| Some(0.5)).map_err(e)?;
        if cy.withdrawn.contains(&fp) {
            if let Ok(mut r) = REPORT.lock() {
                r.push_str(&format!("service drift: withdrawn after {} uses with no gain
", i + 1));
            }
            let mut fresh = G3::new("base-a");
            return if fresh.reload(&st).map_err(e)? == 0 && g.live_packs().is_empty() {
                Ok(())
            } else {
                Err("nmmusc.g3t: a withdrawn pack was loaded again".into())
            };
        }
    }
    Err("nmmusc.g3t: the drifted pack was not withdrawn within 40 uses".into())
}

/// The last G3-T run, for the record (`cargo test --release -p nmmusc g3 -- --ignored --nocapture`).
static REPORT: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Drift test of a promoted gain record (gains of spread `sd` holding for 50 uses, then zero):
/// (false withdrawals, median uses after the drift) for the change rule (in service since
/// 2026-11-18), the recent rule (before it) and the whole-history margin rule, over 60 seeds.
fn drift_uses(promo: &GainRecord, sd: f64, z: f64) -> [(u64, usize); 3] {
    use crate::withdraw::{Recent, RunningGain, Z_RECENT, should_withdraw, should_withdraw_change, should_withdraw_recent};
    let mut res = [(0_u64, Vec::new()), (0_u64, Vec::new()), (0_u64, Vec::new())];
    let d = 50;
    for s in 0..60_u64 {
        for (k, slot) in res.iter_mut().enumerate() {
            let mut rng = crate::synth::Rng::new(s * 7919 + 13);
            let (mut run, mut rec) = (RunningGain::from_promotion(promo), Recent::default());
            let mut at = None;
            for i in 0..d + 400 {
                let x = if i < d { promo.diff } else { 0.0 } + sd * rng.normal();
                run.push(x);
                rec.push_against(promo, x);
                let w = match k {
                    0 => should_withdraw_change(promo, &run, &rec, z),
                    1 => should_withdraw_recent(promo, &run, &rec, z, Z_RECENT),
                    _ => should_withdraw(promo, &run, z),
                };
                if w {
                    at = Some(i);
                    break;
                }
            }
            match at {
                Some(i) if i < d => slot.0 += 1,
                Some(i) => slot.1.push(i - d + 1),
                None => slot.1.push(400),
            }
        }
    }
    let med = |v: &mut Vec<usize>| {
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(400)
    };
    res.map(|(f, mut v)| (f, med(&mut v)))
}

/// MO-T (specification 4.9): a planted repeated sequence is found with its boundaries; a pack
/// round-trips by fingerprint; a sensory item is refused by G3's log and by the store.
/// # Errors
/// One fault line.
pub fn check_mot() -> std::result::Result<(), String> {
    let e = |f: Fault| f.to_string();
    crate::miner::check()?;
    let dir = crate::scratch_dir("mot").map_err(e)?;
    let st = MuscleStore::open(&dir).map_err(e)?;
    let p = crate::pack::sample("base-a").map_err(e)?;
    st.write(&p).map_err(e)?;
    if st.load(&p.header.fingerprint, "base-a").map_err(e)? != p {
        return Err("nmmusc.mot: the pack did not round-trip by fingerprint".into());
    }
    let mut g = G3::new("base-a");
    g.observe(Record {
        id: "q".into(),
        kind: "name".into(),
        area: "p".into(),
        choices: vec![(Site::Allocation, "nbv".into())],
        by_pack: None,
        origin: Origin::Service,
        outcome: None,
        observed: Vec::new(),
    })
    .map_err(e)?;
    let sensory = Message { class: crate::DataClass::Sensory, ..Message::behavioural(MessageKind::Report, "g0_t1", "span:src/lib.rs:1-9") };
    match g.log_step("q", &sensory) {
        Err(f) if f.code == "nmmusc-sensory-refused" => {}
        other => return Err(format!("nmmusc.mot: a sensory item entered G3: {other:?}")),
    }
    crate::store::check().map_err(|x| format!("nmmusc.mot: {x}"))
}

/// Section check of G3: G3-T and MO-T.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    check_mot()?;
    check_g3t()
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "report, run on demand"]
    fn g3t_report() {
        assert_eq!(super::check_g3t(), Ok(()));
        println!("{}", super::REPORT.lock().map(|r| r.clone()).unwrap_or_default());
    }
}
