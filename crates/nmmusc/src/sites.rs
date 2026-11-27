//! Decision sites of the tentacle path as behavioural envelopes (tentacle master plan 3.10, Phase
//! J1; master design 3.6 rules 11 and 19; lane M, 2026-11-19).
//!
//! Every decision the path takes is logged as one `SiteEvent`: a low-dimensional choice (a label,
//! never content) at a named site, by the unit that made it, about a subject (a unit id or `-`;
//! never a piece of the material). Sites: a tentacle's perspective and aspect, its pose, the
//! team's mode (coverage, relation, frontier), moving on, integrations granted, orders issued and
//! whether they landed, the chain chosen at the Core Brain's intake. Each event renders as an
//! in-house `nmtype::Envelope` (behavioural, to G3) and as a behaviour step G3 mines with the
//! call's record (`Record::observed`).
//!
//! Sensory data is refused twice: the log admits only behavioural items at its boundary (`SITES`,
//! `nmtype::class::Boundary`), and a choice or subject that is not a short label (a path, a
//! sentence, code) is refused, so no material can ride in a label. Muscle memory itself stays its
//! own store and switch: this log is G3's input only.
use crate::symbol::{BehaviouralSymbol, DataClass, MessageKind};
use crate::{Fault, Result};
use nmtype::class::{Boundary, Classed};

/// The boundary of the decision-site log: behavioural only.
pub const SITES: Boundary = Boundary::new("g3-decision-sites", nmtype::class::DataClass::Behavioural);
/// G3's unit id as the envelopes' target.
pub const G3_UNIT: &str = "nmbrain.g3";
/// The longest label a choice or subject may be.
pub const LABEL_MAX: usize = 48;
/// The prefix of a site step in a record's observed behaviour.
pub const STEP: &str = "site:";

/// A decision site of the tentacle path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SiteKind {
    /// A tentacle's perspective and aspect (`<lead-kind>+<aspect>`).
    Perspective,
    /// A tentacle's pose.
    Pose,
    /// The team's mode for a round (`coverage`, `relation`, `frontier`).
    Mode,
    /// A tentacle moving on (`stay`, `move`).
    MovingOn,
    /// Integrations granted to a tentacle (`granted:<n>`).
    Integration,
    /// An order issued (its verb), with whether it landed.
    Order,
    /// The chain chosen at the Core Brain's intake (`<job>:<micromodel>`).
    Chain,
    /// The Core Brain's second look (`gated`, `always`, `skip`; Stage 5 WP-C2).
    Look,
    /// The judge's rerank cap (`tight`, `base`, `wide`; Stage 5 WP-C2).
    RerankCap,
}
impl SiteKind {
    pub const ALL: [Self; 9] = [Self::Perspective, Self::Pose, Self::Mode, Self::MovingOn, Self::Integration, Self::Order, Self::Chain, Self::Look, Self::RerankCap];
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Perspective => "tentacle.perspective",
            Self::Pose => "tentacle.pose",
            Self::Mode => "team.mode",
            Self::MovingOn => "tentacle.moving-on",
            Self::Integration => "tentacle.integration",
            Self::Order => "g2.order",
            Self::Chain => "core-brain.chain",
            Self::Look => "core-brain.look",
            Self::RerankCap => "core-brain.rerank-cap",
        }
    }
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == s)
    }
}

/// One decision at one site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteEvent {
    pub site: SiteKind,
    /// The unit that decided (`g1`, `g2`, `g0_t3`, `core-brain`).
    pub unit: String,
    /// What it is about: a unit id, or `-`.
    pub subject: String,
    /// The choice, a short label.
    pub choice: String,
    /// For an order: whether it landed (its read entered the set, its reply came).
    pub landed: Option<bool>,
}

fn label(what: &str, x: &str) -> Result<()> {
    let ok = !x.is_empty() && x.len() <= LABEL_MAX && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.:+#".contains(c));
    if ok { Ok(()) } else { Err(Fault::new("nmmusc-sensory-refused", format!("decision site {what} {x:?} is not a short label"))) }
}

impl SiteEvent {
    /// A decision, checked: unit, subject and choice must be short labels (no path, no text).
    /// # Errors
    /// `nmmusc-sensory-refused` for anything that is not a label.
    pub fn new(site: SiteKind, unit: &str, subject: &str, choice: &str, landed: Option<bool>) -> Result<Self> {
        label("unit", unit)?;
        label("subject", subject)?;
        label("choice", choice)?;
        Ok(Self { site, unit: unit.to_owned(), subject: subject.to_owned(), choice: choice.to_owned(), landed })
    }
    /// The behaviour step: `site:<site>:<subject>:<choice>[:landed|:missed]`.
    #[must_use]
    pub fn step(&self) -> String {
        let l = match self.landed {
            Some(true) => ":landed",
            Some(false) => ":missed",
            None => "",
        };
        format!("{STEP}{}:{}:{}{l}", self.site.name(), self.subject, self.choice)
    }
    /// Read a step back (the unit is not in the step; `unit` names it).
    /// # Errors
    /// Not a site step, or not labels.
    pub fn parse_step(step: &str, unit: &str) -> Result<Self> {
        let bad = || Fault::new("nmmusc-bad-site", format!("not a decision-site step: {step:?}"));
        let rest = step.strip_prefix(STEP).ok_or_else(bad)?;
        let p: Vec<&str> = rest.split(':').collect();
        let site = p.first().and_then(|s| SiteKind::parse(s)).ok_or_else(bad)?;
        let subject = p.get(1).ok_or_else(bad)?;
        let (choice, landed) = match p.get(2..).unwrap_or_default() {
            [] => return Err(bad()),
            [c @ .., "landed"] if !c.is_empty() => (c.join(":"), Some(true)),
            [c @ .., "missed"] if !c.is_empty() => (c.join(":"), Some(false)),
            c => (c.join(":"), None),
        };
        Self::new(site, unit, subject, &choice, landed)
    }
    /// The in-house envelope: behavioural, from the deciding unit to G3, the step as its one-line
    /// description, no references (a decision carries no material).
    /// # Errors
    /// The envelope's own refusals.
    pub fn envelope(&self) -> Result<nmtype::Envelope> {
        let kind = if self.site == SiteKind::Order { nmtype::MessageKind::Instruction } else { nmtype::MessageKind::Report };
        nmtype::Envelope::new(kind, &self.unit, G3_UNIT, 1.0, &self.step(), Vec::new(), Vec::new()).map_err(|f| Fault::new("nmmusc-bad-site", f.to_string()))
    }
}

impl BehaviouralSymbol for SiteEvent {
    fn data_class(&self) -> DataClass {
        DataClass::Behavioural
    }
    fn kind(&self) -> MessageKind {
        if self.site == SiteKind::Order { MessageKind::Instruction } else { MessageKind::Report }
    }
    fn descriptor(&self) -> String {
        format!("{} {} {}", BehaviouralSymbol::kind(self).as_str(), self.unit, self.step())
    }
}
impl Classed for SiteEvent {
    fn data_class(&self) -> nmtype::class::DataClass {
        nmtype::class::DataClass::Behavioural
    }
}

/// The alternatives G3 may replay at each learnable site (Phase J1 learning: the site records feed
/// G3's event log; a recurring choice that gains by paired replay becomes a pack, reloaded as a
/// prior at its site with G2's veto).
#[must_use]
pub fn alternatives(site: crate::priors::Site) -> &'static [&'static str] {
    use crate::priors::Site;
    match site {
        Site::TeamMode => &["coverage", "relation", "frontier"],
        Site::IntakeChain => &["embedder", "judge"],
        Site::Integrations => &["none", "some", "all"],
        Site::SecondLook => &["gated", "always", "skip"],
        Site::RerankCap => &["tight", "base", "wide"],
        _ => &[],
    }
}

impl SiteEvent {
    /// The learnable site and choice this event is, if any: the team's mode, the intake's model for
    /// a close call (the model of `<job>:<model>`), integrations granted as a bucket (`granted:0`
    /// none, up to 2 some, more all).
    #[must_use]
    pub fn learnable(&self) -> Option<(crate::priors::Site, String)> {
        use crate::priors::Site;
        let c = match self.site {
            SiteKind::Mode => (Site::TeamMode, self.choice.clone()),
            SiteKind::Chain => (Site::IntakeChain, self.choice.rsplit(':').next().unwrap_or("").to_owned()),
            SiteKind::Look => (Site::SecondLook, self.choice.clone()),
            SiteKind::RerankCap => (Site::RerankCap, self.choice.clone()),
            SiteKind::Integration => {
                let n: usize = self.choice.strip_prefix("granted:").and_then(|x| x.parse().ok())?;
                (Site::Integrations, match n {
                    0 => "none",
                    1 | 2 => "some",
                    _ => "all",
                }
                .to_owned())
            }
            _ => return None,
        };
        alternatives(c.0).contains(&c.1.as_str()).then_some(c)
    }
}

/// The decision-site log of one call: envelopes in order; sensory items refused at its boundary.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteLog {
    pub envelopes: Vec<nmtype::Envelope>,
    pub events: Vec<SiteEvent>,
    /// Items refused (must stay 0 on the served path).
    pub refused: usize,
}
impl SiteLog {
    /// Log one decision; `carrier` is the item it travelled as, admitted only when behavioural.
    /// # Errors
    /// A sensory carrier, or the envelope's refusal.
    pub fn log<T: Classed + ?Sized>(&mut self, carrier: &T, e: SiteEvent) -> Result<()> {
        if let Err(f) = SITES.admit(carrier) {
            self.refused += 1;
            return Err(Fault::new("nmmusc-sensory-refused", f.to_string()));
        }
        self.envelopes.push(e.envelope()?);
        self.events.push(e);
        Ok(())
    }
    /// Log a decision made directly (the event is its own carrier).
    /// # Errors
    /// As `log`, and `SiteEvent::new`'s refusals (counted).
    pub fn note(&mut self, site: SiteKind, unit: &str, subject: &str, choice: &str, landed: Option<bool>) -> Result<()> {
        let e = match SiteEvent::new(site, unit, subject, choice, landed) {
            Ok(e) => e,
            Err(f) => {
                self.refused += 1;
                return Err(f);
            }
        };
        let carrier = e.clone();
        self.log(&carrier, e)
    }
    /// The behaviour steps for the call's record (`Record::observed`).
    #[must_use]
    pub fn steps(&self) -> Vec<String> {
        self.events.iter().map(SiteEvent::step).collect()
    }
    /// The learnable choices of the call, one per site (the first taken), for its configuration
    /// record (`Record::choices`), which G3 replays and promotes.
    #[must_use]
    pub fn choices(&self) -> Vec<(crate::priors::Site, String)> {
        let mut out: Vec<(crate::priors::Site, String)> = Vec::new();
        for (s, c) in self.events.iter().filter_map(SiteEvent::learnable) {
            if !out.iter().any(|(x, _)| *x == s) {
                out.push((s, c));
            }
        }
        out
    }
    /// Orders issued and how many landed.
    #[must_use]
    pub fn orders(&self) -> (usize, usize) {
        let o: Vec<&SiteEvent> = self.events.iter().filter(|e| e.site == SiteKind::Order).collect();
        (o.len(), o.iter().filter(|e| e.landed == Some(true)).count())
    }
}

/// G3-T for the decision sites (tentacle master plan section 8): every site logs as a behavioural
/// envelope to G3 and reads back; the steps join a record G3 takes and a codebook codes; a sensory
/// carrier and a label carrying material (a path, a sentence, code) are refused and counted.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    let mut log = SiteLog::default();
    let rows: [(SiteKind, &str, &str, &str, Option<bool>); 8] = [
        (SiteKind::Perspective, "g1", "g0_t1", "meaning+definition", None),
        (SiteKind::Pose, "g1", "g0_t1", "aspect:definition", None),
        (SiteKind::Mode, "g1", "-", "relation", None),
        (SiteKind::MovingOn, "g0_t2", "g0_t2", "move", None),
        (SiteKind::Integration, "g1", "g0_t3", "granted:2", None),
        (SiteKind::Order, "g2", "g0_t1", "focus", Some(true)),
        (SiteKind::Order, "g2", "g0_t4", "reshape-window", Some(false)),
        (SiteKind::Chain, "g2", "-", "prose-close:judge", None),
    ];
    for (s, u, sub, c, l) in rows {
        log.note(s, u, sub, c, l).map_err(|f| format!("nmmusc.sites: {} {}", f.code, f.detail))?;
    }
    if log.envelopes.len() != 8
        || log.envelopes.iter().any(|e| e.data_class() != nmtype::class::DataClass::Behavioural || e.target() != G3_UNIT)
        || log.orders() != (2, 1)
    {
        return Err(format!("nmmusc.sites: envelopes {}", log.envelopes.len()));
    }
    for (e, step) in log.events.iter().zip(log.steps()) {
        if SiteEvent::parse_step(&step, &e.unit).as_ref() != Ok(e) {
            return Err(format!("nmmusc.sites: {step} did not read back"));
        }
    }
    for (s, sub, c) in [(SiteKind::Pose, "g0_t1", "src/a.rs"), (SiteKind::Chain, "-", "fn main() {}"), (SiteKind::Mode, "where is it", "x")] {
        if log.note(s, "g1", sub, c, None).is_ok() {
            return Err(format!("nmmusc.sites: material {c:?} entered as a label"));
        }
    }
    let ok = SiteEvent::new(SiteKind::Mode, "g1", "-", "coverage", None).map_err(|f| f.detail)?;
    if log.log(&nmtype::class::Sensory::new(1_u8), ok).is_ok() || log.refused != 4 || log.events.len() != 8 {
        return Err(format!("nmmusc.sites: a sensory carrier was admitted ({} refused)", log.refused));
    }
    let mut g3 = crate::g3::G3::new("base");
    let rec = crate::g3::Record {
        id: "c1".into(),
        kind: "where".into(),
        area: "p".into(),
        choices: Vec::new(),
        by_pack: None,
        origin: crate::g3::Origin::Service,
        outcome: None,
        observed: log.steps(),
    };
    g3.observe(rec).map_err(|f| f.detail)?;
    let mut cb = crate::Codebook::default();
    for e in &log.events {
        cb.code(e).map_err(|f| f.detail)?;
    }
    if g3.records().first().map(|r| r.observed.len()) != Some(8) || cb.len() != 8 {
        return Err("nmmusc.sites: the steps did not reach G3's record".into());
    }
    Ok(())
}

/// G3 learning from the decision sites: calls whose site records choose `relation` at the team's
/// mode gain against `coverage` by paired replay; G3 promotes a pack, which reloads as a prior at
/// `team.mode` and decides there unless G2 vetoes it.
/// # Errors
/// One fault line.
pub fn check_learning() -> std::result::Result<(), String> {
    use crate::g3::{G3, Origin, Record};
    use crate::priors::Site;
    let mut g3 = G3::new("base");
    for i in 0..48 {
        let mut log = SiteLog::default();
        log.note(SiteKind::Mode, "g1", "-", "coverage", None).map_err(|f| f.detail)?;
        log.note(SiteKind::Chain, "g2", "-", "code-close:embedder", None).map_err(|f| f.detail)?;
        let rec = Record {
            id: format!("c{i:03}"),
            kind: "where".into(),
            area: "p".into(),
            choices: log.choices(),
            by_pack: None,
            origin: Origin::Service,
            outcome: None,
            observed: log.steps(),
        };
        if rec.choices != vec![(Site::TeamMode, "coverage".to_owned()), (Site::IntakeChain, "embedder".to_owned())] {
            return Err(format!("nmmusc.sites.learning: choices {:?}", rec.choices));
        }
        g3.observe(rec).map_err(|f| f.detail)?;
        g3.verdict(&format!("c{i:03}"), 0.5).map_err(|f| f.detail)?;
    }
    let alts = |s: Site| alternatives(s).iter().map(|x| (*x).to_owned()).collect::<Vec<String>>();
    let mut replay = |r: &Record, site: Site, choice: &str| -> Option<f64> {
        let jitter = f64::from(u8::try_from(r.id.bytes().map(u32::from).sum::<u32>() % 7).unwrap_or(0)) * 0.005;
        Some(match (site, choice) {
            (Site::TeamMode, "relation") => 0.8 + jitter,
            (Site::TeamMode, _) => 0.5,
            _ => 0.5,
        })
    };
    let mut cycle = None;
    for _ in 0..3 {
        cycle = Some(g3.cycle(None, &alts, &mut replay).map_err(|f| f.detail)?);
    }
    let ctl = g3.control("where", "p");
    let mut ctl2 = ctl.clone();
    let picked = ctl2.decide(Site::TeamMode, "coverage", &["coverage", "relation", "frontier"], &|_| 0.0, &|_| false, &|_| false);
    let mut ctl3 = ctl;
    let vetoed = ctl3.decide(Site::TeamMode, "coverage", &["coverage", "relation", "frontier"], &|_| 0.0, &|_| true, &|_| false);
    if picked != "relation" || vetoed != "coverage" {
        return Err(format!("nmmusc.sites.learning: picked {picked}, vetoed {vetoed}, cycle {cycle:?}, biases {:?}", g3.biases()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn g3_t_sites() {
        assert_eq!(super::check(), Ok(()));
    }
    #[test]
    fn learning() {
        assert_eq!(super::check_learning(), Ok(()));
    }
}

#[cfg(test)]
mod stage5_sites {
    use super::{SiteEvent, SiteKind, alternatives};
    use crate::priors::{Site, site_of};

    /// Stage 5 WP-C2: the second look and the rerank cap are learnable sites; their rule choices
    /// (`gated`, `base`) are among the alternatives, an unknown choice is not learnable, and the
    /// descriptor action reads back to its site.
    #[test]
    fn look_and_rerank_cap_are_learnable() {
        let look = SiteEvent::new(SiteKind::Look, "core-brain", "-", "always", None).expect("look event");
        assert_eq!(look.learnable(), Some((Site::SecondLook, "always".to_owned())));
        let cap = SiteEvent::new(SiteKind::RerankCap, "core-brain", "-", "tight", None).expect("cap event");
        assert_eq!(cap.learnable(), Some((Site::RerankCap, "tight".to_owned())));
        let odd = SiteEvent::new(SiteKind::RerankCap, "core-brain", "-", "huge", None).expect("odd event");
        assert_eq!(odd.learnable(), None);
        assert!(alternatives(Site::SecondLook).contains(&"gated") && alternatives(Site::RerankCap).contains(&"base"));
        // The L1 tuner's numeric look arm is untouched: no named alternatives at it.
        assert!(alternatives(Site::LookArm).is_empty());
        assert_eq!(site_of(&Site::SecondLook.action("skip")), Some((Site::SecondLook, "skip".to_owned())));
        assert_eq!(site_of(&Site::RerankCap.action("wide")), Some((Site::RerankCap, "wide".to_owned())));
        assert_eq!(SiteKind::parse("core-brain.rerank-cap"), Some(SiteKind::RerankCap));
    }
}
