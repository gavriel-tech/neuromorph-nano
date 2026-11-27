//! How a reloaded pack acts (2026-11-13; research file Solution 3, MO8 "priors never below no
//! pack"; analysis E gap 12.3, option 3: priors at decision sites with a G2 veto and termination).
//!
//! A pack changes decisions only, at named decision sites, never weights, windows or content.
//! Each step of a pack whose descriptor names a site and an alternative becomes a prior there.
//! Arbitration (Daw, Niv and Dayan 2005, in arithmetic form, as analysis E gives it: control goes to
//! the less uncertain estimate): at a site the deliberative rule offers its choice with a margin in
//! standard errors over the pack's alternative; the pack decides only when its own evidence (its
//! gain over its standard error, from the record that promoted it, lowered by any opposing record)
//! exceeds that margin. With no deliberative evidence (margin 0) a pack with positive evidence
//! decides; with strong deliberative evidence the rule keeps control. G2 may veto any step; a veto,
//! or a step whose site reports a fresh fact against it, ends the pack's control for the question
//! (the termination marker), and the rest runs on the deliberative rules.
//!
//! Sites named on the whole path (`Site`): the Core Brain's second-look arm, G2's up-count bucket,
//! G1's stop rule; on every path, the Map Coordinator's allocation of the second round. Descriptors
//! as the behaviour log records them: `look:<move|keep>:arm:<t>`, `send-up:<few|some|many>`,
//! `stop:<reason>:rounds:<n>`, `allocation:<mode>`, `chain:<mode>`.
use crate::gain::GainRecord;

/// A decision site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Site {
    /// The Core Brain's second-look threshold (an arm of the L1 tuner).
    LookArm,
    /// G2's up-count bucket (few up to 4, some up to 12, many above).
    UpCount,
    /// G1's stop rule.
    G1Stop,
    /// The Map Coordinator's allocation of the second round (`nbv`, `uniform` or `off`).
    Allocation,
    /// G2's chain choice: which MicroModel chain the stack hands the request to (`fast`, `full`
    /// or `flow`; 2026-11-18).
    Chain,
    /// The team's mode for round 2 on (`coverage`, `relation`, `frontier`; Phase J1, 2026-11-19).
    TeamMode,
    /// The MicroModel the Core Brain's intake gives a close call (`embedder`, `judge`; Phase J1).
    IntakeChain,
    /// Integrations granted to the tentacles (`none`, `some`, `all`; Phase J1).
    Integrations,
    /// The judge's rerank cap for the call (`tight`, `base`, `wide`; Stage 5 WP-C2): `base` is the
    /// cap the engine computes, the deterministic rule.
    RerankCap,
    /// Whether the Core Brain's second look runs (`gated`, `always`, `skip`; Stage 5 WP-C2): `gated`
    /// (run unless the top-two gap reaches the gate) is the deterministic rule. Distinct from
    /// `LookArm`, the numeric switch threshold the L1 tuner moves.
    SecondLook,
}

impl Site {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LookArm => "core-brain.look-arm",
            Self::UpCount => "g2.up-count",
            Self::G1Stop => "g1.stop",
            Self::Allocation => "g1.allocation",
            Self::Chain => "g2.chain",
            Self::TeamMode => "team.mode",
            Self::IntakeChain => "core-brain.intake",
            Self::Integrations => "tentacle.integrations",
            Self::RerankCap => "core-brain.rerank-cap",
            Self::SecondLook => "core-brain.second-look",
        }
    }
    /// The descriptor action that names `choice` at this site, as `site_of` reads it back.
    #[must_use]
    pub fn action(self, choice: &str) -> String {
        match self {
            Self::LookArm => format!("look:move:arm:{choice}"),
            Self::UpCount => format!("send-up:{choice}"),
            Self::G1Stop => format!("stop:{choice}"),
            Self::Allocation => format!("allocation:{choice}"),
            Self::Chain => format!("chain:{choice}"),
            Self::TeamMode => format!("mode:{choice}"),
            Self::IntakeChain => format!("intake:{choice}"),
            Self::Integrations => format!("integrate:{choice}"),
            Self::RerankCap => format!("rerank-cap:{choice}"),
            Self::SecondLook => format!("second-look:{choice}"),
        }
    }
}

/// One prior from one pack step.
#[derive(Clone, Debug, PartialEq)]
pub struct Prior {
    pub site: Site,
    pub alternative: String,
    /// Evidence in standard errors (promotion gain over its error, lowered by opposing records).
    pub evidence: f64,
    pub pack: String,
    pub step: usize,
}

/// The site and alternative a step's descriptor names, if any.
#[must_use]
pub fn site_of(descriptor: &str) -> Option<(Site, String)> {
    let d = descriptor.rsplit(' ').next().unwrap_or(descriptor);
    let p: Vec<&str> = d.split(':').collect();
    match p.as_slice() {
        ["look", _, "arm", t, ..] => Some((Site::LookArm, (*t).to_owned())),
        ["send-up", b, ..] => Some((Site::UpCount, (*b).to_owned())),
        ["stop", reason, ..] => Some((Site::G1Stop, (*reason).to_owned())),
        ["allocation", mode, ..] => Some((Site::Allocation, (*mode).to_owned())),
        ["chain", c, ..] => Some((Site::Chain, (*c).to_owned())),
        ["mode", c, ..] => Some((Site::TeamMode, (*c).to_owned())),
        ["intake", c, ..] => Some((Site::IntakeChain, (*c).to_owned())),
        ["integrate", c, ..] => Some((Site::Integrations, (*c).to_owned())),
        ["rerank-cap", c, ..] => Some((Site::RerankCap, (*c).to_owned())),
        ["second-look", c, ..] => Some((Site::SecondLook, (*c).to_owned())),
        _ => None,
    }
}

/// A pack's evidence: its promotion gain over the error, less the opposing margin when a running
/// record has gone against it (`opposed` in standard errors, 0 when none).
#[must_use]
pub fn evidence(promotion: &GainRecord, opposed: f64) -> f64 {
    if promotion.se > 0.0 { promotion.diff / promotion.se - opposed.max(0.0) } else { 0.0 }
}

/// The priors of a pack's steps (only steps naming a site).
#[must_use]
pub fn priors(pack: &str, steps: &[String], evidence: f64) -> Vec<Prior> {
    steps
        .iter()
        .enumerate()
        .filter_map(|(step, d)| site_of(d).map(|(site, alternative)| Prior { site, alternative, evidence, pack: pack.to_owned(), step }))
        .collect()
}

/// What happened at one site.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub site: Site,
    pub chosen: String,
    /// The pack decided (its fingerprint and step), or the deliberative rule did.
    pub by_pack: Option<(String, usize)>,
    pub why: String,
}

/// The pack's control over one question: arbitrates each site, stops at a veto or a fresh fact.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Control {
    pub priors: Vec<Prior>,
    /// Set once the pack's control ended (a veto or a contradicting fact); later sites run on the
    /// deliberative rules.
    pub ended: Option<String>,
    pub log: Vec<Decision>,
}

impl Control {
    #[must_use]
    pub fn new(priors: Vec<Prior>) -> Self {
        Self { priors, ended: None, log: Vec::new() }
    }
    /// Arbitrate at `site`: the deliberative choice `rule` with its margin in standard errors over
    /// the pack's alternative (`margin_over` gives it for an alternative; 0 when the rule has no
    /// evidence), the alternatives the site allows, G2's veto, and whether a fresh fact at the
    /// site contradicts the pack (`contradicts`).
    #[expect(clippy::too_many_arguments, reason = "a site, the rule's choice and margin, the allowed set, the veto and the fresh fact")]
    pub fn decide(
        &mut self,
        site: Site,
        rule: &str,
        allowed: &[&str],
        margin_over: &dyn Fn(&str) -> f64,
        veto: &dyn Fn(&Prior) -> bool,
        contradicts: &dyn Fn(&Prior) -> bool,
    ) -> String {
        let keep = |me: &mut Self, why: String| {
            me.log.push(Decision { site, chosen: rule.to_owned(), by_pack: None, why });
            rule.to_owned()
        };
        if let Some(e) = &self.ended {
            let why = format!("pack control ended ({e})");
            return keep(self, why);
        }
        let Some(p) = self.priors.iter().find(|p| p.site == site).cloned() else { return keep(self, "no pack step at this site".into()) };
        if !allowed.contains(&p.alternative.as_str()) {
            return keep(self, format!("pack names {} which the site does not offer", p.alternative));
        }
        if veto(&p) {
            self.ended = Some(format!("G2 vetoed step {} of {}", p.step, p.pack));
            let why = self.ended.clone().unwrap_or_default();
            return keep(self, why);
        }
        if contradicts(&p) {
            self.ended = Some(format!("a fresh fact at {} went against step {}", site.name(), p.step));
            let why = self.ended.clone().unwrap_or_default();
            return keep(self, why);
        }
        let m = if p.alternative == rule { 0.0 } else { margin_over(&p.alternative) };
        if p.alternative != rule && p.evidence > m {
            self.log.push(Decision {
                site,
                chosen: p.alternative.clone(),
                by_pack: Some((p.pack.clone(), p.step)),
                why: format!("pack evidence {:.2} over the rule's margin {m:.2}", p.evidence),
            });
            return p.alternative;
        }
        keep(self, format!("the rule keeps control (margin {m:.2} against pack evidence {:.2})", p.evidence))
    }
}

/// Section check: descriptors map to sites; arbitration gives the pack control only when its
/// evidence exceeds the rule's margin; a veto or a contradicting fact ends control for later
/// sites; and on synthetic replays of a site with a known best alternative, priors are never
/// below "no pack" on any fold and at least equal to a hard override, while a bad pack (weak
/// evidence) changes nothing where the rule is sure.
/// # Errors
/// One fault line.
pub fn check() -> std::result::Result<(), String> {
    if site_of("nmbrain.core-brain#0 look:move:arm:0.1") != Some((Site::LookArm, "0.1".into()))
        || site_of("send-up:few") != Some((Site::UpCount, "few".into()))
        || site_of("instruction g1 allocation:uniform") != Some((Site::Allocation, "uniform".into()))
        || site_of(&Site::Allocation.action("off")) != Some((Site::Allocation, "off".into()))
        || site_of(&Site::Chain.action("flow")) != Some((Site::Chain, "flow".into()))
        || site_of("other").is_some()
    {
        return Err("nmmusc.priors: descriptors".into());
    }
    let promo = GainRecord { with_n: 10, with_mean: 0.8, without_n: 10, without_mean: 0.5, diff: 0.3, se: 0.1 };
    if (evidence(&promo, 0.0) - 3.0).abs() > 1e-9 || (evidence(&promo, 1.0) - 2.0).abs() > 1e-9 {
        return Err("nmmusc.priors: evidence".into());
    }
    let steps = vec!["look:move:arm:0.1".to_owned(), "send-up:few".to_owned()];
    let no = |_: &Prior| false;
    let mut c = Control::new(priors("fp1", &steps, 3.0));
    let arms = ["0", "0.1", "0.205"];
    if c.decide(Site::LookArm, "0.205", &arms, &|_| 1.0, &no, &no) != "0.1"
        || c.decide(Site::UpCount, "some", &["few", "some", "many"], &|_| 5.0, &no, &no) != "some"
    {
        return Err(format!("nmmusc.priors: arbitration {:?}", c.log));
    }
    let mut v = Control::new(priors("fp1", &steps, 3.0));
    let yes = |_: &Prior| true;
    if v.decide(Site::LookArm, "0.205", &arms, &|_| 0.0, &yes, &no) != "0.205"
        || v.decide(Site::UpCount, "some", &["few", "some", "many"], &|_| 0.0, &no, &no) != "some"
        || v.ended.is_none()
    {
        return Err("nmmusc.priors: a veto did not end control".into());
    }
    // Replays: per question, a rule choice with its margin (z), the right alternative known.
    // Good pack names the right alternative on the questions where the rule is unsure.
    let mut rng = crate::synth::Rng::new(5);
    let mut folds_ok = true;
    let (mut none_t, mut prior_t, mut over_t) = (0, 0, 0);
    for _fold in 0..5 {
        let (mut none, mut prior, mut over) = (0, 0, 0);
        for _q in 0..40 {
            let sure = rng.unit() < 0.5;
            let rule_right = if sure { rng.unit() < 0.9 } else { rng.unit() < 0.4 };
            let pack_right = rng.unit() < 0.7;
            let (rule, right) = if rule_right { ("a", "a") } else { ("a", "b") };
            let pack_alt = if pack_right {
                right
            } else if right == "a" {
                "b"
            } else {
                "a"
            };
            let margin = if sure { 4.0 } else { 0.5 };
            let mut ctl = Control::new(vec![Prior {
                site: Site::LookArm,
                alternative: pack_alt.into(),
                evidence: 2.0,
                pack: "good".into(),
                step: 0,
            }]);
            let got = ctl.decide(Site::LookArm, rule, &["a", "b"], &|_| margin, &no, &no);
            none += usize::from(rule == right);
            prior += usize::from(got == right);
            over += usize::from(pack_alt == right);
        }
        folds_ok &= prior >= none;
        none_t += none;
        prior_t += prior;
        over_t += over;
    }
    if !folds_ok || prior_t < over_t || prior_t <= none_t {
        return Err(format!(
            "nmmusc.priors: replay none {none_t} priors {prior_t} override {over_t} (every fold at least none: {folds_ok})"
        ));
    }
    // A real pack learned on whole-path events: its steps read back become priors and decide.
    let mut cb = crate::symbol::Codebook::default();
    let msgs = [
        crate::symbol::Message::behavioural(crate::symbol::MessageKind::Report, "g3", "stop:nothing-left-to-ask:rounds:2"),
        crate::symbol::Message::behavioural(crate::symbol::MessageKind::Report, "g3", "send-up:many"),
        crate::symbol::Message::behavioural(crate::symbol::MessageKind::Report, "g3", "look:move:arm:0.1"),
    ];
    let mut chunk = Vec::new();
    for m in &msgs {
        chunk.push(cb.code(m).map_err(|e| e.to_string())?);
    }
    let pack = crate::pack::Pack::build(&chunk, &cb, "chain whole: 3 behavioural messages", "base-a", promo).map_err(|e| e.to_string())?;
    let steps: Vec<String> = pack.steps().into_iter().map(|(_, d)| d).collect();
    let pr = priors(&pack.header.fingerprint, &steps, evidence(&pack.header.promotion, 0.0));
    let mut ctl = Control::new(pr);
    let up = ctl.decide(Site::UpCount, "some", &["some", "many"], &|_| 0.0, &no, &no);
    let arm = ctl.decide(Site::LookArm, "0.205", &arms, &|_| 1.0, &no, &no);
    if ctl.priors.len() != 3
        || up != "many"
        || arm != "0.1"
        || ctl.log.iter().any(|d| d.by_pack.as_ref().is_none_or(|(f, _)| *f != pack.header.fingerprint))
    {
        return Err(format!("nmmusc.priors: pack round trip {steps:?} {:?}", ctl.log));
    }
    let mut weak = Control::new(vec![Prior { site: Site::LookArm, alternative: "b".into(), evidence: 0.3, pack: "weak".into(), step: 0 }]);
    if weak.decide(Site::LookArm, "a", &["a", "b"], &|_| 4.0, &no, &no) != "a" {
        return Err("nmmusc.priors: a weak pack overrode a sure rule".into());
    }
    Ok(())
}
