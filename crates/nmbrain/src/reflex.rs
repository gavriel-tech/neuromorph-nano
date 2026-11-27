//! Controlled reflexes G2 installs below the Core Brain (2026-11-13; analysis C, section 8.3,
//! reflexes R1 to R5; analysis B, the tentacle's local decisions, rule 3 the self-initiated short
//! read; analysis E 11.2, `reflex.skip` and `reflex.reread-low` of G1). Spinal reflexes are local
//! rules whose gain is set from above (analysis B, d); here each reflex is a fixed rule with a
//! threshold, a lifetime in questions, a fire-once flag and an exclusion group, installed by G2.
//!
//! Tentacle reflexes (run by each `g0_tx` on its own window, `crate::g0::G0`):
//! - R1 no information: logged when the read finds nothing above the baseline (the rule exists);
//! - R2 definition companion: a reported span that uses an item defined in a held chunk brings
//!   that definition up beside it;
//! - R3 tie: when the top two are within `threshold` times the window's spread, their order is
//!   settled by order averaging over every reading the tentacle holds (mean rank) before reporting;
//! - R4 self-edit: a span of a file the organism itself just edited is marked self-caused and not
//!   reported as new (it goes to `next`);
//! - R5 overflow: what does not fit the report goes on by reference (`next`), logged;
//! - R6 short read: when the top margin is within `threshold` times the spread, the tentacle reads
//!   once more under its next unused statistic without waiting for G1.
//!   G1 reflexes (`crate::g1::G1`):
//! - skip: a tentacle reporting no information is not re-read this round;
//! - reread-low: a tentacle whose confidence is below half the median is re-read (G1's rule
//!   today, now logged as a reflex).

/// Which reflex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Code {
    R1NoInformation,
    R2Definition,
    R3Tie,
    R4SelfEdit,
    R5Overflow,
    R6ShortRead,
    G1Skip,
    G1RereadLow,
}

impl Code {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::R1NoInformation => "R1 no information",
            Self::R2Definition => "R2 definition companion",
            Self::R3Tie => "R3 tie",
            Self::R4SelfEdit => "R4 self-edit",
            Self::R5Overflow => "R5 overflow",
            Self::R6ShortRead => "R6 short read",
            Self::G1Skip => "G1 skip",
            Self::G1RereadLow => "G1 re-read low",
        }
    }
    /// Its trigger predicate and action (master design 3.6 rule 9: the reflex record).
    #[must_use]
    pub const fn rule(self) -> (&'static str, &'static str) {
        match self {
            Self::R1NoInformation => ("nothing above the window's baseline", "report no information"),
            Self::R2Definition => ("a reported span uses an item defined in a held chunk", "bring the definition up beside it"),
            Self::R3Tie => ("top two within threshold spreads", "settle by mean rank over every reading held"),
            Self::R4SelfEdit => ("a span of a file the organism itself just edited", "mark self-caused, queue it by reference"),
            Self::R5Overflow => ("the report is full", "queue the rest by reference, never drop"),
            Self::R6ShortRead => ("top margin within threshold spreads", "read once more under the next unused statistic"),
            Self::G1Skip => ("a tentacle reported no information", "do not re-read it this round"),
            Self::G1RereadLow => ("confidence under half the median", "re-read that tentacle"),
        }
    }
    /// Whether a tentacle runs it (else G1).
    #[must_use]
    pub const fn tentacle(self) -> bool {
        !matches!(self, Self::G1Skip | Self::G1RereadLow)
    }
}

/// Default threshold (in window spreads) for the margin reflexes R3 and R6 when none is given:
/// half a spread (stated fallback until the run-time noise estimate exists).
pub const MARGIN_FALLBACK: f32 = 0.5;

/// One installed reflex.
#[derive(Clone, Debug, PartialEq)]
pub struct Installed {
    pub code: Code,
    pub threshold: f32,
    /// Questions it lasts (0: until removed).
    pub lifetime: u32,
    pub fire_once: bool,
    /// Reflexes in one group exclude each other within one read: the first to fire wins.
    pub group: String,
    /// Questions it has run so far, and times it fired.
    pub questions: u32,
    pub fired: u32,
    /// Who installed it (from above: G2), empty for a short form not yet installed.
    pub set_by: String,
    /// Expiry in ms on the caller's clock (0: none; the question lifetime still applies).
    pub expires_ms: u64,
}

impl Installed {
    #[must_use]
    pub fn new(code: Code) -> Self {
        let threshold = if matches!(code, Code::R3Tie | Code::R6ShortRead) { MARGIN_FALLBACK } else { 0.0 };
        // R3 and R6 both act on a narrow margin: one group, R3 (cheaper) listed first wins.
        let group = if matches!(code, Code::R3Tie | Code::R6ShortRead) { "margin".to_owned() } else { code.name().to_owned() };
        Self { code, threshold, lifetime: 0, fire_once: false, group, questions: 0, fired: 0, set_by: String::new(), expires_ms: 0 }
    }
    /// Short form `R2`, `R3:0.5`, `R6:0.5:3` (threshold, lifetime in questions), `R4:once`,
    /// `G1skip`, `G1low`.
    /// # Errors
    /// An unknown code or a bad number.
    pub fn parse(t: &str) -> Result<Self, String> {
        let p: Vec<&str> = t.split(':').collect();
        let code = match p.first().copied() {
            Some("R1") => Code::R1NoInformation,
            Some("R2") => Code::R2Definition,
            Some("R3") => Code::R3Tie,
            Some("R4") => Code::R4SelfEdit,
            Some("R5") => Code::R5Overflow,
            Some("R6") => Code::R6ShortRead,
            Some("G1skip") => Code::G1Skip,
            Some("G1low") => Code::G1RereadLow,
            _ => return Err(format!("{t}: unknown reflex (R1 to R6, G1skip, G1low)")),
        };
        let mut r = Self::new(code);
        r.fire_once = p.contains(&"once");
        let nums: Vec<&str> = p.iter().skip(1).copied().filter(|x| *x != "once").collect();
        let num = |x: &str| x.parse::<f32>().map_err(|e| format!("{t}: {e}"));
        let whole = |x: &str| x.parse::<u32>().map_err(|e| format!("{t}: {e}"));
        if matches!(code, Code::R3Tie | Code::R6ShortRead) {
            if let Some(x) = nums.first() {
                r.threshold = num(x)?;
            }
            if let Some(x) = nums.get(1) {
                r.lifetime = whole(x)?;
            }
        } else if let Some(x) = nums.first() {
            r.lifetime = whole(x)?;
        }
        Ok(r)
    }
    /// Whether it may fire now.
    #[must_use]
    pub fn live(&self) -> bool {
        !(self.fire_once && self.fired > 0) && (self.lifetime == 0 || self.questions < self.lifetime)
    }
    /// Whether it may fire at `now_ms` (live and not past its expiry).
    #[must_use]
    pub fn live_at(&self, now_ms: u64) -> bool {
        self.live() && (self.expires_ms == 0 || now_ms <= self.expires_ms)
    }
}

/// The parts allowed to install a reflex: from above only (G2, under both of its ids).
pub const INSTALLERS: [&str; 2] = ["g2", "nmbrain.g2"];

/// The reflexes G2 has installed, across questions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Set {
    pub installed: Vec<Installed>,
}

impl Set {
    /// Install (replacing one of the same code).
    pub fn install(&mut self, r: Installed) {
        self.installed.retain(|x| x.code != r.code);
        self.installed.push(r);
    }
    /// Those live for this question.
    #[must_use]
    pub fn live(&self) -> Vec<Installed> {
        self.installed.iter().filter(|r| r.live()).cloned().collect()
    }
    /// After a question: count it, add the firings reported, drop what has run out.
    pub fn after_question(&mut self, fired: &[Code]) {
        for r in &mut self.installed {
            r.questions += 1;
            r.fired += u32::try_from(fired.iter().filter(|c| **c == r.code).count()).unwrap_or(u32::MAX);
        }
        self.installed.retain(Installed::live);
    }
    /// Install from above (master design 3.6 rule 9): only G2 may install; the threshold comes
    /// with the reflex; `ttl_ms` above 0 sets its expiry from `now_ms`. A threshold that is not
    /// finite or below 0 is refused (a reflex never sets its own rule).
    /// # Errors
    /// An installer other than G2, or a bad threshold.
    pub fn install_from(&mut self, by: &str, mut r: Installed, now_ms: u64, ttl_ms: u64) -> Result<(), String> {
        if !INSTALLERS.contains(&by) {
            return Err(format!("{}: refused, installed by {by:?} (only G2 installs reflexes)", r.code.name()));
        }
        if !r.threshold.is_finite() || r.threshold < 0.0 {
            return Err(format!("{}: threshold {}", r.code.name(), r.threshold));
        }
        by.clone_into(&mut r.set_by);
        r.expires_ms = if ttl_ms > 0 { now_ms.saturating_add(ttl_ms) } else { 0 };
        self.install(r);
        Ok(())
    }
    /// Install exactly the reflexes named in `list` (short forms, comma separated); nothing else
    /// is installed. An empty list installs nothing.
    /// # Errors
    /// As `Installed::parse` and `install_from`.
    pub fn install_named(&mut self, list: &str, by: &str, now_ms: u64, ttl_ms: u64) -> Result<usize, String> {
        let mut n = 0;
        for t in list.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            self.install_from(by, Installed::parse(t)?, now_ms, ttl_ms)?;
            n += 1;
        }
        Ok(n)
    }
    /// Install from a controlled-reflex envelope of the motor path (kind `controlled-reflex`, verb
    /// `gain`: a reflex's threshold is the gain set from above): argument `reflex` (word, the short
    /// form), its expiry the envelope's.
    /// # Errors
    /// Another kind, no `reflex` argument, or `install_from`'s refusals.
    pub fn install_envelope(&mut self, m: &nmtype::message::Motor, now_ms: u64) -> Result<Code, String> {
        use nmtype::message::{Arg, MessageKind, Verb};
        if m.kind() != MessageKind::Reflex || m.verb() != Verb::Gain {
            return Err(format!("envelope {}: not a controlled-reflex install", m.id()));
        }
        let t = m
            .args()
            .iter()
            .find_map(|(n, a)| if let (true, Arg::Word(w)) = (n == "reflex", a) { Some(w.clone()) } else { None })
            .ok_or_else(|| format!("envelope {}: no reflex named", m.id()))?;
        let r = Installed::parse(&t)?;
        let code = r.code;
        self.install_from(m.sender(), r, now_ms, m.expires_ms().saturating_sub(now_ms))?;
        Ok(code)
    }
    /// Drop what has expired at `now_ms`.
    pub fn expire(&mut self, now_ms: u64) {
        self.installed.retain(|r| r.live_at(now_ms));
    }
    /// Whether `code` is installed and live.
    #[must_use]
    pub fn has(&self, code: Code) -> bool {
        self.installed.iter().any(|r| r.code == code && r.live())
    }
}

/// Mean rank of each candidate over several readings (higher value is better in every reading);
/// lower mean rank is better. `readings[r][i]` is reading `r` of candidate `i`.
#[must_use]
pub fn mean_ranks(readings: &[Vec<f32>], candidates: &[usize]) -> Vec<f32> {
    let mut sum = vec![0.0_f32; candidates.len()];
    let mut n = 0.0_f32;
    for row in readings {
        let vals: Vec<f32> = candidates.iter().map(|&i| row.get(i).copied().unwrap_or(f32::NEG_INFINITY)).collect();
        if vals.iter().all(|v| !v.is_finite()) {
            continue;
        }
        n += 1.0;
        for (k, v) in vals.iter().enumerate() {
            let better = vals.iter().filter(|o| **o > *v).count();
            #[expect(clippy::cast_precision_loss, reason = "ranks")]
            let r = better as f32;
            if let Some(s) = sum.get_mut(k) {
                *s += r;
            }
        }
    }
    sum.into_iter().map(|s| if n > 0.0 { s / n } else { 0.0 }).collect()
}

/// Section check: short forms parse with their defaults; lifetime and fire-once end a reflex; the
/// set replaces by code and counts questions; mean ranks over readings.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let r3 = Installed::parse("R3")?;
    let r6 = Installed::parse("R6:0.8:3")?;
    let r4 = Installed::parse("R4:once")?;
    if r3.threshold != MARGIN_FALLBACK
        || r3.group != r6.group
        || (r6.threshold - 0.8).abs() > 1e-6
        || r6.lifetime != 3
        || !r4.fire_once
        || Installed::parse("R9").is_ok()
        || Installed::parse("G1skip")?.code.tentacle()
    {
        return Err(format!("nmbrain.reflex: parse {r3:?} {r6:?} {r4:?}"));
    }
    let mut s = Set::default();
    s.install(r6.clone());
    s.install(r4);
    s.install(Installed::parse("R6:0.5:1")?);
    if s.installed.len() != 2 || s.installed.iter().find(|r| r.code == Code::R6ShortRead).map(|r| r.lifetime) != Some(1) {
        return Err("nmbrain.reflex: install did not replace by code".into());
    }
    s.after_question(&[Code::R4SelfEdit]);
    if s.has(Code::R6ShortRead) || s.has(Code::R4SelfEdit) {
        return Err(format!("nmbrain.reflex: lifetime or fire-once {s:?}"));
    }
    let mut s = Set::default();
    if s.install_from("g0_t1", Installed::parse("R1")?, 0, 0).is_ok() || s.install_named("", "g2", 0, 0)? != 0 || !s.installed.is_empty() {
        return Err("nmbrain.reflex: an install not from above, or one not named".into());
    }
    s.install_named("R1,R2,R3:0.4,R4,R5", "g2", 1_000, 500)?;
    if s.installed.len() != 5 || s.installed.iter().any(|r| r.set_by != "g2" || r.expires_ms != 1_500) {
        return Err(format!("nmbrain.reflex: named install {s:?}"));
    }
    s.expire(1_501);
    if !s.installed.is_empty() {
        return Err("nmbrain.reflex: expiry".into());
    }
    let m = mean_ranks(&[vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0], vec![0.0, 5.0, 1.0]], &[0, 1]);
    if m != vec![2.0 / 3.0, 1.0 / 3.0] {
        return Err(format!("nmbrain.reflex: mean ranks {m:?}"));
    }
    Ok(())
}
