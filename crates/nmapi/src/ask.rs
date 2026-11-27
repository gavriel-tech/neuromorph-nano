//! The Core Brain asking for more context (2026-11-12; research file Part 10 (c), CRAG's rule
//! integrated, not its model; analysis D, gap 10.4: requests Q1 to Q7, cheapest first, and the
//! stopping rule).
//!
//! A judge gives the Core Brain's confidence in its first place; two thresholds derived from
//! development outcomes sort it as correct (no request), ambiguous (request, keep what is held) or
//! incorrect (request deeper, keep what is held: context grows, never shrinks). The requests
//! built here are the millisecond ones, in the design's order: Q3, the next chunks G1 holds in its
//! queue (no read); Q1, the chunks around the first place's span in the same file. A cycle stops
//! when the confidence clears the upper threshold, when it adds no piece, or when the time left is
//! under one measured cycle.
//!
//! Thresholds are derived only from a judge that separates right from wrong first places on
//! development outcomes (area under the curve at least `MIN_AUC`); without one, `derive` gives
//! `None` and the rule is "ask while the budget allows" (FrugalGPT's stop rule needs a scorer).
use nmsubs::Substrate;
use std::path::Path;

/// When the Core Brain asks (`serve --ask judge|always|off`; default judge).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Ask only when a judge with derived thresholds exists and says not correct.
    #[default]
    Judge,
    /// Ask every question, while the budget allows (no judge needed).
    Always,
    Off,
}

impl Mode {
    /// From `--ask <mode>` among `args`.
    /// # Errors
    /// An unknown mode.
    pub fn from_args(args: &[String]) -> Result<Self, String> {
        // 2026-11-25: without `--ask`, the `routing.ask` control (interface, MCP).
        let control = crate::whole::control_str(&nmtype::nano_root(), "routing.ask");
        let given = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--ask")).and_then(|w| w.get(1)).map(String::as_str);
        match given.or(control.as_deref()) {
            None | Some("judge") => Ok(Self::Judge),
            Some("always") => Ok(Self::Always),
            Some("off") => Ok(Self::Off),
            Some(x) => Err(format!("--ask {x}: judge, always or off")),
        }
    }
}

/// The judge: the newest saved L4 calibrator over the reranker's scores, with thresholds derived
/// from the session's decisions that have a verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Judge {
    pub id: String,
    pub calibrator: nmlearn::calibrate::Calibrator,
    pub crag: Crag,
}

impl Judge {
    /// The judge's confidence that the reranker's first place is right.
    #[must_use]
    pub fn confidence(&self, rerank: &[f32]) -> f64 {
        self.calibrator.predict(&nmlearn::calibrate::features(rerank))
    }
}

/// Build the judge from the learned store and the ledger, if a calibrator is saved and its
/// thresholds separate on the ledger's verdicts. (Whether the right chunk was among the pieces is
/// not known in service; it counts as present, so the lower threshold falls to the lowest seen.)
#[must_use]
pub fn judge(home: &Path, store: &Path, ledger: &Path) -> Option<Judge> {
    // 2026-11-26 (plugs): the newest calibrator that is plugged in, in a compartment that is on.
    let compartment = crate::plugs::compartment_of_store(home, store);
    let plugs = crate::plugs::read(home);
    let id = nmlearn::learned::list(store).into_iter().rfind(|x| x.starts_with("l4-calibrator-") && plugs.acts(&compartment, x))?;
    let policy = nmlearn::cartridge::Policy::load(home);
    let loaded = nmlearn::cartridge::Loaded::population(home, "qwen3-reranker-0.6b", vec![crate::learn::CORE_BRAIN.into()]).ok()?;
    let piece = nmlearn::learned::load(store, &id, &policy, &loaded).ok()?;
    let calibrator = nmlearn::learned::calibrator_of(&piece.shape, &piece.numbers);
    let cases: Vec<(f64, bool, bool)> = nmlearn::ledger::load(ledger)
        .ok()?
        .iter()
        .filter_map(|d| Some((calibrator.predict(&nmlearn::calibrate::features(&d.rerank)), d.correct?, true)))
        .collect();
    let crag = derive(&cases)?;
    Some(Judge { id, calibrator, crag })
}

/// Whether to run another request cycle: the mode and class allow it, the last cycle added a
/// piece above the current best (or none ran yet), and the time left covers one measured cycle.
#[must_use]
pub fn another(mode: Mode, class: Option<Class>, cycles: usize, improved: bool, left_s: f64, cycle_s: f64) -> bool {
    let wanted = match mode {
        Mode::Off => false,
        Mode::Always => true,
        Mode::Judge => matches!(class, Some(Class::Ambiguous | Class::Incorrect)),
    };
    wanted && (cycles == 0 || improved) && left_s > cycle_s
}

/// The least separation a judge needs before its thresholds are used.
pub const MIN_AUC: f64 = 0.65;
/// Precision the upper threshold must give (correct: right first at least this often above it).
pub const UPPER_PRECISION: f64 = 0.9;
/// Fewest cases on a side of a threshold.
pub const MIN_SIDE: usize = 5;
/// Q3 chunks per cycle (ambiguous); twice as many when incorrect.
pub const Q3_K: usize = 4;

/// How the judge sorts the Core Brain's first place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Correct,
    Ambiguous,
    Incorrect,
}

/// The two thresholds over a judge's confidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crag {
    pub lower: f64,
    pub upper: f64,
    pub auc: f64,
}

impl Crag {
    #[must_use]
    pub fn class(&self, conf: f64) -> Class {
        if conf >= self.upper {
            Class::Correct
        } else if conf < self.lower {
            Class::Incorrect
        } else {
            Class::Ambiguous
        }
    }
}

/// Area under the curve of `conf` for `right` (ties count half); `None` with one class missing.
#[must_use]
pub fn auc(cases: &[(f64, bool)]) -> Option<f64> {
    let pos: Vec<f64> = cases.iter().filter(|c| c.1).map(|c| c.0).collect();
    let neg: Vec<f64> = cases.iter().filter(|c| !c.1).map(|c| c.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return None;
    }
    let mut s = 0.0;
    for p in &pos {
        for n in &neg {
            s += if p > n {
                1.0
            } else if (p - n).abs() < f64::EPSILON {
                0.5
            } else {
                0.0
            };
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    Some(s / (pos.len() * neg.len()) as f64)
}

/// Thresholds from development outcomes: (confidence, right first, right chunk among the pieces).
/// Upper: the lowest confidence above which right-first holds at least `UPPER_PRECISION` of the
/// time over at least `MIN_SIDE` cases. Lower: the highest confidence below which the right chunk
/// was missing from the pieces more often than present, over at least `MIN_SIDE` cases (else the
/// lowest confidence seen). `None` when the judge does not separate (`MIN_AUC`) or no upper exists.
#[must_use]
pub fn derive(cases: &[(f64, bool, bool)]) -> Option<Crag> {
    let a = auc(&cases.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>())?;
    if a < MIN_AUC {
        return None;
    }
    let mut cuts: Vec<f64> = cases.iter().map(|c| c.0).collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let share = |xs: &[&(f64, bool, bool)], f: &dyn Fn(&(f64, bool, bool)) -> bool| {
        xs.iter().filter(|c| f(c)).count() as f64 / xs.len().max(1) as f64
    };
    let upper = cuts.iter().copied().find(|t| {
        let above: Vec<&(f64, bool, bool)> = cases.iter().filter(|c| c.0 >= *t).collect();
        above.len() >= MIN_SIDE && share(&above, &|c| c.1) >= UPPER_PRECISION
    })?;
    let lower = cuts
        .iter()
        .copied()
        .filter(|t| *t <= upper)
        .rev()
        .find(|t| {
            let below: Vec<&(f64, bool, bool)> = cases.iter().filter(|c| c.0 < *t).collect();
            below.len() >= MIN_SIDE && share(&below, &|c| !c.2) > 0.5
        })
        .unwrap_or_else(|| cuts.first().copied().unwrap_or(0.0));
    Some(Crag { lower, upper, auc: a })
}

/// One request cycle's new chunks, cheapest first: Q3 the next `k` of G1's queue not held, then Q1
/// the chunks just before and after the first place's span in the same file. Returns (request, chunk).
#[must_use]
pub fn request(sub: &Substrate, held: &[usize], queued: &[usize], k: usize) -> Vec<(&'static str, usize)> {
    let mut out: Vec<(&'static str, usize)> = Vec::new();
    let new = |c: usize, out: &[(&str, usize)]| !held.contains(&c) && !out.iter().any(|(_, x)| *x == c);
    for &c in queued {
        if out.len() >= k {
            break;
        }
        if new(c, &out) {
            out.push(("Q3", c));
        }
    }
    if let Some(first) = held.first().and_then(|c| sub.chunks.get(*c)) {
        let file = first.file;
        let before =
            sub.chunks.iter().enumerate().filter(|(_, c)| c.file == file && c.end < first.start).max_by_key(|(_, c)| c.end).map(|(i, _)| i);
        let after = sub
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.file == file && c.start > first.end)
            .min_by_key(|(_, c)| c.start)
            .map(|(i, _)| i);
        for c in [before, after].into_iter().flatten() {
            if new(c, &out) {
                out.push(("Q1", c));
            }
        }
    }
    out
}

/// Section check: AUC of a perfect and a useless judge; thresholds derived on a separating judge
/// and refused on a useless one; the classes; a request takes the queue first, skips held chunks,
/// then the span's neighbours in the same file only.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    if auc(&[(0.9, true), (0.1, false)]) != Some(1.0) || auc(&[(0.5, true), (0.5, false)]) != Some(0.5) {
        return Err("nmapi.ask: auc".into());
    }
    let mut cases: Vec<(f64, bool, bool)> = (0..20).map(|i| (f64::from(i) / 20.0, i >= 10, i >= 4)).collect();
    let c = derive(&cases).ok_or("nmapi.ask: no thresholds on a separating judge")?;
    if c.upper > 0.5 + 1e-9 || c.lower > c.upper || c.class(0.95) != Class::Correct || c.class(0.0) != Class::Incorrect {
        return Err(format!("nmapi.ask: thresholds {c:?}"));
    }
    for (i, x) in cases.iter_mut().enumerate() {
        x.1 = i % 2 == 0;
    }
    if derive(&cases).is_some() {
        return Err("nmapi.ask: thresholds from a judge that does not separate".into());
    }
    let root = nmtype::check_dir("nmapi-ask").map_err(|e| e.to_string())?;
    let body: String = (1..=60).map(|i| format!("fn f{i}() {{}}\n\n")).collect();
    std::fs::write(root.join("a.rs"), &body).map_err(|e| e.to_string())?;
    std::fs::write(root.join("b.rs"), "fn other() {}\n").map_err(|e| e.to_string())?;
    let sub = Substrate::scan(&root, &[], std::sync::Arc::new(|_: &str| Vec::new()));
    let a: Vec<usize> = (0..sub.chunks.len())
        .filter(|i| sub.chunks.get(*i).and_then(|c| sub.files.get(c.file)).is_some_and(|f| f.path.ends_with("a.rs")))
        .collect();
    if a.len() < 3 {
        return Err(format!("nmapi.ask: test file gave {} chunks", a.len()));
    }
    let mid = a[a.len() / 2];
    let other: Vec<usize> = (0..sub.chunks.len()).filter(|i| !a.contains(i)).collect();
    let queued: Vec<usize> = other.iter().copied().chain([mid]).collect();
    if another(Mode::Judge, None, 0, true, 30.0, 1.0)
        || !another(Mode::Judge, Some(Class::Ambiguous), 0, false, 30.0, 1.0)
        || another(Mode::Always, None, 1, false, 30.0, 1.0)
        || another(Mode::Always, None, 0, true, 0.5, 1.0)
        || Mode::from_args(&["--ask".into(), "always".into()]) != Ok(Mode::Always)
    {
        return Err("nmapi.ask: the cycle rule".into());
    }
    let r = request(&sub, &[mid], &queued, 4);
    let q3: Vec<usize> = r.iter().filter(|x| x.0 == "Q3").map(|x| x.1).collect();
    let q1: Vec<usize> = r.iter().filter(|x| x.0 == "Q1").map(|x| x.1).collect();
    if q3.contains(&mid) || q3.len() != other.len().min(4) || q1.len() != 2 || q1.iter().any(|c| !a.contains(c)) {
        return Err(format!("nmapi.ask: request {r:?}"));
    }
    Ok(())
}
