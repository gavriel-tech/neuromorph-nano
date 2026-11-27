//! The tentacle core's exact fallback (2026-11-10, step P4 first form; specification 2.2 and
//! 2.4, design by Luis Jake Gabriel III). A tentacle's core processes what its drone tip read
//! and reports through the tentacle's own `g0_tx` to G1. When a neural core is bound
//! (`nmapi::tentacle_core`, joined by `nmapi::tentacle_unit`) it reads the window through its
//! attention-head drones and its own components; when no model is loaded this exact core runs, so
//! the organism still works with no weights (law 8):
//! - processing: the unit pipeline (reflexes, reader-agreement confidence, anomaly, distillation
//!   into cited packets, a subordinate request when warranted), exactly as before;
//! - report: one score per span of the window, the sum over the readers that found something in
//!   the span of each reader's best finding score (agreement between readers raises it), and the
//!   number of readers per span as a second statistic. A window where no reader found anything
//!   reports no information. Findings of whole-code-base readers that fall outside the window are
//!   counted, not scored (they still reach the Core Brain as cited packets).
//!
//! This crate holds no engine and no G0 type: the adapter turns the report into the tentacle's
//! `g0_tx` state (`nmbrain::g0::G0`).
use crate::tip::Finding;
use crate::window::Window;
use std::collections::BTreeMap;

/// Which core a tentacle ran for a question.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreKind {
    /// The neural tentacle core (attention-head drones and the tentacle's own components).
    Neural,
    /// The stated exact fallback (no model loaded, or the neural core faulted).
    Exact,
}
impl CoreKind {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Neural => "neural",
            Self::Exact => "exact",
        }
    }
}

/// Statistic names of the exact report (the first is the one it ranks by).
pub const STATISTICS: [&str; 2] = ["exact", "agreement"];

/// The exact core's report over one window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExactReport {
    /// Per span of the window: the summed best finding score of each reader that found
    /// something in it (0 when none did).
    pub scores: Vec<f32>,
    /// Per span: how many readers found something in it.
    pub agree: Vec<f32>,
    /// Findings inside the window, and findings outside it (whole-code-base readers).
    pub inside: usize,
    pub outside: usize,
    /// True when no span holds a finding.
    pub no_information: bool,
}

/// Score every span of `window` from the tip's findings (module notes).
#[must_use]
pub fn exact_report(window: &Window, findings: &[Finding]) -> ExactReport {
    let n = window.spans.len();
    let mut best: BTreeMap<(usize, &str), f32> = BTreeMap::new();
    let (mut inside, mut outside) = (0, 0);
    for f in findings {
        match window.span_of(&f.source) {
            Some(i) => {
                inside += 1;
                let s = if f.score.is_finite() { f.score.max(0.0) } else { 0.0 };
                let e = best.entry((i, f.reader.as_str())).or_insert(0.0);
                *e = e.max(s);
            }
            None => outside += 1,
        }
    }
    let mut scores = vec![0.0_f32; n];
    let mut agree = vec![0.0_f32; n];
    for ((i, _), s) in best {
        if let (Some(x), Some(a)) = (scores.get_mut(i), agree.get_mut(i)) {
            *x += s;
            *a += 1.0;
        }
    }
    let no_information = !scores.iter().any(|s| *s > 0.0);
    ExactReport { scores, agree, inside, outside, no_information }
}

/// Self-check of the `core` sub-component: two readers agreeing on one span outrank one reader
/// on another, a finding outside the window is counted and not scored, and a window with no
/// finding reports no information.
pub(crate) fn check() -> Result<(), String> {
    use crate::window::Span;
    let span = |f: &str, a: usize, b: usize| Span { file: f.into(), start: a, end: b, key: format!("{f}:{a}"), tokens: 5 };
    let w = Window { tentacle: 2, lens: "names".into(), budget: 50, spans: vec![span("a/x.rs", 1, 30), span("a/x.rs", 31, 60)] };
    let f = |src: &str, reader: &str, score: f32| Finding {
        key: format!("k:{src}"),
        kind: "grep".into(),
        text: "t".into(),
        source: src.into(),
        evidence: Vec::new(),
        reader: reader.into(),
        score,
        status: nmtype::Status::Observed,
    };
    let r = exact_report(
        &w,
        &[
            f("a/x.rs:3", "lexical", 0.5),
            f("a/x.rs:4", "lexical", 0.4),
            f("a/x.rs:5", "symbols", 0.9),
            f("a/x.rs:40", "semantic", 0.6),
            f("z/q.rs:1", "git", 0.9),
        ],
    );
    if r.scores.len() != 2
        || r.scores[0] <= r.scores[1]
        || (r.scores[0] - 1.4).abs() > 1e-6
        || r.agree != [2.0, 1.0]
        || r.inside != 4
        || r.outside != 1
        || r.no_information
    {
        return Err(format!("nmpod.core: exact report {r:?}"));
    }
    if !exact_report(&w, &[f("z/q.rs:1", "git", 0.9)]).no_information {
        return Err("nmpod.core: a window with no finding reported information".into());
    }
    if CoreKind::Neural.name() == CoreKind::Exact.name() {
        return Err("nmpod.core: core kinds share a name".into());
    }
    Ok(())
}
