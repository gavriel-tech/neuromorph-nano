//! Tentacle windows (2026-11-10, step P4 first form; specification 2.2 and 2.3; design by
//! Luis Jake Gabriel III): what one tentacle reads for one question. Each tentacle owns its own
//! full window of chunk spans laid out over the whole code base by a lens; windows are read in
//! parallel, no chunk is in two windows (the crossover gate), and the total context is expanded,
//! not partitioned. The lens is a temporary specialisation given per question, not a fixed role.
//! The window replaces the file territory as what a tentacle reads; territories remain only as
//! G's coverage segments and in the served control path while `--one-tentacle` is off.
//!
//! Exact readers are of two reaches (`crate::tip::Reach`): window-bound readers (lexical,
//! semantic, deep) read inside the spans of the tentacle's window; whole-code-base readers
//! (symbols, git, build, tests, uses, rules, interface) read the whole tree, and each of them is
//! carried by one tentacle per question (`deal`), so the whole tree is read once by each and no
//! finding is doubled across tentacles.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;

/// One chunk span of a window: file, first and last line, content key, tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub file: String,
    pub start: usize,
    pub end: usize,
    /// Content key of the chunk (the substrate chunk key; the encoded-chunk store key).
    pub key: String,
    pub tokens: usize,
}

/// One tentacle's window for one question.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Window {
    /// The tentacle (slot) that owns it.
    pub tentacle: usize,
    /// The lens that filled it for this question (temporary specialisation).
    pub lens: String,
    /// Token budget of the window (the firm 12,000 of specification 2.3, supplied by the caller).
    pub budget: usize,
    /// Its chunk spans, in window order.
    pub spans: Vec<Span>,
}

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmpod.window", FaultKind::Interface, e.to_string())
}

impl Window {
    /// Tokens the window holds.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.spans.iter().map(|s| s.tokens).sum()
    }
    /// True when the window holds nothing (its lens found nothing for the question).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
    /// Distinct files of the window, sorted (the path filter a window-bound reader starts from).
    #[must_use]
    pub fn files(&self) -> Vec<String> {
        let mut f: Vec<String> = self.spans.iter().map(|s| s.file.clone()).collect();
        f.sort();
        f.dedup();
        f
    }
    /// Does the window hold line `line` of `file`?
    #[must_use]
    pub fn contains(&self, file: &str, line: usize) -> bool {
        self.spans.iter().any(|s| s.file == file && s.start <= line && line <= s.end)
    }
    /// The span index holding a cited source `file:line` (or `file:line text`), if any.
    #[must_use]
    pub fn span_of(&self, source: &str) -> Option<usize> {
        let loc = source.split_whitespace().next().unwrap_or(source);
        let (file, line) = loc.rsplit_once(':')?;
        let line: usize = line.parse().ok()?;
        self.spans.iter().position(|s| s.file == file && s.start <= line && line <= s.end)
    }
    /// JSON form, for the edge from a tentacle to its drone tip.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let n = |x: usize| Value::Number(x.to_string());
        let spans = self
            .spans
            .iter()
            .map(|s| Value::Array(vec![Value::String(s.file.clone()), n(s.start), n(s.end), Value::String(s.key.clone()), n(s.tokens)]))
            .collect();
        Value::Object(
            [
                ("tentacle".to_owned(), n(self.tentacle)),
                ("lens".to_owned(), Value::String(self.lens.clone())),
                ("budget".to_owned(), n(self.budget)),
                ("spans".to_owned(), Value::Array(spans)),
            ]
            .into_iter()
            .collect(),
        )
    }
    /// # Errors
    /// A missing field or a malformed span.
    pub fn from_value(v: &Value) -> Result<Self, Fault> {
        let int = |v: &Value| v.integer().ok().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| bad("window: not a count"));
        let spans = v
            .get("spans")
            .and_then(Value::array)
            .map_err(bad)?
            .iter()
            .map(|s| {
                let a = s.array().map_err(bad)?;
                let at = |i: usize| a.get(i).ok_or_else(|| bad("window: short span"));
                Ok(Span {
                    file: at(0)?.string().map_err(bad)?.to_owned(),
                    start: int(at(1)?)?,
                    end: int(at(2)?)?,
                    key: at(3)?.string().map_err(bad)?.to_owned(),
                    tokens: int(at(4)?)?,
                })
            })
            .collect::<Result<_, Fault>>()?;
        Ok(Self {
            tentacle: int(v.get("tentacle").map_err(bad)?)?,
            lens: v.get("lens").and_then(Value::string).map_err(bad)?.to_owned(),
            budget: int(v.get("budget").map_err(bad)?)?,
            spans,
        })
    }
}

/// The crossover gate over one question's windows: chunks (by content key) held by more than
/// one window, plus chunks repeated inside one window. Must be 0.
#[must_use]
pub fn crossover(windows: &[Window]) -> usize {
    let mut held: BTreeMap<&str, usize> = BTreeMap::new();
    let mut repeated = 0;
    for w in windows {
        let mut own: Vec<&str> = w.spans.iter().map(|s| s.key.as_str()).collect();
        let n = own.len();
        own.sort_unstable();
        own.dedup();
        repeated += n - own.len();
        for k in own {
            *held.entry(k).or_default() += 1;
        }
    }
    repeated + held.values().filter(|c| **c > 1).count()
}

/// Deal whole-code-base readers to tentacles for one question: reader `i` goes to tentacle
/// `(i + offset) % tentacles`, so each is carried once and the load spreads; `offset` lets the
/// caller rotate the dealing between questions (specialisation is temporary). Names keep their
/// order and a name listed twice is dealt once.
#[must_use]
pub fn deal(names: &[&str], tentacles: usize, offset: usize) -> Vec<Vec<String>> {
    let n = tentacles.max(1);
    let mut out = vec![Vec::new(); n];
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if seen.contains(name) {
            continue;
        }
        let i = (seen.len() + offset) % n;
        seen.push(name);
        if let Some(t) = out.get_mut(i) {
            t.push((*name).to_owned());
        }
    }
    out
}

/// Self-check of the `window` sub-component: span lookup, JSON round trip, the crossover gate
/// on a lawful and an unlawful pair of windows, and dealing whole readers once each.
pub(crate) fn check() -> Result<(), String> {
    let span = |f: &str, a: usize, b: usize, k: &str| Span { file: f.into(), start: a, end: b, key: k.into(), tokens: 10 };
    let w0 =
        Window { tentacle: 0, lens: "words".into(), budget: 100, spans: vec![span("a/x.rs", 1, 30, "k1"), span("b/y.rs", 31, 60, "k2")] };
    let w1 = Window { tentacle: 1, lens: "names".into(), budget: 100, spans: vec![span("a/x.rs", 31, 60, "k3")] };
    if !w0.contains("a/x.rs", 7)
        || w0.contains("a/x.rs", 31)
        || w0.span_of("b/y.rs:40 text") != Some(1)
        || w0.files() != ["a/x.rs", "b/y.rs"]
        || w0.tokens() != 20
    {
        return Err("nmpod.window: span lookup".into());
    }
    if Window::from_value(&w0.to_value()).map_err(|e| e.to_string())? != w0 {
        return Err("nmpod.window: JSON round trip".into());
    }
    if crossover(&[w0.clone(), w1.clone()]) != 0 {
        return Err("nmpod.window: lawful windows counted as crossover".into());
    }
    let w2 = Window { spans: vec![span("a/x.rs", 1, 30, "k1")], ..w1 };
    if crossover(&[w0, w2]) != 1 {
        return Err("nmpod.window: a chunk in two windows was not counted".into());
    }
    let d = deal(&["symbols", "git", "symbols", "build"], 2, 1);
    if d != vec![vec!["git".to_owned()], vec!["symbols".to_owned(), "build".to_owned()]] {
        return Err(format!("nmpod.window: dealing {d:?}"));
    }
    Ok(())
}
