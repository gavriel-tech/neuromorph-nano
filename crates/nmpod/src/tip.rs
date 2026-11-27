//! Drone tip readers (guide 4.2, 17.1): exact readers first, a slot reader only when a
//! reflex escalates. Each reader scans the unit's territory through the substrate view and
//! returns findings; it holds no state and no handle to any other unit.
//!
//! 2026-11-10 (step P4 first form, one tentacle system): every reader sits behind this one
//! `Reader` interface, the attention-head drone readers of a bound neural core (built outside
//! this crate, `nmapi::tentacle_unit`) and these exact readers alike. A reader states its
//! `Reach`: window-bound readers read inside the tentacle's window when the context names one
//! (`ReadCtx::window`); whole-code-base readers read the whole tree. With no window in the
//! context every reader behaves exactly as before (territory paths), which is the served
//! control path while `--one-tentacle` is off.
use crate::window::Window;
use nmtype::{Hit, Packet, SlotView, Status, SubstrateView};

/// One thing a reader found: the claim key it bears on, the text, its cited source and the
/// evidence line, the reader that found it, and a status hint for the pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub key: String,
    pub kind: String,
    pub text: String,
    pub source: String,
    pub evidence: Vec<String>,
    pub reader: String,
    pub score: f32,
    pub status: Status,
}
impl Finding {
    /// A finding taken from a substrate packet.
    #[must_use]
    pub fn from_packet(p: &Packet, reader: &str) -> Self {
        Self {
            key: p.key().to_owned(),
            kind: p.kind().to_owned(),
            text: p.text().to_owned(),
            source: p.provenance().source.clone(),
            evidence: p.get_evidence().to_vec(),
            reader: reader.to_owned(),
            score: p.get_confidence(),
            status: p.get_status(),
        }
    }
}

impl Finding {
    /// JSON form, for the edge from a drone tip to its tentacle's pipeline.
    #[must_use]
    pub fn to_value(&self) -> nmjson::Value {
        use nmjson::Value as V;
        let s = |t: &str| V::String(t.to_owned());
        V::Object(
            [
                ("key", s(&self.key)),
                ("kind", s(&self.kind)),
                ("text", s(&self.text)),
                ("source", s(&self.source)),
                ("evidence", V::Array(self.evidence.iter().map(|e| s(e)).collect())),
                ("reader", s(&self.reader)),
                ("score", V::Number(if self.score.is_finite() { self.score.to_string() } else { "0".into() })),
                ("status", self.status.to_value()),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
    /// # Errors
    /// A missing field.
    pub fn from_value(v: &nmjson::Value) -> Result<Self, nmtype::Fault> {
        let bad = |e: String| nmtype::Fault::new("nmpod.tip", nmtype::FaultKind::Interface, e);
        let s = |k: &str| v.get(k).and_then(nmjson::Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
        Ok(Self {
            key: s("key")?,
            kind: s("kind")?,
            text: s("text")?,
            source: s("source")?,
            evidence: v
                .get("evidence")
                .and_then(nmjson::Value::array)
                .map_err(|e| bad(e.to_string()))?
                .iter()
                .filter_map(|e| e.string().ok().map(str::to_owned))
                .collect(),
            reader: s("reader")?,
            score: match v.get("score") {
                Ok(nmjson::Value::Number(n)) => n.parse().unwrap_or(0.0),
                _ => 0.0,
            },
            status: Status::from_value(v.get("status").map_err(|e| bad(e.to_string()))?)?,
        })
    }
}

/// Where a reader looks: the question, the territory's path prefixes (empty: anywhere) and,
/// in the one tentacle system, the tentacle's window (window-bound readers read inside it).
pub struct ReadCtx<'a> {
    pub sub: &'a dyn SubstrateView,
    pub slots: &'a dyn SlotView,
    pub query: &'a str,
    pub paths: &'a [String],
    pub window: Option<&'a Window>,
}
impl ReadCtx<'_> {
    fn inside(&self, source: &str) -> bool {
        self.paths.is_empty() || self.paths.iter().any(|p| source.starts_with(p.as_str()))
    }
    /// Lines holding every term inside the window's spans, at most `limit` (file order of the
    /// substrate, then line order). Used by window-bound readers when a window is given.
    fn grep_window(&self, w: &Window, terms: &[String], limit: usize) -> Vec<Hit> {
        let files = w.files();
        if files.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<Hit> = self.sub.grep(terms, &files, usize::MAX).into_iter().filter(|h| w.contains(&h.file, h.line)).collect();
        hits.truncate(limit);
        hits
    }
    /// The `k` nearest chunks whose first line lies inside the window's spans.
    fn semantic_window(&self, w: &Window, k: usize) -> Vec<Packet> {
        let files = w.files();
        if files.is_empty() {
            return Vec::new();
        }
        self.sub
            .semantic_in(self.query, &files, usize::MAX)
            .into_iter()
            .filter(|p| w.span_of(&p.provenance().source).is_some())
            .take(k)
            .collect()
    }
}

/// How far a reader reaches in the one tentacle system: inside the tentacle's window, or over
/// the whole code base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    Window,
    Whole,
}

/// A drone tip reader.
pub trait Reader: Send + Sync {
    fn name(&self) -> &'static str;
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding>;
    /// Whole-code-base unless the reader reads text inside a window.
    fn reach(&self) -> Reach {
        Reach::Whole
    }
}

/// Reach of a reader by name (`Reach::Whole` for an unknown name).
#[must_use]
pub fn reach_of(name: &str) -> Reach {
    reader(name).map_or(Reach::Whole, |r| r.reach())
}

/// Lines in the territory containing the question's key terms.
pub struct Lexical;
impl Reader for Lexical {
    fn name(&self) -> &'static str {
        "lexical"
    }
    fn reach(&self) -> Reach {
        Reach::Window
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        let mut terms = nmlex::names(ctx.query);
        if terms.is_empty() {
            terms = nmlex::key_terms(ctx.query);
            // Codes and keys made of letters and digits (`s031`, `p-10012`) are the rarest words
            // a question holds: they are read first (stable, so the question's order stays).
            terms.sort_by_key(|t| !(t.chars().any(|c| c.is_ascii_digit()) && t.chars().any(char::is_alphabetic)));
        }
        terms
            .iter()
            .take(3)
            .flat_map(|t| match ctx.window {
                Some(w) => ctx.grep_window(w, std::slice::from_ref(t), 4),
                None => ctx.sub.grep(std::slice::from_ref(t), ctx.paths, 4),
            })
            .map(|h| {
                let src = format!("{}:{}", h.file, h.line);
                Finding {
                    key: format!("grep:{src}"),
                    kind: "grep".into(),
                    text: h.text.clone(),
                    evidence: vec![format!("{src} {}", h.text)],
                    source: src,
                    reader: "lexical".into(),
                    score: 0.5,
                    status: Status::Observed,
                }
            })
            .chain(
                // Table rows the question points at by a column's extreme or empty values.
                ctx.sub
                    .view("table", ctx.query)
                    .iter()
                    .filter(|p| ctx.inside(&p.provenance().source))
                    .filter(|p| ctx.window.is_none_or(|w| w.span_of(&p.provenance().source).is_some()))
                    .map(|p| Finding::from_packet(p, "lexical")),
            )
            .collect()
    }
}
/// Definitions and crate dependencies the question names, inside the territory.
pub struct Symbols;
impl Reader for Symbols {
    fn name(&self) -> &'static str {
        "symbols"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("structural", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "symbols"))
            .collect()
    }
}
/// File history from git for the territory.
pub struct Git;
impl Reader for Git {
    fn name(&self) -> &'static str {
        "git"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("temporal", ctx.query)
            .iter()
            .filter(|p| ctx.inside(p.key().trim_start_matches("history:")))
            .map(|p| Finding::from_packet(p, "git"))
            .collect()
    }
}
/// Build and Step Check records touching the territory: the verified outcomes the build tools
/// log (`build-outcomes.jsonl`), loaded into the substrate's operational view (2026-11-09;
/// before, that view never received them and this reader always came back empty).
pub struct Build;
impl Reader for Build {
    fn name(&self) -> &'static str {
        "build"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("operational", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "build"))
            .collect()
    }
}
/// Where a named item is tested: test files and the lines after a file's test marker
/// (`#[cfg(test)]`, a section `fn check(`) that name it (Verification job, 2026-11-09).
pub struct Tests;
impl Reader for Tests {
    fn name(&self) -> &'static str {
        "tests"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("verification", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "tests"))
            .collect()
    }
}
/// Where a named item is used, and which crates depend on the crate that defines it
/// (Dependency job, 2026-11-09).
pub struct Uses;
impl Reader for Uses {
    fn name(&self) -> &'static str {
        "uses"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("dependency", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "uses"))
            .collect()
    }
}
/// Rules and lints bearing on the question (Constraint job, 2026-11-10).
pub struct Rules;
impl Reader for Rules {
    fn name(&self) -> &'static str {
        "rules"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("constraint", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "rules"))
            .collect()
    }
}
/// The public interface of a named crate or file (Functional job, 2026-11-10).
pub struct Interface;
impl Reader for Interface {
    fn name(&self) -> &'static str {
        "interface"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        ctx.sub
            .view("interface", ctx.query)
            .iter()
            .filter(|p| ctx.inside(&p.provenance().source))
            .map(|p| Finding::from_packet(p, "interface"))
            .collect()
    }
}
/// Key terms a deeper reading pairs up.
pub const DEEP_TERMS: usize = 5;
/// Chunks a deeper reading takes, and its lower chunk floor.
pub const DEEP_K: usize = 4;
pub const DEEP_MIN: f32 = 0.2;
/// A deeper reading of one territory (subordinate drones and the Reasoning job, 2026-11-09):
/// lines holding two of the question's key terms together, and more chunks at a lower floor
/// than the base semantic reader takes.
pub struct Deep;
impl Reader for Deep {
    fn name(&self) -> &'static str {
        "deep"
    }
    fn reach(&self) -> Reach {
        Reach::Window
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        let mut terms = nmlex::key_terms(ctx.query);
        terms.sort_by_key(|t| std::cmp::Reverse(t.len()));
        terms.truncate(DEEP_TERMS);
        let mut out: Vec<Finding> = Vec::new();
        for (i, a) in terms.iter().enumerate() {
            for b in terms.iter().skip(i + 1) {
                let pair = [a.clone(), b.clone()];
                let hits = match ctx.window {
                    Some(w) => ctx.grep_window(w, &pair, 2),
                    None => ctx.sub.grep(&pair, ctx.paths, 2),
                };
                for h in hits {
                    let src = format!("{}:{}", h.file, h.line);
                    if out.iter().any(|f| f.source == src) {
                        continue;
                    }
                    out.push(Finding {
                        key: format!("grep:{src}"),
                        kind: "grep".into(),
                        text: h.text.clone(),
                        evidence: vec![format!("{src} {}", h.text)],
                        source: src,
                        reader: "deep".into(),
                        score: 0.6,
                        status: Status::Observed,
                    });
                }
            }
        }
        let near = match ctx.window {
            Some(w) => ctx.semantic_window(w, DEEP_K),
            None => ctx.sub.semantic_in(ctx.query, ctx.paths, DEEP_K),
        };
        out.extend(
            near.iter()
                .filter(|p| ctx.inside(&p.provenance().source) && p.get_confidence() >= DEEP_MIN)
                .map(|p| Finding::from_packet(p, "deep")),
        );
        out
    }
}
/// Chunks the semantic reader takes from its own territory.
pub const SEMANTIC_K: usize = 2;
/// Chunks scoring below this (cosine and key-term blend, times source weight) are left out at
/// the tip: they never reach the reducer's relevance floor, so scoring them there is waste.
pub const SEMANTIC_MIN: f32 = 0.3;
/// Nearest chunks by the brain's embedder, ranked inside the territory (2026-11-08: before,
/// the global top four were filtered to the territory, so most territories read nothing).
pub struct Semantic;
impl Reader for Semantic {
    fn name(&self) -> &'static str {
        "semantic"
    }
    fn reach(&self) -> Reach {
        Reach::Window
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        let near = match ctx.window {
            Some(w) => ctx.semantic_window(w, SEMANTIC_K),
            None => ctx.sub.semantic_in(ctx.query, ctx.paths, SEMANTIC_K),
        };
        near.iter()
            .filter(|p| ctx.inside(&p.provenance().source) && p.get_confidence() >= SEMANTIC_MIN)
            .map(|p| Finding::from_packet(p, "semantic"))
            .collect()
    }
}
/// Asks the writer slot through the Core Brain; used only when a reflex escalates. Its
/// answer is a hypothesis until a reader agrees.
pub struct Slot;
impl Reader for Slot {
    fn name(&self) -> &'static str {
        "slot"
    }
    fn scan(&self, ctx: &ReadCtx<'_>) -> Vec<Finding> {
        let input = nmjson::Value::Object([("prompt".to_owned(), nmjson::Value::String(ctx.query.to_owned()))].into_iter().collect());
        match ctx.slots.call("writer", &input) {
            Ok(nmjson::Value::String(text)) if !text.is_empty() => vec![Finding {
                key: format!("slot:{}", ctx.query),
                kind: "finding".into(),
                text,
                source: "unit:nmbrain.slot#writer".into(),
                evidence: Vec::new(),
                reader: "slot".into(),
                score: 0.3,
                status: Status::Hypothesized,
            }],
            _ => Vec::new(),
        }
    }
}

/// Readers for a reader name (`lexical`, `symbols`, `git`, `build`, `semantic`, `slot`).
#[must_use]
pub fn reader(name: &str) -> Option<Box<dyn Reader>> {
    match name {
        "lexical" => Some(Box::new(Lexical)),
        "symbols" => Some(Box::new(Symbols)),
        "git" => Some(Box::new(Git)),
        "build" => Some(Box::new(Build)),
        "semantic" => Some(Box::new(Semantic)),
        "tests" => Some(Box::new(Tests)),
        "uses" => Some(Box::new(Uses)),
        "deep" => Some(Box::new(Deep)),
        "rules" => Some(Box::new(Rules)),
        "interface" => Some(Box::new(Interface)),
        "slot" => Some(Box::new(Slot)),
        _ => None,
    }
}

/// Self-check of the `tip` sub-component: the lexical and symbol readers find a constant
/// inside their territory and nothing outside it.
pub(crate) fn check() -> Result<(), String> {
    let (sub, slots) = crate::fixture::views();
    let ctx = |paths: &'static [String]| ReadCtx { sub: &sub, slots: &slots, query: "What is LIMIT?", paths, window: None };
    static A: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| vec!["a/".to_owned()]);
    static Z: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| vec!["zzz/".to_owned()]);
    if Lexical.scan(&ctx(&A)).is_empty() || Symbols.scan(&ctx(&A)).is_empty() {
        return Err("nmpod.tip: readers missed the constant in their territory".into());
    }
    if !Lexical.scan(&ctx(&Z)).is_empty() || !Symbols.scan(&ctx(&Z)).is_empty() {
        return Err("nmpod.tip: readers read outside their territory".into());
    }
    // One tentacle system: a window-bound reader reads inside the window's spans only; a
    // whole-code-base reader ignores the window.
    let span = |f: &str, a: usize, b: usize| crate::window::Span { file: f.into(), start: a, end: b, key: format!("{f}:{a}"), tokens: 5 };
    let wb = Window { tentacle: 0, lens: "words".into(), budget: 50, spans: vec![span("b/y.rs", 1, 1)] };
    let wn = Window { tentacle: 0, lens: "words".into(), budget: 50, spans: vec![span("a/x.rs", 2, 2)] };
    let in_win = |w| ReadCtx { sub: &sub, slots: &slots, query: "What is LIMIT?", paths: &[], window: Some(w) };
    let lex: Vec<String> = Lexical.scan(&in_win(&wb)).into_iter().map(|f| f.source).collect();
    if lex != ["b/y.rs:1"] || !Lexical.scan(&in_win(&wn)).is_empty() {
        return Err(format!("nmpod.tip: window-bound reader read outside its window: {lex:?}"));
    }
    if Symbols.scan(&in_win(&wn)).len() != 2
        || Lexical.reach() != Reach::Window
        || Symbols.reach() != Reach::Whole
        || reach_of("git") != Reach::Whole
        || reach_of("deep") != Reach::Window
    {
        return Err("nmpod.tip: whole-code-base reader limited by a window, or a wrong reach".into());
    }
    Ok(())
}
