//! Recall over the memory book (2026-11-12, memory revision 2, section 12; research memory Part
//! 13: words plus meaning plus links fused by reciprocal rank, as Zep/Graphiti do without a model
//! at query time; spreading over links as HippoRAG 2 does; retrieval strength from ACT-R activation
//! over run-days of use).
//!
//! Three searches over live memories, each a ranked list:
//! - words: BM25 over the memory's keys and text (k1 1.2, b 0.75);
//! - meaning: cosine of the question's vector and each memory's text vector, when the caller gives
//!   them (the engine lives above this crate; vectors of memory text are a cache);
//! - links: one round of spreading from the first word and meaning hits along their links
//!   (cited together, derived from, merged from), weighted by the seed's rank.
//!   The lists are fused by reciprocal rank (k 60); then a small boost for retrieval strength
//!   (activation from distinct run-days of use and run-days since the last) and pending salience.
//!   The reranker gate and the fresh re-read of the source stay with the caller.
use crate::book::Book;
use crate::record::{Record, State};
use std::collections::BTreeMap;

/// Reciprocal-rank constant.
pub const RRF_K: f64 = 60.0;

/// Common words that carry no meaning for recall.
const STOP: [&str; 40] = [
    "the", "of", "and", "or", "to", "in", "on", "at", "is", "are", "was", "be", "it", "its", "this", "that", "for", "with", "as", "by",
    "an", "from", "how", "what", "which", "where", "when", "why", "who", "does", "do", "did", "can", "not", "no", "yes", "if", "then",
    "so", "into",
];

fn words(t: &str) -> Vec<String> {
    t.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.len() > 1)
        .map(str::to_lowercase)
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

fn doc_words(r: &Record) -> Vec<String> {
    let mut w = words(&r.text);
    for k in &r.keys {
        w.extend(words(k));
    }
    w
}

/// ACT-R-style activation over run-days: ln of the sum of t_j^-0.5 over the run-days of use,
/// approximated from the distinct run-days used and the run-days since the last use.
#[must_use]
pub fn activation(r: &Record, day: u64) -> f64 {
    let n = r.retention.days_used.max(1);
    #[expect(clippy::cast_precision_loss, reason = "run-days")]
    let (n, t) = (n as f64, day.saturating_sub(r.retention.last_use_day) as f64 + 1.0);
    (n / (1.0 - 0.5)).ln() - 0.5 * t.ln()
}

/// One recall hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub score: f64,
    /// Which searches found it.
    pub by: Vec<&'static str>,
}

/// Recall `k` memories for `question` on run-day `day`. `meaning` maps a memory id to the cosine
/// of its text vector with the question's (empty when no engine is bound).
#[must_use]
pub fn recall(book: &Book, question: &str, day: u64, meaning: &BTreeMap<String, f32>, k: usize) -> Vec<Hit> {
    let live: Vec<&Record> = book.records.values().filter(|r| r.state == State::Active).collect();
    if live.is_empty() {
        return Vec::new();
    }
    // Words: BM25.
    let docs: Vec<Vec<String>> = live.iter().map(|r| doc_words(r)).collect();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let avg = docs.iter().map(Vec::len).sum::<usize>() as f64 / docs.len() as f64;
    let q = words(question);
    let mut df: BTreeMap<&str, usize> = BTreeMap::new();
    for d in &docs {
        let mut seen: Vec<&str> = d.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for w in seen {
            *df.entry(w).or_default() += 1;
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = docs.len() as f64;
    let bm25: Vec<f64> = docs
        .iter()
        .map(|d| {
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let len = d.len() as f64;
            q.iter()
                .map(|w| {
                    #[expect(clippy::cast_precision_loss, reason = "counts")]
                    let tf = d.iter().filter(|x| *x == w).count() as f64;
                    #[expect(clippy::cast_precision_loss, reason = "counts")]
                    let dfw = df.get(w.as_str()).copied().unwrap_or(0) as f64;
                    let idf = ((n - dfw + 0.5) / (dfw + 0.5) + 1.0).ln();
                    idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len / avg.max(1.0)))
                })
                .sum()
        })
        .collect();
    let ranked = |score: &dyn Fn(usize) -> f64| -> Vec<usize> {
        let mut v: Vec<usize> = (0..live.len()).filter(|i| score(*i) > 0.0).collect();
        v.sort_by(|a, b| score(*b).total_cmp(&score(*a)).then(a.cmp(b)));
        v
    };
    let by_words = ranked(&|i| bm25.get(i).copied().unwrap_or(0.0));
    let by_meaning = ranked(&|i| live.get(i).and_then(|r| meaning.get(&r.id)).map_or(0.0, |c| f64::from(*c)));
    // Links: spread from the first hits of both lists.
    let index: BTreeMap<&str, usize> = live.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect();
    let mut spread = vec![0.0_f64; live.len()];
    for (rank, &i) in by_words.iter().take(5).chain(by_meaning.iter().take(5)).enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "ranks")]
        let w = 1.0 / (rank as f64 + 1.0);
        let Some(r) = live.get(i) else { continue };
        for (_, to) in &r.links {
            if let Some(&j) = index.get(to.as_str())
                && let Some(x) = spread.get_mut(j)
            {
                *x += w;
            }
        }
        for (j, other) in live.iter().enumerate() {
            if other.links.iter().any(|(_, to)| *to == r.id)
                && let Some(x) = spread.get_mut(j)
            {
                *x += w;
            }
        }
    }
    let by_links = ranked(&|i| spread.get(i).copied().unwrap_or(0.0));
    // Fusion.
    let mut fused: BTreeMap<usize, (f64, Vec<&'static str>)> = BTreeMap::new();
    for (name, list) in [("words", &by_words), ("meaning", &by_meaning), ("links", &by_links)] {
        for (rank, &i) in list.iter().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "ranks")]
            let add = 1.0 / (RRF_K + rank as f64 + 1.0);
            let e = fused.entry(i).or_insert((0.0, Vec::new()));
            e.0 += add;
            e.1.push(name);
        }
    }
    let mut hits: Vec<Hit> = fused
        .into_iter()
        .filter_map(|(i, (s, by))| {
            let r = live.get(i)?;
            let boost = 0.001 * activation(r, day).max(-5.0) + 0.002 * r.retention.salience_pending;
            Some(Hit { id: r.id.clone(), score: s + boost, by })
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    hits.truncate(k);
    hits
}

/// Section check: words find the matching memory first; a memory linked to it surfaces through
/// links though it shares no word; meaning alone finds a memory with no shared word; of two equal
/// matches the one used more recently on more run-days ranks first; archived memories are not
/// recalled.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    use crate::record::{Class, Kind, Retention, Tier, Validity};
    let mk = |id: &str, text: &str, links: Vec<(&str, &str)>, days: u64, last: u64, state: State| Record {
        id: id.into(),
        content: id.into(),
        kind: Kind::Fact,
        class: Class::Normal,
        tier: Tier::Active,
        state,
        topic: "context".into(),
        text: text.into(),
        anchors: Vec::new(),
        keys: Vec::new(),
        links: links.into_iter().map(|(a, b)| (a.to_owned(), b.to_owned())).collect(),
        validity: Validity::default(),
        retention: Retention { days_used: days, last_use_day: last, stability: 8.0, difficulty: 5.0, ..Retention::default() },
        verdicts: Vec::new(),
    };
    let mut b = Book::default();
    for r in [
        mk("gate", "the gate threshold of the second look", vec![("cited-together", "margin")], 1, 5, State::Active),
        mk("margin", "how the margin is chosen out of fold", vec![], 1, 5, State::Active),
        mk("other", "unrelated text about tokens", vec![], 1, 5, State::Active),
        mk("gate-old", "the gate threshold of the second look", vec![], 1, 1, State::Active),
        mk("arch", "the gate threshold of the second look", vec![], 9, 10, State::Archived),
        mk("paraphrase", "where first place can move", vec![], 1, 5, State::Active),
    ] {
        b.records.insert(r.id.clone(), r);
    }
    let hits = recall(&b, "second look gate threshold", 10, &BTreeMap::new(), 10);
    let ids: Vec<&str> = hits.iter().map(|h| h.id.as_str()).collect();
    let pos = |id: &str| ids.iter().position(|x| *x == id);
    if pos("gate") != Some(0) || pos("gate-old").is_none_or(|p| p < 1) || pos("margin").is_none() || pos("arch").is_some() {
        return Err(format!("nmmem.recall2: order {ids:?}"));
    }
    let mut meaning = BTreeMap::new();
    meaning.insert("paraphrase".to_owned(), 0.9_f32);
    let hits = recall(&b, "which step decides the final ordering", 10, &meaning, 3);
    if hits.first().map(|h| h.id.as_str()) != Some("paraphrase") {
        return Err(format!("nmmem.recall2: meaning lane {:?}", hits.iter().map(|h| &h.id).collect::<Vec<_>>()));
    }
    Ok(())
}
