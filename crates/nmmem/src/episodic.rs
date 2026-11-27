//! Episodic memory (guide 7; PLAN 4c, from `record-store` and `bm25-recall`): append-only
//! outcomes at drone, tentacle, brain or Neuromorph level. An episode's identity is the
//! SHA-256 of level, unit, query and answer, so the same outcome is stored once. Verdicts
//! from feedback are appended as their own records, never written over an episode.
use crate::store::{self, obj, s, u};
use nmjson::Value;
use nmtype::class::{Against, FromMaterial, FromRequest};
use nmtype::{Fault, MemoryRef, Written};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Episode {
    pub id: String,
    pub at: u64,
    pub level: String,
    pub unit: String,
    pub query: String,
    pub answer: String,
    pub evidence: Vec<String>,
    pub correct: Option<bool>,
    pub version: u64,
}
impl Episode {
    /// A new episode; the id is derived from what it records.
    #[must_use]
    pub fn new(level: &str, unit: &str, query: &str, answer: &str, evidence: Vec<String>, version: u64) -> Self {
        let id = nmhash::sha256_bytes(format!("{level}\u{1f}{unit}\u{1f}{query}\u{1f}{answer}").as_bytes());
        Self {
            id,
            at: nmtype::now_ms(),
            level: level.into(),
            unit: unit.into(),
            query: query.into(),
            answer: answer.into(),
            evidence,
            correct: None,
            version,
        }
    }
    fn to_value(&self) -> Value {
        obj(vec![
            ("record", s("episode")),
            ("id", s(&self.id)),
            ("at", u(self.at)),
            ("level", s(&self.level)),
            ("unit", s(&self.unit)),
            ("query", s(&self.query)),
            ("answer", s(&self.answer)),
            ("evidence", Value::Array(self.evidence.iter().map(|e| s(e)).collect())),
            ("version", u(self.version)),
        ])
    }
}

/// The append-only episode file and its in-memory index.
#[derive(Debug)]
pub struct Episodic {
    log: store::Appender,
    by_id: BTreeMap<String, Episode>,
    order: Vec<String>,
}
impl Episodic {
    /// Open (and replay) `episodes.jsonl` in a memory folder.
    #[must_use]
    pub fn open(dir: &std::path::Path) -> Self {
        let path: PathBuf = dir.join("episodes.jsonl");
        let mut e = Self { log: store::Appender::new(path), by_id: BTreeMap::new(), order: Vec::new() };
        for v in store::lines(e.log.path()) {
            let id = store::text(&v, "id");
            match store::text(&v, "record").as_str() {
                "episode" if !e.by_id.contains_key(&id) => {
                    let ep = Episode {
                        id: id.clone(),
                        at: store::num(&v, "at"),
                        level: store::text(&v, "level"),
                        unit: store::text(&v, "unit"),
                        query: store::text(&v, "query"),
                        answer: store::text(&v, "answer"),
                        evidence: store::strings(&v, "evidence"),
                        correct: None,
                        version: store::num(&v, "version"),
                    };
                    e.order.push(id.clone());
                    e.by_id.insert(id, ep);
                }
                "verdict" => {
                    if let Some(ep) = e.by_id.get_mut(&id) {
                        ep.correct = v.get("correct").ok().and_then(|c| if let Value::Bool(b) = c { Some(*b) } else { None });
                    }
                }
                _ => {}
            }
        }
        e
    }
    /// Store an episode once; the same id again is a duplicate, not a second record.
    /// # Errors
    /// The file could not be appended.
    pub fn write(&mut self, ep: Episode) -> Result<Written, Fault> {
        let r = MemoryRef { tier: 2, id: ep.id.clone(), version: ep.version };
        if self.by_id.contains_key(&ep.id) {
            return Ok(Written::Duplicate(r));
        }
        self.log.append(&ep.to_value())?;
        self.order.push(ep.id.clone());
        self.by_id.insert(ep.id.clone(), ep);
        Ok(Written::Stored(r))
    }
    /// Append a verdict for an episode (feedback); the episode itself is never rewritten.
    /// # Errors
    /// Unknown episode, or the file could not be appended.
    pub fn verdict(&mut self, id: &str, correct: bool) -> Result<(), Fault> {
        let ep = self.by_id.get_mut(id).ok_or_else(|| store::fault(format!("no episode {id}")))?;
        self.log.append(&obj(vec![("record", s("verdict")), ("id", s(id)), ("at", u(nmtype::now_ms())), ("correct", Value::Bool(correct))]))?;
        ep.correct = Some(correct);
        Ok(())
    }
    /// Episodes whose query shares the most key terms with `q`, best first, never one marked
    /// wrong.
    #[must_use]
    pub fn recall(&self, q: &str, k: usize) -> Vec<&Episode> {
        let terms = nmlex::key_terms(q);
        let mut scored: Vec<(usize, &Episode)> = self
            .order
            .iter()
            .filter_map(|id| self.by_id.get(id))
            .filter(|e| e.correct != Some(false))
            .map(|e| {
                let ql = e.query.to_lowercase();
                (terms.iter().filter(|t| ql.contains(t.as_str())).count(), e)
            })
            .filter(|(n, _)| *n > 0 && *n * 2 >= terms.len().max(1))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.at.cmp(&a.1.at)));
        scored.into_iter().take(k).map(|(_, e)| e).collect()
    }
    /// M1 (2026-11-09): episodes matched on what they cite as well as on their question. Score
    /// = rarity-weighted share of `q`'s key terms found in the episode's query and its first
    /// `cites` citations (the lines that will be re-read; long evidence lists matched almost
    /// anything in development, M1 revision 2). Each term's weight is its rarity in the tree
    /// (`idf`) times its rarity among the stored questions, so a phrase most questions share
    /// (a template) does not recall them all (BM25 over the episode collection); kept when the
    /// share stands out from the request's own null, its shares with the other episodes
    /// (`evidence::recalled`), best first with their scores, never one marked wrong.
    ///
    /// `fresh` gives a citation's lines as the tree holds them now: stored citations keep only a
    /// place and a hash, so the words matched are the current code's, not a stored copy. A
    /// citation `fresh` cannot read matches on its own text (an older episode's cited line).
    ///
    /// The request's data and the material's are never crossed: the stored question is request
    /// data and is compared with `q` alone; the cited lines are material and are compared with `q`
    /// alone. A term counts once when either comparison finds it.
    #[must_use]
    pub fn recall_cited(
        &self,
        q: &str,
        k: usize,
        cites: usize,
        idf: &dyn Fn(&str) -> f32,
        fresh: &dyn Fn(&str) -> Option<FromMaterial<String>>,
    ) -> Vec<(f32, &Episode)> {
        let terms = nmlex::key_terms(q);
        let live: Vec<&Episode> = self.order.iter().filter_map(|id| self.by_id.get(id)).filter(|e| e.correct != Some(false)).collect();
        let queries: Vec<String> = live.iter().map(|e| e.query.to_lowercase()).collect();
        #[expect(clippy::cast_precision_loss, reason = "episode counts are small")]
        let n = queries.len() as f32;
        let weight = |t: &str| {
            #[expect(clippy::cast_precision_loss, reason = "episode counts are small")]
            let df = queries.iter().filter(|x| x.contains(t)).count() as f32;
            idf(t) * ((n + 1.0) / (df + 0.5)).ln().max(0.05)
        };
        let w: Vec<f32> = terms.iter().map(|t| weight(t)).collect();
        let total: f32 = w.iter().sum();
        if total <= 0.0 {
            return Vec::new();
        }
        // The share the lightest term carries: the finest difference a match can show.
        let resolution = w.iter().copied().filter(|x| *x > 0.0).fold(f32::INFINITY, f32::min) / total;
        let all: Vec<(f32, &Episode)> = live
            .into_iter()
            .map(|e| {
                let asked = holds(&terms, Against::Request(&FromRequest::new(e.query.to_lowercase())));
                let cited: Vec<String> = e
                    .evidence
                    .iter()
                    .filter(|c| !c.starts_with("note:"))
                    .take(cites)
                    .map(|c| fresh(c).map_or_else(|| c.clone(), |now| format!("{c} {}", now.into_inner())))
                    .collect();
                let read = holds(&terms, Against::Material(&FromMaterial::new(cited.join(" ").to_lowercase())));
                let got: f32 = w.iter().zip(asked.iter().zip(&read)).filter(|(_, (a, b))| **a || **b).map(|(x, _)| x).sum();
                (got / total, e)
            })
            .collect();
        let mut scored: Vec<(f32, &Episode)> = all
            .iter()
            .enumerate()
            .filter(|(i, (sc, _))| {
                let others: Vec<f32> = all.iter().enumerate().filter(|(j, _)| j != i).map(|(_, x)| x.0).collect();
                crate::evidence::recalled(*sc, &others, resolution)
            })
            .map(|(_, x)| *x)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.at.cmp(&a.1.at)));
        scored.into_iter().take(k).collect()
    }
    /// Every episode not marked wrong, oldest first (recall by meaning, 2026-11-10).
    #[must_use]
    pub fn live(&self) -> Vec<&Episode> {
        self.order.iter().filter_map(|id| self.by_id.get(id)).filter(|e| e.correct != Some(false)).collect()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Episode> {
        self.by_id.get(id)
    }
    /// Latest episodes, newest first.
    #[must_use]
    pub fn latest(&self, k: usize) -> Vec<&Episode> {
        self.order.iter().rev().take(k).filter_map(|id| self.by_id.get(id)).collect()
    }
}

/// Which of a request's `terms` one item of one origin holds (lowercase text).
fn holds(terms: &[String], item: Against<'_, String>) -> Vec<bool> {
    terms.iter().map(|t| item.item().contains(t.as_str())).collect()
}

/// Self-check of the `episodic` sub-component on a temporary folder inside `scratch/`:
/// store, duplicate refused, verdict appended, replay after reopening, recall skips wrong.
pub(crate) fn check() -> Result<(), String> {
    let dir = crate::scratch_dir("episodic")?;
    let mut e = Episodic::open(&dir);
    let a = Episode::new("neuromorph", "b0", "Where is ParMap defined?", "a.rs:3", vec!["a.rs:3".into()], 1);
    if !matches!(e.write(a.clone()).map_err(|x| x.to_string())?, Written::Stored(_))
        || !matches!(e.write(a.clone()).map_err(|x| x.to_string())?, Written::Duplicate(_))
    {
        return Err("nmmem.episodic: duplicate not refused".into());
    }
    e.verdict(&a.id, false).map_err(|x| x.to_string())?;
    let again = Episodic::open(&dir);
    if again.len() != 1 || again.get(&a.id).and_then(|x| x.correct) != Some(false) || !again.recall("Where is ParMap?", 3).is_empty() {
        return Err("nmmem.episodic: replay or verdict lost".into());
    }
    // M1: a question sharing words only with the cited line is recalled; one marked wrong is not.
    let b = Episode::new("neuromorph", "b0", "Where is Q defined?", "b.rs:7", vec!["b.rs:7 pub fn cleanup_folder()".into()], 1);
    e.write(b.clone()).map_err(|x| x.to_string())?;
    let one = |_: &str| 1.0;
    let none = |_: &str| -> Option<FromMaterial<String>> { None };
    if e.recall_cited("which function does cleanup_folder", 3, 3, &one, &none).first().map(|x| &x.1.id) != Some(&b.id)
        || !e.recall_cited("ParMap", 3, 3, &one, &none).is_empty()
    {
        return Err("nmmem.episodic: recall_cited".into());
    }
    // A stored citation holds a place and a hash only; its words come from the fresh read.
    let c = Episode::new("neuromorph", "b0", "Where is R defined?", "c.rs:2", vec!["c.rs:2 lines 2..2 h:0123456789abcdef".into()], 1);
    e.write(c.clone()).map_err(|x| x.to_string())?;
    let now = |cite: &str| cite.starts_with("c.rs:2").then(|| FromMaterial::new("pub fn sweep_drawer()".to_owned()));
    if e.recall_cited("which function does sweep_drawer", 3, 3, &one, &now).first().map(|x| &x.1.id) != Some(&c.id)
        || !e.recall_cited("which function does sweep_drawer", 3, 3, &one, &none).is_empty()
    {
        return Err("nmmem.episodic: recall_cited through the fresh read".into());
    }
    Ok(())
}
