//! Working memory (guide 7; PLAN 4c): in RAM, per scope (`d<n>`, `t<n>`, `task:<id>`, `brain`).
//! Temporary by purpose: a task's scope is dropped at task end unless consolidation turns a
//! checked outcome into an episode. No count limit: a scope holds what its task noted, and the
//! task's end bounds it.
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Default)]
pub struct Working {
    scopes: BTreeMap<String, VecDeque<String>>,
    /// Tags set during a task (2026-11-12, memory revision 2, section 10): (scope, note) -> why.
    tags: BTreeMap<(String, String), String>,
    /// Scopes each note text has appeared in (recurrence across tasks).
    seen: BTreeMap<String, std::collections::BTreeSet<String>>,
}
impl Working {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Note one line in a scope.
    pub fn note(&mut self, scope: &str, text: &str) {
        self.seen.entry(text.to_owned()).or_default().insert(scope.to_owned());
        self.scopes.entry(scope.to_owned()).or_default().push_back(text.to_owned());
    }
    /// Task end for every scope at once (the tentacles' per-frame notes of one task, none tagged):
    /// all scopes and their tags are dropped; recurrence keeps only notes seen in two scopes.
    pub fn end_all(&mut self) {
        self.scopes.clear();
        self.tags.clear();
        self.seen.retain(|_, s| s.len() >= 2);
    }
    #[must_use]
    pub fn notes(&self, scope: &str) -> Vec<String> {
        self.scopes.get(scope).map(|q| q.iter().cloned().collect()).unwrap_or_default()
    }
    /// Drop a scope (task end); returns what it held, for consolidation.
    pub fn drop_scope(&mut self, scope: &str) -> Vec<String> {
        self.scopes.remove(scope).map(Vec::from).unwrap_or_default()
    }
    /// Tag a note of a scope as worth keeping (a surprise: an error, a correction, the Core Brain
    /// changing its mind), with the reason.
    pub fn tag(&mut self, scope: &str, text: &str, why: &str) {
        self.tags.insert((scope.to_owned(), text.to_owned()), why.to_owned());
    }
    /// Task end with tag and capture (memory revision 2, section 10): the scope is dropped and the
    /// notes worth keeping come back with their reason: tagged ones, ones that reached the answer's
    /// `evidence`, and ones that recurred in another scope. They become temporary notes and still
    /// earn a lasting place by retention; everything else is dropped.
    pub fn end_task(&mut self, scope: &str, evidence: &[String]) -> Vec<(String, String)> {
        let notes = self.drop_scope(scope);
        let mut out = Vec::new();
        for n in notes {
            let why = if let Some(w) = self.tags.remove(&(scope.to_owned(), n.clone())) {
                Some(w)
            } else if evidence.iter().any(|e| e.contains(n.as_str()) || n.contains(e.as_str())) {
                Some("reached the answer's evidence".to_owned())
            } else if self.seen.get(&n).is_some_and(|s| s.len() >= 2) {
                Some("recurred in another task".to_owned())
            } else {
                None
            };
            if let Some(w) = why
                && !out.iter().any(|(x, _): &(String, String)| *x == n)
            {
                out.push((n, w));
            }
        }
        self.tags.retain(|(s, _), _| s != scope);
        out
    }
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.scopes.values().flatten().map(String::len).sum()
    }
}

/// Self-check of the `working` sub-component: a scope keeps every note of its task (no count
/// limit); a dropped scope returns its notes and is gone; ending every scope empties it.
pub(crate) fn check() -> Result<(), String> {
    let mut w = Working::new();
    let many: Vec<String> = (0..100).map(|i| format!("note {i}")).collect();
    for t in &many {
        w.note("t1", t);
    }
    if w.notes("t1") != many || w.drop_scope("t1") != many || !w.notes("t1").is_empty() {
        return Err("nmmem.working: a note was lost, or the scope drop".into());
    }
    w.note("t2", "a");
    w.note("t3", "b");
    w.end_all();
    if w.bytes() != 0 {
        return Err("nmmem.working: end_all left notes".into());
    }
    // Tag and capture: tagged, evidenced and recurring notes come back; the rest are dropped.
    let mut w = Working::new();
    w.note("task:1", "the gate lives in decide.rs");
    w.note("task:2", "the gate lives in decide.rs");
    for n in ["plain thought", "surprise: second look moved first place", "decide.rs:241 lines 241..270"] {
        w.note("task:2", n);
    }
    w.tag("task:2", "surprise: second look moved first place", "the Core Brain changed its mind");
    let kept = w.end_task("task:2", &["decide.rs:241 lines 241..270".to_owned()]);
    let texts: Vec<&str> = kept.iter().map(|(t, _)| t.as_str()).collect();
    if texts.len() != 3 || texts.contains(&"plain thought") || !w.notes("task:2").is_empty() {
        return Err(format!("nmmem.working: tag and capture kept {texts:?}"));
    }
    Ok(())
}
