//! The Parameter Map observer (guide 10.4; PLAN 4c, from `parctl/parmap`): every structural,
//! functional or parameter change is observed; while recording is on (off by default) it is
//! appended to `edits.jsonl` with time, scope, target, kind, detail and run. `guarded_edit`
//! refuses frozen clusters and scopes that are not recorded. The Parameter Map stays
//! separate: it knows clusters and changes, never drones or context mapping.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::io::Write as _;
use std::path::PathBuf;

#[derive(Debug)]
pub struct ParMap {
    pub recording: bool,
    pub seen: u64,
    pub written: u64,
    path: PathBuf,
    frozen: Vec<String>,
}

fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}

impl ParMap {
    /// An observer writing `edits.jsonl` in a folder; recording off.
    #[must_use]
    pub fn new(dir: &std::path::Path) -> Self {
        Self { recording: false, seen: 0, written: 0, path: dir.join("edits.jsonl"), frozen: Vec::new() }
    }
    /// Clusters no edit may touch (from `tensors.json` `control: frozen`, S6).
    pub fn freeze(&mut self, clusters: Vec<String>) {
        self.frozen = clusters;
    }
    /// Observe one change. Counted always; appended only while recording.
    /// # Errors
    /// Recording is on and the line could not be appended.
    pub fn observe(&mut self, scope: &str, target: &str, kind: &str, detail: &str, run: &str) -> Result<(), Fault> {
        self.seen += 1;
        if !self.recording {
            return Ok(());
        }
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d).map_err(|e| Fault::new("nmnpcs.parmap", FaultKind::Resource, e.to_string()))?;
        }
        let line = obj(vec![
            ("at", Value::Number(nmtype::now_ms().to_string())),
            ("scope", s(scope)),
            ("target", s(target)),
            ("kind", s(kind)),
            ("detail", s(detail)),
            ("run", s(run)),
        ])
        .encode()
        .unwrap_or_default();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| Fault::new("nmnpcs.parmap", FaultKind::Resource, e.to_string()))?;
        writeln!(f, "{line}").map_err(|e| Fault::new("nmnpcs.parmap", FaultKind::Resource, e.to_string()))?;
        self.written += 1;
        Ok(())
    }
    /// Allow an edit of a cluster only when recording is on and the cluster is not frozen.
    /// # Errors
    /// A frozen cluster, or recording off (an unrecorded scope).
    pub fn guarded_edit(&self, cluster: &str) -> Result<(), Fault> {
        if self.frozen.iter().any(|c| c == cluster) {
            return Err(Fault::new("nmnpcs.parmap", FaultKind::Policy, format!("cluster {cluster} is frozen")));
        }
        if !self.recording {
            return Err(Fault::new("nmnpcs.parmap", FaultKind::Policy, "edits need the Parameter Map recording"));
        }
        Ok(())
    }
}

/// Self-check of the `parmap` sub-component: off by default (counted, not written), recorded
/// when on, frozen and unrecorded edits refused.
pub(crate) fn check() -> Result<(), String> {
    let dir = nmtype::check_dir("nmnpcs-parmap").map_err(|e| e.to_string())?;
    let mut p = ParMap::new(&dir);
    p.observe("structure", "t#0", "swap", "x", "check").map_err(|e| e.to_string())?;
    if p.seen != 1 || p.written != 0 || p.guarded_edit("layer0.head0").is_ok() {
        return Err("nmnpcs.parmap: recording should be off by default".into());
    }
    p.recording = true;
    p.freeze(vec!["layer0.head0".into()]);
    p.observe("structure", "t#0", "swap", "x", "check").map_err(|e| e.to_string())?;
    if p.written != 1 || p.guarded_edit("layer0.head0").is_ok() || p.guarded_edit("layer0.head1").is_err() {
        return Err("nmnpcs.parmap: recording or guard".into());
    }
    Ok(())
}
