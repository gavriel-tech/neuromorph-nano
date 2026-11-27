//! The preference cartridge fitter (roadmap 2026-11-23, Phase 2 item 2: preference cartridges at
//! the top of the stack). It reads a project's preference signals (`training/preferences.jsonl`),
//! learns how likely each circumstance is to end well (an edit kept, an answer accepted) or badly (an
//! edit reverted, a correction, a refused tool call), and writes the result as a `preference`
//! settings cartridge through the cartridge library's shelf (layer 4 of the stacking order,
//! `library::family_rank`), plugged by a stack for the project's preference use.
//!
//! The gate: the signals' own split (fixed by hash when they were captured) gives a held-out part.
//! The candidate is fitted on the rest and scored on the held-out part by the Brier score (mean
//! squared error of its predictions; lower is better). It is written only if it beats its absence
//! (a prediction of one half for every signal: no preference known) and, when a preference cartridge
//! for the project already exists, also beats that one on the same held-out part.
//!
//! Circumstances are behavioural labels only: the file kind (extension) and top folder of the paths
//! a signal names, and the tool a refusal named. No prompt, passage or code text enters a cartridge.
use crate::idle::{Learner, LearnerReport};
use crate::library::{CartridgeSpec, Filter, Library, PREFERENCE, Stack};
use crate::unit::{n, obj, s};
use nmjson::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Held-out signals needed before a cartridge can be judged and written.
pub const MIN_HELD: usize = 4;
/// Signals a circumstance needs before its own rate is used (else the overall rate).
pub const MIN_KEY_N: u64 = 2;
/// The margin a candidate must win by.
const EPS: f64 = 1e-9;
/// The Brier score of predicting one half for every signal: the absence of a cartridge.
pub const ABSENT_BRIER: f64 = 0.25;

/// One preference signal as the fitter sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefRow {
    /// Circumstance labels.
    pub features: Vec<String>,
    /// Whether it ended well.
    pub positive: bool,
    /// Whether it is in the held-out part.
    pub held: bool,
}

/// Whether a signal of `kind` ended well (`Some(true)`), badly (`Some(false)`), or says nothing
/// about an outcome (`None`: a stated rule, or a kind this build does not know).
#[must_use]
pub fn polarity(kind: &str) -> Option<bool> {
    match kind {
        "edit-kept" | "acceptance" => Some(true),
        "edit-reverted" | "correction" | "refusal" => Some(false),
        _ => None,
    }
}

/// A label is short and plain (no path separators, spaces or quotes), so no material rides in it.
fn label_ok(t: &str) -> bool {
    !t.is_empty() && t.len() <= 32 && t.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// The circumstance labels of a signal row: `ext:<extension>` and `dir:<top folder>` of each path,
/// `tool:<tool>`, always `any`.
#[must_use]
pub fn features(row: &Value) -> Vec<String> {
    let mut out = vec!["any".to_owned()];
    let mut push = |f: String| {
        if !out.contains(&f) {
            out.push(f);
        }
    };
    for p in row.get("paths").and_then(Value::array).unwrap_or_default() {
        let Ok(p) = p.string() else { continue };
        let p = p.replace('\\', "/");
        if let Some(ext) = p.rsplit('/').next().and_then(|f| f.rsplit_once('.')).map(|(_, e)| e.to_ascii_lowercase())
            && label_ok(&ext)
        {
            push(format!("ext:{ext}"));
        }
        if let Some((top, _)) = p.split_once('/')
            && label_ok(top)
        {
            push(format!("dir:{top}"));
        }
    }
    if let Ok(t) = row.get("tool").and_then(Value::string)
        && label_ok(t)
    {
        push(format!("tool:{t}"));
    }
    out
}

/// The rows that carry an outcome, with their features and split.
#[must_use]
pub fn parse_rows(rows: &[Value]) -> Vec<PrefRow> {
    rows.iter()
        .filter_map(|r| {
            let kind = r.get("kind").and_then(Value::string).ok()?;
            let positive = polarity(kind)?;
            let held = r.get("split").and_then(Value::string).ok().and_then(nmtype::examples::Split::parse) == Some(nmtype::examples::Split::Held);
            Some(PrefRow { features: features(r), positive, held })
        })
        .collect()
}

/// A fitted preference model: (good outcomes, signals) overall and per circumstance.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefModel {
    pub global: (u64, u64),
    pub keys: BTreeMap<String, (u64, u64)>,
}

#[expect(clippy::cast_precision_loss, reason = "counts")]
fn rate((pos, all): (u64, u64)) -> f64 {
    (pos as f64 + 1.0) / (all as f64 + 2.0)
}

impl PrefModel {
    /// Counts over `rows`.
    #[must_use]
    pub fn fit(rows: &[&PrefRow]) -> Self {
        let mut m = Self::default();
        for r in rows {
            m.global.1 += 1;
            m.global.0 += u64::from(r.positive);
            for f in r.features.iter().filter(|f| f.as_str() != "any") {
                let e = m.keys.entry(f.clone()).or_default();
                e.1 += 1;
                e.0 += u64::from(r.positive);
            }
        }
        m
    }
    /// The chance a signal with `features` ends well: the mean smoothed rate of its circumstances
    /// seen at least `MIN_KEY_N` times, else the smoothed overall rate.
    #[must_use]
    pub fn predict(&self, features: &[String]) -> f64 {
        let rates: Vec<f64> = features.iter().filter_map(|f| self.keys.get(f)).filter(|c| c.1 >= MIN_KEY_N).map(|c| rate(*c)).collect();
        if rates.is_empty() {
            rate(self.global)
        } else {
            #[expect(clippy::cast_precision_loss, reason = "counts")]
            let k = rates.len() as f64;
            rates.iter().sum::<f64>() / k
        }
    }
    /// The model as a settings cartridge's values: `pref.global` and `pref.key.<label>`, each
    /// `<good>/<signals>`.
    #[must_use]
    pub fn to_settings(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("pref.global".to_owned(), format!("{}/{}", self.global.0, self.global.1));
        for (k, (p, a)) in &self.keys {
            m.insert(format!("pref.key.{k}"), format!("{p}/{a}"));
        }
        m
    }
    /// The model a settings cartridge's values hold (values that do not read are left out).
    #[must_use]
    pub fn from_settings(m: &BTreeMap<String, String>) -> Self {
        let pair = |v: &str| -> Option<(u64, u64)> {
            let (a, b) = v.split_once('/')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        };
        let mut out = Self::default();
        for (k, v) in m {
            let Some(c) = pair(v) else { continue };
            if k == "pref.global" {
                out.global = c;
            } else if let Some(key) = k.strip_prefix("pref.key.") {
                out.keys.insert(key.to_owned(), c);
            }
        }
        out
    }
}

/// The Brier score of `predict` over `rows` (`None` when empty).
#[must_use]
pub fn brier(rows: &[&PrefRow], predict: &dyn Fn(&[String]) -> f64) -> Option<f64> {
    if rows.is_empty() {
        return None;
    }
    let sum: f64 = rows
        .iter()
        .map(|r| {
            let y = if r.positive { 1.0 } else { 0.0 };
            let p = predict(&r.features);
            (p - y) * (p - y)
        })
        .sum();
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let k = rows.len() as f64;
    Some(sum / k)
}

/// Below this chance of ending well, a file's circumstances earn a caution line at serve time.
pub const CAUTION_BELOW: f64 = 0.4;
/// Caution lines one reply carries at most.
pub const CAUTION_LINES: usize = 3;

/// The serve-time reading of a preference cartridge (roadmap Phase 2 item 2, 2026-11-24): for each
/// file a reply cites (in order, each once), the chance an edit there ends well, from the
/// circumstances of the file (kind and top folder) the cartridge has seen at least `MIN_KEY_N` times;
/// a file whose chance is below `CAUTION_BELOW` gets one caution line. A file with no such
/// circumstance gets none: the overall rate says nothing about that file. Behavioural labels only.
#[must_use]
pub fn guidance(model: &PrefModel, paths: &[String]) -> Vec<String> {
    let mut seen: Vec<&str> = Vec::new();
    let mut out = Vec::new();
    for p in paths {
        if out.len() >= CAUTION_LINES || seen.contains(&p.as_str()) {
            continue;
        }
        seen.push(p);
        let feats: Vec<String> = features(&obj(vec![("paths", Value::Array(vec![s(p)]))])).into_iter().filter(|f| model.keys.get(f).is_some_and(|c| c.1 >= MIN_KEY_N)).collect();
        if feats.is_empty() {
            continue;
        }
        let chance = model.predict(&feats);
        if chance < CAUTION_BELOW {
            let signals: u64 = feats.iter().filter_map(|f| model.keys.get(f)).map(|c| c.1).sum();
            out.push(format!(
                "preference: edits to files like {p} ({}) have ended well about {:.0}% of the time in this project ({signals} signals); check with the user before larger edits there",
                feats.join(", "),
                chance * 100.0
            ));
        }
    }
    out
}

/// The cartridge id of a project's preference cartridge.
#[must_use]
pub fn cartridge_id(project: &str) -> String {
    let clean: String = project.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    format!("pref-{}", clean.trim_start_matches('.'))
}

/// The stack use a project's preference cartridge is plugged for.
#[must_use]
pub fn use_for(project: &str) -> String {
    format!("{PREFERENCE}:project:{project}")
}

/// What one fit did.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefFit {
    pub fit_rows: usize,
    pub held_rows: usize,
    /// Held-out Brier score of the candidate, of the cartridge in use (if any), and of no cartridge.
    pub candidate: Option<f64>,
    pub current: Option<f64>,
    pub absent: f64,
    /// The cartridge written (`id@version`), when the candidate won.
    pub written: Option<String>,
    pub note: String,
}

/// The project's preference cartridge in the library now, if any, as a model.
fn current_model(lib: &Library, id: &str) -> Option<PrefModel> {
    let c = lib.cartridge(id)?;
    match lib.filter(c).ok()? {
        Filter::Settings(m) => Some(PrefModel::from_settings(&m)),
        _ => None,
    }
}

/// Fits the project's preference cartridge from its preference signals and writes it to the
/// library under `library_home` only when it beats both its absence and the cartridge in use on the
/// held-out part (module note).
/// # Errors
/// The library cannot be opened or written.
pub fn fit_preference_cartridge(library_home: &Path, memory: &Path, project: &str) -> Result<PrefFit, String> {
    let raw = nmtype::examples::read_preferences(memory)?;
    let rows = parse_rows(&raw);
    let fit: Vec<&PrefRow> = rows.iter().filter(|r| !r.held).collect();
    let held: Vec<&PrefRow> = rows.iter().filter(|r| r.held).collect();
    let mut out = PrefFit { fit_rows: fit.len(), held_rows: held.len(), candidate: None, current: None, absent: ABSENT_BRIER, written: None, note: String::new() };
    if held.len() < MIN_HELD || fit.is_empty() {
        out.note = format!("not written: {} held-out and {} fit signals with an outcome (needs {MIN_HELD} held out and one to fit)", held.len(), fit.len());
        return Ok(out);
    }
    let model = PrefModel::fit(&fit);
    let cand = brier(&held, &|f| model.predict(f)).unwrap_or(ABSENT_BRIER);
    out.candidate = Some(cand);
    let mut lib = Library::open(library_home).map_err(|e| e.to_string())?;
    let id = cartridge_id(project);
    let cur = current_model(&lib, &id);
    out.current = cur.as_ref().and_then(|m| brier(&held, &|f| m.predict(f)));
    if cand >= ABSENT_BRIER - EPS {
        out.note = format!("not written: held-out Brier {cand:.4} does not beat {ABSENT_BRIER:.4} without a cartridge");
        return Ok(out);
    }
    if let Some(c) = out.current
        && cand >= c - EPS
    {
        out.note = format!("not written: held-out Brier {cand:.4} does not beat the cartridge in use ({c:.4})");
        return Ok(out);
    }
    let version = lib.cartridges.iter().filter(|c| c.id == id).map(|c| c.version).max().unwrap_or(0) + 1;
    let spec = CartridgeSpec {
        id: id.clone(),
        version,
        family: PREFERENCE.to_owned(),
        domain: "preferences".to_owned(),
        scope: format!("project:{project}"),
        model: "settings".to_owned(),
        weights: String::new(),
        layers: None,
        name: format!("preferences of project {project}"),
        description: "How likely each circumstance (file kind, folder, tool) is to end well for the user, learned from kept and reverted edits, acceptances, corrections and refusals.".to_owned(),
        made_from: format!("{} preference signals of the project's training store ({} held out for the gate)", fit.len() + held.len(), held.len()),
        effect: format!("held-out Brier {cand:.4} against {ABSENT_BRIER:.4} without it{}", out.current.map_or(String::new(), |c| format!(" and {c:.4} with the previous version"))),
    };
    let notes = obj(vec![
        ("fit_rows", n(fit.len() as u64)),
        ("held_rows", n(held.len() as u64)),
        ("held_brier", Value::Number(format!("{cand:.6}"))),
        ("absent_brier", Value::Number(format!("{ABSENT_BRIER:.6}"))),
        ("previous_brier", out.current.map_or(Value::Null, |c| Value::Number(format!("{c:.6}")))),
        ("gate", s("held-out Brier score: written only when it beats no cartridge and the cartridge in use")),
    ]);
    let key = lib.add_cartridge(&spec, &Filter::Settings(model.to_settings()), notes, Vec::new(), false).map_err(|e| e.to_string())?;
    let stack_id = format!("{id}-stack");
    let sv = lib.stacks.iter().filter(|st| st.id == stack_id).map(|st| st.version).max().unwrap_or(0) + 1;
    let st = Stack {
        id: stack_id,
        version: sv,
        use_for: use_for(project),
        piece: None,
        cartridges: vec![key.clone()],
        description: format!("the preference cartridge of project {project}, applied after every other stack"),
        hash: String::new(),
        trial: false,
    };
    let sk = lib.add_stack(st).map_err(|e| e.to_string())?;
    lib.switch_use(&use_for(project), Some(&sk)).map_err(|e| e.to_string())?;
    lib.save().map_err(|e| e.to_string())?;
    out.note = format!("written {key}, plugged by {sk}");
    out.written = Some(key);
    Ok(out)
}

/// The preference cartridge fitter as a learner of the idle cycle.
pub struct PreferenceLearner {
    pub library_home: PathBuf,
    pub memory: PathBuf,
    pub project: String,
}

impl PreferenceLearner {
    #[must_use]
    pub fn new(library_home: &Path, memory: &Path, project: &str) -> Self {
        Self { library_home: library_home.to_path_buf(), memory: memory.to_path_buf(), project: project.to_owned() }
    }
}

impl Learner for PreferenceLearner {
    fn name(&self) -> &str {
        "preference-cartridge"
    }
    fn state_paths(&self) -> Vec<PathBuf> {
        let lib = self.library_home.join("library");
        vec![crate::library::catalogue_file(&lib), lib.join("cartridges").join(PREFERENCE).join(cartridge_id(&self.project))]
    }
    fn run(&mut self) -> Result<LearnerReport, String> {
        let f = fit_preference_cartridge(&self.library_home, &self.memory, &self.project)?;
        // Scores reported as 1 - Brier, so higher is better like the other learners'.
        let up = |b: f64| 1.0 - b;
        Ok(LearnerReport {
            name: self.name().to_owned(),
            ran: true,
            gate: "held-out preference outcomes (1 - Brier); must beat no cartridge and the cartridge in use".into(),
            held_out: f.held_rows,
            current: Some(up(f.current.unwrap_or(f.absent))),
            candidate: f.candidate.map(up),
            promoted: f.written.is_some(),
            note: f.note,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: &str, path: &str, split: &str) -> Value {
        obj(vec![("kind", s(kind)), ("paths", Value::Array(vec![s(path)])), ("split", s(split))])
    }

    #[test]
    fn rows_carry_outcomes_and_plain_labels_only() {
        let rows = vec![
            row("edit-kept", "crates/a/src/x.rs", "fit"),
            row("edit-reverted", "docs/notes.md", "held-out"),
            row("stated-rule", "x.rs", "fit"),
            obj(vec![("kind", s("refusal")), ("tool", s("Bash")), ("split", s("tune"))]),
            obj(vec![("kind", s("refusal")), ("tool", s("rm -rf /")), ("split", s("fit"))]),
        ];
        let p = parse_rows(&rows);
        assert_eq!(p.len(), 4, "a stated rule says nothing about an outcome");
        assert_eq!(p.first().map(|r| r.features.clone()), Some(vec!["any".into(), "ext:rs".into(), "dir:crates".into()]));
        assert!(p.get(1).is_some_and(|r| r.held && !r.positive));
        assert!(p.get(2).is_some_and(|r| r.features.contains(&"tool:Bash".to_owned())));
        assert_eq!(p.get(3).map(|r| r.features.len()), Some(1), "a label that is not plain is dropped");
    }

    #[test]
    fn the_model_round_trips_through_settings_and_predicts_by_circumstance() {
        let a = PrefRow { features: vec!["any".into(), "ext:rs".into()], positive: true, held: false };
        let b = PrefRow { features: vec!["any".into(), "ext:md".into()], positive: false, held: false };
        let rows = [&a, &a, &a, &b, &b];
        let m = PrefModel::fit(&rows);
        assert_eq!(m.global, (3, 5));
        assert!(m.predict(&["ext:rs".into()]) > 0.7 && m.predict(&["ext:md".into()]) < 0.3);
        assert!((m.predict(&["ext:py".into()]) - rate((3, 5))).abs() < 1e-12, "an unseen circumstance takes the overall rate");
        assert_eq!(PrefModel::from_settings(&m.to_settings()), m);
        assert_eq!(brier(&[], &|_| 0.5), None);
        assert!((brier(&[&a, &b], &|_| 0.5).unwrap_or_default() - ABSENT_BRIER).abs() < 1e-12);
    }

    #[test]
    fn guidance_cautions_only_files_whose_seen_circumstances_end_badly() {
        let bad = PrefRow { features: vec!["any".into(), "ext:md".into(), "dir:docs".into()], positive: false, held: false };
        let good = PrefRow { features: vec!["any".into(), "ext:rs".into(), "dir:crates".into()], positive: true, held: false };
        let m = PrefModel::fit(&[&bad, &bad, &bad, &good, &good, &good]);
        let g = guidance(&m, &["docs/a.md".into(), "docs/a.md".into(), "crates/x/src/lib.rs".into(), "tools/run.ps1".into()]);
        assert_eq!(g.len(), 1, "{g:?}");
        assert!(g.first().is_some_and(|l| l.contains("docs/a.md") && l.contains("ext:md")));
        assert!(guidance(&PrefModel::default(), &["docs/a.md".into()]).is_empty());
    }

    fn home(name: &str) -> PathBuf {
        let d = nmtype::nano_root().join("scratch").join("unit-tests").join(format!("nmlearn-pref-{name}-{}-{}", std::process::id(), nmtype::now_ms()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(nmtype::examples::dir(&d.join("mem"))).expect("dir");
        d
    }

    fn write_prefs(h: &Path, rows: &[Value]) {
        let text: String = rows.iter().map(|r| r.encode().unwrap_or_default() + "\n").collect();
        std::fs::write(nmtype::examples::dir(&h.join("mem")).join(nmtype::examples::PREFERENCES), text).expect("write");
    }

    #[test]
    fn a_cartridge_is_written_only_when_it_beats_its_absence_and_the_one_in_use() {
        let h = home("gate");
        let mem = h.join("mem");
        // Too little held out: nothing written.
        write_prefs(&h, &[row("edit-kept", "src/a.rs", "fit")]);
        let f = fit_preference_cartridge(&h, &mem, "proj").expect("fit");
        assert!(f.written.is_none() && f.candidate.is_none());
        // Rust edits are kept, Markdown edits reverted, in both parts: the candidate wins.
        let mut rows = Vec::new();
        for _ in 0..6 {
            rows.push(row("edit-kept", "src/a.rs", "fit"));
            rows.push(row("edit-reverted", "docs/b.md", "fit"));
        }
        for _ in 0..3 {
            rows.push(row("edit-kept", "src/c.rs", "held-out"));
            rows.push(row("edit-reverted", "docs/d.md", "held-out"));
        }
        write_prefs(&h, &rows);
        let f = fit_preference_cartridge(&h, &mem, "proj").expect("fit");
        assert_eq!(f.written.as_deref(), Some("pref-proj@1"), "{f:?}");
        let lib = Library::open(&h).expect("library");
        assert_eq!(lib.active.get(&use_for("proj")).map(String::as_str), Some("pref-proj-stack@1"));
        assert!(current_model(&lib, "pref-proj").is_some());
        // The same signals again: no gain over the cartridge in use, nothing new written.
        let again = fit_preference_cartridge(&h, &mem, "proj").expect("fit");
        assert!(again.written.is_none() && again.current.is_some(), "{again:?}");
        // Signals that carry no information: no gain over no cartridge.
        let h2 = home("flat");
        let mut flat = Vec::new();
        for _ in 0..4 {
            flat.push(row("edit-kept", "src/a.rs", "fit"));
            flat.push(row("edit-reverted", "src/a.rs", "fit"));
            flat.push(row("edit-kept", "src/a.rs", "held-out"));
            flat.push(row("edit-reverted", "src/a.rs", "held-out"));
        }
        write_prefs(&h2, &flat);
        let f2 = fit_preference_cartridge(&h2, &h2.join("mem"), "proj").expect("fit");
        assert!(f2.written.is_none(), "{f2:?}");
    }

    #[test]
    fn the_learner_names_its_state_for_the_snapshot() {
        let l = PreferenceLearner::new(Path::new("H"), Path::new("M"), "p.x");
        let paths = l.state_paths();
        assert!(paths.iter().any(|p| p.ends_with("catalogue.json")));
        assert!(paths.iter().any(|p| p.ends_with("pref-p.x")));
    }
}
