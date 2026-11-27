//! The GCX context map (CM1, 2026-11-09; design by Luis Jake Gabriel III). One map per question:
//! the segments the tentacles were given (whole folders or files, never a split file), what
//! each segment's reading found and at which level (1 overview, 2 a focus or job reader,
//! 3 a refining drone), the findings by position with every unit and reader that reported
//! them, merges and conflicts, and which units of the Neuromorph read what. Coordination
//! reads it: the no-overlap and no-gap gates (LG0, LG0b, LG0c), and next-best-view refinement
//! (salience times the unread share). It maps context only; parameters are the Parameter
//! Map's, and the two meet only through the Map Coordinator's summaries (`nmapi::coordinator`).
use nmjson::Value;
use nmtype::{Pack, Packet, Region};
use std::collections::BTreeMap;

/// One segment: a territory region given to one tentacle.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub territory: u8,
    pub path: String,
    pub files: usize,
}

/// What reading a segment has produced so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Coverage {
    /// Highest level that read it: 0 not read, 1 overview, 2 focus or job reader, 3 refining drone.
    pub level: u8,
    pub findings: usize,
    /// Sum of the findings' confidences.
    pub salience: f32,
    /// Distinct files of the segment with a finding.
    pub files_hit: usize,
    /// Findings that reached the final pack.
    pub kept: usize,
    /// Units of the Neuromorph that read the segment (tentacles, drone tips, refining drones).
    pub units: Vec<String>,
}

/// The no-overlap and no-gap gates of one map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gates {
    /// LG0: segments claimed by more than one territory, or nested inside another segment.
    pub overlap: usize,
    /// LG0b: files outside every segment.
    pub unassigned: usize,
    /// LG0c: segments whose path ends inside a file or folder name (would split a unit).
    pub split: usize,
    /// LG0d (2026-11-10): chunks held by more than one tentacle window (no crossover).
    pub crossover: usize,
}

/// One tentacle window of one question (the window axis, 2026-11-10): which chunks one
/// tentacle held under which lens and pose, how deeply it read them and what it found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WindowRecord {
    pub tentacle: u8,
    /// The lens that filled the window (`nmsubs::lens::NAMES`).
    pub lens: String,
    /// The pose the window was read under (`plain` or `focused`).
    pub pose: String,
    /// Substrate chunk ids held, in window order.
    pub chunks: Vec<usize>,
    pub tokens: usize,
    /// Read level: 0 held only, 1 scored by the lane, 2 read by the drone heads.
    pub level: u8,
    /// Candidates the window passed on.
    pub findings: usize,
    /// Best calibrated chunk score, and the no-information baseline it is measured against.
    pub best: f32,
    pub baseline: f32,
    /// True when no chunk scored above the baseline: the window reports that it holds nothing.
    pub no_information: bool,
    /// Distinct files of the window's chunks.
    pub files: Vec<String>,
}

/// The context map of one question.
#[derive(Clone, Debug)]
pub struct GMap {
    pub question: String,
    pub focus: Vec<String>,
    pub segments: Vec<Segment>,
    pub cover: Vec<Coverage>,
    /// Position (`file:line`, or `history:file`) to the (unit, kind) pairs that reported it.
    pub findings: BTreeMap<String, Vec<(String, String)>>,
    /// Positions reported by more than one unit or reader (merged, every source kept).
    pub merged: usize,
    /// Keys with two different texts (kept side by side, never averaged).
    pub conflicts: Vec<String>,
    pub gates: Gates,
    pub rounds: u8,
    /// Segments chosen for refinement, in order, and how they were chosen.
    pub refined: Vec<usize>,
    pub refine_mode: String,
    /// The Map Coordinator's decision for this question (rules fired, budgets).
    pub coordinator: Value,
    /// The tentacle windows of this question, when the tentacle pass ran.
    pub windows: Vec<WindowRecord>,
    /// Stage 5 WP-C4 W2: the project outline the writer wrote at idle, when one is held.
    pub outline: Option<String>,
    keys: BTreeMap<String, String>,
    files_of: Vec<Vec<String>>,
}

impl Default for GMap {
    fn default() -> Self {
        Self {
            question: String::new(),
            focus: Vec::new(),
            segments: Vec::new(),
            cover: Vec::new(),
            findings: BTreeMap::new(),
            merged: 0,
            conflicts: Vec::new(),
            gates: Gates::default(),
            rounds: 0,
            refined: Vec::new(),
            refine_mode: "off".into(),
            coordinator: Value::Null,
            windows: Vec::new(),
            outline: None,
            keys: BTreeMap::new(),
            files_of: Vec::new(),
        }
    }
}

/// The file a packet bears on: its source path, or for history packets the file in the key.
#[must_use]
pub fn file_of(p: &Packet) -> String {
    let src = &p.provenance().source;
    if src.starts_with("git:") {
        return p.key().trim_start_matches("history:").to_owned();
    }
    match src.rsplit_once(':') {
        Some((f, l)) if l.chars().all(|c| c.is_ascii_digit()) => f.to_owned(),
        _ => src.clone(),
    }
}

impl GMap {
    /// A map over the territories of one question, with its gates computed against `files`.
    #[must_use]
    pub fn new(question: &str, focus: &[String], territories: &[Vec<Region>], files: &[String]) -> Self {
        let mut segments = Vec::new();
        for t in territories {
            for r in t {
                segments.push(Segment { territory: r.territory, path: r.path.clone(), files: 0 });
            }
        }
        let mut files_of: Vec<Vec<String>> = vec![Vec::new(); segments.len()];
        let mut gates = Gates::default();
        for f in files {
            let owners: Vec<usize> = segments.iter().enumerate().filter(|(_, s)| f.starts_with(s.path.as_str())).map(|(i, _)| i).collect();
            match owners.as_slice() {
                [] => gates.unassigned += 1,
                [i] => {
                    if let Some(v) = files_of.get_mut(*i) {
                        v.push(f.clone());
                    }
                }
                _ => {}
            }
            // A segment path that is neither a folder (ends with '/') nor the whole file splits a unit.
            gates.split += owners.iter().filter(|i| segments.get(**i).is_some_and(|s| !s.path.ends_with('/') && s.path != *f)).count();
        }
        for (i, a) in segments.iter().enumerate() {
            if segments
                .iter()
                .enumerate()
                .any(|(j, b)| j != i && (a.path == b.path || (b.path.ends_with('/') && a.path.starts_with(b.path.as_str()))))
            {
                gates.overlap += 1;
            }
        }
        for (s, f) in segments.iter_mut().zip(&files_of) {
            s.files = f.len();
        }
        let n = segments.len();
        Self {
            question: question.to_owned(),
            focus: focus.to_vec(),
            segments,
            cover: vec![Coverage::default(); n],
            gates,
            files_of,
            refine_mode: "off".into(),
            ..Self::default()
        }
    }
    /// The segment owning a file (the one whose path it starts with).
    #[must_use]
    pub fn segment_of(&self, file: &str) -> Option<usize> {
        self.segments.iter().position(|s| file.starts_with(s.path.as_str()))
    }
    /// Record what one reading round produced, at `level`; each packet's unit is its origin.
    pub fn record(&mut self, level: u8, packets: &[Packet]) {
        self.rounds = self.rounds.max(level);
        let mut hit: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for p in packets.iter().filter(|p| p.kind() != "question") {
            let file = file_of(p);
            let unit = p.provenance().origin.clone();
            let pos = if p.provenance().source.starts_with("git:") { format!("history:{file}") } else { p.provenance().source.clone() };
            let seen = self.findings.entry(pos).or_default();
            if !seen.is_empty() && !seen.iter().any(|(u, k)| *u == unit && k == p.kind()) {
                self.merged += 1;
            }
            seen.push((unit.clone(), p.kind().to_owned()));
            match self.keys.get(p.key()) {
                Some(t) if t != p.text() && !self.conflicts.iter().any(|k| k == p.key()) => self.conflicts.push(p.key().to_owned()),
                None => {
                    self.keys.insert(p.key().to_owned(), p.text().to_owned());
                }
                _ => {}
            }
            let Some(i) = self.segment_of(&file) else { continue };
            hit.entry(i).or_default().push(file);
            if let Some(c) = self.cover.get_mut(i) {
                c.level = c.level.max(level);
                c.findings += 1;
                c.salience += p.get_confidence().clamp(0.0, 1.0);
                if !c.units.contains(&unit) {
                    c.units.push(unit);
                }
            }
        }
        for (i, mut fs) in hit {
            fs.sort();
            fs.dedup();
            if let Some(c) = self.cover.get_mut(i) {
                c.files_hit = c.files_hit.max(fs.len());
            }
        }
    }
    /// Record the tentacle windows: the crossover gate (a chunk id in more than one window) and
    /// segment coverage from the files each window held (the unit is the window's tentacle).
    pub fn record_windows(&mut self, windows: Vec<WindowRecord>) {
        let mut held: BTreeMap<usize, usize> = BTreeMap::new();
        for w in &windows {
            let mut own = w.chunks.clone();
            own.sort_unstable();
            own.dedup();
            // A chunk repeated inside one window is crossover too (it would be read twice).
            self.gates.crossover += w.chunks.len() - own.len();
            for c in own {
                *held.entry(c).or_default() += 1;
            }
        }
        self.gates.crossover += held.values().filter(|n| **n > 1).count();
        for w in &windows {
            if w.level == 0 {
                continue;
            }
            self.rounds = self.rounds.max(w.level);
            let unit = format!("nmpod.tentacle#t{}/lens:{}/pose:{}", w.tentacle, w.lens, w.pose);
            for f in &w.files {
                let Some(i) = self.segment_of(f) else { continue };
                if let Some(c) = self.cover.get_mut(i) {
                    c.level = c.level.max(w.level);
                    if !c.units.contains(&unit) {
                        c.units.push(unit.clone());
                    }
                }
            }
        }
        self.windows = windows;
    }
    /// Count the findings each segment contributed to the final pack.
    pub fn mark_kept(&mut self, pack: &Pack) {
        for p in &pack.packets {
            if let Some(i) = self.segment_of(&file_of(p))
                && let Some(c) = self.cover.get_mut(i)
            {
                c.kept += 1;
            }
        }
    }
    /// Share of a segment not yet read at file level (at least 0.1, so a fully hit segment can
    /// still be refined at a finer level).
    fn unread(&self, i: usize) -> f32 {
        let (Some(s), Some(c)) = (self.segments.get(i), self.cover.get(i)) else { return 0.0 };
        if s.files == 0 {
            return 0.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "file counts are small")]
        let share = 1.0 - c.files_hit as f32 / s.files as f32;
        share.max(0.1)
    }
    /// Next-best-view: up to `k` segments by salience times unread share (times 1.5 for a
    /// territory whose tentacle asked for a subordinate drone), never one already refined.
    #[must_use]
    pub fn next_best_view(&self, k: usize, asked: &[u8]) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..self.segments.len())
            .filter(|i| !self.refined.contains(i))
            .filter_map(|i| {
                let c = self.cover.get(i)?;
                let s = self.segments.get(i)?;
                let boost = if asked.contains(&s.territory) { 1.5 } else { 1.0 };
                (c.salience > 0.0).then(|| (c.salience * self.unread(i) * boost, i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }
    /// The uniform baseline for refinement: `k` segments spread evenly over the map, whatever
    /// they hold (spec gate P2 compares guided with uniform poses).
    #[must_use]
    pub fn uniform(&self, k: usize) -> Vec<usize> {
        let n = self.segments.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        (0..k).map(|j| j * n / k).collect()
    }
    /// The region of one segment, for a refining drone.
    #[must_use]
    pub fn region(&self, i: usize) -> Option<Region> {
        let s = self.segments.get(i)?;
        #[expect(clippy::cast_precision_loss, reason = "file counts are small")]
        let priority = s.files as f32;
        Some(Region { territory: s.territory, path: s.path.clone(), start: 0, end: s.files, priority })
    }
    /// Share of segments with any reading (coverage of the map).
    #[must_use]
    pub fn coverage(&self) -> f32 {
        let n = self.segments.iter().filter(|s| s.files > 0).count();
        if n == 0 {
            return 0.0;
        }
        let read = self.segments.iter().zip(&self.cover).filter(|(s, c)| s.files > 0 && c.level > 0).count();
        #[expect(clippy::cast_precision_loss, reason = "segment counts are small")]
        let share = read as f32 / n as f32;
        share
    }
    /// Segments with salience above zero.
    #[must_use]
    pub fn salient(&self) -> usize {
        self.cover.iter().filter(|c| c.salience > 0.0).count()
    }
    /// Files in the map's scope.
    #[must_use]
    pub fn scope_files(&self) -> usize {
        self.files_of.iter().map(Vec::len).sum()
    }
    /// The map as JSON: gates, coverage, the read segments with their units, refinement and
    /// the coordinator's decision.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let n = |x: usize| Value::Number(x.to_string());
        let f = |x: f32| Value::Number(format!("{x:.3}"));
        let s = |t: &str| Value::String(t.to_owned());
        let read: Vec<Value> = self
            .segments
            .iter()
            .zip(&self.cover)
            .enumerate()
            .filter(|(_, (_, c))| c.level > 0)
            .map(|(i, (sg, c))| {
                Value::Object(
                    [
                        ("segment", n(i)),
                        ("path", s(&sg.path)),
                        ("territory", n(usize::from(sg.territory))),
                        ("files", n(sg.files)),
                        ("level", n(usize::from(c.level))),
                        ("findings", n(c.findings)),
                        ("salience", f(c.salience)),
                        ("files_hit", n(c.files_hit)),
                        ("kept", n(c.kept)),
                        ("units", Value::Array(c.units.iter().map(|u| s(u)).collect())),
                    ]
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v))
                    .collect(),
                )
            })
            .collect();
        let windows: Vec<Value> = self
            .windows
            .iter()
            .map(|w| {
                Value::Object(
                    [
                        ("tentacle", n(usize::from(w.tentacle))),
                        ("lens", s(&w.lens)),
                        ("pose", s(&w.pose)),
                        ("chunks", n(w.chunks.len())),
                        ("tokens", n(w.tokens)),
                        ("files", n(w.files.len())),
                        ("level", n(usize::from(w.level))),
                        ("findings", n(w.findings)),
                        ("best", f(w.best)),
                        ("baseline", f(w.baseline)),
                        ("no_information", Value::Bool(w.no_information)),
                    ]
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v))
                    .collect(),
                )
            })
            .collect();
        Value::Object(
            [
                ("segments", n(self.segments.len())),
                ("scope_files", n(self.scope_files())),
                ("coverage", f(self.coverage())),
                (
                    "gates",
                    Value::Object(
                        [
                            ("overlap", n(self.gates.overlap)),
                            ("unassigned", n(self.gates.unassigned)),
                            ("split", n(self.gates.split)),
                            ("crossover", n(self.gates.crossover)),
                        ]
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v))
                        .collect(),
                    ),
                ),
                ("positions", n(self.findings.len())),
                ("merged", n(self.merged)),
                ("conflicts", Value::Array(self.conflicts.iter().map(|k| s(k)).collect())),
                ("rounds", n(usize::from(self.rounds))),
                ("focus", Value::Array(self.focus.iter().map(|j| s(j)).collect())),
                ("refine_mode", s(&self.refine_mode)),
                ("refined", Value::Array(self.refined.iter().filter_map(|i| self.segments.get(*i)).map(|sg| s(&sg.path)).collect())),
                ("coordinator", self.coordinator.clone()),
                ("read", Value::Array(read)),
                ("windows", Value::Array(windows)),
                ("outline", self.outline.as_deref().map_or(Value::Null, s)),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect(),
        )
    }
}

/// Self-check of the `gmap` sub-component: gates on a clean and a faulty partition, coverage
/// and merges from a round, next-best-view prefers a salient, partly read segment.
pub(crate) fn check() -> Result<(), String> {
    use nmtype::{Provenance, Status};
    let files: Vec<String> = ["a/x.rs", "a/y.rs", "b/z.rs", "c.md"].iter().map(|s| (*s).to_owned()).collect();
    let r = |t: u8, p: &str| Region { territory: t, path: p.into(), start: 0, end: 1, priority: 1.0 };
    let good = GMap::new("q", &[], &[vec![r(0, "a/")], vec![r(1, "b/"), r(1, "c.md")]], &files);
    if good.gates != Gates::default() {
        return Err(format!("nmbrain.gmap: clean partition gates {:?}", good.gates));
    }
    let bad = GMap::new("q", &[], &[vec![r(0, "a/"), r(0, "a/x")], vec![r(1, "a/")]], &files);
    if bad.gates.overlap == 0 || bad.gates.unassigned != 2 || bad.gates.split == 0 {
        return Err(format!("nmbrain.gmap: faulty partition gates {:?}", bad.gates));
    }
    let mut m = good;
    let pk = |src: &str, unit: &str, kind: &str| {
        Packet::new(kind, &format!("{kind}:{src}"), "t", Provenance::new(src, unit, 1, 0))
            .status(Status::Observed)
            .confidence(0.8)
            .evidence(vec![src.into()])
    };
    m.record(
        1,
        &[pk("a/x.rs:3", "nmpod.unit#t0", "grep"), pk("a/x.rs:3", "nmpod.unit#t0", "symbol"), pk("b/z.rs:1", "nmpod.unit#t1", "grep")],
    );
    if m.merged != 1 || m.cover.first().map(|c| c.files_hit) != Some(1) || (m.coverage() - 2.0 / 3.0).abs() > 1e-3 {
        return Err(format!("nmbrain.gmap: record merged {} coverage {}", m.merged, m.coverage()));
    }
    if m.next_best_view(1, &[]).first() != Some(&0) || m.uniform(2) != vec![0, 1] {
        return Err(format!("nmbrain.gmap: next best view {:?}", m.next_best_view(2, &[])));
    }
    // Window axis: clean windows pass the crossover gate and mark their segments read; a chunk
    // held twice fails it.
    let win = |t: u8, chunks: &[usize], file: &str| WindowRecord {
        tentacle: t,
        lens: "meaning".into(),
        pose: "focused".into(),
        chunks: chunks.to_vec(),
        tokens: 10,
        level: 2,
        findings: 1,
        best: 0.4,
        files: vec![file.to_owned()],
        ..WindowRecord::default()
    };
    m.record_windows(vec![win(0, &[0, 1], "a/y.rs"), win(1, &[2], "c.md")]);
    if m.gates.crossover != 0 || (m.coverage() - 1.0).abs() > 1e-3 || m.cover.first().map(|c| c.level) != Some(2) {
        return Err(format!("nmbrain.gmap: clean windows crossover {} coverage {}", m.gates.crossover, m.coverage()));
    }
    let mut x = GMap::new("q", &[], &[vec![r(0, "a/")], vec![r(1, "b/"), r(1, "c.md")]], &files);
    x.record_windows(vec![win(0, &[0, 1], "a/y.rs"), win(1, &[1, 2], "c.md")]);
    if x.gates.crossover != 1 {
        return Err(format!("nmbrain.gmap: a chunk in two windows gives crossover {}", x.gates.crossover));
    }
    Ok(())
}
