//! Scoreboard, first form (2026-11-10; Gavriel Technologies; direction of Luis Jake Gabriel
//! III: assemble the whole system first, then improve the part measured worst, with the state
//! of every part visible at a glance; plan section 11). Data: `designs/scoreboard.json`, one
//! row per part of the specification with state, latest measurement and its record, target and
//! its source. Components: `parse` (rows from JSON), `Part::distance` (distance to target),
//! `sorted` (worst first), `render` (a Markdown table), `next_step` (the worst part by distance
//! among parts at least in first form, the parts with no distance yet, and the absent parts in
//! dependency order: what the planning session uses to choose the next step), `cli` (prints
//! the table and writes a new `runs/scoreboard-<utc ms>.md` each run; registration in the
//! binary is one line: `Some("scoreboard") => nmtool::scoreboard::cli(&nano_root())`).
use nmjson::Value;
use std::path::{Path, PathBuf};

/// The scoreboard file, relative to the mini folder.
pub const FILE: &str = "designs/scoreboard.json";

/// A part's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    Absent,
    FirstForm,
    Refined,
}
impl State {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "absent" => Ok(Self::Absent),
            "first form" => Ok(Self::FirstForm),
            "refined" => Ok(Self::Refined),
            other => Err(format!("state {other:?} is not absent, first form or refined")),
        }
    }
    /// The word the data file uses.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::FirstForm => "first form",
            Self::Refined => "refined",
        }
    }
}

/// One row.
#[derive(Clone, Debug)]
pub struct Part {
    pub id: String,
    pub part: String,
    pub spec: String,
    pub state: State,
    pub value: Option<f64>,
    pub measured: String,
    pub record: String,
    pub target: Option<f64>,
    pub target_text: String,
    pub target_source: String,
    pub higher_is_better: bool,
    pub depends_on: Vec<String>,
    pub plan_step: String,
    pub plan_order: u64,
}
impl Part {
    /// Distance to target in 0..=1 (0 when met); none without a measured value and a target.
    #[must_use]
    pub fn distance(&self) -> Option<f64> {
        let (v, t) = (self.value?, self.target?);
        let d = if self.higher_is_better {
            if t <= 0.0 { 0.0 } else { (t - v) / t }
        } else if v <= 0.0 {
            0.0
        } else {
            (v - t) / v
        };
        Some(d.clamp(0.0, 1.0))
    }
}

fn num(v: Option<&Value>) -> Result<Option<f64>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n.parse::<f64>().map(Some).map_err(|e| format!("number {n}: {e}")),
        Some(_) => Err("a value must be a number or null".into()),
    }
}

fn st(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

/// Rows from the scoreboard JSON.
/// # Errors
/// A missing field, an unknown state, a duplicate id or an unknown dependency.
pub fn parse(v: &Value) -> Result<Vec<Part>, String> {
    let mut out: Vec<Part> = Vec::new();
    for p in v.get("parts").and_then(Value::array).map_err(|e| format!("parts: {e}"))? {
        let id = p.get("id").and_then(Value::string).map_err(|e| format!("part id: {e}"))?.to_owned();
        let m = p.get("measurement").map_err(|e| format!("{id}: {e}"))?;
        let t = p.get("target").map_err(|e| format!("{id}: {e}"))?;
        let deps = p
            .get("depends_on")
            .and_then(Value::array)
            .map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
            .unwrap_or_default();
        let part = Part {
            state: State::parse(&st(p, "state")).map_err(|e| format!("{id}: {e}"))?,
            value: num(m.get("value").ok()).map_err(|e| format!("{id}: {e}"))?,
            measured: st(m, "text"),
            record: st(m, "record"),
            target: num(t.get("value").ok()).map_err(|e| format!("{id}: {e}"))?,
            target_text: st(t, "text"),
            target_source: st(t, "source"),
            higher_is_better: st(p, "direction") != "lower",
            part: st(p, "part"),
            spec: st(p, "spec"),
            depends_on: deps,
            plan_step: st(p, "plan_step"),
            plan_order: p.get("plan_order").and_then(Value::integer).unwrap_or(u64::MAX),
            id,
        };
        if out.iter().any(|x| x.id == part.id) {
            return Err(format!("part {} twice", part.id));
        }
        if part.value.is_some() && (part.record.is_empty() || part.record == "none") {
            return Err(format!("{}: a measurement needs its record", part.id));
        }
        out.push(part);
    }
    for p in &out {
        if let Some(d) = p.depends_on.iter().find(|d| !out.iter().any(|x| &&x.id == d)) {
            return Err(format!("{}: unknown dependency {d}", p.id));
        }
    }
    Ok(out)
}

/// Load `designs/scoreboard.json` under `root`.
/// # Errors
/// Unreadable or invalid file.
pub fn load(root: &Path) -> Result<Vec<Part>, String> {
    let p = root.join(FILE);
    let bytes = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    parse(&nmjson::parse(&bytes).map_err(|e| format!("{FILE}: {e}"))?)
}

/// Absent parts in dependency order (a part after every absent part it depends on; ties by
/// plan order, then id). A cycle is broken by plan order and named in the result.
#[must_use]
pub fn absent_order(parts: &[Part]) -> Vec<String> {
    let mut left: Vec<&Part> = parts.iter().filter(|p| p.state == State::Absent).collect();
    left.sort_by(|a, b| a.plan_order.cmp(&b.plan_order).then(a.id.cmp(&b.id)));
    let mut out: Vec<String> = Vec::new();
    while !left.is_empty() {
        let ready = left.iter().position(|p| p.depends_on.iter().all(|d| !left.iter().any(|x| &x.id == d))).unwrap_or(0);
        let p = left.remove(ready);
        out.push(p.id.clone());
    }
    out
}

/// What the planning session reads to choose the next step.
#[derive(Clone, Debug, PartialEq)]
pub struct Next {
    /// The part at least in first form farthest from its target, with the distance.
    pub worst: Option<(String, f64)>,
    /// Parts at least in first form with no distance yet (no measurement or no target).
    pub unmeasured: Vec<String>,
    /// Absent parts in dependency order.
    pub absent: Vec<String>,
}

/// The worst part by distance among parts at least in first form, the ones with no distance,
/// and the absent parts in dependency order.
#[must_use]
pub fn next_step(parts: &[Part]) -> Next {
    let present: Vec<&Part> = parts.iter().filter(|p| p.state != State::Absent).collect();
    let worst = present.iter().filter_map(|p| p.distance().map(|d| (p.id.clone(), d))).fold(None::<(String, f64)>, |best, x| match best {
        Some(b) if b.1 >= x.1 => Some(b),
        _ => Some(x),
    });
    let unmeasured = present.iter().filter(|p| p.distance().is_none()).map(|p| p.id.clone()).collect();
    Next { worst, unmeasured, absent: absent_order(parts) }
}

/// Worst first: absent parts in dependency order, then parts with no distance, then the rest by
/// distance, largest first.
#[must_use]
pub fn sorted(parts: &[Part]) -> Vec<&Part> {
    let order = absent_order(parts);
    let mut v: Vec<&Part> = parts.iter().collect();
    v.sort_by(|a, b| {
        let rank = |p: &Part| match (p.state, p.distance()) {
            (State::Absent, _) => (0, order.iter().position(|x| *x == p.id).unwrap_or(usize::MAX), 0.0),
            (_, None) => (1, 0, 0.0),
            (_, Some(d)) => (2, 0, -d),
        };
        let (x, y) = (rank(a), rank(b));
        x.0.cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.total_cmp(&y.2)).then(a.id.cmp(&b.id))
    });
    v
}

fn cell(s: &str) -> String {
    s.replace('|', "/").replace('\n', " ")
}

/// The table as Markdown, worst first, with the next-step summary under it.
#[must_use]
pub fn render(parts: &[Part], stamp: &str) -> String {
    let mut s = format!(
        "# Scoreboard {stamp}\n\nGavriel Technologies. Rendered by nmtool::scoreboard from `{FILE}`; worst first (absent parts in dependency order, then parts with no distance, then by distance to target). Distance: (target - value) / target when higher is better, (value - target) / value when lower is better.\n\n| Part | Spec | State | Latest measurement | Record | Target | Distance |\n|---|---|---|---|---|---|---|\n"
    );
    for p in sorted(parts) {
        let m = p.value.map_or_else(|| cell(&p.measured), |v| format!("{v} ({})", cell(&p.measured)));
        let t = p.target.map_or_else(|| cell(&p.target_text), |v| format!("{v} ({}; {})", cell(&p.target_text), cell(&p.target_source)));
        let d = p.distance().map_or_else(|| "none".to_owned(), |d| format!("{d:.3}"));
        s.push_str(&format!(
            "| {} (`{}`) | {} | {} | {m} | {} | {t} | {d} |\n",
            cell(&p.part),
            p.id,
            cell(&p.spec),
            p.state.as_str(),
            cell(&p.record)
        ));
    }
    let n = next_step(parts);
    let worst = n.worst.map_or_else(|| "none".to_owned(), |(id, d)| format!("{id} (distance {d:.3})"));
    s.push_str(&format!("\nWorst part by distance (first form or refined): {worst}.\nAt least first form with no distance yet: {}.\nAbsent, in dependency order: {}.\n", if n.unmeasured.is_empty() { "none".into() } else { n.unmeasured.join(", ") }, if n.absent.is_empty() { "none".into() } else { n.absent.join(", ") }));
    s
}

fn now_ms() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())
}

/// Render and write a new `scoreboard-<utc ms>.md` in `dir`; returns the path and the text.
/// # Errors
/// The file cannot be written or already exists.
pub fn write_record(parts: &[Part], dir: &Path) -> Result<(PathBuf, String), String> {
    let ms = now_ms();
    let text = render(parts, &format!("(UTC ms {ms})"));
    let path = dir.join(format!("scoreboard-{ms}.md"));
    if path.exists() {
        return Err(format!("{} exists; never overwritten", path.display()));
    }
    std::fs::write(&path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((path, text))
}

/// The command face: print the table and write a new record under `runs/`.
#[must_use]
pub fn cli(root: &Path) -> std::process::ExitCode {
    match load(root).and_then(|p| write_record(&p, &root.join("runs"))) {
        Ok((path, text)) => {
            print!("{text}");
            println!("written {}", path.display());
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            println!("scoreboard: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Section check: the real file loads; on a fixture the rows sort worst first, the worst part
/// is named by distance among parts at least in first form, absent parts come out in
/// dependency order, and a record is written as a new file.
/// # Errors
/// One line naming what failed.
pub fn check() -> Result<(), String> {
    let real = load(&nmtype::nano_root())?;
    if real.is_empty() {
        return Err("the scoreboard has no parts".into());
    }
    let row = |id: &str, state: &str, v: &str, t: &str, dir: &str, deps: &str, order: u32| {
        format!(
            "{{\"id\": \"{id}\", \"part\": \"{id}\", \"spec\": \"x\", \"state\": \"{state}\", \"measurement\": {{\"value\": {v}, \"text\": \"m\", \"record\": \"runs/r.txt\"}}, \"target\": {{\"value\": {t}, \"text\": \"t\", \"source\": \"s\"}}, \"direction\": \"{dir}\", \"depends_on\": [{deps}], \"plan_step\": \"P\", \"plan_order\": {order}}}"
        )
    };
    let rows = [
        row("near", "first form", "18", "20", "higher", "", 1),
        row("slow", "first form", "100", "25", "lower", "", 1),
        row("blind", "first form", "null", "10", "higher", "", 1),
        row("late", "absent", "null", "null", "higher", "\"early\"", 1),
        row("early", "absent", "null", "null", "higher", "", 5),
        row("done", "refined", "20", "20", "higher", "", 1),
    ];
    let v = nmjson::parse(format!("{{\"parts\": [{}]}}", rows.join(", ")).as_bytes()).map_err(|e| e.to_string())?;
    let parts = parse(&v)?;
    let n = next_step(&parts);
    let worst = n.worst.clone().ok_or("no worst part")?;
    if worst.0 != "slow" || (worst.1 - 0.75).abs() > 1e-9 || n.unmeasured != ["blind"] || n.absent != ["early", "late"] {
        return Err(format!("next step on the fixture: {n:?}"));
    }
    let order: Vec<&str> = sorted(&parts).iter().map(|p| p.id.as_str()).collect();
    if order != ["early", "late", "blind", "slow", "near", "done"] {
        return Err(format!("sort order on the fixture: {order:?}"));
    }
    let dir = crate::scratch_dir("scoreboard")?;
    let (path, text) = write_record(&parts, &dir)?;
    if !path.is_file() || !text.contains("Worst part by distance (first form or refined): slow (distance 0.750)") {
        return Err(format!("record {}: {text}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn section_check() {
        super::check().unwrap();
    }
    #[test]
    #[ignore = "first render: writes a new runs/scoreboard-<utc ms>.md"]
    fn first_render() {
        let root = nmtype::nano_root();
        let (path, text) = super::write_record(&super::load(&root).unwrap(), &root.join("runs")).unwrap();
        print!("{text}");
        println!("written {}", path.display());
    }
}
