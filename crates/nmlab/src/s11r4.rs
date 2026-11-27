//! S11 revision 4 (runbook `runs/2026-11-08-S11-runbook-r4.md`): the readout learner grown
//! on verified cases from Claude's code. Parts: `mine` (cases from the mini's own source,
//! verified, the rest quarantined), `pool <seconds>` (features in shards under 60 s each, one
//! process, the model loaded once, append-only cache), `learn` (fit, per-scope 3a/3b, benefit
//! gate, shuffled control through the same rule, unit written once), `decay` (the life ledger).
use super::{bad, feature, field, line, n, obj, s};
use nmjson::Value;
use nmlearn::readout::{self, Gate, Life, Paired, SCOPES};
use nmlearn::rls::Rls;
use nmlearn::{UnitCard, UnitWeights, Units};
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Mining caps per scope (about 400 cases in all).
const CAP: [(&str, usize); 3] = [("reranker:value", 120), ("reranker:where-fn", 220), ("reranker:where-type", 80)];
/// Scopes the unit trains on; the last scope is kept unseen for 3b.
const TRAINED: [&str; 2] = ["reranker:value", "reranker:where-fn"];
const UNIT: &str = "lwu-readout-002";
const TARGET: &str = "nmengn.components#final_norm";
const POOL: &str = "tests/stage-gates/s11/s11-pool-features.jsonl";

/// One verified case: docs[0] is the positive.
#[derive(Clone)]
struct Case {
    id: String,
    scope: String,
    split: String,
    question: String,
    docs: Vec<String>,
}
impl Case {
    fn to_value(&self) -> Value {
        obj(vec![
            ("id", s(&self.id)),
            ("scope", s(&self.scope)),
            ("split", s(&self.split)),
            ("question", s(&self.question)),
            ("docs", Value::Array(self.docs.iter().map(|d| s(d)).collect())),
        ])
    }
    fn from_value(v: &Value) -> Self {
        Self {
            id: field(v, "id"),
            scope: field(v, "scope"),
            split: field(v, "split"),
            question: field(v, "question"),
            docs: v
                .get("docs")
                .and_then(Value::array)
                .map(|a| a.iter().filter_map(|d| d.string().ok().map(str::to_owned)).collect())
                .unwrap_or_default(),
        }
    }
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut items: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    items.sort();
    for p in items {
        if p.is_dir() {
            if p.file_name().is_some_and(|f| f != "target") {
                rs_files(&p, out);
            }
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}
/// Does `text` hold `name` as a whole word (followed by `(` when `call`)?
fn has_word(text: &str, name: &str, call: bool) -> bool {
    text.match_indices(name).any(|(i, _)| {
        let before = text.get(..i).and_then(|t| t.chars().last()).is_none_or(|c| !is_word(c));
        let rest = text.get(i + name.len()..).unwrap_or("");
        let after = rest.chars().next();
        before && after.is_none_or(|c| !is_word(c)) && (!call || after == Some('('))
    })
}
/// The (kind scope, name) a line defines, if any.
fn definition(text: &str) -> Option<(&'static str, String)> {
    let t = text.strip_prefix("pub(crate) ").or_else(|| text.strip_prefix("pub ")).unwrap_or(text);
    let name = |rest: &str| rest.chars().take_while(|c| is_word(*c)).collect::<String>();
    if let Some(r) = t.strip_prefix("const ") {
        let nm = name(r);
        if !nm.is_empty()
            && nm.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            && r.get(nm.len()..).is_some_and(|x| x.starts_with(':'))
            && text.ends_with(';')
        {
            return Some(("reranker:value", nm));
        }
        if let Some(f) = r.strip_prefix("fn ") {
            let nm = name(f);
            return (!nm.is_empty()).then_some(("reranker:where-fn", nm));
        }
        return None;
    }
    if let Some(f) = t.strip_prefix("fn ") {
        let nm = name(f);
        return (!nm.is_empty() && nm != "main").then_some(("reranker:where-fn", nm));
    }
    for k in ["struct ", "enum "] {
        if let Some(r) = t.strip_prefix(k) {
            let nm = name(r);
            return (!nm.is_empty()).then_some(("reranker:where-type", nm));
        }
    }
    None
}
fn hash_byte(name: &str) -> u8 {
    u8::from_str_radix(nmhash::sha256_bytes(name.as_bytes()).get(..2).unwrap_or("00"), 16).unwrap_or(0)
}
fn question(scope: &str, name: &str) -> String {
    match scope {
        "reranker:value" => format!("What value is the constant {name} set to, and where is it defined?"),
        "reranker:where-fn" => format!("Where is the function {name} defined?"),
        _ => format!("Where is the type {name} defined?"),
    }
}

/// `mine`: verified cases from the mini's own source.
pub(super) fn mine(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let mut files = Vec::new();
    rs_files(&home.join("crates"), &mut files);
    // (source "path:line", trimmed text)
    let mut lines: Vec<(String, String)> = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(home).unwrap_or(f).to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(f).unwrap_or_default();
        for (i, l) in text.lines().enumerate() {
            let t = l.trim();
            if !t.is_empty() && !t.starts_with("//") {
                lines.push((format!("{rel}:{}", i + 1), t.chars().take(200).collect()));
            }
        }
    }
    let mut defs: BTreeMap<(&str, String), Vec<usize>> = BTreeMap::new();
    for (i, (_, t)) in lines.iter().enumerate() {
        if let Some((k, nm)) = definition(t) {
            defs.entry((k, nm)).or_default().push(i);
        }
    }
    let by_scope = |k: &str| -> Vec<usize> {
        defs.iter().filter(|((kk, _), v)| *kk == k && v.len() == 1).filter_map(|(_, v)| v.first().copied()).collect()
    };
    let mut cases = Vec::new();
    let mut quarantine = Vec::new();
    for (scope, cap) in CAP {
        let pool = by_scope(scope);
        let mut kept = 0;
        for ((k, nm), at) in defs.iter().filter(|((k, _), _)| *k == scope) {
            if kept >= cap {
                break;
            }
            let q = |reason: &str| obj(vec![("scope", s(k)), ("name", s(nm)), ("reason", s(reason))]);
            let Some(&di) = at.first() else { continue };
            if at.len() > 1 {
                quarantine.push(q(&format!("ambiguous: defined {} times", at.len())));
                continue;
            }
            let call = scope == "reranker:where-fn";
            let uses: Vec<usize> = lines
                .iter()
                .enumerate()
                .filter(|(i, (_, t))| *i != di && definition(t).is_none_or(|d| d.1 != *nm) && has_word(t, nm, call))
                .map(|(i, _)| i)
                .take(2)
                .collect();
            if uses.len() < 2 {
                quarantine.push(q("fewer than two uses"));
                continue;
            }
            let h = usize::from(hash_byte(nm));
            let other = pool.iter().copied().cycle().skip(h % pool.len().max(1)).take(pool.len()).find(|j| *j != di);
            let Some(other) = other else { continue };
            let doc = |i: usize| lines.get(i).map(|(src, t)| format!("{src} {t}")).unwrap_or_default();
            let docs = vec![doc(di), doc(*uses.first().unwrap_or(&0)), doc(*uses.get(1).unwrap_or(&0)), doc(other)];
            // Verification: the positive re-reads from disk and defines the name; no negative
            // defines it; the four documents are distinct.
            let reread = lines.get(di).and_then(|(src, t)| {
                let (p, ln) = src.rsplit_once(':')?;
                let ln: usize = ln.parse().ok()?;
                let got = std::fs::read_to_string(home.join(p)).ok()?.lines().nth(ln - 1)?.trim().chars().take(200).collect::<String>();
                Some(got == *t)
            });
            let defines = |d: &str| d.split_once(' ').and_then(|(_, t)| definition(t)).is_some_and(|x| x.1 == *nm);
            let distinct = docs.iter().collect::<BTreeSet<_>>().len() == 4;
            if reread != Some(true) || !docs.first().is_some_and(|d| defines(d)) || docs.iter().skip(1).any(|d| defines(d)) || !distinct {
                quarantine.push(q("failed verification"));
                continue;
            }
            let split = if !TRAINED.contains(&scope) {
                "heldout"
            } else {
                match h % 4 {
                    0 | 1 => "train",
                    2 => "gate",
                    _ => "heldout",
                }
            };
            cases.push(Case {
                id: format!("{}-{nm}", scope.trim_start_matches("reranker:")),
                scope: scope.to_owned(),
                split: split.to_owned(),
                question: question(scope, nm),
                docs,
            });
            kept += 1;
        }
    }
    let stamp = nmtype::now_ms();
    std::fs::create_dir_all(home.join("units/cases")).map_err(|e| bad(e.to_string()))?;
    let cpath = home.join(format!("units/cases/s11-cases-{stamp}.json"));
    std::fs::write(&cpath, Value::Array(cases.iter().map(Case::to_value).collect()).encode().map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let qpath = home.join(format!("units/cases/s11-quarantine-{stamp}.jsonl"));
    std::fs::write(&qpath, quarantine.iter().filter_map(|v| v.encode().ok()).collect::<Vec<_>>().join("\n"))
        .map_err(|e| bad(e.to_string()))?;
    line(&mut out, format!("source: {} .rs files, {} code lines, {} distinct definitions", files.len(), lines.len(), defs.len()));
    for sc in SCOPES {
        let c = |sp: &str| cases.iter().filter(|x| x.scope == sc && x.split == sp).count();
        line(&mut out, format!("{sc}: train {}, gate {}, heldout {}", c("train"), c("gate"), c("heldout")));
    }
    line(
        &mut out,
        format!("verified cases {} written to {}; quarantined {} to {}", cases.len(), cpath.display(), quarantine.len(), qpath.display()),
    );
    line(&mut out, format!("GATE V1: {} (run {} ms)", if cases.len() >= 300 { "PASS" } else { "MISS" }, t0.elapsed().as_millis()));
    Ok(out)
}

fn newest(home: &Path, dir: &str, prefix: &str) -> Result<PathBuf, Fault> {
    std::fs::read_dir(home.join(dir))
        .map_err(|e| bad(e.to_string()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|f| f.to_string_lossy().starts_with(prefix)))
        .max()
        .ok_or_else(|| bad(format!("no {dir}/{prefix}*")))
}
fn load_cases(home: &Path) -> Result<(PathBuf, Vec<Case>), Fault> {
    let p = newest(home, "units/cases", "s11-cases-")?;
    let v = nmjson::parse(&std::fs::read(&p).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    Ok((p, v.array().map_err(|e| bad(e.to_string()))?.iter().map(Case::from_value).collect()))
}
fn cached(home: &Path) -> BTreeMap<String, Vec<(Vec<f64>, f64)>> {
    let mut m: BTreeMap<String, Vec<(Vec<f64>, f64)>> = BTreeMap::new();
    for l in std::fs::read_to_string(home.join(POOL)).unwrap_or_default().lines() {
        let Ok(v) = nmjson::parse(l.as_bytes()) else { continue };
        let f = |x: &Value| if let Value::Number(t) = x { t.parse::<f64>().ok() } else { None };
        let docs: Vec<(Vec<f64>, f64)> = v
            .get("docs")
            .and_then(Value::array)
            .map(|a| {
                a.iter()
                    .map(|d| {
                        (
                            d.get("h").and_then(Value::array).map(|x| x.iter().filter_map(f).collect()).unwrap_or_default(),
                            d.get("base").ok().and_then(f).unwrap_or(0.0),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        m.insert(field(&v, "case"), docs);
    }
    m
}

/// `pool <seconds>`: features for uncached cases, in shards under 60 s, one process.
pub(super) fn pool(home: &Path, total_s: u64) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let (cpath, cases) = load_cases(home)?;
    let have = cached(home);
    let todo: Vec<&Case> = cases.iter().filter(|c| !have.contains_key(&c.id)).collect();
    line(&mut out, format!("cases {} ({}), cached {}, to do {}", cases.len(), cpath.display(), have.len(), todo.len()));
    let m = super::model(home)?;
    let dev = super::device(&m);
    line(&mut out, format!("model loaded on {} in {} ms", dev.name(), t0.elapsed().as_millis()));
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(home.join(POOL)).map_err(|e| bad(e.to_string()))?;
    let (mut shard, mut shard_t0, mut shard_cases, mut done) = (1_u32, Instant::now(), 0_usize, 0_usize);
    // The first shard carries the model load, so it gets less compute time.
    let mut limit_ms = 60_000_u128.saturating_sub(t0.elapsed().as_millis()).saturating_sub(12_000);
    let stamp = nmtype::now_ms();
    for c in todo {
        if t0.elapsed().as_secs() >= total_s {
            break;
        }
        let mut docs = Vec::new();
        for d in &c.docs {
            let (h, base) = feature(&m, dev, &c.question, d)?;
            docs.push(obj(vec![
                ("base", Value::Number(format!("{base}"))),
                ("h", Value::Array(h.iter().map(|x| Value::Number(format!("{x}"))).collect())),
            ]));
        }
        let rec = obj(vec![("case", s(&c.id)), ("docs", Value::Array(docs))]);
        writeln!(f, "{}", rec.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
        shard_cases += 1;
        done += 1;
        if shard_t0.elapsed().as_millis() >= limit_ms {
            let ms =
                shard_t0.elapsed().as_millis() + if shard == 1 { t0.elapsed().as_millis() - shard_t0.elapsed().as_millis() } else { 0 };
            let txt = format!(
                "shard {shard}: {shard_cases} cases, {} forwards, {ms} ms ({})\n",
                shard_cases * 4,
                if ms < 60_000 { "under 60 s" } else { "OVER 60 s" }
            );
            let _ = std::fs::write(home.join(format!("runs/S11-pool-{stamp}-{shard}.txt")), &txt);
            out.push_str(&txt);
            shard += 1;
            shard_t0 = Instant::now();
            shard_cases = 0;
            limit_ms = 48_000;
        }
    }
    if shard_cases > 0 {
        let ms = shard_t0.elapsed().as_millis();
        let txt = format!(
            "shard {shard}: {shard_cases} cases, {} forwards, {ms} ms ({})\n",
            shard_cases * 4,
            if ms < 60_000 { "under 60 s" } else { "OVER 60 s" }
        );
        let _ = std::fs::write(home.join(format!("runs/S11-pool-{stamp}-{shard}.txt")), &txt);
        out.push_str(&txt);
    }
    line(&mut out, format!("this process: {done} cases; cached now {}; process {} ms", have.len() + done, t0.elapsed().as_millis()));
    line(&mut out, super::resources(&m));
    Ok(out)
}

/// One case with its features.
struct Row {
    case: Case,
    x: Vec<Vec<f64>>,
    base: Vec<f64>,
}

fn feed(r: &mut Rls, rows: &[&Row], noise: bool) {
    for row in rows {
        for (j, x) in row.x.iter().enumerate() {
            let pos = if noise { j == 1 } else { j == 0 };
            r.update(x, if pos { 1.0 } else { -1.0 });
        }
    }
}
fn unit_scores(r: &Rls, row: &Row) -> Vec<f64> {
    row.x.iter().map(|x| r.predict(x)).collect()
}

/// The whole pathway for one fitted readout: gate fit, per-scope 3a/3b verdicts.
struct Pathway {
    gate: Gate,
    verdicts: Vec<readout::ScopeVerdict>,
    table: Vec<Value>,
    gate_rows: usize,
}

#[expect(clippy::cast_precision_loss, reason = "case counts are small")]
fn share(xs: &[bool]) -> f64 {
    xs.iter().filter(|x| **x).count() as f64 / xs.len().max(1) as f64
}

fn pathway(unit: &Rls, gate_set: &[&Row], held: &[&Row]) -> Pathway {
    let control = nmbrain::slot::Slots::default();
    let benefit: Vec<(Vec<f64>, f64)> = gate_set
        .iter()
        .map(|r| {
            let u = unit_scores(unit, r);
            let gain = f64::from(u8::from(readout::first(&u))) - f64::from(u8::from(readout::first(&r.base)));
            (Gate::features(&u, &r.base), gain)
        })
        .collect();
    let gate = Gate::fit(&benefit);
    let mut verdicts = Vec::new();
    let mut table = Vec::new();
    for sc in SCOPES {
        let rows: Vec<&&Row> = held.iter().filter(|r| r.case.scope == sc).collect();
        let trained = TRAINED.contains(&sc);
        let (mut un, mut ba, mut ga, mut ct, mut sh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in &rows {
            let u = unit_scores(unit, r);
            let open = trained && gate.open(&Gate::features(&u, &r.base));
            un.push(readout::first(&u));
            ba.push(readout::first(&r.base));
            ga.push(if open { readout::first(&u) } else { readout::first(&r.base) });
            ct.push(readout::first(&r.case.docs.iter().map(|d| f64::from(control.rerank(&r.case.question, d))).collect::<Vec<_>>()));
            #[expect(clippy::cast_precision_loss, reason = "document lengths are small")]
            sh.push(readout::first(&r.case.docs.iter().map(|d| -(d.len() as f64)).collect::<Vec<_>>()));
        }
        let best = share(&ba).max(share(&ct)).max(share(&sh));
        let near: Option<Paired> = trained.then(|| readout::paired(&ga, &ba));
        let v = readout::judge(sc, near.as_ref(), share(&ga), best);
        table.push(obj(vec![
            ("scope", s(sc)),
            ("cases", n(rows.len())),
            ("unit", n(format!("{:.3}", share(&un)))),
            ("gated", n(format!("{:.3}", share(&ga)))),
            ("base", n(format!("{:.3}", share(&ba)))),
            ("control", n(format!("{:.3}", share(&ct)))),
            ("shortest", n(format!("{:.3}", share(&sh)))),
            ("paired", near.as_ref().map_or(Value::Null, Paired::to_value)),
            ("verdict", s(&v.verdict)),
            ("reason", s(&v.reason)),
        ]));
        verdicts.push(v);
    }
    Pathway { gate, verdicts, table, gate_rows: benefit.len() }
}

/// `learn`: R2 to R5 of revision 4, CPU only, on the cached features.
#[expect(clippy::too_many_lines, reason = "the revision 4 gates in runbook order")]
pub(super) fn learn(home: &Path) -> Result<(String, Value), Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let (cpath, cases) = load_cases(home)?;
    let feats = cached(home);
    let rows: Vec<Row> = cases
        .iter()
        .filter_map(|c| {
            let f = feats.get(&c.id)?;
            (f.len() == c.docs.len()).then(|| Row {
                case: c.clone(),
                x: f.iter().map(|(h, b)| h.iter().copied().chain([*b, 1.0]).collect()).collect(),
                base: f.iter().map(|d| d.1).collect(),
            })
        })
        .collect();
    let pick = |sp: &str| -> Vec<&Row> {
        rows.iter().filter(|r| r.case.split == sp && (sp == "heldout" || TRAINED.contains(&r.case.scope.as_str()))).collect()
    };
    let (train, gate_set, held) = (pick("train"), pick("gate"), pick("heldout"));
    let d = train.first().and_then(|r| r.x.first()).map_or(0, Vec::len);
    line(
        &mut out,
        format!(
            "cases {} ({}), with features {}; train {}, gate {}, held-out {}; {d} features",
            cases.len(),
            cpath.display(),
            rows.len(),
            train.len(),
            gate_set.len(),
            held.len()
        ),
    );
    if train.len() < 16 || d == 0 {
        return Err(bad("fewer than 16 training cases have features; run `s11 pool` first"));
    }
    let count = |r: &Rls, xs: &[&Row]| xs.iter().filter(|row| readout::first(&unit_scores(r, row))).count();
    // Delta by prequential accuracy on the first 48 training cases, under the retention check.
    let stream = train.get(..48.min(train.len())).unwrap_or(&[]);
    let mut chosen: Option<(f64, usize)> = None;
    for delta in [0.1, 1.0, 10.0] {
        let mut r = Rls::new(d, delta);
        let (mut preq, mut after8) = (0, 0);
        for (i, row) in stream.iter().enumerate() {
            preq += usize::from(readout::first(&unit_scores(&r, row)));
            feed(&mut r, &[*row], false);
            if i == 7 {
                after8 = count(&r, stream.get(..8).unwrap_or(&[]));
            }
        }
        let end8 = count(&r, stream.get(..8).unwrap_or(&[]));
        let kept = end8 + 1 >= after8;
        line(
            &mut out,
            format!(
                "delta {delta}: prequential {preq} of {}, first 8 after learning {after8}, at the end {end8}, retention {kept}",
                stream.len()
            ),
        );
        if kept && chosen.is_none_or(|c| preq > c.1) {
            chosen = Some((delta, preq));
        }
    }
    let (delta, _) = chosen.ok_or_else(|| bad("no delta passed the retention check"))?;
    // R2: memorize 16 checked cases, else stop.
    let first16 = train.get(..16).unwrap_or(&[]);
    let mut m16 = Rls::new(d, delta);
    feed(&mut m16, first16, false);
    let memorized = count(&m16, first16);
    line(
        &mut out,
        format!("GATE R2 memorize: {memorized} of 16 (delta {delta}): {}", if memorized == 16 { "PASS" } else { "MISS (stop)" }),
    );
    if memorized != 16 {
        return Ok((out, obj(vec![("stopped", s("R2 memorize"))])));
    }
    // R3: the unit; R4 benefit gate; per-scope verdicts.
    let mut unit = Rls::new(d, delta);
    let tu = Instant::now();
    feed(&mut unit, &train, false);
    #[expect(clippy::cast_precision_loss, reason = "update counts are small")]
    let per_update = tu.elapsed().as_secs_f64() * 1000.0 / unit.updates.max(1) as f64;
    let pw = pathway(&unit, &gate_set, &held);
    line(
        &mut out,
        format!(
            "GATE R4 benefit gate: fitted on {} measured benefits, w {:?}",
            pw.gate_rows,
            pw.gate.w.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
        ),
    );
    for t in &pw.table {
        line(&mut out, format!("GATE R3 {}", t.encode().unwrap_or_default()));
    }
    // R5: the shuffled-label control through the same pathway and the same accept rule.
    let mut noise = Rls::new(d, delta);
    feed(&mut noise, &train, true);
    let pn = pathway(&noise, &gate_set, &held);
    let noise_accepted: Vec<&str> = pn.verdicts.iter().filter(|v| v.verdict == "accept").map(|v| v.scope.as_str()).collect();
    for t in &pn.table {
        line(&mut out, format!("noise control {}", t.encode().unwrap_or_default()));
    }
    line(
        &mut out,
        format!(
            "GATE R5 noise control rejected by the accept rule: {} (accepted scopes {noise_accepted:?})",
            if noise_accepted.is_empty() { "PASS" } else { "MISS" }
        ),
    );
    // The unit, written once, beside the base weights; never loaded here.
    let accepted: Vec<String> = pw.verdicts.iter().filter(|v| v.verdict == "accept").map(|v| v.scope.clone()).collect();
    let mut units = Units::open(home);
    if units.get(UNIT).is_some() {
        return Err(Fault::new("s11", FaultKind::Policy, format!("{UNIT} exists; files are written once")));
    }
    let base_hash = units.get("lwu-rerank-001").map(|c| c.base_sha256.clone()).unwrap_or_default();
    let mut card = UnitCard::new(UNIT, "qwen3-0.6b", &base_hash, "reranker", TARGET, 1, 0);
    card.lineage = vec!["lwu-readout-001".into()];
    // The parent is retired when the child registers (one open unit per target; files kept).
    if units.get("lwu-readout-001").is_some_and(|c| !matches!(c.status.as_str(), "rejected" | "rolled-back" | "retired")) {
        units.set_status("lwu-readout-001", "retired")?;
        line(&mut out, "lwu-readout-001 retired (its child lwu-readout-002 registers on the same target; files kept)");
    }
    units.register(card)?;
    #[expect(clippy::cast_possible_truncation, reason = "readout weights are small values")]
    let w32: Vec<f32> = unit.w.iter().map(|x| *x as f32).collect();
    let now = nmtype::now_ms();
    let verdict_json = obj(vec![
        ("unit", s(UNIT)),
        ("kind", s("rls readout with a benefit gate")),
        ("delta", n(delta)),
        ("features", n(d)),
        ("trained_scopes", Value::Array(TRAINED.iter().map(|x| s(x)).collect())),
        ("train_cases", n(train.len())),
        ("memorized_of_16", n(memorized)),
        ("ms_per_update", n(format!("{per_update:.3}"))),
        ("state_numbers", n(unit.state_numbers())),
        ("scopes", Value::Array(pw.table.clone())),
        ("noise_control", Value::Array(pn.table.clone())),
        ("accepted_scopes", Value::Array(accepted.iter().map(|x| s(x)).collect())),
        ("active", Value::Bool(false)),
        (
            "note",
            s(
                "active stays false until Luis switches the unit on; the reranker uses it only in accepted scopes and only where the gate opens",
            ),
        ),
    ]);
    let mut files = vec![("readout.q8".to_owned(), units.commit_file(UNIT, "readout.q8", &UnitWeights::encode_q8(&w32, 1, d)?)?)];
    files.push((
        "gate.json".to_owned(),
        units.commit_file(UNIT, "gate.json", pw.gate.to_value().encode().map_err(|e| bad(e.to_string()))?.as_bytes())?,
    ));
    files.push((
        "verdict.json".to_owned(),
        units.commit_file(UNIT, "verdict.json", verdict_json.encode().map_err(|e| bad(e.to_string()))?.as_bytes())?,
    ));
    let status = if accepted.is_empty() { "candidate" } else { "accepted-inactive" };
    let status =
        if pw.verdicts.iter().any(|v| v.verdict == "reject") && accepted.is_empty() && pw.verdicts.iter().all(|v| v.verdict != "candidate")
        {
            "rejected"
        } else {
            status
        };
    units.set_files(UNIT, files.clone(), status)?;
    append_life(home, UNIT, &Life::new(now, 0.0, status))?;
    for (f, h) in &files {
        line(&mut out, format!("wrote units/{UNIT}/{f} sha256 {h}"));
    }
    let wall = t0.elapsed().as_millis();
    line(
        &mut out,
        format!(
            "unit {UNIT}: status {status}, accepted scopes {accepted:?}, not loaded; {per_update:.3} ms per update; run {wall} ms ({})",
            if wall < 60_000 { "under 60 s" } else { "OVER 60 s" }
        ),
    );
    Ok((out, verdict_json))
}

fn append_life(home: &Path, id: &str, life: &Life) -> Result<(), Fault> {
    let mut v = life.to_value();
    if let Value::Object(m) = &mut v {
        m.insert("unit".into(), s(id));
        m.insert("at_ms".into(), n(nmtype::now_ms()));
    }
    let mut f =
        std::fs::OpenOptions::new().create(true).append(true).open(home.join("units/life.jsonl")).map_err(|e| bad(e.to_string()))?;
    writeln!(f, "{}", v.encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))
}

/// `decay`: decay every readout unit's benefit to now and retire by the rule (append-only).
pub(super) fn decay(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let mut last: BTreeMap<String, Life> = BTreeMap::new();
    for l in std::fs::read_to_string(home.join("units/life.jsonl")).unwrap_or_default().lines() {
        if let Ok(v) = nmjson::parse(l.as_bytes())
            && let Ok(life) = Life::from_value(&v)
        {
            last.insert(field(&v, "unit"), life);
        }
    }
    let mut units = Units::open(home);
    let now = nmtype::now_ms();
    for (id, mut life) in last {
        let retired = life.decay(now);
        if retired && units.get(&id).is_some() {
            units.set_status(&id, "retired")?;
        }
        line(
            &mut out,
            format!(
                "{id}: benefit {:.4}, uses {}, status {}{}",
                life.benefit,
                life.uses,
                life.status,
                if retired { " (retired now; files kept)" } else { "" }
            ),
        );
        append_life(home, &id, &life)?;
    }
    Ok(out)
}

/// `verify`: the unit as the reranker would load it (q8 readout, gate, accepted scopes,
/// through `readout::Active`) on the held-out cases; matches the f64 result of `learn`?
pub(super) fn verify(home: &Path) -> Result<String, Fault> {
    let mut out = String::new();
    let dir = home.join(format!("units/{UNIT}"));
    let (w, _, _) = UnitWeights::decode_q8(&std::fs::read(dir.join("readout.q8")).map_err(|e| bad(e.to_string()))?)?;
    let gate = Gate::from_value(
        &nmjson::parse(&std::fs::read(dir.join("gate.json")).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?,
    )
    .map_err(bad)?;
    let v = nmjson::parse(&std::fs::read(dir.join("verdict.json")).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    let scopes: Vec<String> = v
        .get("accepted_scopes")
        .and_then(Value::array)
        .map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
        .unwrap_or_default();
    let act = readout::Active { unit: UNIT.into(), w: w.into_iter().map(f64::from).collect(), gate, scopes };
    let (_, cases) = load_cases(home)?;
    let feats = cached(home);
    for sc in SCOPES {
        let (mut path, mut base, mut k) = (0, 0, 0);
        for c in cases.iter().filter(|c| c.scope == sc && c.split == "heldout") {
            let Some(f) = feats.get(&c.id) else { continue };
            #[expect(clippy::cast_possible_truncation, reason = "features were f32")]
            let unit: Vec<f64> = f.iter().map(|(h, b)| act.score(&h.iter().map(|x| *x as f32).collect::<Vec<_>>(), *b as f32)).collect();
            let b: Vec<f64> = f.iter().map(|d| d.1).collect();
            let used = if act.serves(&c.question) { act.order(&unit, &b) } else { b.clone() };
            path += usize::from(readout::first(&used));
            base += usize::from(readout::first(&b));
            k += 1;
        }
        line(
            &mut out,
            format!("{sc}: served {}, held-out {k}: reranker path (q8 readout, gates) {path}, base {base}", act.serves(&question(sc, "x"))),
        );
    }
    Ok(out)
}
