//! S11 runs (runbook `runs/2026-11-08-S11-runbook.md`): `outcomes` (gate 2: outcomes from
//! Claude's work through the tools), `fit <k>` and `check <k>` (gates 3 and 4: link k of the
//! chain of rank-1 units, each a one-minute fit behind its own Step Check). New files only.
#[path = "s11r4.rs"]
mod r4;
use crate::s8::{batches, device, model, parmap, resources, view_hashes};
use nmapi::learn::{self, Scorer, TARGET};
use nmjson::Value;
use nmlearn::{Fitter, Mezo, Rung, Scope, StepCheckSpec, UnitCard, UnitWeights, Units};
use nmtype::{Budget, Fault, FaultKind};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEED: u64 = 11;
const LINKS: usize = 7;
/// Revision 2 chain starts at link 4 (revision 1 used links 1 to 3); link 7 is the noise control.
const CHAIN2: usize = 4;
const NOISE: usize = 7;
/// The compartment's declared scope (slot and question kind).
const SCOPE: &str = "reranker:value";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s11", FaultKind::Execution, m)
}
fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: impl ToString) -> Value {
    Value::Number(v.to_string())
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}
fn unit_id(k: usize) -> String {
    format!("lwu-rerank-{:03}", k + 1)
}
fn write_ahead(path: &Path, text: &str) -> Result<(), Fault> {
    std::fs::write(path, text).map_err(|e| bad(e.to_string()))?;
    std::fs::File::options()
        .write(true)
        .open(path)
        .and_then(|f| f.set_modified(std::time::SystemTime::now() + Duration::from_secs(2)))
        .map_err(|e| bad(e.to_string()))
}
fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap_or_default().lines().filter_map(|l| nmjson::parse(l.as_bytes()).ok()).collect()
}
fn field(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}

const LIB: &str = "//! Probe.\n/// How many polls a reader waits.\npub const POLL_LIMIT: u32 = 4;\n/// One.\n#[must_use]\npub const fn one() -> u32 {\n    1\n}\n";

/// Gate 2: outcomes from Claude's work, through the tools.
fn outcomes(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let stamp = nmtype::now_ms();
    let tree = home.join(format!("tests/stage-gates/s11/s11-tree-{stamp}"));
    let mem = home.join(format!("tests/stage-gates/s11/s11-mem-{stamp}"));
    std::fs::create_dir_all(tree.join("src")).map_err(|e| bad(e.to_string()))?;
    std::fs::write(tree.join("Cargo.toml"), "[package]\nname = \"s11-probe\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[workspace]\n")
        .map_err(|e| bad(e.to_string()))?;
    let lib = tree.join("src/lib.rs");
    std::fs::write(&lib, LIB).map_err(|e| bad(e.to_string()))?;
    let lock = nmtool::run::run(&tree, "cargo", &["generate-lockfile", "--offline"], &[], Duration::from_secs(20));
    line(&mut out, format!("lock file: ok {}", lock.ok));
    let org = nmapi::forage::Organism::open_with(home, &tree, &[], true, &mem)?;
    nmapi::tools::install(org).map_err(|e| bad(e.message))?;
    let api = nmapi::tools::api();
    let call = |tool: &str, args: Vec<(&str, Value)>| api.dispatch(tool, &obj(args)).map_err(|e| bad(format!("{tool}: {}", e.message)));
    let good = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -7,1 +7,1 @@\n-    1\n+    3\n";
    let broken = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -7,1 +7,1 @@\n-    1\n+    \"one\"\n";
    let g = call("nm_stepcheck", vec![("diff", s(good))])?;
    let b = call("nm_stepcheck", vec![("diff", s(broken))])?;
    line(&mut out, format!("nm_stepcheck good diff: {}; broken diff: {}", field(&g, "verdict"), field(&b, "verdict")));
    let r1 = call("nm_build_run", vec![])?;
    write_ahead(&lib, &LIB.replace("    1\n", "    let x: u32 = \"one\";\n    x\n"))?;
    let r2 = call("nm_build_run", vec![])?;
    write_ahead(&lib, LIB)?;
    let r3 = call("nm_build_run", vec![])?;
    let ok = |v: &Value| matches!(v.get("ok"), Ok(Value::Bool(true)));
    line(&mut out, format!("nm_build_run clean {}, seeded {}, clean again {}", ok(&r1), ok(&r2), ok(&r3)));
    let pack = call(
        "nm_context_pack",
        vec![("question", s("What value is the constant POLL_LIMIT set to, and where is it defined?")), ("format", s("json"))],
    )?;
    let episode = field(&pack, "episode");
    let fb = call("nm_feedback", vec![("episode", s(&episode)), ("correct", Value::Bool(true))])?;
    line(
        &mut out,
        format!(
            "nm_context_pack answer {:?}, episode {episode}; nm_feedback logged {}",
            field(&pack, "answer"),
            matches!(fb.get("outcome_logged"), Ok(Value::Bool(true)))
        ),
    );
    let recs = records(&mem.join("build-outcomes.jsonl"));
    let verified = |r: &&Value| matches!(r.get("verified"), Ok(Value::Bool(true)));
    let sc_pass = recs.iter().filter(verified).any(|r| field(r, "event") == "stepcheck" && field(r, "verdict") == "pass");
    let sc_fail = recs
        .iter()
        .filter(verified)
        .any(|r| field(r, "event") == "stepcheck" && field(r, "verdict") == "fail" && field(r, "failed_stage") == "check");
    let fixed = recs.iter().any(|r| field(r, "event") == "fixed");
    let fbrec = records(&mem.join("outcomes.jsonl"))
        .iter()
        .any(|r| field(r, "episode") == episode && matches!(r.get("correct"), Ok(Value::Bool(true))));
    for r in &recs {
        line(&mut out, format!("record: {}", r.encode().unwrap_or_default()));
    }
    line(
        &mut out,
        format!(
            "GATE 2: stepcheck pass {sc_pass}, stepcheck fail at check {sc_fail}, fixed {fixed}, feedback {fbrec}: {}",
            if sc_pass && sc_fail && fixed && fbrec { "PASS" } else { "MISS" }
        ),
    );
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// The last link before `k` that was not rejected (the parent of link k).
fn parent(units: &Units, k: usize) -> Option<UnitCard> {
    if k == NOISE {
        return None;
    }
    let start = if k >= CHAIN2 { CHAIN2 } else { 1 };
    (start..k).rev().map(unit_id).filter_map(|id| units.get(&id).cloned()).find(|c| c.status != "rejected" && !c.files.is_empty())
}

/// Gates 3 and 4, the fit of link k.
fn fit(home: &Path, k: usize, noise: bool) -> Result<(String, Value), Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let id = unit_id(k);
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    if units.get(&id).is_some() {
        return Err(Fault::new("s11", FaultKind::Policy, format!("{id} exists; a fit is a new unit and files are written once")));
    }
    let base = units
        .get("lwu-rerank-001")
        .map(|c| c.base_sha256.clone())
        .ok_or_else(|| bad("the S8 unit lwu-rerank-001 (base hash) is missing"))?;
    let par = parent(&units, k);
    if let Some(p) = &par
        && !matches!(p.status.as_str(), "retired" | "rolled-back")
    {
        units.set_status(&p.id, "retired")?;
    }
    let mut card = UnitCard::new(&id, "qwen3-0.6b", &base, "reranker", TARGET, 1, SEED);
    card.lineage = par.iter().map(|p| p.id.clone()).collect();
    units.register(card.clone())?;
    let (kk, nn) = learn::compat(&m, &card, None)?;
    let start = match &par {
        Some(p) => UnitWeights::read(p, &units.unit_dir(&p.id))?,
        None => UnitWeights::unfitted(&card, kk, nn),
    };
    let mut items = learn::items(home, "train")?;
    if noise {
        // The standing noise control: each item's positive swapped with its first negative.
        for it in &mut items {
            if let Some(neg) = it.negatives.first_mut() {
                std::mem::swap(&mut it.positive, neg);
            }
        }
    }
    let tokens_per = items
        .iter()
        .map(|it| m.pop.tokenizer.encode(&nmengn::roles::prompt(&it.question, &it.positive)).map_or(0, |v| v.len()))
        .sum::<usize>()
        / items.len().max(1);
    let cap = Mezo::pair_cap(4, tokens_per);
    let mezo = Mezo { lr: 1e-4, ..Mezo::standard(SEED, cap) };
    line(
        &mut out,
        format!(
            "link {k}: unit {id}, rank 1, seed {SEED}, parent {}, device {}; MeZO eps {} lr {} cap {cap} before the first pair, wall {} ms, ceiling {} ms",
            par.as_ref().map_or("none (zero up)", |p| p.id.as_str()),
            dev.name(),
            mezo.eps,
            mezo.lr,
            mezo.wall_ms,
            mezo.budget_ms
        ),
    );
    let rows: Vec<usize> = (0..nn).collect();
    let bt = batches(items.len(), 4);
    let mut scorer =
        Scorer { model: Arc::clone(&m), card: card.clone(), weights: start.clone(), items, device: dev, negatives: 1, forwards: 0 };
    let mut up = start.up.clone();
    let result = mezo.fit(&mut up, 1, &rows, &bt, &mut scorer);
    m.set_unit(TARGET, None)?;
    let log = match result {
        Ok(l) => l,
        Err(f) if f.kind == FaultKind::Policy => {
            line(&mut out, format!("REFUSED by the budget rule: {}", f.message));
            return Ok((out, obj(vec![("refused", s(&f.message))])));
        }
        Err(f) => return Err(f),
    };
    let mut fitted = start;
    fitted.up = up;
    let mut files = Vec::new();
    for (name, bytes) in fitted.files()? {
        files.push((name.clone(), units.commit_file(&id, &name, &bytes)?));
    }
    let fit_json = obj(vec![
        ("unit", s(&id)),
        ("scope", s(SCOPE)),
        ("noise_control", Value::Bool(noise)),
        ("parent", par.as_ref().map_or(Value::Null, |p| s(&p.id))),
        ("pairs", n(log.pairs.len())),
        ("cap_before", n(cap)),
        ("cap_after_first_pair", n(log.cap)),
        ("estimate_ms", n(log.estimate_ms)),
        ("wall_ms", n(log.wall_ms)),
        ("first_loss", n(log.first_loss)),
        ("last_loss", n(log.last_loss)),
        ("best_loss", n(log.best_loss)),
        ("stability", n(log.stability())),
        ("stopped", s(&log.stopped)),
        ("forwards", n(scorer.forwards)),
    ]);
    files.push(("fit.json".into(), units.commit_file(&id, "fit.json", fit_json.encode().map_err(|e| bad(e.to_string()))?.as_bytes())?));
    units.set_files(&id, files.clone(), "candidate")?;
    for (f, h) in &files {
        line(&mut out, format!("wrote units/{id}/{f} sha256 {h}"));
    }
    line(
        &mut out,
        format!(
            "fit: {} pairs (cap after the first pair {}, estimate {} ms), loop {} ms, stopped: {}; loss first {:.4}, last {:.4}, best {:.4}; stability {:.2}; {} forwards",
            log.pairs.len(),
            log.cap,
            log.estimate_ms,
            log.wall_ms,
            log.stopped,
            log.first_loss,
            log.last_loss,
            log.best_loss,
            log.stability(),
            scorer.forwards
        ),
    );
    line(&mut out, resources(&m));
    let wall = t0.elapsed().as_millis();
    line(&mut out, format!("run {wall} ms: {}", if wall < 60_000 { "under 60 s" } else { "OVER 60 s" }));
    Ok((out, fit_json))
}

/// Gate 4, the Step Check of link k.
fn check(home: &Path, k: usize) -> Result<(String, Value), Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let id = unit_id(k);
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    let mut pm = parmap(home, nmtype::now_ms());
    let card = units.get(&id).cloned().ok_or_else(|| bad(format!("{id} is not registered; run `s11 fit {k}` first")))?;
    let fit_v = nmjson::parse(&std::fs::read(units.unit_dir(&id).join("fit.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let fit_wall = fit_v.get("wall_ms").and_then(Value::integer).unwrap_or(0);
    let stability = fit_v.get("stability").ok().and_then(nmlearn::unit::float_of).unwrap_or(0.0);
    let held = learn::items(home, "heldout")?;
    let views0 = view_hashes(&m);
    // Out-of-scope probes (where-defined and which-crates questions) run without the unit
    // (the local gate) before the load and after the unload: they must stay bit-identical.
    let probes0 = learn::probes(&m, dev)?;
    let _ = learn::load(&m, &mut units, &id)?;
    let with = learn::shard(&m, dev, &held)?;
    learn::unload(&m, &mut units, &id)?;
    let without = learn::shard(&m, dev, &held)?;
    let probes1 = learn::probes(&m, dev)?;
    let probes_same = probes0.iter().zip(&probes1).all(|(a, b)| a.bytes == b.bytes) && probes0.len() == probes1.len();
    // Policy v2 references (EAMM V79.2): the base, the exact control's order, a trivial prior.
    let control = nmbrain::slot::Slots::default();
    let first = |score: &dyn Fn(&str, &str) -> f32| -> f32 {
        let wins = held.iter().filter(|it| {
            let pos = score(&it.question, &it.positive);
            it.negatives.iter().all(|d| score(&it.question, d) < pos)
        });
        #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
        let share = wins.count() as f32 / held.len().max(1) as f32;
        share
    };
    #[expect(clippy::cast_precision_loss, reason = "document lengths are small")]
    let prior = first(&|_, d: &str| -(d.len() as f32));
    let references = vec![
        ("base without the unit".to_owned(), without.score),
        ("exact control order".to_owned(), first(&|q, d| control.rerank(q, d))),
        ("shortest document first".to_owned(), prior),
    ];
    let integrity = views0 == view_hashes(&m) && !views0.is_empty();
    // Only `up` may change along the chain: link 1's down equals the seeded down; a child's
    // down equals its parent's.
    let down = |c: &UnitCard| c.files.iter().find(|f| f.0 == "down.q8").map(|f| f.1.clone()).unwrap_or_default();
    let expected = match card.lineage.first().and_then(|p| units.get(p)) {
        Some(p) => down(p),
        None => {
            let (kk, nn) = learn::compat(&m, &card, None)?;
            nmhash::sha256_bytes(&UnitWeights::encode_q8(&UnitWeights::unfitted(&card, kk, nn).down, 1, kk)?)
        }
    };
    let alignment = down(&card) == expected;
    // Each run is under 60 s on its own (law 7); the larger run is the cost.
    let cost = fit_wall.max(u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX));
    let spec = StepCheckSpec {
        scope: Scope::Component(TARGET.into()),
        rung: Rung::S1,
        metrics: vec!["rank-first".into(), "cross-entropy".into()],
        baseline: Some("qwen3-reference without the unit".into()),
        budget: Budget { tokens: 0, ms: 60_000, depth: 0, units: 1 },
        overlap: Vec::new(),
    };
    let before = learn::shard_value(&without, Vec::new());
    let after = learn::shard_value(
        &with,
        vec![
            ("integrity", Value::Bool(integrity)),
            ("compat", Value::Bool(true)),
            ("alignment", Value::Bool(alignment)),
            ("stability", n(stability)),
            ("cost_ms", n(cost)),
        ],
    );
    let mut report = nmlearn::stepspec::compare(&spec, &before, &after);
    report.verdict = nmlearn::stepspec::promote(&report, with.score, &references);
    if !probes_same {
        report.verdict = nmlearn::Verdict::Reject;
    }
    let applied = nmlearn::adapt::apply(&nmlearn::Adaptation::Parameter { unit: id.clone() }, report.verdict, &mut units)?;
    line(
        &mut out,
        format!(
            "policy v2 references: {}; unit {:.3}; out-of-scope probes bit-identical {probes_same}",
            references.iter().map(|(n, v)| format!("{n} {v:.3}")).collect::<Vec<_>>().join(", "),
            with.score
        ),
    );
    pm.observe("parameter", TARGET, "stepcheck", &format!("chain link {k} unit {id}: verdict {}", report.verdict.name()), "s11-check")?;
    line(
        &mut out,
        format!(
            "link {k} unit {id} (lineage {:?}): held-out 8 items, without the unit score {:.3} loss {:.4} ranks {:?}; with it score {:.3} loss {:.4} ranks {:?}",
            card.lineage, without.score, without.loss, without.ranks, with.score, with.loss, with.ranks
        ),
    );
    line(
        &mut out,
        format!(
            "criteria: improvement {:.3}, regression {:.3}, integrity {integrity}, compat true, stability {stability:.2}, alignment {alignment}, cost {cost} ms; verdict {}; applied {}",
            report.improvement,
            report.regression,
            report.verdict.name(),
            applied.to_value().encode().unwrap_or_default()
        ),
    );
    line(&mut out, resources(&m));
    let wall = t0.elapsed().as_millis();
    line(&mut out, format!("run {wall} ms: {}", if wall < 60_000 { "under 60 s" } else { "OVER 60 s" }));
    let json = obj(vec![
        ("link", n(k)),
        ("unit", s(&id)),
        ("scope", s(SCOPE)),
        (
            "references",
            Value::Array(references.iter().map(|(n, v)| obj(vec![("name", s(n)), ("score", Value::Number(v.to_string()))])).collect()),
        ),
        ("probes_identical", Value::Bool(probes_same)),
        ("before", before),
        ("after", after),
        ("report", report.to_value()),
        ("applied", applied.to_value()),
    ]);
    Ok((out, json))
}

/// Revision 3, gate R1: features of every (question, document) pair, computed once.
const READOUT: &str = "lwu-readout-001";
const READOUT_TARGET: &str = "nmengn.components#final_norm";

/// (hidden state at the last position of `final_norm`, base rerank score) for one pair.
fn feature(m: &nmengn::model::Model, dev: nmengn::placement::Device, q: &str, d: &str) -> Result<(Vec<f32>, f32), Fault> {
    use nmengn::assembly::nid;
    let tok = |s: &str| {
        m.pop
            .tokenizer
            .encode(s)
            .ok()
            .and_then(|v| v.first().and_then(|x| usize::try_from(*x).ok()))
            .ok_or_else(|| bad(format!("no token for {s:?}")))
    };
    let (yes, no) = (tok(" yes")?, tok(" no")?);
    let mut d_ids = m.pop.tokenizer.encode(d).map_err(|e| bad(e.to_string()))?;
    d_ids.truncate(nmengn::roles::MAX_TOKENS);
    let doc = m.pop.tokenizer.decode(&d_ids);
    let ids = m.pop.tokenizer.encode(&nmengn::roles::prompt(q, &doc)).map_err(|e| bad(e.to_string()))?;
    let targets = [(nid("final_norm"), "y".to_owned()), (nid("head"), "logits".to_owned())];
    let out = m.forward(dev, &ids, &targets, &nmengn::model::RunOpts::default())?;
    let y = nmengn::model::values(out.get(&targets[0]).ok_or_else(|| bad("no hidden state"))?)?;
    let l = nmengn::model::values(out.get(&targets[1]).ok_or_else(|| bad("no logits"))?)?;
    let width = 1024.min(y.len());
    let h = y.get(y.len() - width..).map(<[f32]>::to_vec).unwrap_or_default();
    Ok((h, l.get(yes).copied().unwrap_or(0.0) - l.get(no).copied().unwrap_or(0.0)))
}

fn features(home: &Path) -> Result<String, Fault> {
    let t0 = Instant::now();
    let mut out = String::new();
    let m = model(home)?;
    let dev = device(&m);
    let mut rows = Vec::new();
    for (split, items) in [("train", learn::items(home, "train")?), ("heldout", learn::items(home, "heldout")?)] {
        for it in &items {
            for (j, d) in std::iter::once(&it.positive).chain(&it.negatives).enumerate() {
                let (h, base) = feature(&m, dev, &it.question, d)?;
                rows.push(obj(vec![
                    ("split", s(split)),
                    ("item", s(&it.id)),
                    ("doc", n(j)),
                    ("base", Value::Number(format!("{base}"))),
                    ("h", Value::Array(h.iter().map(|x| Value::Number(format!("{x}"))).collect())),
                ]));
            }
        }
    }
    let first = learn::items(home, "train")?.into_iter().next().ok_or_else(|| bad("no train item"))?;
    let (a, ab) = feature(&m, dev, &first.question, &first.positive)?;
    let again = rows
        .first()
        .map(|r| r.get("h").and_then(Value::array).map(|v| v.iter().filter_map(num_f).collect::<Vec<f32>>()).unwrap_or_default())
        .unwrap_or_default();
    let same = a == again
        && rows.first().and_then(|r| r.get("base").ok()).map(|b| matches!(b, Value::Number(x) if x == &format!("{ab}"))).unwrap_or(false);
    let path = home.join(format!("tests/stage-gates/s11/s11-features-{}.json", nmtype::now_ms()));
    std::fs::write(&path, Value::Array(rows.clone()).encode().map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    line(&mut out, format!("features: {} pairs on {}, written to {}", rows.len(), dev.name(), path.display()));
    line(&mut out, format!("probe pair recomputed: bit-identical {same}"));
    line(&mut out, resources(&m));
    let wall = t0.elapsed().as_millis();
    line(&mut out, format!("GATE R1: {} (run {wall} ms)", if same && rows.len() == 128 && wall < 60_000 { "PASS" } else { "MISS" }));
    Ok(out)
}

fn num_f(v: &Value) -> Option<f32> {
    if let Value::Number(x) = v { x.parse().ok() } else { None }
}

/// One document's row: features (hidden state, base score, constant) and the base score.
struct Row {
    x: Vec<f64>,
    base: f32,
}

/// The positive is document 0 of an item; it is first when it outscores every negative.
fn first_ok(score: &dyn Fn(&Row) -> f64, rows: &[Row]) -> bool {
    rows.first().is_some_and(|p| rows.iter().skip(1).all(|r| score(r) < score(p)))
}

/// Feed items to a readout: +1 for the positive (document 0, or document 1 for the noise
/// control's shuffled labels), -1 otherwise.
fn feed(r: &mut nmlearn::rls::Rls, items: &[&Vec<Row>], noise: bool) {
    for rows in items {
        for (j, row) in rows.iter().enumerate() {
            let positive = if noise { j == 1 } else { j == 0 };
            r.update(&row.x, if positive { 1.0 } else { -1.0 });
        }
    }
}

/// Revision 3, gates R2 to R4: the RLS readout fitted on the cached features.
#[expect(clippy::too_many_lines, reason = "the revision 3 gates in runbook order")]
fn rls(home: &Path) -> Result<(String, Value), Fault> {
    use nmlearn::rls::Rls;
    let t0 = Instant::now();
    let mut out = String::new();
    let newest = std::fs::read_dir(home.join("tests/stage-gates/s11"))
        .map_err(|e| bad(e.to_string()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("s11-features-")))
        .max()
        .ok_or_else(|| bad("no cached features; run `s11 features` first"))?;
    let v = nmjson::parse(&std::fs::read(&newest).map_err(|e| bad(e.to_string()))?).map_err(|e| bad(e.to_string()))?;
    let mut by: std::collections::BTreeMap<(String, String), Vec<Row>> = std::collections::BTreeMap::new();
    let mut order: Vec<(String, String)> = Vec::new();
    for r in v.array().map_err(|e| bad(e.to_string()))? {
        let key = (field(r, "split"), field(r, "item"));
        let base = r.get("base").ok().and_then(num_f).unwrap_or(0.0);
        let mut x: Vec<f64> =
            r.get("h").and_then(Value::array).map(|a| a.iter().filter_map(num_f).map(f64::from).collect()).unwrap_or_default();
        x.extend([f64::from(base), 1.0]);
        if !by.contains_key(&key) {
            order.push(key.clone());
        }
        by.entry(key).or_default().push(Row { x, base });
    }
    let split = |name: &str| -> Vec<&Vec<Row>> { order.iter().filter(|k| k.0 == name).filter_map(|k| by.get(k)).collect() };
    let (train, held) = (split("train"), split("heldout"));
    let d = train.first().and_then(|r| r.first()).map_or(0, |r| r.x.len());
    line(
        &mut out,
        format!("cached features {}: {} train items, {} held-out items, {d} features each", newest.display(), train.len(), held.len()),
    );
    let count = |r: &Rls, items: &[&Vec<Row>]| items.iter().filter(|rows| first_ok(&|x: &Row| r.predict(&x.x), rows)).count();
    // Delta by prequential accuracy under the retention check (the V108.1 selection rule).
    let mut chosen: Option<(f64, usize)> = None;
    for delta in [0.1, 1.0, 10.0] {
        let mut r = Rls::new(d, delta);
        let (mut preq, mut after8) = (0, 0);
        for (i, rows) in train.iter().enumerate() {
            if first_ok(&|x: &Row| r.predict(&x.x), rows) {
                preq += 1;
            }
            feed(&mut r, &[*rows], false);
            if i == 7 {
                after8 = count(&r, &train[..8]);
            }
        }
        let end8 = count(&r, &train[..8.min(train.len())]);
        let retained = end8 + 1 >= after8;
        line(
            &mut out,
            format!(
                "delta {delta}: prequential {preq} of {}, first 8 right after learning {after8}, at the end {end8}, retention {retained}",
                train.len()
            ),
        );
        if retained && chosen.is_none_or(|c| preq > c.1) {
            chosen = Some((delta, preq));
        }
    }
    let (delta, _) = chosen.ok_or_else(|| bad("no delta passed the retention check"))?;
    // R2: memorize the first 16 checked items.
    let mut m16 = Rls::new(d, delta);
    feed(&mut m16, &train[..16.min(train.len())], false);
    let memorized = count(&m16, &train[..16.min(train.len())]);
    line(
        &mut out,
        format!("GATE R2 memorize: {memorized} of 16 with delta {delta}: {}", if memorized == 16 { "PASS" } else { "MISS (stop)" }),
    );
    // The unit on every training item, timed; the noise control on shuffled labels.
    let mut unit = Rls::new(d, delta);
    let tu = Instant::now();
    feed(&mut unit, &train, false);
    #[expect(clippy::cast_precision_loss, reason = "update counts are small")]
    let per_update = tu.elapsed().as_secs_f64() * 1000.0 / unit.updates.max(1) as f64;
    let mut noise = Rls::new(d, delta);
    feed(&mut noise, &train, true);
    #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
    let share = |score: &dyn Fn(&Row) -> f64| held.iter().filter(|rows| first_ok(score, rows)).count() as f32 / held.len().max(1) as f32;
    let unit_score = share(&|x: &Row| unit.predict(&x.x));
    let noise_score = share(&|x: &Row| noise.predict(&x.x));
    let base_score = share(&|x: &Row| f64::from(x.base));
    let items = learn::items(home, "heldout")?;
    let control = nmbrain::slot::Slots::default();
    let text_share = |score: &dyn Fn(&str, &str) -> f32| -> f32 {
        let wins = items.iter().filter(|it| {
            let p = score(&it.question, &it.positive);
            it.negatives.iter().all(|d| score(&it.question, d) < p)
        });
        #[expect(clippy::cast_precision_loss, reason = "item counts are small")]
        let s = wins.count() as f32 / items.len().max(1) as f32;
        s
    };
    #[expect(clippy::cast_precision_loss, reason = "document lengths are small")]
    let prior = text_share(&|_, d: &str| -(d.len() as f32));
    let references = [
        ("base without the unit", base_score),
        ("exact control order", text_share(&|q, d| control.rerank(q, d))),
        ("shortest document first", prior),
    ];
    let best = references.iter().map(|r| r.1).fold(f32::NEG_INFINITY, f32::max);
    let verdict = if memorized == 16 && unit_score >= best { "candidate" } else { "rejected" };
    line(&mut out, format!("references: {}", references.iter().map(|(nm, v)| format!("{nm} {v:.3}")).collect::<Vec<_>>().join(", ")));
    line(
        &mut out,
        format!(
            "GATE R3: 3a gain near seen cases: sample too small ({} held-out items; 400 paired cases needed), so Accept is not possible; 3b no loss on unseen: unit {unit_score:.3} against best reference {best:.3}: {}; verdict {verdict}",
            held.len(),
            if unit_score >= best { "holds" } else { "fails" }
        ),
    );
    #[expect(clippy::cast_precision_loss, reason = "state sizes are small")]
    let mb = unit.state_numbers() as f64 * 8.0 / 1_048_576.0;
    line(
        &mut out,
        format!(
            "GATE R4: {per_update:.3} ms per update over {} updates, state {} numbers ({mb:.1} MB in f64); noise unit (shuffled labels) held-out {noise_score:.3}, 3b {}",
            unit.updates,
            unit.state_numbers(),
            if noise_score >= best { "holds (recorded)" } else { "fails (not promoted)" }
        ),
    );
    let mut units = Units::open(home);
    if units.get(READOUT).is_some() {
        return Err(Fault::new("s11", FaultKind::Policy, format!("{READOUT} exists; files are written once")));
    }
    let base_hash = units.get("lwu-rerank-001").map(|c| c.base_sha256.clone()).unwrap_or_default();
    units.register(UnitCard::new(READOUT, "qwen3-0.6b", &base_hash, "reranker", READOUT_TARGET, 1, 0))?;
    #[expect(clippy::cast_possible_truncation, reason = "readout weights are small values")]
    let w32: Vec<f32> = unit.w.iter().map(|x| *x as f32).collect();
    let mut files = vec![("readout.q8".to_owned(), units.commit_file(READOUT, "readout.q8", &UnitWeights::encode_q8(&w32, 1, d)?)?)];
    let json = obj(vec![
        ("unit", s(READOUT)),
        ("kind", s("rls readout")),
        ("scope", s(SCOPE)),
        ("delta", Value::Number(delta.to_string())),
        ("features", n(d)),
        ("memorized_of_16", n(memorized)),
        ("heldout_unit", Value::Number(unit_score.to_string())),
        ("heldout_noise", Value::Number(noise_score.to_string())),
        (
            "references",
            Value::Array(references.iter().map(|(nm, v)| obj(vec![("name", s(nm)), ("score", Value::Number(v.to_string()))])).collect()),
        ),
        ("ms_per_update", Value::Number(format!("{per_update:.3}"))),
        ("state_numbers", n(unit.state_numbers())),
        ("verdict", s(verdict)),
    ]);
    files.push(("rls.json".to_owned(), units.commit_file(READOUT, "rls.json", json.encode().map_err(|e| bad(e.to_string()))?.as_bytes())?));
    units.set_files(READOUT, files, verdict)?;
    let wall = t0.elapsed().as_millis();
    line(
        &mut out,
        format!(
            "unit {READOUT} written once, status {verdict}; run {wall} ms ({})",
            if wall < 60_000 { "under 60 s" } else { "OVER 60 s" }
        ),
    );
    Ok((out, json))
}

/// `run s11 <outcomes|fit <k>|check <k>>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("", String::as_str);
    let k: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(0);
    let stamp = nmtype::now_ms();
    let result: Result<(String, Option<Value>), Fault> = match part {
        "outcomes" => outcomes(home).map(|t| (t, None)),
        "features" => features(home).map(|t| (t, None)),
        "rls" => rls(home).map(|(t, j)| (t, Some(j))),
        "mine" => r4::mine(home).map(|t| (t, None)),
        "pool" => r4::pool(home, args.get(3).and_then(|x| x.parse().ok()).unwrap_or(50)).map(|t| (t, None)),
        "learn" => r4::learn(home).map(|(t, j)| (t, Some(j))),
        "decay" => r4::decay(home).map(|t| (t, None)),
        "verify" => r4::verify(home).map(|t| (t, None)),
        "fit" if (1..=LINKS).contains(&k) => {
            fit(home, k, k == NOISE && args.get(4).map(String::as_str) == Some("noise")).map(|(t, j)| (t, Some(j)))
        }
        "check" if (1..=LINKS).contains(&k) => check(home, k).map(|(t, j)| (t, Some(j))),
        other => Err(bad(format!("unknown S11 part {other:?}: outcomes, fit <1-{LINKS}>, check <1-{LINKS}>"))),
    };
    let label = if k > 0 && !matches!(part, "pool") { format!("{part}-{k}") } else { part.to_owned() };
    match result {
        Ok((text, json)) => {
            let p = home.join(format!("runs/S11-{label}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            if let Some(j) = json {
                let _ = std::fs::write(home.join(format!("runs/S11-{label}-{stamp}.json")), j.encode().unwrap_or_default());
            }
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(home.join(format!("runs/S11-{label}-{stamp}.txt")), format!("FAULT {e}\n"));
            eprintln!("s11 {part}: {e}");
            ExitCode::FAILURE
        }
    }
}
