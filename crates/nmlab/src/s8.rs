//! S8 runs (runbook `runs/2026-11-08-S8-runbook.md`): `isolate` (gate 1), `overlap` (gate 2),
//! `fit` (gate 3), `check` (gate 4), `rollback` (gate 5). Each run writes new files under
//! `runs/` and a Parameter Map log under `tests/stage-gates/s8/s8-<ms>/`; every run is under one minute.
use nmapi::learn::{self, Scorer, TARGET};
use nmengn::model::{Model, Population};
use nmengn::placement::Device;
use nmjson::Value;
use nmlearn::{Fitter, Mezo, Outcome, Rung, Scope, StepCheckSpec, UnitCard, UnitWeights, Units, Verdict};
use nmtype::{Budget, Fault, FaultKind};
use std::fmt::Write as _;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const MODEL: &str = "qwen3-0.6b";
/// The unit of the S8 runbook.
pub const UNIT: &str = "lwu-rerank-001";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s8", FaultKind::Execution, m)
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
pub(crate) fn model(home: &Path) -> Result<Arc<Model>, Fault> {
    let pop = Population::open(home, MODEL, true, true)?;
    Ok(Arc::new(Model::bind(&pop, nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?))
}
pub(crate) fn device(m: &Model) -> Device {
    if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu }
}
pub(crate) fn resources(m: &Model) -> String {
    let peak = nmffi::dynlib::process_memory().map_or(0, |(p, _)| p >> 20);
    let gpu = m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok()).map_or(0, |(f, t)| (t - f) >> 20);
    format!("peak RAM {peak} MB (process working set), GPU in use {gpu} MB (device total minus free)")
}
/// The stored hash of every crate card, sorted (the base cards must not change).
fn card_hashes(home: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(home.join("crates"))
        .map(|d| {
            d.flatten()
                .filter_map(|e| std::fs::read(e.path().join("part.json")).ok())
                .filter_map(|b| nmjson::parse(&b).ok())
                .filter_map(|c| c.get("hash").and_then(Value::string).ok().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}
pub(crate) fn parmap(home: &Path, stamp: u64) -> nmnpcs::ParMap {
    let mut p = nmnpcs::ParMap::new(&home.join("tests/stage-gates/s8").join(format!("s8-{stamp}")));
    p.recording = true;
    p
}
/// The target node's cluster view hashes (the frozen base the unit stacks beside).
pub(crate) fn view_hashes(m: &Model) -> Vec<String> {
    m.bound.iter().filter(|b| b.node.id == TARGET).flat_map(|b| b.views.iter().map(|v| v.hash().to_owned())).collect()
}

/// Gate 1: register the unfitted unit; probes with no unit, with it loaded, after unloading.
fn isolate(home: &Path, stamp: u64) -> Result<String, Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    let mut pm = parmap(home, stamp);
    let card = match units.get(UNIT) {
        Some(c) if c.files.is_empty() => c.clone(),
        Some(c) => return Err(bad(format!("{UNIT} already has files (status {}); the isolation gate runs on an unfitted unit", c.status))),
        None => {
            let c = UnitCard::new(UNIT, MODEL, &m.pop.card.sha_main, "reranker", TARGET, 8, 8);
            units.register(c.clone())?;
            pm.observe("parameter", TARGET, "register", &format!("unit {UNIT} rank 8 seed 8"), "s8-isolate")?;
            c
        }
    };
    let (k, nn) = learn::compat(&m, &card, None)?;
    let w = UnitWeights::unfitted(&card, k, nn);
    line(
        &mut out,
        format!(
            "unit {UNIT} on {TARGET}: rank {}, down {}x{k}, up {nn}x{}, up all zero: {}; device {}",
            w.rank,
            w.rank,
            w.rank,
            w.up.iter().all(|v| *v == 0.0),
            dev.name()
        ),
    );
    let cards0 = card_hashes(home);
    let before = learn::probes(&m, dev)?;
    m.set_unit(TARGET, Some(learn::adapter(&card, &w)?))?;
    pm.observe("parameter", TARGET, "load", &format!("unit {UNIT} unfitted"), "s8-isolate")?;
    let loaded = learn::probes(&m, dev)?;
    m.set_unit(TARGET, None)?;
    pm.observe("parameter", TARGET, "unload", &format!("unit {UNIT}"), "s8-isolate")?;
    let after = learn::probes(&m, dev)?;
    let cards1 = card_hashes(home);
    let (ok, why) = nmlearn::isolation::isolation(&before, &loaded, &after, &cards0, &cards1);
    line(&mut out, format!("probe bytes per probe: {:?}", before.iter().map(|p| p.bytes.len()).collect::<Vec<_>>()));
    line(&mut out, format!("gate 1 isolation: {} ({why}; {} crate cards compared)", if ok { "PASS" } else { "MISS" }, cards0.len()));
    line(&mut out, format!("parameter map: {} events recorded under tests/stage-gates/s8/s8-{stamp}", pm.written));
    line(&mut out, resources(&m));
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// Gate 2: a second unit on the same target is refused; one on another node is accepted.
fn overlap(home: &Path, _stamp: u64) -> Result<String, Fault> {
    let mut out = String::new();
    let mut units = Units::open(home);
    let first = units.get(UNIT).cloned().ok_or_else(|| bad(format!("{UNIT} is not registered; run `s8 isolate` first")))?;
    let second = UnitCard::new("lwu-rerank-002", MODEL, &first.base_sha256, "reranker", TARGET, 8, 9);
    let refused = match units.register(second) {
        Err(f) if f.kind == FaultKind::Policy => {
            line(&mut out, format!("second unit on {TARGET}: refused ({})", f.message));
            true
        }
        Err(f) => {
            line(&mut out, format!("second unit on {TARGET}: refused for another reason ({f})"));
            false
        }
        Ok(()) => {
            line(&mut out, format!("second unit on {TARGET}: ACCEPTED (wrong)"));
            false
        }
    };
    let other = "nmengn.components#L26.down";
    let third = UnitCard::new("lwu-probe-003", MODEL, &first.base_sha256, "reranker", other, 8, 10);
    let accepted = units.register(third).is_ok();
    line(&mut out, format!("unit on {other}: {}", if accepted { "accepted" } else { "refused (wrong)" }));
    if accepted {
        units.unregister("lwu-probe-003")?;
        line(&mut out, "lwu-probe-003 unregistered again (no files)");
    }
    line(&mut out, format!("gate 2 overlap: {}", if refused && accepted { "PASS" } else { "MISS" }));
    Ok(out)
}

pub(crate) fn batches(count: usize, per: usize) -> Vec<Vec<usize>> {
    (0..count).collect::<Vec<_>>().chunks(per).map(<[usize]>::to_vec).collect()
}

/// Gate 3: the fit under the budget, then the quantized files, written once.
#[expect(clippy::too_many_lines, reason = "the fit procedure in runbook order")]
fn fit(home: &Path, stamp: u64) -> Result<(String, Value), Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    let mut pm = parmap(home, stamp);
    let card = units.get(UNIT).cloned().ok_or_else(|| bad(format!("{UNIT} is not registered; run `s8 isolate` first")))?;
    if !card.files.is_empty() {
        return Err(Fault::new("s8", FaultKind::Policy, format!("{UNIT} already has files; a new fit is a new unit in the lineage")));
    }
    let items = learn::items(home, "train")?;
    let at = nmtype::now_ms();
    let mut tokens_total = 0_usize;
    for it in &items {
        let tokens = m.pop.tokenizer.encode(&nmengn::roles::prompt(&it.question, &it.positive)).map(|v| v.len()).unwrap_or(0);
        tokens_total += tokens;
        nmlearn::outcomes::append(
            home,
            &Outcome {
                at,
                slot: "reranker".into(),
                task: it.id.clone(),
                query: it.question.clone(),
                answer: it.positive.clone(),
                correct: true,
                tokens: u64::try_from(tokens).unwrap_or(0),
                verified: true,
            },
        )?;
    }
    let logged = nmlearn::outcomes::since(home, at);
    let training = nmlearn::outcomes::training(&logged, true).len();
    let tokens_per = tokens_total / items.len().max(1);
    line(
        &mut out,
        format!(
            "outcomes logged: {} (verified, correct: {training}); mean prompt tokens {tokens_per}; device {}",
            logged.len(),
            dev.name()
        ),
    );
    let (k, nn) = learn::compat(&m, &card, None)?;
    let weights = UnitWeights::unfitted(&card, k, nn);
    let cap = Mezo::pair_cap(4, tokens_per);
    let mezo = Mezo::standard(card.seed, cap);
    line(
        &mut out,
        format!(
            "fitter: MeZO eps {} lr {} seed {} pairs cap {cap} (min(64, 50000 / (4 x {tokens_per}))), wall limit {} ms, ceiling {} ms, patience {}",
            mezo.eps, mezo.lr, mezo.seed, mezo.wall_ms, mezo.budget_ms, mezo.patience
        ),
    );
    let rows: Vec<usize> = (0..nn).collect();
    let bt = batches(items.len(), 4);
    let mut scorer =
        Scorer { model: Arc::clone(&m), card: card.clone(), weights: weights.clone(), items, device: dev, negatives: 1, forwards: 0 };
    let mut up = weights.up.clone();
    let result = mezo.fit(&mut up, weights.rank, &rows, &bt, &mut scorer);
    m.set_unit(TARGET, None)?;
    let log = match result {
        Ok(l) => l,
        Err(f) if f.kind == FaultKind::Policy => {
            line(&mut out, format!("fit refused by the budget rule: {}", f.message));
            line(&mut out, format!("gate 3 fit: REFUSED (recorded; the unit stays unfitted); forwards {}", scorer.forwards));
            line(&mut out, resources(&m));
            return Ok((out, obj(vec![("refused", s(&f.message)), ("forwards", n(scorer.forwards))])));
        }
        Err(f) => return Err(f),
    };
    let mut fitted = weights.clone();
    fitted.up = up;
    let moved = fitted.up.iter().filter(|v| **v != 0.0).count();
    let mut files = Vec::new();
    for (name, bytes) in fitted.files()? {
        let h = units.commit_file(UNIT, &name, &bytes)?;
        line(&mut out, format!("wrote units/{UNIT}/{name}: {} bytes, sha256 {h}", bytes.len()));
        files.push((name, h));
    }
    let pairs: Vec<Value> = log
        .pairs
        .iter()
        .map(|p| obj(vec![("plus", n(p.plus)), ("minus", n(p.minus)), ("projected", n(p.projected)), ("ms", n(format!("{:.0}", p.ms)))]))
        .collect();
    let fit_json = obj(vec![
        ("unit", s(UNIT)),
        ("target", s(TARGET)),
        ("device", s(dev.name())),
        ("pairs", n(log.pairs.len())),
        ("pair_cap", n(cap)),
        ("estimate_ms", n(log.estimate_ms)),
        ("wall_ms", n(log.wall_ms)),
        ("first_loss", n(log.first_loss)),
        ("last_loss", n(log.last_loss)),
        ("best_loss", n(log.best_loss)),
        ("stability", n(log.stability())),
        ("stopped", s(&log.stopped)),
        ("forwards", n(scorer.forwards)),
        ("tokens_per_outcome", n(tokens_per)),
        ("up_values_moved", n(moved)),
        ("outcomes_at", n(at)),
        ("steps", Value::Array(pairs)),
    ]);
    let fit_bytes = fit_json.encode().map_err(|e| bad(e.to_string()))?;
    let fh = units.commit_file(UNIT, "fit.json", fit_bytes.as_bytes())?;
    files.push(("fit.json".into(), fh));
    units.set_files(UNIT, files, "candidate")?;
    pm.observe(
        "parameter",
        TARGET,
        "fit",
        &format!("unit {UNIT}: {} pairs, loss {:.4} to {:.4}, {} up values moved", log.pairs.len(), log.first_loss, log.last_loss, moved),
        "s8-fit",
    )?;
    line(
        &mut out,
        format!(
            "fit: {} pairs in {} ms (estimate {} ms, stopped: {}); training loss first {:.4}, last {:.4}, best {:.4}; stability {:.2}; {} forwards; {moved} of {} up values moved",
            log.pairs.len(),
            log.wall_ms,
            log.estimate_ms,
            log.stopped,
            log.first_loss,
            log.last_loss,
            log.best_loss,
            log.stability(),
            scorer.forwards,
            fitted.up.len()
        ),
    );
    line(
        &mut out,
        format!(
            "gate 3 fit: {} (run {} ms, files written once with hashes in the card; status candidate)",
            if t0.elapsed().as_secs() < 60 { "PASS" } else { "MISS (over 60 s)" },
            t0.elapsed().as_millis()
        ),
    );
    line(&mut out, resources(&m));
    Ok((out, fit_json))
}

/// Gate 4: the held-out shard with the quantized unit loaded and unloaded; the StepReport.
fn check(home: &Path, stamp: u64) -> Result<(String, Value), Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    let mut pm = parmap(home, stamp);
    let card = units.get(UNIT).cloned().ok_or_else(|| bad(format!("{UNIT} is not registered")))?;
    let fit_v = nmjson::parse(
        &std::fs::read(units.unit_dir(UNIT).join("fit.json")).map_err(|e| bad(format!("fit.json: {e}; run `s8 fit` first")))?,
    )
    .map_err(|e| bad(e.to_string()))?;
    let fit_wall = fit_v.get("wall_ms").and_then(Value::integer).unwrap_or(0);
    let stability = fit_v.get("stability").ok().and_then(nmlearn::unit::float_of).unwrap_or(0.0);
    let held = learn::items(home, "heldout")?;
    let views0 = view_hashes(&m);
    let w = learn::load(&m, &mut units, UNIT)?;
    pm.observe("parameter", TARGET, "load", &format!("unit {UNIT} for its Step Check"), "s8-check")?;
    let with = learn::shard(&m, dev, &held)?;
    learn::unload(&m, &mut units, UNIT)?;
    let without = learn::shard(&m, dev, &held)?;
    let views1 = view_hashes(&m);
    let integrity = views0 == views1 && !views0.is_empty();
    let (k, nn) = learn::compat(&m, &card, Some(&w))?;
    let compat = true;
    let down_hash = card.files.iter().find(|f| f.0 == "down.q8").map(|f| f.1.clone()).unwrap_or_default();
    let expected_down = nmhash::sha256_bytes(&UnitWeights::encode_q8(&UnitWeights::unfitted(&card, k, nn).down, card.rank, k)?);
    let alignment = down_hash == expected_down;
    let check_wall = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
    let cost = fit_wall + check_wall;
    let others: Vec<Scope> = units.candidates().iter().filter(|c| c.id != UNIT).map(|c| Scope::Component(c.target.clone())).collect();
    let spec = StepCheckSpec {
        scope: Scope::Component(TARGET.into()),
        rung: Rung::S1,
        metrics: vec!["rank-first".into(), "cross-entropy".into()],
        baseline: Some("qwen3-reference without the unit".into()),
        budget: Budget { tokens: 0, ms: 60_000, depth: 0, units: 1 },
        overlap: others,
    };
    let before = learn::shard_value(&without, Vec::new());
    let after = learn::shard_value(
        &with,
        vec![
            ("integrity", Value::Bool(integrity)),
            ("compat", Value::Bool(compat)),
            ("alignment", Value::Bool(alignment)),
            ("stability", n(stability)),
            ("cost_ms", n(cost)),
        ],
    );
    let report = nmlearn::stepspec::compare(&spec, &before, &after);
    let applied = nmlearn::adapt::apply(&nmlearn::Adaptation::Parameter { unit: UNIT.into() }, report.verdict, &mut units)?;
    pm.observe(
        "parameter",
        TARGET,
        "stepcheck",
        &format!(
            "unit {UNIT}: verdict {} (improvement {:.3}, regression {:.3})",
            report.verdict.name(),
            report.improvement,
            report.regression
        ),
        "s8-check",
    )?;
    line(
        &mut out,
        format!(
            "held-out shard ({} items, four candidates each) on {}: without the unit score {:.3} loss {:.4} ranks {:?}; with the quantized unit score {:.3} loss {:.4} ranks {:?}",
            held.len(),
            dev.name(),
            without.score,
            without.loss,
            without.ranks,
            with.score,
            with.loss,
            with.ranks
        ),
    );
    line(
        &mut out,
        format!(
            "criteria: improvement {:.3}; regression {:.3}; integrity {integrity} (target cluster views {} unchanged, unit files verified at load); compat {compat} (base hash, node {TARGET} {nn}x{k}, rank {}); stability {stability:.2}; alignment {alignment} (down.q8 equals the seeded down); cost {cost} ms (fit {fit_wall} + check {check_wall})",
            report.improvement,
            report.regression,
            views0.len(),
            card.rank
        ),
    );
    line(&mut out, format!("verdict: {}; applied: {}", report.verdict.name(), applied.to_value().encode().unwrap_or_default()));
    line(
        &mut out,
        format!(
            "gate 4 step check: {} (report written with every criterion; verdict recorded)",
            if report.verdict == Verdict::Reject { "PASS (verdict Reject recorded)" } else { "PASS" }
        ),
    );
    line(&mut out, resources(&m));
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    let json = obj(vec![
        ("spec", spec.to_value()),
        ("before", before),
        ("after", after),
        ("report", report.to_value()),
        ("applied", applied.to_value()),
    ]);
    Ok((out, json))
}

/// Gate 5: load, probe, unload, compare with the pre-load bytes; mark rolled-back.
fn rollback(home: &Path, stamp: u64) -> Result<String, Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let m = model(home)?;
    let dev = device(&m);
    let mut units = Units::open(home);
    let mut pm = parmap(home, stamp);
    let before = learn::probes(&m, dev)?;
    learn::load(&m, &mut units, UNIT)?;
    let loaded = learn::probes(&m, dev)?;
    learn::unload(&m, &mut units, UNIT)?;
    let after = learn::probes(&m, dev)?;
    let changed = before.iter().zip(&loaded).filter(|(a, b)| a.bytes != b.bytes).count();
    let restored = before == after;
    units.rollback(UNIT)?;
    pm.observe("parameter", TARGET, "rollback", &format!("unit {UNIT} unloaded and marked rolled-back"), "s8-rollback")?;
    let c = units.get(UNIT).cloned().ok_or_else(|| bad("unit lost"))?;
    line(
        &mut out,
        format!(
            "probes: {} of {} changed while the unit was loaded; after unloading all {} equal the pre-load bytes: {restored}",
            changed,
            before.len(),
            before.len()
        ),
    );
    line(&mut out, format!("card: status {}, loaded {}, lineage {:?}, files kept {}", c.status, c.loaded, c.lineage, c.files.len()));
    line(&mut out, format!("gate 5 rollback: {}", if restored && !c.loaded && c.status == "rolled-back" { "PASS" } else { "MISS" }));
    line(&mut out, resources(&m));
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    Ok(out)
}

/// `run s8 <isolate|overlap|fit|check|rollback>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("isolate", String::as_str);
    let stamp = nmtype::now_ms();
    let result: Result<(String, Option<Value>), Fault> = match part {
        "isolate" => isolate(home, stamp).map(|t| (t, None)),
        "overlap" => overlap(home, stamp).map(|t| (t, None)),
        "fit" => fit(home, stamp).map(|(t, v)| (t, Some(v))),
        "check" => check(home, stamp).map(|(t, v)| (t, Some(v))),
        "rollback" => rollback(home, stamp).map(|t| (t, None)),
        other => Err(bad(format!("unknown S8 part {other}"))),
    };
    match result {
        Ok((text, json)) => {
            let p = home.join(format!("runs/S8-{part}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            if let Some(j) = json {
                let _ = std::fs::write(home.join(format!("runs/S8-{part}-{stamp}.json")), j.encode().unwrap_or_default());
            }
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(home.join(format!("runs/S8-{part}-{stamp}.txt")), format!("FAULT {e}\n"));
            eprintln!("s8 {part}: {e}");
            ExitCode::FAILURE
        }
    }
}
