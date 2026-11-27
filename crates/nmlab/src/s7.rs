//! S7 runs (runbook `runs/2026-11-08-S7-runbook.md`): `decode` (gate 1), `writer <id>` (one
//! fixture item: gates 2 to 4; the intent patterns are scored by the harness from the JSON
//! result), `router` (gate 5, shadow). Each run writes new files under `runs/`.
use nmengn::decode;
use nmengn::model::{Model, Population};
use nmengn::placement::Device;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::time::Instant;

const MODEL: &str = "qwen3-0.6b";

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s7", FaultKind::Execution, m)
}
fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn model(home: &Path) -> Result<Arc<Model>, Fault> {
    let pop = Population::open(home, MODEL, true, true)?;
    Ok(Arc::new(Model::bind(&pop, nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?)?))
}
/// A writer population other than the locked 0.6B (2026-11-12, runbook "if the writer misses gate
/// 2": the 1.7B base), on the same assembly; linear weights at `precision` (`q8` fits the 4 GB card).
fn model_of(home: &Path, id: &str, precision: &str) -> Result<Arc<Model>, Fault> {
    let pop = Population::open(home, id, true, true)?;
    let mut a = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference")?;
    if !precision.is_empty() {
        for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
            n.precision = precision.to_owned();
        }
    }
    Ok(Arc::new(Model::bind(&pop, a)?))
}
fn resources(m: &Model) -> String {
    let peak = nmffi::dynlib::process_memory().map_or(0, |(p, _)| p >> 20);
    let gpu = m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok()).map_or(0, |(f, t)| (t - f) >> 20);
    format!("peak RAM {peak} MB (process working set), GPU in use {gpu} MB (device total minus free)")
}

/// One cargo check of a package in the scratch tree: (errors with code and line, raw JSON).
fn cargo_check(tree: &Path, target: &Path, package: &str) -> Result<(Vec<Value>, f64), Fault> {
    let t0 = Instant::now();
    let o = Command::new("cargo")
        .args(["check", "-p", package, "--message-format", "json"])
        .current_dir(tree)
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_INCREMENTAL", "0")
        .output()
        .map_err(|e| bad(format!("cargo: {e}")))?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut msgs = Vec::new();
    for l in String::from_utf8_lossy(&o.stdout).lines() {
        if let Ok(v) = nmjson::parse(l.as_bytes())
            && v.get("reason").and_then(Value::string).ok() == Some("compiler-message")
            && let Ok(m) = v.get("message")
            && m.get("level").and_then(Value::string).ok() == Some("error")
            && m.get("spans").and_then(Value::array).is_ok_and(|s| !s.is_empty())
        {
            msgs.push(m.clone());
        }
    }
    Ok((msgs, ms))
}
fn code_of(m: &Value) -> String {
    m.get("code").and_then(|c| c.get("code")).and_then(Value::string).unwrap_or("").to_owned()
}
fn primary(m: &Value) -> Option<(String, u64)> {
    m.get("spans").and_then(Value::array).ok()?.iter().find(|s| matches!(s.get("is_primary"), Ok(Value::Bool(true)))).and_then(|s| {
        Some((s.get("file_name").and_then(Value::string).ok()?.replace('\\', "/"), s.get("line_start").and_then(Value::integer).ok()?))
    })
}
/// Machine-applicable replacements in a diagnostic: (byte start, byte end, text).
fn suggestions(m: &Value) -> Vec<(usize, usize, String)> {
    let mut out = Vec::new();
    let mut take = |s: &Value| {
        if s.get("suggestion_applicability").and_then(Value::string).ok() == Some("MachineApplicable")
            && let (Ok(a), Ok(b), Ok(t)) = (
                s.get("byte_start").and_then(Value::integer),
                s.get("byte_end").and_then(Value::integer),
                s.get("suggested_replacement").and_then(Value::string),
            )
        {
            out.push((usize::try_from(a).unwrap_or(0), usize::try_from(b).unwrap_or(0), t.to_owned()));
        }
    };
    for s in m.get("spans").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default() {
        take(&s);
    }
    for c in m.get("children").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default() {
        for s in c.get("spans").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default() {
            take(&s);
        }
    }
    out
}
/// Lines removed and added between two texts (common prefix and suffix stripped).
fn changed(old: &str, new: &str) -> (Vec<String>, Vec<String>) {
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suf = a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count().min(a.len().min(b.len()) - pre);
    (a[pre..a.len() - suf].iter().map(|s| (*s).to_owned()).collect(), b[pre..b.len() - suf].iter().map(|s| (*s).to_owned()).collect())
}
fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: impl ToString) -> Value {
    Value::Number(v.to_string())
}

/// Restores the scratch file on drop, whatever happens in between.
struct Restore(PathBuf, Vec<u8>);
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = std::fs::write(&self.0, &self.1);
    }
}

/// Try one fixed text: write it, check, and describe the outcome as JSON.
fn try_fix(tree: &Path, target: &Path, item: &Value, file: &Path, seeded: &str, fixed: &str) -> Result<Value, Fault> {
    let pkg = item.get("package").and_then(Value::string).map_err(|e| bad(e.to_string()))?;
    std::fs::write(file, fixed).map_err(|e| bad(e.to_string()))?;
    let (errs, ms) = cargo_check(tree, target, pkg)?;
    let (removed, added) = changed(seeded, fixed);
    let at = usize::try_from(item.get("expected_line").and_then(Value::integer).unwrap_or(0)).unwrap_or(0);
    let fixed_line = fixed.lines().nth(at.saturating_sub(1)).unwrap_or("").to_owned();
    Ok(Value::Object(
        [
            ("errors_after", n(errs.len())),
            (
                "first_error_after",
                s(&errs
                    .first()
                    .map(|m| format!("{} {}", code_of(m), m.get("message").and_then(Value::string).unwrap_or("")))
                    .unwrap_or_default()),
            ),
            ("check_ms", n(format!("{ms:.0}"))),
            ("removed", Value::Array(removed.iter().map(|x| s(x)).collect())),
            ("added", Value::Array(added.iter().map(|x| s(x)).collect())),
            ("fixed_line", s(&fixed_line)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    ))
}

const EXAMPLES: &str = "Rust compiler errors and the corrected line.\n\nerror[E0423]: expected value, found struct `Point`\n --> src/geo.rs:12\nLine 12 now:     let p = Point;\nLine 12 fixed:     let p = Point { x: 0, y: 0 };\n\nerror[E0604]: only `u8` can be cast as `char`, not `u32`\n --> src/text.rs:7\nLine 7 now:     let c = 66_u32 as char;\nLine 7 fixed:     let c = char::from_u32(66_u32).unwrap_or('?');\n\n";

/// Gates 2 to 4 for one fixture item.
#[expect(clippy::too_many_lines, reason = "one item's whole procedure, in runbook order")]
/// `writer <id> [--sampling card] [--seed n]` (2026-11-12, research file Part 10 (d)): with
/// `--sampling card` the writer decodes by the Qwen3 card's non-thinking settings, read from the
/// weights folder (`nmengn::decode::Sampling::from_card`), seeded; otherwise greedy as locked in the
/// runbook. Everything else is unchanged.
/// `chat`: 0 the locked completion prompt; 1 that prompt in the card's chat template; 2 the chat
/// template with one instruction line and the item only (no worked examples).
fn writer(
    home: &Path,
    id: &str,
    sampling: Option<&decode::Sampling>,
    population: Option<(&str, &str)>,
    chat: u8,
) -> Result<(String, Value), Fault> {
    let mut out = String::new();
    let t0 = Instant::now();
    let fx = nmjson::parse(&std::fs::read(home.join("designs/writer-errors.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let item = fx
        .get("items")
        .and_then(Value::array)
        .ok()
        .and_then(|a| a.iter().find(|i| i.get("id").and_then(Value::string).ok() == Some(id)))
        .cloned()
        .ok_or_else(|| bad(format!("no item {id}")))?;
    let g = |k: &str| item.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| bad(format!("{k}: {e}")));
    let (rel, pkg, code, insert) = (g("file")?, g("package")?, g("expected_code")?, g("insert")?);
    let after = usize::try_from(item.get("insert_after_line").and_then(Value::integer).map_err(|e| bad(e.to_string()))?).unwrap_or(0);
    let expected = after + 1;
    let tree = home.join("tests/stage-gates/s7/s7-tree");
    let target = home.join("scratch/s7-target");
    let file = tree.join(&rel);
    let original = std::fs::read(&file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    let _restore = Restore(file.clone(), original.clone());
    let text = String::from_utf8_lossy(&original).into_owned();
    let mut lines: Vec<&str> = text.lines().collect();
    lines.insert(after, &insert);
    let seeded = lines.join("\n") + "\n";
    std::fs::write(&file, &seeded).map_err(|e| bad(e.to_string()))?;
    let (errs, seed_ms) = cargo_check(&tree, &target, &pkg)?;
    let diag =
        errs.iter().find(|m| code_of(m) == code).cloned().ok_or_else(|| bad(format!("{id}: the seeded error {code} did not appear")))?;
    let message = diag.get("message").and_then(Value::string).unwrap_or("").to_owned();
    let (dfile, dline) = primary(&diag).unwrap_or_default();
    line(
        &mut out,
        format!(
            "item {id} ({pkg}, {rel}:{expected}, {code}): seeded check {seed_ms:.0} ms; rustc says {code} at {dfile}:{dline}: {message}"
        ),
    );
    // Context through the organism (nm_context_pack path) on the scratch tree.
    let mem = home.join(format!("tests/stage-gates/s7/s7-mem-{}", nmtype::now_ms()));
    let org = nmapi::forage::Organism::open_with(home, &tree, &[], false, &mem)?;
    let question = format!("error[{code}] at {rel}:{expected}: {message}. What is the code around this line?");
    let a = org.ask(&question, 4000)?;
    let mut ctx: Vec<(u64, String)> = Vec::new();
    for p in &a.pack.packets {
        for e in p.get_evidence().iter().chain(std::iter::once(&p.provenance().source)) {
            if let Some(rest) = e.strip_prefix(&format!("{rel}:"))
                && let Some((ln, txt)) = rest.split_once(' ').or(Some((rest, "")))
                && let Ok(ln) = ln.trim_end_matches(':').parse::<u64>()
                && !ctx.iter().any(|c| c.0 == ln)
            {
                let body = if txt.is_empty() { p.text().to_owned() } else { txt.to_owned() };
                ctx.push((ln, body));
            }
        }
    }
    ctx.sort_by_key(|c| c.0);
    ctx.truncate(12);
    line(
        &mut out,
        format!("context pack: {} tokens, {} packets, {} lines from the cited file", a.pack.tokens, a.pack.packets.len(), ctx.len()),
    );
    let seeded_line = seeded.lines().nth(after).unwrap_or("").to_owned();
    let mut prompt = if chat == 2 {
        String::from("Fix this Rust compile error. Reply with the corrected line only, in a code block.\n")
    } else {
        String::from(EXAMPLES)
    };
    let _ = writeln!(prompt, "error[{code}]: {message}\n --> {rel}:{expected}");
    for (ln, t) in &ctx {
        let _ = writeln!(prompt, "{rel}:{ln} {t}");
    }
    let _ = write!(prompt, "Line {expected} now: {seeded_line}\nLine {expected} fixed:");
    // Writer over the base population (or `--model <id> [--precision q8]`).
    let m = match population {
        Some((id, precision)) => {
            line(&mut out, format!("writer population: {id} (linear {})", if precision.is_empty() { "as released" } else { precision }));
            model_of(home, id, precision)?
        }
        None => model(home)?,
    };
    let tw = Instant::now();
    let (answer, choice, gen_ms) = match sampling {
        _ if chat > 0 => {
            line(
                &mut out,
                format!(
                    "prompt form: the card's chat template, non-thinking (empty think block); decoding {}",
                    sampling.map_or_else(
                        || "greedy".to_owned(),
                        |s| format!(
                            "sampled by the card ({}): temperature {}, top-p {}, top-k {}, min-p {}, seed {}",
                            s.source, s.temperature, s.top_p, s.top_k, s.min_p, s.seed
                        )
                    )
                ),
            );
            nmengn::roles::write_chat(&m, &prompt, 96, sampling)?
        }
        Some(s) => {
            line(
                &mut out,
                format!(
                    "decoding: sampled by the card ({}): temperature {}, top-p {}, top-k {}, min-p {}, seed {}",
                    s.source, s.temperature, s.top_p, s.top_k, s.min_p, s.seed
                ),
            );
            nmengn::roles::write_sampled(&m, &prompt, 48, s)?
        }
        None => nmengn::roles::write(&m, &prompt, 48)?,
    };
    let answer = answer.trim_end().to_owned();
    // The prompt writes "Line N fixed:" followed by one space, then the line as written in
    // the file (the worked examples show it); that one separating space is removed.
    let writer_line = if answer.trim().is_empty() { String::new() } else { answer.strip_prefix(' ').unwrap_or(&answer).to_owned() };
    let writer_ms = tw.elapsed().as_secs_f64() * 1000.0;
    line(&mut out, format!("writer on {} ({}): {gen_ms:.0} ms generation; line: {writer_line:?}", choice.device.name(), choice.why));
    let writer_result = if writer_line.is_empty() {
        Value::Null
    } else {
        let mut fixed: Vec<&str> = seeded.lines().collect();
        if let Some(l) = fixed.get_mut(after) {
            *l = &writer_line;
        }
        try_fix(&tree, &target, &item, &file, &seeded, &(fixed.join("\n") + "\n"))?
    };
    // Exact control: rustc's machine-applicable suggestions.
    let tc = Instant::now();
    let sug = suggestions(&diag);
    let control_result = if sug.is_empty() {
        Value::Null
    } else {
        let mut bytes = seeded.clone().into_bytes();
        let mut ordered = sug.clone();
        ordered.sort_by_key(|x| std::cmp::Reverse(x.0));
        for (a, b, t) in ordered {
            if a <= b && b <= bytes.len() {
                bytes.splice(a..b, t.into_bytes());
            }
        }
        try_fix(&tree, &target, &item, &file, &seeded, &String::from_utf8_lossy(&bytes))?
    };
    let control_ms = tc.elapsed().as_secs_f64() * 1000.0;
    line(
        &mut out,
        format!(
            "control: {} machine-applicable suggestion(s); {}",
            sug.len(),
            if sug.is_empty() { "no fix offered".to_owned() } else { format!("{control_ms:.0} ms") }
        ),
    );
    line(&mut out, format!("writer result: {}", writer_result.encode().unwrap_or_default()));
    line(&mut out, format!("control result: {}", control_result.encode().unwrap_or_default()));
    line(&mut out, resources(&m));
    line(&mut out, format!("wall {} ms", t0.elapsed().as_millis()));
    let result = Value::Object(
        [
            ("id", s(id)),
            ("explanation_file", s(&dfile)),
            ("explanation_line", n(dline)),
            ("explanation_code", s(&code_of(&diag))),
            ("pack_tokens", n(a.pack.tokens)),
            ("context_lines", n(ctx.len())),
            ("writer_line", s(&writer_line)),
            ("writer_ms", n(format!("{writer_ms:.0}"))),
            ("writer_device", s(choice.device.name())),
            ("writer", writer_result),
            ("control", control_result),
            ("control_ms", n(format!("{control_ms:.0}"))),
            ("peak_ram_mb", n(nmffi::dynlib::process_memory().map_or(0, |(p, _)| p >> 20))),
            ("gpu_mb", n(m.pop.gpu.as_ref().and_then(|g| g.gpu.memory().ok()).map_or(0, |(f, t)| (t - f) >> 20))),
            ("wall_ms", n(t0.elapsed().as_millis())),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    );
    Ok((out, result))
}

/// Gate 5: the router over the base population in shadow, beside the keyword control, on the
/// 23 context questions. A job is chosen when logit(" yes") exceeds logit(" no") after a fixed
/// prompt; nothing is decided by it.
fn router(home: &Path) -> Result<String, Fault> {
    use nmbrain::slot::Exact;
    let mut out = String::new();
    let m = model(home)?;
    let qs = nmjson::parse(&std::fs::read(home.join("designs/questions-context.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let jobs = ["structural", "retrieval", "semantic", "verification", "reasoning", "dependency", "historical", "execution"];
    let (mut agree, mut total, mut ms) = (0_usize, 0_usize, 0.0_f64);
    for q in qs.get("items").and_then(Value::array).map_err(|e| bad(e.to_string()))? {
        let text = q.get("question").and_then(Value::string).unwrap_or("");
        let control: Vec<String> = nmbrain::slot::Controls.route(text);
        let mut neural = Vec::new();
        for j in jobs {
            let doc = format!("A {j} search of the codebase");
            let t = Instant::now();
            let s = nmengn::roles::rerank_on(&m, Device::Gpu, text, &doc)?;
            ms += t.elapsed().as_secs_f64() * 1000.0;
            if s > 0.0 {
                neural.push(j.to_owned());
            }
            total += 1;
            if (s > 0.0) == control.iter().any(|c| c == j) {
                agree += 1;
            }
        }
        line(&mut out, format!("  {}: control {:?}; router {:?}", q.get("id").and_then(Value::string).unwrap_or("?"), control, neural));
    }
    #[expect(clippy::cast_precision_loss, reason = "counts are small")]
    let rate = 100.0 * agree as f64 / total.max(1) as f64;
    line(
        &mut out,
        format!(
            "gate 5 router shadow: {agree} of {total} job decisions agree with the control ({rate:.1} percent); {ms:.0} ms of engine time; recorded, decides nothing"
        ),
    );
    line(&mut out, resources(&m));
    Ok(out)
}

/// `run s7 <decode|writer <id>|router>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("decode", String::as_str);
    let stamp = nmtype::now_ms();
    let result: Result<(String, Option<Value>), Fault> = match part {
        "decode" => {
            let devices = match args.get(3).map(String::as_str) {
                Some("cpu") => vec![Device::Cpu],
                Some("gpu") => vec![Device::Gpu],
                _ => vec![Device::Cpu, Device::Gpu],
            };
            nmengn::selftest::decode(home, &devices).map(|t| (t, None))
        }
        "writer" => {
            let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
            let sampling = (opt("--sampling").as_deref() == Some("card")).then(|| {
                let dir = nmtype::models_dir(home).join(opt("--model").unwrap_or_else(|| MODEL.to_owned())).join("files");
                decode::Sampling::from_card(&dir, false, opt("--seed").and_then(|x| x.parse().ok()).unwrap_or(1))
            });
            let pop = opt("--model");
            let precision = opt("--precision").unwrap_or_default();
            writer(
                home,
                args.get(3).map_or("w1", String::as_str),
                sampling.as_ref(),
                pop.as_deref().map(|p| (p, precision.as_str())),
                match opt("--template").as_deref() {
                    Some("chat") => 1,
                    Some("chat-item") => 2,
                    _ => 0,
                },
            )
            .map(|(t, v)| (t, Some(v)))
        }
        "router" => router(home).map(|t| (t, None)),
        other => Err(bad(format!("unknown S7 part {other}"))),
    };
    let label = if part == "decode" {
        format!("decode-{}", args.get(3).map_or("both", String::as_str))
    } else if part == "writer" {
        format!("writer-{}", args.get(3).map_or("w1", String::as_str))
    } else {
        part.to_owned()
    };
    match result {
        Ok((text, json)) => {
            let p = home.join(format!("runs/S7-{label}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            if let Some(j) = json {
                let _ = std::fs::write(home.join(format!("runs/S7-{label}-{stamp}.json")), j.encode().unwrap_or_default());
            }
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(home.join(format!("runs/S7-{label}-{stamp}.txt")), format!("FAULT {e}\n"));
            eprintln!("s7 {part}: {e}");
            ExitCode::FAILURE
        }
    }
}
