//! S9 runs (runbook `runs/2026-11-08-S9-runbook.md`): `preflight <id>` (gate 4), `item <id>`
//! (gates 5 and 6, one breakage through `nm_build_run` and `nm_build_explain`), `hashes
//! <label> --live <emma-labs root>` (gate 7). Seeds go into the scratch tree only and are
//! put back by a guard; every run writes new files under `runs/`.
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

const TREE: &str = "tests/stage-gates/s7/s7-tree";
const TARGET: &str = "scratch/s7-target";
/// A clean check slower than this ends the preflight run as a warm-up (split runs, law 7).
const WARM_MS: u64 = 30_000;
/// Deadline per cargo call inside an item run (two calls plus the organism fit in 60 s).
const ITEM_DEADLINE: Duration = Duration::from_secs(25);

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("s9", FaultKind::Execution, m)
}
fn line(out: &mut String, s: impl AsRef<str>) {
    let _ = writeln!(out, "{}", s.as_ref());
}
fn st(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::string).map(str::to_owned).unwrap_or_default()
}
fn num(v: &Value, k: &str) -> usize {
    v.get(k).and_then(Value::integer).ok().and_then(|x| usize::try_from(x).ok()).unwrap_or(0)
}

/// One fixture item.
struct Item {
    package: String,
    file: String,
    after: usize,
    insert: String,
    code: String,
    expected: usize,
    anchor: String,
}
fn item(home: &Path, id: &str) -> Result<Item, Fault> {
    let v = nmjson::parse(&std::fs::read(home.join("designs/breakages.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let it = v
        .get("items")
        .and_then(Value::array)
        .map_err(|e| bad(e.to_string()))?
        .iter()
        .find(|i| st(i, "id") == id)
        .ok_or_else(|| bad(format!("no item {id}")))?;
    Ok(Item {
        package: st(it, "package"),
        file: st(it, "file"),
        after: num(it, "insert_after_line"),
        insert: st(it, "insert"),
        code: st(it, "expected_code"),
        expected: num(it, "expected_line"),
        anchor: st(it, "anchor"),
    })
}

/// The anchor rule: (insert after line, expected line, note) or the exclusion reason.
fn place(text: &str, it: &Item) -> Result<(usize, usize, String), String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.get(it.after.wrapping_sub(1)).is_some_and(|l| l.trim() == it.anchor) {
        return Ok((it.after, it.expected, "anchor at the recorded line".into()));
    }
    let lo = it.after.saturating_sub(3).max(1);
    if (lo..it.after).any(|n| lines.get(n - 1).is_some_and(|l| l.trim() == it.anchor)) {
        return Ok((it.after, it.expected, "anchor within 3 lines above the recorded line".into()));
    }
    let found: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.trim() == it.anchor).map(|(i, _)| i + 1).collect();
    match found.as_slice() {
        [f] => Ok((
            *f,
            f + 1,
            format!(
                "anchor moved from line {} to {f}: shifted {:+}",
                it.after,
                i64::try_from(*f).unwrap_or(0) - i64::try_from(it.after).unwrap_or(0)
            ),
        )),
        [] => Err("anchor absent".into()),
        _ => Err(format!("anchor found {} times", found.len())),
    }
}

/// Write `bytes` and set the file's modification time 2 s ahead: exFAT stores whole
/// seconds, so a write in the same second as cargo's last build of the package would look
/// older than the build output and cargo would not see it (runbook revision 2).
fn write_ahead(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)?;
    std::fs::File::options().write(true).open(path)?.set_modified(std::time::SystemTime::now() + Duration::from_secs(2))
}

/// Puts the scratch file back on drop, whatever happens in between.
struct Restore(PathBuf, Vec<u8>);
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = write_ahead(&self.0, &self.1);
    }
}

fn seeded(text: &str, after: usize, insert: &str) -> String {
    let nl = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut v: Vec<&str> = text.lines().collect();
    v.insert(after.min(v.len()), insert);
    let mut s = v.join(nl);
    if text.ends_with('\n') {
        s.push_str(nl);
    }
    s
}

fn plain_check(home: &Path, pkg: &str, deadline: Duration) -> nmtool::run::Output {
    let target = home.join(TARGET).to_string_lossy().into_owned();
    let env = [("CARGO_TARGET_DIR", target.as_str()), ("CARGO_INCREMENTAL", "0"), ("CARGO_TERM_COLOR", "never")];
    nmtool::run::run(&home.join(TREE), "cargo", &["check", "-p", pkg, "--color", "never", "--offline", "--locked"], &env, deadline)
}

fn peak() -> String {
    nmffi::dynlib::process_memory().map_or(0, |(p, _)| p >> 20).to_string()
}

/// Gate 4 for one item.
fn preflight(home: &Path, id: &str) -> Result<String, Fault> {
    let it = item(home, id)?;
    let mut out = String::new();
    let path = home.join(TREE).join(&it.file);
    let original = std::fs::read(&path).map_err(|e| bad(format!("{}: {e}", it.file)))?;
    let text = String::from_utf8_lossy(&original).into_owned();
    let h0 = nmhash::sha256_bytes(&original);
    line(&mut out, format!("item {id} ({}, {}) file sha256 {h0}", it.package, it.file));
    let (after, expected, note) = match place(&text, &it) {
        Ok(p) => p,
        Err(why) => {
            line(&mut out, format!("EXCLUDED {id}: {why}"));
            return Ok(out);
        }
    };
    line(&mut out, format!("anchor: {note}; insert after line {after}, expected line {expected}"));
    let clean = plain_check(home, &it.package, nmtool::run::DEADLINE);
    line(&mut out, format!("clean check: ok {} in {} ms{}", clean.ok, clean.ms, if clean.timed_out { " (deadline: warm-up)" } else { "" }));
    if clean.timed_out || (clean.ok && clean.ms > WARM_MS) {
        line(&mut out, format!("WARM-UP {id}: clean check took {} ms; rerun the preflight in a new run", clean.ms));
        return Ok(out);
    }
    if !clean.ok {
        let first = clean.stderr.lines().find(|l| l.starts_with("error")).unwrap_or("").to_owned();
        line(&mut out, format!("EXCLUDED {id}: the clean package does not build: {first}"));
        return Ok(out);
    }
    {
        let _restore = Restore(path.clone(), original.clone());
        write_ahead(&path, seeded(&text, after, &it.insert).as_bytes()).map_err(|e| bad(e.to_string()))?;
        let s = plain_check(home, &it.package, nmtool::run::DEADLINE);
        let named = s.stdout.contains(&it.code) || s.stderr.contains(&it.code);
        line(&mut out, format!("seeded check: ok {} in {} ms, names {}: {named}", s.ok, s.ms, it.code));
        if s.ok {
            line(&mut out, format!("EXCLUDED {id}: the seed does not break the build"));
        } else if !named {
            line(&mut out, format!("EXCLUDED {id}: the seed breaks the build without {}", it.code));
        } else {
            line(&mut out, format!("INCLUDED {id}"));
        }
    }
    let h1 = nmhash::sha256_bytes(&std::fs::read(&path).map_err(|e| bad(e.to_string()))?);
    line(&mut out, format!("restored: sha256 equal {}", h0 == h1));
    if h0 != h1 {
        return Err(bad(format!("{id}: file not restored")));
    }
    line(&mut out, format!("peak RAM {} MB (process working set)", peak()));
    Ok(out)
}

fn tokens(tok: &nmtokz::Tokenizer, text: &str) -> Result<usize, Fault> {
    tok.encode(text).map(|v| v.len()).map_err(|e| bad(e.to_string()))
}

/// Gates 5 and 6 for one item.
fn run_item(home: &Path, id: &str, lines_form: bool) -> Result<(String, Value), Fault> {
    let t0 = Instant::now();
    let it = item(home, id)?;
    let mut out = String::new();
    let tree = home.join(TREE);
    let path = tree.join(&it.file);
    let original = std::fs::read(&path).map_err(|e| bad(format!("{}: {e}", it.file)))?;
    let text = String::from_utf8_lossy(&original).into_owned();
    let h0 = nmhash::sha256_bytes(&original);
    let (after, expected, note) = place(&text, &it).map_err(|why| bad(format!("{id} excluded: {why}")))?;
    line(&mut out, format!("item {id} ({}, {}:{expected}, {}): {note}", it.package, it.file, it.code));
    let tok = nmtokz::Tokenizer::open(&nmtype::models_dir(home).join("qwen3-0.6b/files/tokenizer.json"))
        .map_err(|e| bad(format!("tokenizer: {e}")))?;
    let (base, r1, r2, raw_json) = {
        let _restore = Restore(path.clone(), original.clone());
        write_ahead(&path, seeded(&text, after, &it.insert).as_bytes()).map_err(|e| bad(e.to_string()))?;
        let base = plain_check(home, &it.package, ITEM_DEADLINE);
        let mem = home.join(format!("tests/stage-gates/s9/s9-mem-{}", nmtype::now_ms()));
        let org = nmapi::forage::Organism::open_with(home, &tree, &[], false, &mem)?;
        nmapi::build::set_target(home.join(TARGET));
        nmapi::tools::install(org).map_err(|e| bad(e.message))?;
        let api = nmapi::tools::api();
        let mut fields =
            vec![("package".to_owned(), Value::String(it.package.clone())), ("kind".to_owned(), Value::String("check".into()))];
        if !lines_form {
            fields.push(("form".to_owned(), Value::String("json".into())));
        }
        let req = Value::Object(fields.into_iter().collect());
        let r1 = api.dispatch("nm_build_run", &req).map_err(|e| bad(format!("nm_build_run: {}", e.message)))?;
        let run_id = st(&r1, "run");
        let r2 = if lines_form {
            Value::Null
        } else {
            api.dispatch("nm_build_explain", &Value::Object([("run".to_owned(), Value::String(run_id.clone()))].into_iter().collect()))
                .map_err(|e| bad(format!("nm_build_explain: {}", e.message)))?
        };
        let raw_json = nmapi::build::by_id(&run_id).map_or(0, |r| r.raw_bytes);
        (base, r1, r2, raw_json)
    };
    let h1 = nmhash::sha256_bytes(&std::fs::read(&path).map_err(|e| bad(e.to_string()))?);
    let t1 = r1.encode().map_err(|e| bad(e.to_string()))?;
    let t2 = if lines_form { String::new() } else { r2.encode().map_err(|e| bad(e.to_string()))? };
    let base_text = format!("{}{}", base.stdout, base.stderr);
    let (tool_tok, base_tok) = (tokens(&tok, &t1)? + if lines_form { 0 } else { tokens(&tok, &t2)? }, tokens(&tok, &base_text)?);
    // The lines form's first line: `<code> <file>:<line>:<col> <message>`.
    let first = r1.get("lines").and_then(Value::array).ok().and_then(|l| l.first()).and_then(|x| x.string().ok()).unwrap_or("").to_owned();
    let (code_l, place_l) = first
        .split_once(' ')
        .map_or((String::new(), String::new()), |(c, rest)| (c.to_owned(), rest.split(' ').next().unwrap_or("").to_owned()));
    let mut parts = place_l.rsplitn(3, ':');
    let (_, line_l, file_l) =
        (parts.next(), parts.next().and_then(|x| x.parse::<usize>().ok()).unwrap_or(0), parts.next().unwrap_or("").to_owned());
    let file = if lines_form { file_l } else { st(&r2, "file") };
    let got_line = if lines_form { line_l } else { num(&r2, "line") };
    let localized = file == it.file && got_line == expected;
    let cause_right = if lines_form { code_l == it.code } else { st(&r2, "code") == it.code };
    let fix_source = if lines_form { "rustc-help".to_owned() } else { st(&r2, "fix_source") };
    line(
        &mut out,
        format!(
            "baseline cargo: ok {} in {} ms, {} bytes, {base_tok} tokens{}",
            base.ok,
            base.ms,
            base_text.len(),
            if base.timed_out { " (DEADLINE)" } else { "" }
        ),
    );
    line(
        &mut out,
        format!(
            "nm_build_run: {} bytes; nm_build_explain: {} bytes; {tool_tok} tokens together; raw JSON cargo output {raw_json} bytes",
            t1.len(),
            t2.len()
        ),
    );
    line(
        &mut out,
        format!(
            "explained at {file}:{got_line} (expected {}:{expected}): localized {localized}; code {} (expected {}): cause right {cause_right}; fix source {fix_source}",
            it.file,
            st(&r2, "code"),
            it.code
        ),
    );
    line(&mut out, format!("restored: sha256 equal {}", h0 == h1));
    line(&mut out, format!("wall {} ms, peak RAM {} MB", t0.elapsed().as_millis(), peak()));
    line(&mut out, format!("reply nm_build_run: {t1}"));
    line(&mut out, format!("reply nm_build_explain: {t2}"));
    if h0 != h1 {
        return Err(bad(format!("{id}: file not restored")));
    }
    let n = |x: usize| Value::Number(x.to_string());
    let j = Value::Object(
        [
            ("id", Value::String(id.to_owned())),
            ("localized", Value::Bool(localized)),
            ("cause_right", Value::Bool(cause_right)),
            ("fix_source", Value::String(fix_source)),
            ("tool_tokens", n(tool_tok)),
            ("baseline_tokens", n(base_tok)),
            ("tool_bytes", n(t1.len() + t2.len())),
            ("baseline_bytes", n(base_text.len())),
            ("raw_json_bytes", n(raw_json)),
            ("baseline_ms", n(usize::try_from(base.ms).unwrap_or(0))),
            ("wall_ms", n(usize::try_from(t0.elapsed().as_millis()).unwrap_or(0))),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect(),
    );
    Ok((out, j))
}

/// Gate 7: hashes of the fixture files and lock files in the scratch tree and the live tree.
fn hashes(home: &Path, live: Option<&Path>) -> Result<String, Fault> {
    let v = nmjson::parse(&std::fs::read(home.join("designs/breakages.json")).map_err(|e| bad(e.to_string()))?)
        .map_err(|e| bad(e.to_string()))?;
    let mut out = String::new();
    let mut files: Vec<String> =
        v.get("items").and_then(Value::array).map_err(|e| bad(e.to_string()))?.iter().map(|i| st(i, "file")).collect();
    files.push("Cargo.lock".into());
    let h = |p: PathBuf| std::fs::read(&p).map_or_else(|_| "missing".to_owned(), |b| nmhash::sha256_bytes(&b));
    for f in &files {
        let a = h(home.join(TREE).join(f));
        let b = live.map_or_else(|| "not given".to_owned(), |l| h(l.join(f)));
        line(&mut out, format!("{f} scratch {a} live {b} same {}", a == b));
    }
    if let Some(l) = live {
        let o = nmtool::run::run(l, "git", &["-c", "safe.directory=*", "rev-parse", "HEAD"], &[], Duration::from_secs(10));
        line(
            &mut out,
            format!(
                "live git HEAD (read only): {}",
                if o.ok { o.stdout.trim().to_owned() } else { format!("not read: {}", o.stderr.trim()) }
            ),
        );
    }
    Ok(out)
}

/// `run s9 <preflight <id>|item <id>|hashes <label> [--live <path>]>`.
pub fn run(args: &[String], home: &Path) -> ExitCode {
    let part = args.get(2).map_or("", String::as_str);
    let arg = args.get(3).map_or("", String::as_str);
    let stamp = nmtype::now_ms();
    let live = args.iter().position(|a| a == "--live").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let result: Result<(String, Option<Value>), Fault> = match part {
        "preflight" => preflight(home, arg).map(|t| (t, None)),
        "item" => run_item(home, arg, args.iter().any(|a| a == "--lines")).map(|(t, j)| (t, Some(j))),
        "hashes" => hashes(home, live.as_deref()).map(|t| (t, None)),
        other => Err(bad(format!("unknown S9 part {other:?}: preflight <id>, item <id>, hashes <label> [--live <path>]"))),
    };
    let label = if args.iter().any(|a| a == "--lines") { format!("{part}-{arg}-lines") } else { format!("{part}-{arg}") };
    match result {
        Ok((text, json)) => {
            let p = home.join(format!("runs/S9-{label}-{stamp}.txt"));
            let _ = std::fs::write(&p, &text);
            if let Some(j) = json {
                let _ = std::fs::write(home.join(format!("runs/S9-{label}-{stamp}.json")), j.encode().unwrap_or_default());
            }
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = std::fs::write(home.join(format!("runs/S9-{label}-{stamp}.txt")), format!("FAULT {e}\n"));
            eprintln!("s9 {part}: {e}");
            ExitCode::FAILURE
        }
    }
}
