//! The situation record's reader: the records the hook intake wrote for a project, filtered by
//! session and cut to the newest ones, and the `situation` command that prints them plainly, shows
//! the kept prompt text beside its prompt, and clears that text on request.
use crate::hook::{prompts_file, situation_file};
use nmjson::Value;
use std::path::Path;
use std::process::ExitCode;

/// The records of a project memory folder, oldest first: only those of `session` when given, the
/// newest `last` when given. Lines that do not parse are passed over.
#[must_use]
pub fn read(memory: &Path, session: Option<&str>, last: Option<usize>) -> Vec<Value> {
    let text = std::fs::read_to_string(situation_file(memory)).unwrap_or_default();
    let mut rows: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter(|v| session.is_none_or(|s| v.get("session").and_then(Value::string).ok() == Some(s)))
        .collect();
    if let Some(n) = last {
        let skip = rows.len().saturating_sub(n);
        rows.drain(..skip);
    }
    rows
}

/// The kept prompt text by its hash.
fn kept_prompts(memory: &Path) -> Vec<(String, String)> {
    std::fs::read_to_string(prompts_file(memory))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter_map(|v| Some((v.get("sha").and_then(Value::string).ok()?.to_owned(), v.get("text").and_then(Value::string).ok()?.to_owned())))
        .collect()
}

fn field<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::string).unwrap_or_default()
}

/// One record as one plain line.
#[must_use]
pub fn line(v: &Value, prompts: &[(String, String)]) -> String {
    let t = v.get("t").and_then(Value::integer).unwrap_or(0);
    let mut parts = vec![
        nmmusc::pack::utc_stamp(t).replace('T', " ").trim_end_matches('Z').to_owned(),
        field(v, "session").chars().take(8).collect(),
        field(v, "event").to_owned(),
        field(v, "kind").to_owned(),
    ];
    for k in ["tool", "source", "trigger", "cmd"] {
        let x = field(v, k);
        if !x.is_empty() {
            parts.push(x.to_owned());
        }
    }
    if let Ok(nano) = v.get("nano") {
        parts.push(format!("{} episode {}", field(nano, "tool"), field(nano, "episode")));
    }
    match v.get("ok") {
        Ok(Value::Bool(true)) => parts.push("ok".into()),
        Ok(Value::Bool(false)) => parts.push("FAILED".into()),
        _ => {}
    }
    if let Ok(Value::Number(e)) = v.get("exit") {
        parts.push(format!("exit {e}"));
    }
    if let Ok(Value::Array(c)) = v.get("tests") {
        let num = |i: usize| c.get(i).and_then(|x| x.integer().ok()).unwrap_or(0);
        parts.push(format!("tests {} passed {} failed", num(0), num(1)));
    }
    let lines: Vec<String> = v
        .get("lines")
        .and_then(Value::array)
        .unwrap_or_default()
        .iter()
        .filter_map(|l| {
            let a = l.array().ok()?;
            Some(format!("{}:{}-{}", a.first()?.string().ok()?, a.get(1)?.integer().ok()?, a.get(2)?.integer().ok()?))
        })
        .collect();
    if lines.is_empty() {
        let paths: Vec<&str> = v.get("paths").and_then(Value::array).unwrap_or_default().iter().filter_map(|p| p.string().ok()).collect();
        if !paths.is_empty() {
            parts.push(paths.join(" "));
        }
    } else {
        parts.push(lines.join(" "));
    }
    for e in v.get("errors").and_then(Value::array).unwrap_or_default() {
        parts.push(format!("{} {}:{} {}", field(e, "code"), field(e, "file"), e.get("line").and_then(Value::integer).unwrap_or(0), field(e, "msg")));
    }
    if let Ok(p) = v.get("prompt") {
        let sha = field(p, "sha");
        let chars = p.get("chars").and_then(Value::integer).unwrap_or(0);
        match prompts.iter().find(|(h, _)| h == sha) {
            Some((_, text)) => parts.push(format!("\"{}\"", text.replace('\n', " "))),
            None if matches!(p.get("kept"), Ok(Value::Bool(true))) => parts.push(format!("({chars} characters; text cleared or expired)")),
            None => parts.push(format!("({chars} characters; text not kept)")),
        }
    }
    if let Ok(d) = v.get("digest") {
        parts.push(format!("output {} lines, last: {}", d.get("lines").and_then(Value::integer).unwrap_or(0), field(d, "last")));
    }
    parts.retain(|p| !p.is_empty());
    parts.join("  ")
}

/// Removes the kept prompt text of a project memory folder; returns how many prompts it held.
/// # Errors
/// The file cannot be removed.
pub fn clear_prompts(memory: &Path) -> Result<usize, String> {
    let f = prompts_file(memory);
    let n = std::fs::read_to_string(&f).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
    if f.exists() {
        std::fs::remove_file(&f).map_err(|e| format!("{}: {e}", f.display()))?;
    }
    Ok(n)
}

/// `neuromorph-nano situation --project <repo> [--session <id>] [--last <n>] [--json]
/// [--clear-prompts]`.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let arg = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let Some(repo) = arg("--project") else {
        eprintln!("usage: neuromorph-nano situation --project <repo> [--session <id>] [--last <n>] [--json] [--clear-prompts]");
        return ExitCode::FAILURE;
    };
    let repo = nmsubs::repo::canon(Path::new(&repo));
    let memory = nmtype::project::memory_dir(home, &repo);
    if args.iter().any(|a| a == "--clear-prompts") {
        return match clear_prompts(&memory) {
            Ok(n) => {
                let _ = crate::controls::audit(
                    home,
                    "privacy",
                    vec![
                        ("label", Value::String(format!("Kept prompt text cleared: {n} prompts"))),
                        ("project", Value::String(nmtype::project::label(home, &repo))),
                        ("cleared", Value::Number(n.to_string())),
                    ],
                );
                println!("cleared the kept text of {n} prompts in {}", prompts_file(&memory).display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("situation: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let days = crate::controls::current(home).get("prompts.days").and_then(|d| d.parse::<u64>().ok()).unwrap_or(30);
    if let Err(e) = crate::hook::prune(&prompts_file(&memory), days) {
        eprintln!("situation: {e}");
    }
    let last = arg("--last").and_then(|n| n.parse::<usize>().ok());
    let rows = read(&memory, arg("--session").as_deref(), last.or(Some(50)));
    if args.iter().any(|a| a == "--json") {
        for r in &rows {
            println!("{}", r.encode().unwrap_or_default());
        }
        return ExitCode::SUCCESS;
    }
    println!("situation record: {} ({} records shown)", situation_file(&memory).display(), rows.len());
    let prompts = kept_prompts(&memory);
    for r in &rows {
        println!("{}", line(r, &prompts));
    }
    ExitCode::SUCCESS
}
