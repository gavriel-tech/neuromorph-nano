//! Outcomes for memory: what the evidence ledger hands the memory system. Two kinds, each decided
//! by stated rules from what really happened, and each written to the ledger as one `outcome` row:
//!
//! - `answered`: a context pack whose cited lines Claude Code read or edited directly, then a
//!   verdict on what followed. Verified by a passing build or test after an edit of a cited place
//!   (indirect) or an accepting next prompt (weak); refuted by a failing build or test after such an
//!   edit (indirect), an undo of the edited cited file (its strength), a correcting next prompt
//!   (weak) or the same question asked again (its strength). The stronger side wins; a tie is
//!   refuted. The memory keeps the question asked of the Nano, the cited spans whose lines were
//!   read or edited, and the other places edited in the same turn (each as the read around it).
//! - `fixed`: a build or test that failed with a located error, edits after it, then the same
//!   check (class and package) passing in the same session. Direct when an edit was in the error's
//!   file within 20 lines of it, else indirect. An undo (git revert, restore, checkout, reset,
//!   stash) naming a fixed file later in the session gives a second, refuting row.
//!
//! Each place keeps the hash of its lines read fresh when the row is written, so recall can tell
//! later whether the lines are still the ones that were verified. Nothing of the lines is kept.
//! The memory system takes the rows into its journal (`nmmem::outcomes`); this file only decides
//! them.
use crate::ledger::PackState;
use crate::sessions::Rec;
use nmjson::Value;
use nmmem::outcomes::{Failure, Known, Outcome, Place};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::ExitCode;

/// Lines within this distance of an error count as the error's place.
pub const NEAR_LINES: u64 = 20;
/// Most files one `fixed` outcome names.
const MAX_FIX_FILES: usize = 3;
/// Most places one `answered` outcome keeps.
const MAX_ANSWER_PLACES: usize = 8;

fn sha16(t: &str) -> String {
    nmhash::sha256_bytes(t.as_bytes()).chars().take(16).collect()
}

/// A place read fresh now: its lines' hash, or `None` when the file or the lines are not there.
#[must_use]
pub fn place_now(root: &Path, path: &str, start: u64, end: u64) -> Option<Place> {
    if path == "<outside>" || path.contains("..") {
        return None;
    }
    let (start, end) = (usize::try_from(start).ok()?, usize::try_from(end).ok()?);
    let file = path.split('/').fold(root.to_path_buf(), |a, c| a.join(c));
    let body = std::fs::read_to_string(file).ok()?;
    let hash = nmmem::outcomes::lines_hash_of(&body, start, end)?;
    Some(Place { path: path.to_owned(), start, end, hash })
}

/// The `answered` outcome of a pack whose window has closed, if its joins decide one.
#[must_use]
pub fn answered(root: &Path, recs: &[Rec], st: &PackState) -> Option<Outcome> {
    st.closed?;
    let question = st.pack.rec.nano.as_ref()?.1.clone().filter(|q| !q.trim().is_empty())?;
    if !st.evs.iter().any(|e| matches!(e.rule, "read" | "edit") && e.state == "used" && e.strength == Some("direct")) {
        return None;
    }
    // The cited spans Claude Code's reads and edits in the pack's turn touched directly (their
    // lines overlap the span; an edit's lines are where its old text was): only those are
    // remembered as the answer, not every span the pack cited in the same file.
    let w = crate::ledger::window(recs, &st.pack, u64::MAX);
    let pre: BTreeMap<&str, &Vec<(String, u64, u64)>> = recs
        .iter()
        .filter(|r| r.event == "PreToolUse" && r.session == st.pack.rec.session)
        .filter_map(|r| Some((r.use_id.as_deref()?, &r.lines)))
        .collect();
    let mut touched: Vec<(String, u64, u64)> = Vec::new();
    let mut reads: Vec<(String, u64, u64)> = Vec::new();
    let mut edits: Vec<(String, u64, u64)> = Vec::new();
    for r in w.turn.iter().filter(|r| r.done() && r.ok == Some(true) && matches!(r.kind.as_str(), "read" | "edit" | "write")) {
        let mut lines = r.lines.clone();
        lines.extend(r.use_id.as_deref().and_then(|u| pre.get(u)).into_iter().flat_map(|l| l.iter().cloned()));
        if r.kind == "read" { reads.extend(lines.iter().cloned()) } else { edits.extend(lines.iter().cloned()) }
        touched.extend(lines);
    }
    let mut places: Vec<Place> = st
        .pack
        .cited
        .iter()
        .filter_map(|(p, sp)| {
            let (a, b) = (*sp)?;
            touched.iter().any(|(q, x, y)| q == p && *x <= b && a <= *y).then(|| place_now(root, p, a, b)).flatten()
        })
        .collect();
    if places.is_empty() {
        return None;
    }
    // Where else Claude Code worked on the answer in the turn: each edit the pack's places do not
    // hold, as the smallest read of its file around it (else its own lines), so a question whose
    // answer the pack held only in part is remembered whole once the work is verified.
    for (p, a, b) in &edits {
        let wide = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        if places.iter().any(|x| &x.path == p && wide(x.start) <= *b && *a <= wide(x.end)) || places.len() >= MAX_ANSWER_PLACES {
            continue;
        }
        let around = reads.iter().filter(|(q, x, y)| q == p && x <= a && b <= y).min_by_key(|(_, x, y)| y - x).map_or((*a, *b), |(_, x, y)| (*x, *y));
        if let Some(pl) = place_now(root, p, around.0, around.1) {
            places.push(pl);
        }
    }
    let side = |state: &str| {
        st.evs
            .iter()
            .filter(|e| e.state == state)
            .filter(|e| match e.rule {
                "check" => e.strength == Some("indirect"),
                "revert" | "next-prompt" | "asked-again" => true,
                _ => false,
            })
            .max_by_key(|e| nmmem::outcomes::rank(e.strength.unwrap_or("")))
    };
    let (good, bad) = (side("verified"), side("refuted"));
    let r = |e: Option<&crate::ledger::Ev>| e.map_or(0, |e| nmmem::outcomes::rank(e.strength.unwrap_or("")));
    let decider = match (good, bad) {
        (None, None) => return None,
        _ if r(bad) >= r(good) => bad?,
        _ => good?,
    };
    Some(Outcome {
        id: format!("answered:{}", st.pack.id),
        kind: "answered".into(),
        verified: decider.state == "verified",
        strength: decider.strength.unwrap_or("weak").to_owned(),
        rule: decider.rule.to_owned(),
        question: Some(question),
        ask_sha: None,
        failure: None,
        places,
        session: st.pack.rec.session.clone(),
        at: decider.event.as_ref().map_or(st.pack.at, |x| x.0.t),
    })
}

fn is_undo(r: &Rec) -> bool {
    matches!(r.cmd_class.as_deref(), Some("git-revert" | "git-restore" | "git-checkout" | "git-reset" | "git-stash"))
}

/// Edits made since a failure: path and lines.
type Edits = Vec<(String, u64, u64)>;

/// The `fixed` outcomes of a project's records (every session), in time order.
#[must_use]
pub fn fixes(root: &Path, recs: &[Rec]) -> Vec<Outcome> {
    let mut out = Vec::new();
    let mut by_session: BTreeMap<&str, Vec<&Rec>> = BTreeMap::new();
    for r in recs.iter().filter(|r| r.done() && !r.session.is_empty()) {
        by_session.entry(r.session.as_str()).or_default().push(r);
    }
    for (session, mut list) in by_session {
        list.sort_by_key(|r| (r.t, r.i));
        // Open failures by check, each with the edits made since: (failing record, edits).
        let mut open: BTreeMap<String, (&Rec, Edits)> = BTreeMap::new();
        // Fixes made, with the files they changed, for the undo rule.
        let mut made: Vec<(Outcome, BTreeSet<String>)> = Vec::new();
        for r in list {
            match r.kind.as_str() {
                "build" | "test" => {
                    let key = format!("{} {}", r.cmd_class.as_deref().unwrap_or("check"), r.pkg.as_deref().unwrap_or(""));
                    if r.ok == Some(false) {
                        if r.errors.first().is_some_and(|e| !e.1.is_empty() && e.1 != "<outside>") {
                            open.insert(key, (r, Vec::new()));
                        }
                    } else if r.ok == Some(true)
                        && let Some((f, edits)) = open.remove(&key)
                        && !edits.is_empty()
                        && let Some(o) = fixed(root, session, f, r, &edits)
                    {
                        let files = edits.iter().map(|e| e.0.clone()).collect();
                        out.push(o.clone());
                        made.push((o, files));
                    }
                }
                "edit" | "write" if r.ok == Some(true) => {
                    for (_, edits) in open.values_mut() {
                        for p in r.paths.iter().filter(|p| !matches!(p.as_str(), "." | "<outside>")) {
                            if let Some((a, b)) = r.span_of(p) {
                                edits.push((p.clone(), a, b));
                            }
                        }
                    }
                }
                _ if is_undo(r) && r.ok == Some(true) => {
                    for (o, files) in &made {
                        let undone = format!("{}:undone", o.id);
                        if r.paths.iter().any(|p| files.contains(p)) && !out.iter().any(|x: &Outcome| x.id == undone) {
                            out.push(Outcome { id: undone, verified: false, strength: "direct".into(), rule: "revert".into(), at: r.t, ..o.clone() });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn fixed(root: &Path, session: &str, fail: &Rec, pass: &Rec, edits: &[(String, u64, u64)]) -> Option<Outcome> {
    let (code, file, line) = fail.errors.first()?.clone();
    let mut spans: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    for (p, a, b) in edits {
        spans.entry(p.clone()).or_default().push((*a, *b));
    }
    // The error's own file first, then the others in name order.
    let mut files: Vec<String> = spans.keys().cloned().collect();
    files.sort_by_key(|p| (p != &file, p.clone()));
    let mut places = Vec::new();
    let mut near = false;
    for p in files.iter().take(MAX_FIX_FILES) {
        for (a, b) in crate::sessions::merge(spans.get(p).cloned().unwrap_or_default()) {
            near |= p == &file && a <= line.saturating_add(NEAR_LINES) && line <= b.saturating_add(NEAR_LINES);
            if let Some(pl) = place_now(root, p, a, b) {
                places.push(pl);
            }
        }
    }
    if places.is_empty() {
        return None;
    }
    let failure = Failure {
        check: fail.cmd_class.clone().unwrap_or_else(|| "check".into()),
        pkg: fail.pkg.clone().unwrap_or_default(),
        code,
        file,
        line,
        msg: fail.error_msg.clone().unwrap_or_default(),
    };
    Some(Outcome {
        id: format!("fixed:{}", sha16(&format!("{session}|{}|{}", fail.t, pass.t))),
        kind: "fixed".into(),
        verified: true,
        strength: if near { "direct" } else { "indirect" }.into(),
        rule: "check".into(),
        question: None,
        ask_sha: None,
        failure: Some(failure),
        places,
        session: session.to_owned(),
        at: pass.t,
    })
}

/// Every outcome the ledger can write now: the answered outcomes of closed packs and the fixes in
/// the records.
#[must_use]
pub fn all(root: &Path, recs: &[Rec], packs: &[PackState]) -> Vec<Outcome> {
    let mut v: Vec<Outcome> = packs.iter().filter_map(|st| answered(root, recs, st)).collect();
    v.extend(fixes(root, recs));
    v
}

/// One known outcome as one plain line.
#[must_use]
pub fn known_line(k: &Known, root: &Path) -> String {
    let places: Vec<String> = k
        .places
        .iter()
        .map(|p| format!("{}{}", p.label(), if nmmem::outcomes::fresh(root, p) { "" } else { " (changed since)" }))
        .collect();
    let what = match (&k.question, &k.failure) {
        (Some(q), _) => format!("answered \"{q}\""),
        (None, Some(f)) => format!("fixed {} {} at {}:{}", f.check_label(), f.code, f.file, f.line),
        _ => k.kind.clone(),
    };
    format!(
        "{} {what} at {}: {} (verified {}, refuted {}{}){}",
        k.record,
        places.join(", "),
        if k.standing { "verified" } else { "REFUTED" },
        k.verified,
        k.refuted,
        if k.strength.is_empty() { String::new() } else { format!(", strongest {}", k.strength) },
        if k.in_memory { "" } else { " [in the ledger, not yet in the journal]" }
    )
}

fn known_value(k: &Known, root: &Path) -> Value {
    let s = |t: &str| Value::String(t.to_owned());
    let n = |x: u64| Value::Number(x.to_string());
    let places = k
        .places
        .iter()
        .map(|p| {
            Value::Object(
                [("place".to_owned(), s(&p.label())), ("fresh".to_owned(), Value::Bool(nmmem::outcomes::fresh(root, p)))].into_iter().collect(),
            )
        })
        .collect();
    let mut m: BTreeMap<String, Value> = BTreeMap::new();
    m.insert("record".into(), s(&k.record));
    m.insert("kind".into(), s(&k.kind));
    m.insert("question".into(), k.question.as_deref().map_or(Value::Null, s));
    m.insert(
        "failure".into(),
        k.failure.as_ref().map_or(Value::Null, |f| {
            Value::Object(
                [("check", s(&f.check_label())), ("code", s(&f.code)), ("file", s(&f.file)), ("line", n(f.line)), ("msg", s(&f.msg))]
                    .into_iter()
                    .map(|(a, b)| (a.to_owned(), b))
                    .collect(),
            )
        }),
    );
    m.insert("places".into(), Value::Array(places));
    m.insert("verified".into(), n(u64::from(k.verified)));
    m.insert("refuted".into(), n(u64::from(k.refuted)));
    m.insert("strength".into(), s(&k.strength));
    m.insert("standing".into(), s(if k.standing { "verified" } else { "refuted" }));
    m.insert("in_memory".into(), Value::Bool(k.in_memory));
    Value::Object(m)
}

/// The studied root of a project memory folder: the Nano's own folder for `nano`, else the root
/// `project.json` names (written by `serve`) when its key matches the folder's.
#[must_use]
pub fn project_root_of(home: &Path, key: &str, memory: &Path) -> Option<std::path::PathBuf> {
    if key == "nano" {
        return Some(home.to_path_buf());
    }
    let v = std::fs::read(memory.join("project.json")).ok().and_then(|b| nmjson::parse(&b).ok())?;
    let root = nmsubs::repo::canon(Path::new(v.get("root").ok()?.string().ok()?));
    (root.is_dir() && nmtype::project::key(&root) == key).then_some(root)
}

/// `/api/outcomes?project=<key>`: what memory knows of verified and refuted outcomes in a project,
/// with each place read fresh. Read only.
#[must_use]
pub fn api_view(home: &Path, query: &str) -> Value {
    let s = |t: &str| Value::String(t.to_owned());
    let Some((key, memory)) = crate::ledger::project_memory(home, query) else {
        return Value::Object([("error".to_owned(), s("project: give the key of a project as /api/projects lists it"))].into_iter().collect());
    };
    let root = project_root_of(home, &key, &memory);
    let known = nmmem::outcomes::known(&memory);
    let list = known.iter().map(|k| known_value(k, root.as_deref().unwrap_or(Path::new("")))).collect();
    Value::Object(
        [("project".to_owned(), s(&key)), ("root_known".to_owned(), Value::Bool(root.is_some())), ("outcomes".to_owned(), Value::Array(list))].into_iter().collect(),
    )
}

/// `neuromorph-nano outcomes --project <repo> [--recall "<question>"] [--import] [--json]`: what
/// memory knows of verified outcomes, each place read fresh; `--recall` lists the answered ones a
/// question recalls; `--import` brings the ledger's outcomes into the memory journal now (refused
/// while a server holds the journal; the server imports them when it next opens the memory).
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let arg = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
    let Some(repo) = arg("--project") else {
        eprintln!("usage: neuromorph-nano outcomes --project <repo> [--recall \"<question>\"] [--import] [--json]");
        return ExitCode::FAILURE;
    };
    let root = nmsubs::repo::canon(Path::new(&repo));
    let memory = nmtype::project::memory_dir(home, &root);
    if args.iter().any(|a| a == "--import") {
        if crate::controls::memory_off(home) {
            println!("memory is off: nothing imported");
            return ExitCode::SUCCESS;
        }
        let r = nmmem::mem2::Memory2::open(&memory, &root).and_then(|mut m| nmmem::outcomes::import_ledger(&mut m, &memory));
        match r {
            Ok((n, refused)) => {
                println!("imported into the memory journal: {n} events");
                for x in refused {
                    println!("refused (never saved): {x}");
                }
            }
            Err(e) => {
                eprintln!("outcomes: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let known = nmmem::outcomes::known(&memory);
    if let Some(q) = arg("--evidence") {
        print_evidence(&root, &known, &q);
        return ExitCode::SUCCESS;
    }
    if let Some(q) = arg("--recall") {
        let hits = nmmem::outcomes::recall(&known, &q);
        if args.iter().any(|a| a == "--json") {
            let list = hits
                .iter()
                .map(|(sc, k)| {
                    let mut v = known_value(k, &root);
                    if let Value::Object(m) = &mut v {
                        m.insert("score".into(), Value::Number(format!("{sc:.3}")));
                    }
                    v
                })
                .collect();
            println!("{}", Value::Array(list).encode().unwrap_or_default());
            return ExitCode::SUCCESS;
        }
        println!("recalled for \"{q}\": {}", hits.len());
        for (sc, k) in hits {
            println!("  {sc:.2}  {}", known_line(k, &root));
        }
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--json") {
        println!("{}", Value::Array(known.iter().map(|k| known_value(k, &root)).collect()).encode().unwrap_or_default());
        return ExitCode::SUCCESS;
    }
    println!("outcomes known to memory in {} ({}):", memory.display(), known.len());
    for k in &known {
        println!("  {}", known_line(k, &root));
    }
    ExitCode::SUCCESS
}

/// `--evidence "<question>"`: memory's evidence for a question as a serving process computes it
/// (`nmmem::evidence`), each place read fresh from the files, with the match of every outcome.
fn print_evidence(root: &Path, known: &[Known], q: &str) {
    let sub = nmsubs::Substrate::scan(root, &["neuromorph-nano".to_owned()], std::sync::Arc::new(|_: &str| Vec::new()));
    let idf = |t: &str| sub.idf(t);
    let read = |p: &Place| -> Option<(Place, nmtype::class::FromMaterial<String>)> {
        let file = p.path.split('/').fold(root.to_path_buf(), |a, c| a.join(c));
        let body = std::fs::read_to_string(file).ok()?;
        let lines: Vec<&str> = body.lines().collect();
        let len = p.end.checked_sub(p.start)? + 1;
        let at = (1..=lines.len().saturating_sub(len - 1))
            .filter(|s| nmmem::outcomes::lines_hash_of(&body, *s, s + len - 1).is_some_and(|h| h == p.hash))
            .min_by_key(|s| s.abs_diff(p.start))?;
        Some((Place { start: at, end: at + len - 1, ..p.clone() }, nmtype::class::FromMaterial::new(lines.get(at - 1..at - 1 + len)?.join("\n"))))
    };
    let ev = nmmem::evidence::evidence(known, &nmmem::evidence::Request { text: q, idf: &idf, read: &read });
    println!("evidence for \"{q}\": {} pieces, {} stale", ev.pieces.len(), ev.stale);
    for p in &ev.pieces {
        println!("  {:.2}  {:<8}  {}  {} (\"{}\")", p.strength, p.kind.name(), p.place.label(), p.record, p.question);
    }
    let mut shares = ev.shares.clone();
    shares.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (rec, s) in shares {
        println!("  match {s:.2}  {rec}");
    }
}

/// Section check (temporary folder): a failing test with a located error, an edit near it and the
/// same test passing give one direct `fixed` outcome with the edit's place; an undo of the file
/// gives a refuting row; a failure fixed by no edit gives nothing.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let root = std::env::temp_dir().join(format!("nano-outcomes-check-{}-{}", std::process::id(), nmtype::now_ms()));
    std::fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(root.join("src/a.rs"), "fn a() {}\nfn b() -> u8 {\n    1\n}\n").map_err(|e| e.to_string())?;
    let lines = [
        r#"{"t":1,"event":"PostToolUseFailure","kind":"test","session":"s1","tool":"Bash","cmd_class":"cargo-test","pkg":"a","ok":false,"errors":[{"code":"E0308","file":"src/a.rs","line":3,"msg":"mismatched types"}]}"#,
        r#"{"t":2,"event":"PostToolUse","kind":"edit","session":"s1","tool":"Edit","paths":["src/a.rs"],"lines":[["src/a.rs",2,4]],"ok":true}"#,
        r#"{"t":3,"event":"PostToolUse","kind":"test","session":"s1","tool":"Bash","cmd_class":"cargo-test","pkg":"a","ok":true}"#,
        r#"{"t":4,"event":"PostToolUse","kind":"vcs","session":"s1","tool":"Bash","cmd_class":"git-restore","paths":["src/a.rs"],"ok":true}"#,
        r#"{"t":5,"event":"PostToolUseFailure","kind":"build","session":"s2","tool":"Bash","cmd_class":"cargo-check","ok":false,"errors":[{"file":"src/a.rs","line":1}]}"#,
        r#"{"t":6,"event":"PostToolUse","kind":"build","session":"s2","tool":"Bash","cmd_class":"cargo-check","ok":true}"#,
    ];
    let recs: Vec<Rec> = lines.iter().enumerate().filter_map(|(i, l)| nmjson::parse(l.as_bytes()).ok().map(|v| Rec::from_value(i, &v))).collect();
    let got = fixes(&root, &recs);
    let _ = std::fs::remove_dir_all(&root);
    let summary: Vec<String> = got.iter().map(|o| format!("{} {} {} {}", o.kind, o.verified, o.strength, o.places.iter().map(Place::label).collect::<Vec<_>>().join(","))).collect();
    if summary != ["fixed true direct src/a.rs:2-4", "fixed false direct src/a.rs:2-4"] {
        return Err(format!("outcomes: fixes {summary:?}"));
    }
    if got.first().and_then(|o| o.failure.as_ref()).map(|f| (f.code.as_str(), f.msg.as_str())) != Some(("E0308", "mismatched types")) {
        return Err("outcomes: the failure's code and message".into());
    }
    Ok(())
}
