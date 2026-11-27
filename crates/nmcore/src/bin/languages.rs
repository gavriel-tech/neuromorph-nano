//! `neuromorph-nano check languages`: the language fixtures of `tests/languages/` against their
//! hand-written expected results.
//! - Lexer: every snippet of `lexer/<language>.txt` gives exactly its hand token list.
//! - Reader: each fixture repository is read file for file as `expected.json` lists it, and every
//!   planted file (installed packages, build output, minified, lock, binary, generated, ignored)
//!   is skipped with its reason.
//! - Tagger: definitions (file, name, kind, first line) found against the hand list: recall and
//!   precision at least 0.95 (0.9 for C and CUDA, where macros hide names), last lines exact for at
//!   least 0.9 of the definitions found.
//! - Imports and packages: at least 0.9 of the hand-listed imports resolved to the right file or
//!   outside package; the package list equal to the hand list.
//!
//! The questions of each fixture (`questions.json`) run through the context pack with
//! `neuromorph-nano bench --mode plain --root <repo> --questions <file>`.
use nmjson::Value;
use nmsubs::repo::{self, FileInfo, Skipped};
use std::collections::BTreeSet;
use std::path::Path;

fn s(v: &Value, k: &str) -> String {
    v.get(k).ok().and_then(|x| x.string().ok()).unwrap_or("").to_owned()
}
fn u(v: &Value, k: &str) -> usize {
    v.get(k).ok().and_then(|x| x.integer().ok()).and_then(|x| usize::try_from(x).ok()).unwrap_or(0)
}
fn list<'a>(v: &'a Value, k: &str) -> &'a [Value] {
    v.get(k).ok().and_then(|x| x.array().ok()).unwrap_or(&[])
}

/// The lexer snippets of one file: (name, source, hand tokens).
fn snippets(text: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut cur: Option<(String, Vec<&str>, Vec<&str>, bool)> = None;
    for line in text.lines() {
        if let Some(name) = line.strip_prefix("=== ") {
            if let Some((n, src, toks, _)) = cur.take() {
                out.push((n, src.join("\n"), toks.join(" ")));
            }
            cur = Some((name.trim().to_owned(), Vec::new(), Vec::new(), false));
        } else if line == "---" {
            if let Some(c) = cur.as_mut() {
                c.3 = true;
            }
        } else if let Some(c) = cur.as_mut() {
            if c.3 {
                if !line.trim().is_empty() {
                    c.2.push(line.trim());
                }
            } else {
                c.1.push(line);
            }
        }
    }
    if let Some((n, src, toks, _)) = cur {
        out.push((n, src.join("\n"), toks.join(" ")));
    }
    out
}

fn lexer(base: &Path) -> Result<usize, String> {
    let dir = base.join("lexer");
    let mut entries: Vec<_> = std::fs::read_dir(&dir).map_err(|e| format!("lexer: {e}"))?.flatten().map(|e| e.path()).collect();
    entries.sort();
    let mut total = 0;
    for p in entries.iter().filter(|p| p.extension().is_some_and(|e| e == "txt")) {
        let id = p.file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
        let lang = nmlex::lang::by_id(&id).ok_or_else(|| format!("lexer: no language {id}"))?;
        let text = std::fs::read_to_string(p).map_err(|e| format!("lexer: {e}"))?;
        let cases = snippets(&text);
        let mut wrong = Vec::new();
        for (name, src, want) in &cases {
            let got = nmlex::lang_lex::render(&nmlex::lang_lex::lex(src, lang));
            if got != *want {
                wrong.push(format!("{name}: got {got:?}, want {want:?}"));
            }
        }
        println!("     lexer {id}: {} of {} snippets equal the hand lists", cases.len() - wrong.len(), cases.len());
        if let Some(w) = wrong.first() {
            return Err(format!("lexer {id}: {} wrong; first {w}", wrong.len()));
        }
        total += cases.len();
    }
    Ok(total)
}

/// What one fixture measured against its rules.
struct Measure {
    name: String,
    failures: Vec<String>,
}

fn ratio(a: usize, b: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    if b == 0 { 1.0 } else { a as f64 / b as f64 }
}

fn fixture(dir: &Path) -> Result<Measure, String> {
    let name = dir.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    let exp = nmjson::parse(&std::fs::read(dir.join("expected.json")).map_err(|e| format!("{name}: {e}"))?)
        .map_err(|e| format!("{name}: {e}"))?;
    let root = dir.join("repo");
    let mut m = Measure { name: name.clone(), failures: Vec::new() };
    // Reader.
    let (mut files, mut skipped): (Vec<FileInfo>, Vec<Skipped>) = (Vec::new(), Vec::new());
    repo::walk_counted(&root, &root, &[], &mut files, &mut skipped);
    let got_read: BTreeSet<String> = files.iter().map(|f| f.path.clone()).collect();
    let want_read: BTreeSet<String> = list(&exp, "read").iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect();
    let got_skip: BTreeSet<String> = skipped.iter().map(|x| format!("{} | {}", x.path, x.reason)).collect();
    let want_skip: BTreeSet<String> = list(&exp, "skipped").iter().map(|x| format!("{} | {}", s(x, "path"), s(x, "reason"))).collect();
    let reader_ok = got_read == want_read && got_skip == want_skip;
    println!(
        "     {name} reader: {} read, {} skipped; file list {}",
        got_read.len(),
        got_skip.len(),
        if reader_ok { "exact" } else { "DIFFERS" }
    );
    if !reader_ok {
        let extra: Vec<&String> = got_read.difference(&want_read).chain(got_skip.difference(&want_skip)).collect();
        let missing: Vec<&String> = want_read.difference(&got_read).chain(want_skip.difference(&got_skip)).collect();
        m.failures.push(format!("reader: extra {extra:?}, missing {missing:?}"));
    }
    // Tagger.
    let mut got_defs: Vec<(String, String, String, usize, usize, String)> = Vec::new();
    for f in &files {
        if let Some(t) = nmsubs::tags::tag(&f.path, &f.text) {
            got_defs.extend(t.symbols.into_iter().map(|x| (x.file, x.name, x.kind, x.line, x.end, x.parent)));
        }
    }
    let want_defs: Vec<(String, String, String, usize, usize, String)> = list(&exp, "definitions")
        .iter()
        .map(|d| (s(d, "file"), s(d, "name"), s(d, "kind"), u(d, "line"), u(d, "end"), s(d, "parent")))
        .collect();
    let key = |d: &(String, String, String, usize, usize, String)| (d.0.clone(), d.1.clone(), d.2.clone(), d.3);
    let got_keys: BTreeSet<_> = got_defs.iter().map(key).collect();
    let want_keys: BTreeSet<_> = want_defs.iter().map(key).collect();
    let found = want_keys.intersection(&got_keys).count();
    let ends = want_defs.iter().filter(|w| got_defs.iter().any(|g| key(g) == key(w) && g.4 == w.4)).count();
    let parents = want_defs.iter().filter(|w| got_defs.iter().any(|g| key(g) == key(w) && g.5 == w.5)).count();
    let (recall, precision, end_ok) = (ratio(found, want_keys.len()), ratio(found, got_keys.len()), ratio(ends, found));
    let floor = exp.get("recall_floor").ok().and_then(|x| x.string().ok()).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.95);
    println!(
        "     {name} tagger: {} hand definitions, {} found by the tagger, {found} matched: recall {recall:.3}, precision {precision:.3}, last line exact {ends} of {found} ({end_ok:.3}), parent right {parents} of {found}",
        want_keys.len(),
        got_keys.len()
    );
    let missed: Vec<String> = want_keys.difference(&got_keys).map(|k| format!("{}:{} {} {}", k.0, k.3, k.2, k.1)).collect();
    let extra: Vec<String> = got_keys.difference(&want_keys).map(|k| format!("{}:{} {} {}", k.0, k.3, k.2, k.1)).collect();
    if !missed.is_empty() || !extra.is_empty() {
        println!("     {name} tagger: missed [{}]; not in the hand list [{}]", missed.join("; "), extra.join("; "));
    }
    if recall < floor || precision < 0.95 || end_ok < 0.9 {
        let missed: Vec<_> = want_keys.difference(&got_keys).take(6).collect();
        let extra: Vec<_> = got_keys.difference(&want_keys).take(6).collect();
        m.failures.push(format!(
            "tagger: recall {recall:.3} (floor {floor}), precision {precision:.3}, ends {end_ok:.3}; missed {missed:?}; extra {extra:?}"
        ));
    }
    let wrong_end: Vec<String> = want_defs
        .iter()
        .filter_map(|w| {
            got_defs.iter().find(|g| key(g) == key(w) && g.4 != w.4).map(|g| format!("{}:{} {} end {} want {}", w.0, w.3, w.1, g.4, w.4))
        })
        .take(4)
        .collect();
    if !wrong_end.is_empty() {
        println!("     {name} tagger: last lines that differ: {}", wrong_end.join("; "));
    }
    // Imports and packages.
    let pairs: Vec<(&str, &str)> = files.iter().map(|f| (f.path.as_str(), f.text.as_str())).collect();
    let sub = nmsubs::from_files(&pairs, std::sync::Arc::new(|_: &str| Vec::new()));
    let want_imp = list(&exp, "imports");
    let mut right = 0;
    let mut wrong = Vec::new();
    for w in want_imp {
        let (file, line) = (s(w, "file"), u(w, "line"));
        let edge = sub.import_edges.iter().find(|e| e.from == file && e.line == line);
        let ok = match edge {
            Some(e) => {
                let to = s(w, "to");
                if to.is_empty() {
                    e.to.is_none() && e.external.as_deref() == Some(s(w, "external").as_str())
                } else {
                    e.to.as_deref() == Some(to.as_str())
                }
            }
            None => false,
        };
        if ok {
            right += 1;
        } else {
            wrong.push(format!("{file}:{line} got {:?}", edge.map(|e| (e.to.clone(), e.external.clone()))));
        }
    }
    let imp_ratio = ratio(right, want_imp.len());
    let got_pk: BTreeSet<String> = sub.crates.iter().map(|c| format!("{} | {} | {}", c.name, c.dir, c.deps.join(","))).collect();
    let want_pk: BTreeSet<String> = list(&exp, "packages")
        .iter()
        .map(|p| {
            format!(
                "{} | {} | {}",
                s(p, "name"),
                s(p, "dir"),
                list(p, "deps").iter().filter_map(|d| d.string().ok()).collect::<Vec<_>>().join(",")
            )
        })
        .collect();
    println!(
        "     {name} imports: {right} of {} resolved right ({imp_ratio:.3}); packages {}",
        want_imp.len(),
        if got_pk == want_pk { "equal the hand list" } else { "DIFFER" }
    );
    if imp_ratio < 0.9 {
        m.failures.push(format!("imports: {imp_ratio:.3}; wrong {:?}", wrong.iter().take(5).collect::<Vec<_>>()));
    }
    if got_pk != want_pk {
        m.failures.push(format!("packages: got {got_pk:?}, want {want_pk:?}"));
    }
    Ok(m)
}

/// Run every part; the number of cases checked, or the first rule that failed.
pub fn check(home: &Path) -> Result<usize, String> {
    let base = home.join("tests/languages");
    let mut n = lexer(&base)?;
    let mut dirs: Vec<_> = std::fs::read_dir(&base).map_err(|e| format!("{}: {e}", base.display()))?.flatten().map(|e| e.path()).collect();
    dirs.sort();
    let mut failed = Vec::new();
    for d in dirs.iter().filter(|d| d.join("expected.json").is_file()) {
        let m = fixture(d)?;
        for f in &m.failures {
            failed.push(format!("{}: {f}", m.name));
        }
        n += 1;
    }
    if let Some(f) = failed.first() {
        return Err(format!("{} rule(s) failed; first {f}", failed.len()));
    }
    Ok(n)
}
