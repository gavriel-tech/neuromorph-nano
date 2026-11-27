//! Capability gates C2, C3 and C4 on the live crates (2026-11-10), with the questions, seed
//! rules and thresholds of the locked self gates (`gates.rs`, runbook
//! `runs/2026-11-08-self-gates-runbook.md`), which run the earlier `nmcore` copies. Here C2 asks
//! `nmsubs`, C3 uses `nmtool::improve` and `nmtool::stepcheck`, C4 uses `nmtool::build` and
//! `nmtool::explain`; seeds go into `crates/nmbrain/src` (a live part) where the self gates used
//! `crates/nmcore/src`. Plus pack repeatability on the live organism (the one thin-pass test with
//! no live counterpart). Runbook `runs/2026-11-10-live-self-gates-runbook.md`.
//! Run: `cargo test --release --test live_self_gates -- --ignored --test-threads 1 --nocapture`.
#![expect(clippy::unwrap_used, reason = "gate harness")]
use nmsubs::Substrate;
use nmtool::build::{Kind, Request};
use nmtool::{build, explain, improve, stepcheck};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The live crate the C3 and C4 seeds go into.
const SEED_CRATE: &str = "nmbrain";
const DEADLINE: Duration = Duration::from_secs(900);

fn mini() -> PathBuf {
    nmsubs::repo::canon(&nmtype::nano_root())
}
fn target() -> PathBuf {
    mini().join("scratch/stepcheck-target")
}
fn scratch(name: &str) -> PathBuf {
    let base = mini().join("scratch/gates");
    let _ = std::fs::create_dir_all(&base);
    let p = base.join(format!("nm-live-gate-{name}"));
    let _ = std::fs::remove_dir_all(&p);
    p
}
fn report(name: &str, text: &str) {
    println!("{text}");
    std::fs::write(mini().join(format!("runs/live-gate-{name}-{}.txt", nmtype::now_ms())), text).unwrap();
}
fn walk_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        let n = e.file_name().to_string_lossy().into_owned();
        if p.is_dir() {
            if !n.starts_with('.')
                && ![
                    "target",
                    "neuromorph-nano",
                    "9-weight",
                    "golden",
                    "golden-data",
                    "node_modules",
                    "home",
                    "scratch",
                    "suites",
                    "tests",
                    "private",
                    "retired",
                ]
                .contains(&n.as_str())
            {
                walk_rs(&p, out);
            }
        } else if n.ends_with(".rs") {
            out.push(p);
        }
    }
}
fn ident(s: &str) -> String {
    s.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect()
}
fn src_dir(root: &Path) -> PathBuf {
    root.join(format!("crates/{SEED_CRATE}/src"))
}
fn src_files(root: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(src_dir(root))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".rs"))
        .collect();
    v.sort();
    v
}
fn even<T: Clone>(v: &[T], n: usize) -> Vec<T> {
    if v.is_empty() {
        return Vec::new();
    }
    (0..n.min(v.len())).map(|i| v[i * v.len() / n.min(v.len())].clone()).collect()
}
fn code_line(l: &str) -> bool {
    !l.trim_start().starts_with("//")
}
fn embed() -> nmsubs::Embed {
    std::sync::Arc::new(|t: &str| vec![t.len() as f32, 1.0])
}
fn check_run(root: &Path) -> build::BuildRun {
    build::run(&Request {
        root: root.to_path_buf(),
        target: target(),
        kind: Kind::Check,
        package: Some(SEED_CRATE.into()),
        deadline: DEADLINE,
    })
}

#[test]
#[ignore = "capability gate"]
fn live_c2_understanding() {
    let root = mini();
    let sub = Substrate::scan(&root, &[], embed());
    let ws = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let list = &ws[ws.find("members = [").unwrap()..];
    let list = &list[..list.find(']').unwrap()];
    let members: Vec<&str> = list.split('"').skip(1).step_by(2).collect();
    let mut names = BTreeMap::new();
    let mut raw: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in &members {
        let t = std::fs::read_to_string(root.join(m).join("Cargo.toml")).unwrap();
        let name = t.lines().find_map(|l| l.trim().strip_prefix("name = \"").map(|x| x.trim_end_matches('"').to_owned())).unwrap();
        let mut sect = String::new();
        let mut deps = Vec::new();
        for l in t.lines() {
            let l = l.trim();
            if l.starts_with('[') {
                sect = l.to_owned();
            } else if sect.ends_with("dependencies]") && l.contains("path =") {
                let key = l.split('=').next().unwrap().trim().to_owned();
                let pkg = l.split("package = \"").nth(1).map(|x| x[..x.find('"').unwrap()].to_owned());
                deps.push(pkg.unwrap_or(key));
            }
        }
        names.insert(name.clone(), (*m).to_owned());
        raw.insert(name, deps);
    }
    let truth: BTreeMap<String, BTreeSet<String>> =
        raw.iter().map(|(n, d)| (n.clone(), d.iter().filter(|x| names.contains_key(*x) && *x != n).cloned().collect())).collect();
    let mut log = String::new();
    let (mut g_ok, mut g_n) = (0, 0);
    for (i, m) in members.iter().enumerate() {
        let name = names.iter().find(|(_, d)| d == m).unwrap().0.clone();
        let (want, got): (BTreeSet<String>, BTreeSet<String>) = if i % 2 == 0 {
            (truth[&name].clone(), sub.crate_named(&name).map(|c| c.deps.iter().cloned().collect()).unwrap_or_default())
        } else {
            (truth.iter().filter(|(_, d)| d.contains(&name)).map(|(n, _)| n.clone()).collect(), sub.dependents(&name).into_iter().collect())
        };
        let ok = want == got;
        g_n += 1;
        g_ok += usize::from(ok);
        let _ = writeln!(
            log,
            "graph {} {name}: {}{}",
            if i % 2 == 0 { "deps" } else { "dependents" },
            if ok { "ok" } else { "WRONG" },
            if ok { String::new() } else { format!(" want {want:?} got {got:?}") }
        );
    }
    let mut files = Vec::new();
    walk_rs(&root, &mut files);
    let mut defs: Vec<(String, String, usize, bool)> = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        for (n, l) in std::fs::read_to_string(f).unwrap_or_default().lines().enumerate() {
            let t = l.trim_start();
            let public = t.starts_with("pub struct ") || t.starts_with("pub fn ");
            let t = t.trim_start_matches("pub(crate) ").trim_start_matches("pub ");
            for kw in ["fn ", "struct ", "enum ", "trait ", "type "] {
                if let Some(r) = t.strip_prefix(kw) {
                    let id = ident(r);
                    if !id.is_empty() {
                        defs.push((id, rel.clone(), n + 1, public));
                    }
                }
            }
        }
    }
    let mut count: BTreeMap<&str, usize> = BTreeMap::new();
    for d in &defs {
        *count.entry(d.0.as_str()).or_default() += 1;
    }
    let cands: Vec<&(String, String, usize, bool)> = defs.iter().filter(|d| d.3 && count[d.0.as_str()] == 1).collect();
    let step = (cands.len() / 15).max(1);
    let (mut s_ok, mut s_n) = (0, 0);
    for d in cands.iter().step_by(step).take(15) {
        let ok = sub.symbols.iter().any(|s| s.name == d.0 && s.file == d.1 && s.line == d.2);
        s_n += 1;
        s_ok += usize::from(ok);
        let _ = writeln!(log, "symbol {} at {}:{}: {}", d.0, d.1, d.2, if ok { "ok" } else { "WRONG" });
    }
    let pass = g_ok * 10 >= g_n * 9 && s_ok * 10 >= s_n * 7;
    let head = format!(
        "live C2 understanding (nmsubs) on {} ({} crates, {} candidate symbols): graph {g_ok}/{g_n} (gate 90%), symbol {s_ok}/{s_n} (gate 70%): {}",
        root.display(),
        members.len(),
        cands.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    report("c2", &format!("{head}\n{log}"));
    assert!(pass, "{head}");
}

#[test]
#[ignore = "capability gate"]
fn live_c4_build() {
    let sc = scratch("c4");
    stepcheck::copy_tree(&mini(), &sc).unwrap();
    assert!(check_run(&sc).ok, "clean scratch builds");
    let dir = src_dir(&sc);
    let mut cands: [Vec<(String, usize, String, String)>; 3] = Default::default();
    for f in src_files(&sc) {
        let text = std::fs::read_to_string(dir.join(&f)).unwrap();
        let end = text.lines().position(|l| l.contains("#[test]") || l.contains("#[cfg(test)]")).unwrap_or(usize::MAX);
        for (i, l) in text.lines().enumerate().take(end).filter(|(_, l)| code_line(l)) {
            let t = l.trim();
            let name = ident(t.strip_prefix("let ").unwrap_or("").trim_start_matches("mut "));
            if !name.is_empty() && t.ends_with(';') && t.contains(".len()") && t.contains(&format!("{name} = ")) {
                cands[2].push((f.clone(), i + 1, format!("{name} = "), format!("{name}: () = ")));
            } else if l.contains(".len()") {
                cands[0].push((f.clone(), i + 1, ".len()".into(), ".lenx()".into()));
            } else if l.contains("String::new()") {
                cands[1].push((f.clone(), i + 1, "String::new()".into(), "Strin::new()".into()));
            }
        }
    }
    let order = |v: &[(String, usize, String, String)], q: usize| {
        let mut o = even(v, q);
        o.extend(v.iter().filter(|x| !o.contains(x)).cloned().collect::<Vec<_>>());
        o
    };
    let (mut ok, mut n, mut excluded, mut pk, mut raw, mut log) = (0, 0, 0, 0, 0, String::new());
    for (kind, quota) in [(0, 7), (1, 7), (2, 6)] {
        let mut valid = 0;
        for (f, line, a, b) in &order(&cands[kind], quota) {
            if valid == quota {
                break;
            }
            let path = dir.join(f);
            let orig = std::fs::read_to_string(&path).unwrap();
            let mutated: String = orig
                .lines()
                .enumerate()
                .map(|(i, l)| if i + 1 == *line { l.replacen(a.as_str(), b, 1) } else { l.to_owned() })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n";
            std::fs::write(&path, &mutated).unwrap();
            let r = check_run(&sc);
            std::fs::write(&path, &orig).unwrap();
            if r.ok {
                excluded += 1;
                let _ = writeln!(log, "EXCLUDED (still builds) {f}:{line} {a} -> {b}");
                continue;
            }
            valid += 1;
            n += 1;
            let ex = explain::explain(&sc, &r, None);
            let want = format!("crates/{SEED_CRATE}/src/{f}");
            let good = ex.as_ref().is_some_and(|e| e.file == want && e.line.abs_diff(*line) <= 1);
            ok += usize::from(good);
            let bytes = ex.as_ref().map_or(0, |e| e.file.len() + e.cause.len() + e.fix.len() + 8);
            pk += bytes;
            raw += r.raw_bytes;
            let _ = writeln!(
                log,
                "{} {f}:{line} {a} -> {b} | got {:?} | {bytes} vs {} bytes | {} ms",
                if good { "ok" } else { "WRONG" },
                ex.as_ref().map(|e| (e.file.clone(), e.line)),
                r.raw_bytes,
                r.cargo_ms
            );
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "report ratio")]
    let ratio = pk as f64 / raw.max(1) as f64;
    let pass = ok * 5 >= n * 4 && n >= 16 && ratio <= 0.3;
    let head = format!(
        "live C4 build (nmtool, seeds in crates/{SEED_CRATE}/src): localized {ok}/{n} (gate 16/20, {excluded} excluded), bytes {pk} vs raw {raw} = {ratio:.3} (gate 0.3): {}",
        if pass { "PASS" } else { "FAIL" }
    );
    report("c4", &format!("{head}\n{log}"));
    assert!(pass, "{head}");
}

fn tree_hash(root: &Path) -> String {
    let mut files = Vec::new();
    walk_rs(&root.join("crates"), &mut files);
    files.sort();
    nmhash::sha256_bytes(files.iter().map(|f| std::fs::read_to_string(f).unwrap_or_default()).collect::<String>().as_bytes())
}

#[test]
#[ignore = "capability gate"]
fn live_c3_improve() {
    let src = mini();
    let before = tree_hash(&src);
    let tg = target();
    let mut docs = Vec::new();
    let mut empties = Vec::new();
    for f in src_files(&src) {
        let text = std::fs::read_to_string(src_dir(&src).join(&f)).unwrap();
        for (i, l) in text.lines().enumerate() {
            if l.trim_start().starts_with("/// ") {
                docs.push((f.clone(), i + 1));
            } else if code_line(l) && l.contains(".is_empty()") && !l.contains('!') && !l.contains("is_empty())") {
                empties.push((f.clone(), i + 1));
            }
        }
    }
    let dash = format!(" {} ", '\u{2014}');
    let word = ["spec", "ialist"].concat();
    let mut seeds: Vec<(String, usize, &str)> =
        even(&docs, 5).into_iter().enumerate().map(|(i, (f, l))| (f, l, if i < 3 { "dash" } else { "word" })).collect();
    seeds.extend(even(&empties, 5).into_iter().map(|(f, l)| (f, l, "clippy")));
    let (mut pass_n, mut made, mut excluded, mut found, mut log) = (0, 0, 0, 0, String::new());
    for (k, (f, line, kind)) in seeds.iter().enumerate() {
        let sc = scratch(&format!("c3-{k}"));
        stepcheck::copy_tree(&src, &sc).unwrap();
        let rel = format!("crates/{SEED_CRATE}/src/{f}");
        let path = sc.join(&rel);
        let text = std::fs::read_to_string(&path).unwrap();
        let mutated: String = text
            .lines()
            .enumerate()
            .map(|(i, l)| {
                if i + 1 != *line {
                    return l.to_owned();
                }
                match *kind {
                    "dash" => format!("{l}{dash}seeded note"),
                    "word" => format!("{l} (a {word} note)"),
                    _ => {
                        let at = l.find(".is_empty()").unwrap();
                        let start = l[..at].rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')).map_or(0, |p| p + 1);
                        format!("{}{}.len() == 0{}", &l[..start], &l[start..at], &l[at + ".is_empty()".len()..])
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&path, mutated).unwrap();
        let fs = if *kind == "clippy" {
            improve::clippy(&sc, &tg, SEED_CRATE).into_iter().filter(|x| x.file == rel).collect::<Vec<_>>()
        } else {
            improve::scan(&sc, &rel)
        };
        let hit = fs.iter().find(|x| x.fix.is_some() && x.line == *line);
        found += usize::from(hit.is_some());
        let Some(diff) = hit.and_then(improve::propose) else {
            excluded += 1;
            let _ = writeln!(log, "EXCLUDED (no proposal) {kind} {f}:{line}");
            continue;
        };
        made += 1;
        let st = stepcheck::run(&stepcheck::Plan {
            root: sc.clone(),
            scratch: scratch(&format!("c3-{k}-check")),
            target: tg.clone(),
            diff,
            tests: true,
            package: Some(SEED_CRATE.into()),
            deadline: DEADLINE,
        });
        let ok = st.len() >= 6 && st.iter().all(|s| s.ok);
        pass_n += usize::from(ok);
        let _ = writeln!(
            log,
            "{} {kind} {f}:{line} | {}",
            if ok { "ok" } else { "FAILED" },
            st.iter()
                .map(|s| format!(
                    "{}={}{}",
                    s.name,
                    s.ok,
                    if s.detail.is_empty() || s.ok {
                        String::new()
                    } else {
                        format!("({})", s.detail.chars().take(100).collect::<String>())
                    }
                ))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    let untouched = before == tree_hash(&src);
    let pass = made > 0 && pass_n * 10 >= made * 6 && untouched;
    let head = format!(
        "live C3 improvement (nmtool, seeds in crates/{SEED_CRATE}/src): {pass_n}/{made} proposals pass Step Check (gate 60%), detected {found}/{}, {excluded} excluded, source tree unchanged: {untouched}: {}",
        seeds.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    report("c3", &format!("{head}\n{log}"));
    assert!(pass, "{head}");
}

#[test]
#[ignore = "capability gate"]
fn live_pack_repeatable() {
    let dir = nmtype::tests_dir("capability-gates/c1-context/live-repeat");
    let _ = std::fs::remove_dir_all(&dir);
    let org = nmapi::forage::Organism::open_with(&mini(), &mini(), &[], false, &dir).unwrap();
    let q = "where is the shared Qwen3 structure generated";
    let a = org.ask(q, 4000).unwrap();
    let b = org.ask(q, 4000).unwrap();
    let (ta, tb) = (a.emitted.text(), b.emitted.text());
    let pass = ta == tb && a.pack.withheld == b.pack.withheld;
    let head = format!(
        "live pack repeatability (memory off): same question twice, same pack {}: {}",
        ta == tb,
        if pass { "PASS" } else { "FAIL" }
    );
    report("repeat", &format!("{head}\n{ta}"));
    assert!(pass, "{head}");
}
