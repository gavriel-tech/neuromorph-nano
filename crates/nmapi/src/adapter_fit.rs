//! Fitting the project's adapter cartridges at idle (`adapter` holds the loading path). Closed form
//! only, no backward pass (orchestration study 9.4): the MicroModel reads each pair once, the
//! numbers go into running sums or RAM for the length of the fit, the solve runs on the processor
//! in Rust with the standard library, and only the fitted parameters are written. No read of the
//! user's code base is written to disk (the material's text and its vectors stay in RAM).
//!
//! Labels come from the user's own material, with no use data and no Claude Code verdict
//! (direction 2026-11-18 05:56 UTC: a fitted adapter cartridge, not continuous learning):
//! - a doc comment hidden from its own definition: the comment is the question, the definition's
//!   piece with the comment lines taken out is the item;
//! - a commit message to the functions it changed: `git log -U0` of the material's repository, the
//!   functions named by each hunk's context line and by changed definition lines, found in the
//!   current pieces by name in the same file.
//!
//! Split by file (C42's lesson and 9.5): a pair is held out when its item's file is, by the SHA-256
//! of the file's path; the held-out files are split again into a half that chooses the fit's
//! numbers and a half that only reports. A shuffled-label control is fitted with the same numbers
//! and reported beside it; an adapter is written only when the report half gains and the control
//! does not.
//!
//! The lens (study 9.2 D1 with D2 on the project's own pairs): the material's strongest shared
//! directions from its stored vectors (subspace iteration on their second moment), shrunk on the
//! request's side, then reduced-rank ridge from the request vectors to their items' vectors:
//! `D = (Xf'Xf + l I)^-1 Xf'(Y - Xf)`, truncated to rank r by the top right singular vectors of D.
//! The judge's code read-out (study 9.2 A1): ridge on the judge's final-norm state differences
//! between the true item and its hardest rival (the code piece the embedder ranks highest for the
//! question that is not the item), `w = (S + l I)^-1 s`, scaled to the spread of the judge's own
//! log-odds differences, used only in code-against-code close calls.
//!
//! Gavriel Technologies; architecture by Luis Jake Gabriel III.
use crate::adapter::{self, Lens, Part, Readout};
use crate::forage::Organism;
use nmjson::Value;
use nmtype::{Fault, FaultKind};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.adapter_fit", FaultKind::Execution, m)
}

// ---------------------------------------------------------------- the file form

/// What an adapter's label says, before its parts are encoded.
pub struct LabelIn<'a> {
    pub id: &'a str,
    pub version: u64,
    pub model: &'a str,
    pub weights: &'a str,
    pub kind: &'a str,
    pub does: &'a str,
    pub made_from: &'a str,
    pub notes: Value,
}

fn s(x: &str) -> Value {
    Value::String(x.to_owned())
}
fn n(x: u64) -> Value {
    Value::Number(x.to_string())
}
fn f(x: f64) -> Value {
    Value::Number(if x.is_finite() { format!("{x:e}") } else { "0".to_owned() })
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// The label of an adapter file (the library's fields: `schema`, `id`, `version`, `family`,
/// `domain`, `scope`, `model`, `weights`, `name`, `does`, `made_from`, `made`, `notes`); `parts` and
/// `parameters_sha256` are added by `encode_parts`.
#[must_use]
pub fn label(l: &LabelIn) -> Value {
    obj(vec![
        ("schema", s(nmlearn::library::CARTRIDGE_SCHEMA)),
        ("id", s(l.id)),
        ("version", n(l.version)),
        ("family", s(nmlearn::library::ADAPTER)),
        ("kind", s(l.kind)),
        ("domain", s("project")),
        ("scope", s("project")),
        ("model", s(l.model)),
        ("weights", s(l.weights)),
        ("name", s(l.id)),
        ("does", s(l.does)),
        ("made_from", s(l.made_from)),
        ("made", s(&nmlearn::cartridge::date_of(nmtype::now_ms()))),
        ("notes", l.notes.clone()),
    ])
}

/// The file's bytes: each part at 16 bits (`F16`), its place in the parameters on the label, the
/// parameters' SHA-256 on the label (`nmlearn::library::cartridge_file`).
#[must_use]
pub fn encode_parts(mut label: Value, parts: &[Part]) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    let mut rows = Vec::new();
    for (name, shape, xs) in parts {
        let at = body.len() as u64;
        body.extend(xs.iter().flat_map(|x| nmengn::half::bits16(*x).to_le_bytes()));
        rows.push(obj(vec![
            ("socket", s(name)),
            ("form", s("adapter")),
            ("shape", Value::Array(shape.iter().map(|x| n(*x as u64)).collect())),
            ("encoding", s("F16")),
            ("at", n(at)),
            ("bytes", n(body.len() as u64 - at)),
        ]));
    }
    if let Value::Object(o) = &mut label {
        o.insert("parts".into(), Value::Array(rows));
        o.insert("parameters_sha256".into(), s(&nmhash::sha256_bytes(&body)));
    }
    nmlearn::library::cartridge_file(&label, &body)
}

/// Writes one adapter file (through a temporary file in the same folder, then renamed).
/// # Errors
/// The folder or the file cannot be written.
pub fn write_parts(file: &Path, label: Value, parts: &[Part]) -> Result<PathBuf, Fault> {
    let dir = file.parent().ok_or_else(|| bad("no folder"))?;
    std::fs::create_dir_all(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))?;
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, encode_parts(label, parts)).map_err(|e| bad(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, file).map_err(|e| bad(format!("{}: {e}", file.display())))?;
    Ok(file.to_path_buf())
}

/// Writes the next version of `id` on the shelf, its parent (the version before) named in its
/// notes. Earlier versions stay in place (roll back: pull, or remove the newer file).
/// # Errors
/// As `write_parts`.
pub fn write_next(home: &Path, mut l: LabelIn, parts: &[Part]) -> Result<(PathBuf, u64), Fault> {
    let have = adapter::versions(home, l.id);
    let version = have.last().map_or(1, |(v, _)| v + 1);
    if let Value::Object(o) = &mut l.notes {
        o.insert("parent".into(), s(&have.last().map_or_else(|| "none".to_owned(), |(v, _)| format!("{}@{v}", l.id))));
    }
    l.version = version;
    let file = adapter::shelf_file(home, l.id, version);
    write_parts(&file, label(&l), parts).map(|p| (p, version))
}

// ---------------------------------------------------------------- labels

/// One labelled pair from the user's material.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// The question (a doc comment, a commit message).
    pub query: String,
    /// The item's file (forward slashes) and chunk index.
    pub file: String,
    pub target: usize,
    /// The text the item is read as (for a doc pair, its piece with the comment hidden).
    pub target_text: String,
    /// `doc` or `commit`.
    pub source: &'static str,
}

/// Which fold a file is in, by the SHA-256 of its path: 0 fit, 1 choose the numbers, 2 report.
#[must_use]
pub fn fold(path: &str) -> u8 {
    let h = nmhash::sha256_bytes(path.replace('\\', "/").as_bytes());
    match h.as_bytes().first().copied().unwrap_or(b'0') {
        b'0'..=b'9' | b'a' => 0,
        b'b'..=b'c' => 1,
        _ => 2,
    }
}

const DEF_WORDS: [&str; 17] = ["fn", "struct", "enum", "trait", "impl", "type", "const", "static", "mod", "class", "def", "function", "interface", "func", "union", "macro_rules!", "record"];
const MODIFIERS: [&str; 17] =
    ["pub", "pub(crate)", "pub(super)", "export", "async", "public", "private", "protected", "static", "default", "unsafe", "extern", "abstract", "final", "override", "virtual", "inline"];

/// The name a definition line defines, if it is one.
#[must_use]
pub fn def_name(line: &str) -> Option<String> {
    let mut words = line.split_whitespace().peekable();
    while let Some(w) = words.peek() {
        if MODIFIERS.contains(w) || w.starts_with("pub(") || w.starts_with("extern\"") || w.starts_with('"') {
            words.next();
        } else {
            break;
        }
    }
    let kw = words.next()?;
    if !DEF_WORDS.contains(&kw) {
        return None;
    }
    let rest = words.next()?;
    let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    (!name.is_empty() && name.chars().next().is_some_and(|c| !c.is_ascii_digit())).then_some(name)
}

/// The comment text of a line in a language's line-comment form, or None when it is not one.
fn comment_text(t: &str, hash: bool) -> Option<&str> {
    if t.starts_with("//!") || t.starts_with("/*!") {
        return None;
    }
    for m in ["///", "//", "/**", "/*", "*/", "*"] {
        if let Some(r) = t.strip_prefix(m) {
            return Some(r.trim_end_matches("*/").trim());
        }
    }
    if hash && t.starts_with('#') && !t.starts_with("#[") && !t.starts_with("#!") {
        return Some(t.trim_start_matches('#').trim());
    }
    None
}

fn words(t: &str) -> usize {
    t.split_whitespace().filter(|w| w.chars().any(char::is_alphabetic)).count()
}

/// Doc-comment pairs of one code chunk: each definition line with a comment directly above it
/// (attributes and decorators may sit between), or a Python docstring directly below it.
fn chunk_doc_pairs(path: &str, decoder: &str, idx: usize, text: &str, out: &mut Vec<Pair>) {
    let hash = matches!(decoder, "python" | "ruby" | "shell" | "bash" | "toml" | "perl" | "r" | "yaml");
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if def_name(t).is_none() {
            continue;
        }
        let mut hidden: Vec<usize> = Vec::new();
        let mut said: Vec<String> = Vec::new();
        let mut j = i;
        while j > 0 {
            j -= 1;
            let u = lines.get(j).map_or("", |x| x.trim());
            if u.starts_with("#[") || u.starts_with('@') {
                continue;
            }
            match comment_text(u, hash) {
                Some(c) => {
                    hidden.push(j);
                    if !c.is_empty() {
                        said.push(c.to_owned());
                    }
                }
                None => break,
            }
        }
        said.reverse();
        if said.is_empty() && decoder == "python" {
            // A docstring directly below the definition line.
            let mut k = i + 1;
            if let Some(first) = lines.get(k).map(|x| x.trim()).filter(|x| x.starts_with("\"\"\"") || x.starts_with("'''")) {
                let quote = &first[..3];
                let mut body = first[3..].to_owned();
                hidden.push(k);
                let closed = body.contains(quote);
                while !closed && k + 1 < lines.len() {
                    k += 1;
                    let u = lines.get(k).map_or("", |x| x.trim());
                    hidden.push(k);
                    body.push(' ');
                    body.push_str(u);
                    if u.contains(quote) {
                        break;
                    }
                }
                said.push(body.replace(quote, " "));
            }
        }
        let query: String = said.join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
        if words(&query) < 4 {
            continue;
        }
        let query: String = query.chars().take(600).collect();
        let target_text: String = lines.iter().enumerate().filter(|(k, _)| !hidden.contains(k)).map(|(_, l)| *l).collect::<Vec<_>>().join("\n");
        if words(&target_text) < 3 {
            continue;
        }
        out.push(Pair { query, file: path.to_owned(), target: idx, target_text, source: "doc" });
    }
}

/// Doc-comment pairs over the material's code, at most `cap`, spread over the files.
#[must_use]
pub fn doc_pairs(sub: &nmsubs::Substrate, cap: usize) -> Vec<Pair> {
    let mut all = Vec::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        let Some(file) = sub.files.get(c.file).filter(|f| f.is_code()) else { continue };
        chunk_doc_pairs(&file.path.replace('\\', "/"), file.decoder, i, &c.text, &mut all);
    }
    spread(all, cap)
}

/// At most `cap` pairs, taken evenly over the list (deterministic).
fn spread(all: Vec<Pair>, cap: usize) -> Vec<Pair> {
    if all.len() <= cap || cap == 0 {
        return all;
    }
    let step = all.len() as f64 / cap as f64;
    (0..cap).filter_map(|k| all.get((k as f64 * step) as usize).cloned()).collect()
}

/// Commits read from the material's repository.
pub const COMMITS: usize = 400;
/// A commit touching more functions than this is too diffuse to label any of them.
const COMMIT_TARGETS: usize = 6;

/// Commit-message pairs: the last `commits` commits (no merges) of the material's repository,
/// each message to the current pieces defining the functions its hunks changed. None when the
/// material is not a git repository.
#[must_use]
pub fn commit_pairs(sub: &nmsubs::Substrate, commits: usize, cap: usize) -> Vec<Pair> {
    let mut cmd = std::process::Command::new("git");
    cmd.args(["-c", "safe.directory=*", "log", "--no-merges", "-U0", "--no-color", "--no-ext-diff", "--no-textconv", &format!("-n{commits}"), "--format=%x1e%s%n%b%x1d"]).current_dir(&sub.root);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        cmd.creation_flags(0x0800_0000);
    }
    let Ok(out) = cmd.output() else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    let log = String::from_utf8_lossy(&out.stdout);
    // Definitions by (file, name) in the current pieces.
    let mut by_name: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, c) in sub.chunks.iter().enumerate() {
        let Some(file) = sub.files.get(c.file).filter(|f| f.is_code()) else { continue };
        let path = file.path.replace('\\', "/");
        // The piece's own definitions (`piece::Meta`, empty for line windows) and every definition
        // line in its text.
        let named = c.piece.defines.iter().map(|d| d.rsplit("::").next().unwrap_or(d).to_owned()).chain(c.text.lines().filter_map(|l| def_name(l.trim())));
        for short in named {
            let v = by_name.entry((path.clone(), short)).or_default();
            if !v.contains(&i) {
                v.push(i);
            }
        }
    }
    let mut all = Vec::new();
    for rec in log.split('\u{1e}').filter(|r| !r.trim().is_empty()) {
        let (msg, diff) = rec.split_once('\u{1d}').unwrap_or((rec, ""));
        let query: String = msg.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(400).collect();
        if words(&query) < 3 {
            continue;
        }
        let mut file = String::new();
        let mut hit: BTreeSet<usize> = BTreeSet::new();
        for line in diff.lines() {
            if let Some(rest) = line.strip_prefix("+++ ") {
                file = rest.strip_prefix("b/").unwrap_or(rest).trim().to_owned();
            } else if let Some(h) = line.strip_prefix("@@") {
                // The hunk's context line names the function it sits in.
                let ctx = h.split_once("@@").map_or("", |(_, c)| c).trim();
                if let Some(name) = def_name(ctx)
                    && let Some(v) = by_name.get(&(file.clone(), name))
                {
                    hit.extend(v.iter().copied());
                }
            } else if let Some(added) = line.strip_prefix('+').or_else(|| line.strip_prefix('-'))
                && let Some(name) = def_name(added.trim())
                && let Some(v) = by_name.get(&(file.clone(), name))
            {
                hit.extend(v.iter().copied());
            }
        }
        if hit.is_empty() || hit.len() > COMMIT_TARGETS {
            continue;
        }
        for i in hit {
            let Some(c) = sub.chunks.get(i) else { continue };
            let path = sub.files.get(c.file).map_or_else(String::new, |f| f.path.replace('\\', "/"));
            all.push(Pair { query: query.clone(), file: path, target: i, target_text: c.text.clone(), source: "commit" });
        }
    }
    spread(all, cap)
}

/// Both kinds of pairs, commits first, the same question to the same item once; at most `cap`.
#[must_use]
pub fn labels(sub: &nmsubs::Substrate, cap: usize) -> Vec<Pair> {
    let mut seen = BTreeSet::new();
    let mut out: Vec<Pair> = commit_pairs(sub, COMMITS, cap / 2).into_iter().chain(doc_pairs(sub, cap)).filter(|p| seen.insert((p.query.clone(), p.target))).collect();
    out.truncate(cap);
    out
}

/// A hash of the pairs (question and item key), so a fit is redone only when the labels change.
#[must_use]
pub fn labels_key(sub: &nmsubs::Substrate, pairs: &[Pair]) -> String {
    let mut t = String::new();
    for p in pairs {
        let key = sub.chunks.get(p.target).map_or("", |c| c.key.as_str());
        let _ = writeln!(t, "{}\t{}\t{key}", p.source, p.query);
    }
    nmhash::sha256_bytes(t.as_bytes())
}

// ---------------------------------------------------------------- linear algebra (f64)

/// In-place Cholesky factor of a symmetric positive definite `n x n` matrix (lower triangle).
/// False when it is not positive definite.
#[must_use]
pub fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    true
}

/// Solves `L L' X = B` in place, `B` `n x m` row-major, `L` from `cholesky`.
pub fn chol_solve(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    for i in 0..n {
        for k in 0..i {
            let c = l[i * n + k];
            if c != 0.0 {
                let (lo, hi) = b.split_at_mut(i * m);
                for (x, y) in hi[..m].iter_mut().zip(&lo[k * m..k * m + m]) {
                    *x -= c * y;
                }
            }
        }
        let d = l[i * n + i];
        b[i * m..i * m + m].iter_mut().for_each(|x| *x /= d);
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            let c = l[k * n + i];
            if c != 0.0 {
                let (lo, hi) = b.split_at_mut(k * m);
                for (x, y) in lo[i * m..i * m + m].iter_mut().zip(&hi[..m]) {
                    *x -= c * y;
                }
            }
        }
        let d = l[i * n + i];
        b[i * m..i * m + m].iter_mut().for_each(|x| *x /= d);
    }
}

/// Eigenvalues and eigenvectors of a small symmetric matrix by cyclic Jacobi rotations, largest
/// first; vectors are the columns of the returned `n x n` row-major matrix.
#[must_use]
pub fn jacobi_eig(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..60 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |j| *j != i).map(move |j| (i, j))).map(|(i, j)| a[i * n + j] * a[i * n + j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - sn * akq;
                    a[k * n + q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - sn * vkq;
                    v[k * n + q] = sn * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|x, y| a[y * n + y].total_cmp(&a[x * n + x]));
    let vals = order.iter().map(|i| a[i * n + i]).collect();
    let mut vecs = vec![0.0; n * n];
    for (col, i) in order.iter().enumerate() {
        for k in 0..n {
            vecs[k * n + col] = v[k * n + i];
        }
    }
    (vals, vecs)
}

/// Orthonormalises the columns (modified Gram-Schmidt); a column that vanishes is reseeded.
fn orthonormalise(cols: &mut [Vec<f64>], seed: &mut u64) {
    for i in 0..cols.len() {
        for _try in 0..3 {
            for j in 0..i {
                let (a, b) = cols.split_at_mut(i);
                let c: f64 = b[0].iter().zip(&a[j]).map(|(x, y)| x * y).sum();
                b[0].iter_mut().zip(&a[j]).for_each(|(x, y)| *x -= c * y);
            }
            let nrm = cols[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-12 {
                cols[i].iter_mut().for_each(|x| *x /= nrm);
                break;
            }
            let len = cols[i].len();
            cols[i] = random_vec(len, seed);
        }
    }
}

fn random_vec(n: usize, seed: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((*seed >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// `y = A x` for a symmetric `n x n` row-major `A`.
fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|i| a[i * n..i * n + n].iter().zip(x).map(|(p, q)| p * q).sum()).collect()
}

/// The `k` leading eigenpairs of a symmetric positive semi-definite `n x n` matrix by subspace
/// iteration with a Rayleigh-Ritz step: (values, unit vectors).
#[must_use]
pub fn top_eig(a: &[f64], n: usize, k: usize, iters: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let p = (k + 6).min(n);
    let mut seed = 0x5eed_u64;
    let mut z: Vec<Vec<f64>> = (0..p).map(|_| random_vec(n, &mut seed)).collect();
    orthonormalise(&mut z, &mut seed);
    for _ in 0..iters {
        let mut y: Vec<Vec<f64>> = z.iter().map(|c| matvec(a, n, c)).collect();
        orthonormalise(&mut y, &mut seed);
        z = y;
    }
    let az: Vec<Vec<f64>> = z.iter().map(|c| matvec(a, n, c)).collect();
    let mut b = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            b[i * p + j] = z[i].iter().zip(&az[j]).map(|(x, y)| x * y).sum();
        }
    }
    let (vals, vecs) = jacobi_eig(b, p);
    let mut out = Vec::new();
    for c in 0..k.min(p) {
        let mut v = vec![0.0; n];
        for (i, zi) in z.iter().enumerate() {
            let w = vecs[i * p + c];
            v.iter_mut().zip(zi).for_each(|(x, y)| *x += w * y);
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v.iter_mut().for_each(|x| *x /= nrm);
        out.push(v);
    }
    (vals.into_iter().take(k.min(p)).collect(), out)
}

// ---------------------------------------------------------------- the lens fit

/// What the lens fit reads: request vectors, their items' vectors, each pair's fold and item.
pub struct LensData {
    pub q: Vec<Vec<f32>>,
    pub y: Vec<Vec<f32>>,
    pub fold: Vec<u8>,
    pub own: Vec<usize>,
}

/// One lens fit's result.
#[derive(Clone, Debug)]
pub struct LensFit {
    pub lens: Lens,
    pub accepted: bool,
    pub report: String,
    /// Report half: mean reciprocal rank and first places, base, lens, shuffled control.
    pub mrr: (f64, f64, f64),
    pub first: (usize, usize, usize),
    pub held: usize,
    pub params: (f64, f64, usize),
}

/// Flattened directions: the `k` leading eigenvectors of the material vectors' second moment (a
/// sample of at most `cap` vectors).
fn shared_directions(material: &[&[f32]], d: usize, k: usize, cap: usize) -> Vec<Vec<f64>> {
    if material.is_empty() || k == 0 {
        return Vec::new();
    }
    let step = (material.len() / cap.max(1)).max(1);
    let mut m = vec![0.0_f64; d * d];
    let mut count = 0_usize;
    for v in material.iter().step_by(step) {
        if v.len() != d {
            continue;
        }
        count += 1;
        for i in 0..d {
            let vi = f64::from(v[i]);
            if vi == 0.0 {
                continue;
            }
            for (x, vj) in m[i * d..i * d + d].iter_mut().zip(v.iter()) {
                *x += vi * f64::from(*vj);
            }
        }
    }
    if count == 0 {
        return Vec::new();
    }
    m.iter_mut().for_each(|x| *x /= count as f64);
    top_eig(&m, d, k, 12).1
}

fn flatten(q: &[f32], dirs: &[Vec<f64>], beta: f64) -> Vec<f64> {
    let mut out: Vec<f64> = q.iter().map(|x| f64::from(*x)).collect();
    if beta == 0.0 {
        return out;
    }
    for u in dirs {
        let c: f64 = beta * u.iter().zip(q).map(|(a, b)| a * f64::from(*b)).sum::<f64>();
        out.iter_mut().zip(u).for_each(|(x, ui)| *x -= c * ui);
    }
    out
}

/// The low-rank map fitted on the training pairs (inputs already flattened): `(down [r, d], up
/// [d, r])`, or None when the system is not solvable.
#[must_use]
pub fn ridge_low_rank(xf: &[Vec<f64>], y: &[&[f32]], d: usize, c: f64, rank: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut g = vec![0.0_f64; d * d];
    let mut cy = vec![0.0_f64; d * d];
    for (x, yv) in xf.iter().zip(y) {
        for i in 0..d {
            let xi = x[i];
            if xi == 0.0 {
                continue;
            }
            for (gv, xj) in g[i * d..i * d + d].iter_mut().zip(x) {
                *gv += xi * xj;
            }
            for (cv, yj) in cy[i * d..i * d + d].iter_mut().zip(yv.iter()) {
                *cv += xi * f64::from(*yj);
            }
        }
    }
    ridge_low_rank_sums(g, &cy, d, c, rank)
}

/// `ridge_low_rank` from its sums, so a grid over the ridge strength reads the pairs once: `g` the
/// inputs' second moment `X'X` and `cy` the cross moment `X'Y`, both `d x d` row-major (row = input
/// channel). The ridge is `c` times the mean input energy, as in `ridge_low_rank`.
#[must_use]
pub fn ridge_low_rank_sums(g: Vec<f64>, cy: &[f64], d: usize, c: f64, rank: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let trace: f64 = (0..d).map(|i| g[i * d + i]).sum();
    let lambda = c * (trace / d as f64).max(1e-9);
    // Right side: X'(Y - X) = C - G.
    let mut rhs: Vec<f64> = cy.iter().zip(&g).map(|(a, b)| a - b).collect();
    let mut a = g;
    for i in 0..d {
        a[i * d + i] += lambda;
    }
    if !cholesky(&mut a, d) {
        return None;
    }
    chol_solve(&a, d, &mut rhs, d);
    let dm = rhs; // D, d x d row-major (row = input channel).
    // D'D, then its leading eigenvectors Q (the right singular vectors of D).
    let mut dtd = vec![0.0_f64; d * d];
    for r in 0..d {
        let row = &dm[r * d..r * d + d];
        for i in 0..d {
            let ri = row[i];
            if ri == 0.0 {
                continue;
            }
            for (x, rj) in dtd[i * d..i * d + d].iter_mut().zip(row) {
                *x += ri * rj;
            }
        }
    }
    let (_, q) = top_eig(&dtd, d, rank, 20);
    let r = q.len();
    // x D_r = (x D Q) Q': down row j = (D Q)[:, j], up[m][j] = Q[m][j].
    let mut down = vec![0.0_f64; r * d];
    for (j, qj) in q.iter().enumerate() {
        for i in 0..d {
            down[j * d + i] = dm[i * d..i * d + d].iter().zip(qj).map(|(a, b)| a * b).sum();
        }
    }
    let mut up = vec![0.0_f64; d * r];
    for (j, qj) in q.iter().enumerate() {
        for m in 0..d {
            up[m * r + j] = qj[m];
        }
    }
    Some((down, up))
}

fn to32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|v| *v as f32).collect()
}

/// Mean reciprocal rank and first places of the pairs `which` under the request map `map`, each
/// item ranked against the pool (pool entries of the pair's own item left out).
fn rank_pairs(data: &LensData, which: &[usize], pool: &[(usize, &[f32])], map: &dyn Fn(&[f32]) -> Vec<f32>) -> (f64, usize) {
    let mut rr = 0.0;
    let mut first = 0;
    for &i in which {
        let q = map(&data.q[i]);
        let target = adapter::dot(&q, &data.y[i]);
        let above = pool.iter().filter(|(c, v)| *c != data.own[i] && adapter::dot(&q, v) > target).count();
        rr += 1.0 / (above + 1) as f64;
        first += usize::from(above == 0);
    }
    (if which.is_empty() { 0.0 } else { rr / which.len() as f64 }, first)
}

/// `rank_pairs` on the drones' calibrated scale (roadmap Phase 0 item 3, 2026-11-23): each item's
/// score is its cosine with the mapped question less its mean cosine with the mapped content-free
/// questions of that question's instruction (`nulls[i]`), as the drones' meaning signal reads it.
/// The flattening raised the lane's raw ranking yet lowered this one (stage ladders: drones.M val08
/// 28 to 61), because the calibration already takes out what the shared directions carry.
fn rank_calibrated(data: &LensData, nulls: &[Vec<Vec<f32>>], pool: &[(usize, &[f32])], map: &dyn Fn(&[f32]) -> Vec<f32>) -> (f64, usize) {
    let mut rr = 0.0;
    let mut first = 0;
    for (i, ns) in nulls.iter().enumerate().take(data.q.len()) {
        let q = map(&data.q[i]);
        let nm: Vec<Vec<f32>> = ns.iter().map(|n| map(n)).collect();
        #[expect(clippy::cast_precision_loss, reason = "a few content-free questions")]
        let k = nm.len().max(1) as f32;
        let cal = |v: &[f32]| adapter::dot(&q, v) - nm.iter().map(|n| adapter::dot(n, v)).sum::<f32>() / k;
        let target = cal(&data.y[i]);
        let above = pool.iter().filter(|(c, v)| *c != data.own[i] && cal(v) > target).count();
        rr += 1.0 / (above + 1) as f64;
        first += usize::from(above == 0);
    }
    (if nulls.is_empty() { 0.0 } else { rr / nulls.len() as f64 }, first)
}

/// The question check of `lens` as served (roadmap Phase 0 item 3, 2026-11-24): the pieces (the
/// pool and the answer pieces) flattened by the lens's shared directions as well as the question
/// (the lane flattens both sides since 2026-11-23); raw and calibrated `(MRR, first places)`. The
/// question-side-only score refused strength 0.1 (calibrated 0.3041 against 0.3047) though the
/// served quick board and full sets measured it better.
fn both_sided(lens: &crate::adapter::Lens, qd: &LensData, nulls: &[Vec<Vec<f32>>], pool: &[(usize, &[f32])]) -> ((f64, usize), (f64, usize)) {
    let fpool: Vec<(usize, Vec<f32>)> = pool.iter().map(|(i, v)| (*i, lens.apply_flat(v))).collect();
    let fref: Vec<(usize, &[f32])> = fpool.iter().map(|(i, v)| (*i, v.as_slice())).collect();
    let fqd = LensData { q: qd.q.clone(), y: qd.y.iter().map(|y| lens.apply_flat(y)).collect(), fold: qd.fold.clone(), own: qd.own.clone() };
    let all: Vec<usize> = (0..fqd.q.len()).collect();
    (rank_pairs(&fqd, &all, &fref, &|q| lens.apply(q)), rank_calibrated(&fqd, nulls, &fref, &|q| lens.apply(q)))
}

/// Lens fit grid: shrink of the shared directions, ridge strength (times the mean input energy),
/// rank. Widened 2026-11-23 (roadmap Phase 0 item 3, the refit on embedder-base@4): the stronger
/// ridge won at every step of the first grid (0.3 to 3), which ended there; the lens then lost first
/// places on the report half (98 to 94 of 289) and was refused.
const BETAS: [f64; 3] = [0.0, 0.5, 0.8];
const RIDGES: [f64; 4] = [0.3, 3.0, 10.0, 30.0];
const RANKS: [usize; 3] = [8, 16, 48];
/// Shared directions flattened.
const FLAT_K: usize = 8;
/// One lens chosen on the tuning questions: calibrated MRR, its name, the lens, and (raw MRR, raw
/// first places, calibrated MRR, calibrated first places).
type QuestionPick = (f64, String, crate::adapter::Lens, (f64, usize, f64, usize));
/// One lens setting chosen on fold 1: (keeps the base's first places, MRR), the lens, (beta, ridge,
/// rank).
type LensChoice = ((bool, f64), Lens, (f64, f64, usize));

/// Fits the lens: the grid chosen on fold 1, reported on fold 2 against the base (no lens) and a
/// shuffled-label control fitted with the chosen numbers. `material` is the lane's stored vectors
/// (for the shared directions); `pool` the items every pair is ranked against.
/// # Errors
/// Too few pairs in a fold, or no solvable fit.
pub fn fit_lens(data: &LensData, material: &[&[f32]], pool: &[(usize, &[f32])], stop: &dyn Fn() -> bool) -> Result<LensFit, Fault> {
    let d = data.q.first().map_or(0, Vec::len);
    let idx = |f: u8| -> Vec<usize> { (0..data.q.len()).filter(|i| data.fold[*i] == f).collect() };
    let (train, choose, report) = (idx(0), idx(1), idx(2));
    if d == 0 || train.len() < 32 || choose.len() < 8 || report.len() < 8 {
        return Err(bad(format!("too few pairs (fit {}, choose {}, report {})", train.len(), choose.len(), report.len())));
    }
    let dirs = shared_directions(material, d, FLAT_K, 3000);
    let ty: Vec<&[f32]> = train.iter().map(|i| data.y[*i].as_slice()).collect();
    let make = |beta: f64, down: &[f64], up: &[f64], r: usize| Lens {
        key: String::new(),
        values: String::new(),
        dim: d,
        flat: dirs.iter().flat_map(|u| to32(u)).collect(),
        beta: vec![beta as f32; dirs.len()],
        rank: r,
        down: to32(down),
        up: to32(up),
    };
    // The choice on fold 1: among the settings that keep the base's first places there, the best
    // MRR; when none keeps them, the best MRR (the report half's acceptance also asks for first
    // places not lower, so a setting trading first places for MRR was chosen and then refused).
    let mut best: Option<LensChoice> = None;
    let mut log = String::new();
    let (base_choose, base_first) = rank_pairs(data, &choose, pool, &|q| q.to_vec());
    for beta in BETAS {
        let xf: Vec<Vec<f64>> = train.iter().map(|i| flatten(&data.q[*i], &dirs, beta)).collect();
        for c in RIDGES {
            if stop() {
                return Err(bad("stopped: a call arrived"));
            }
            let Some((down, up)) = ridge_low_rank(&xf, &ty, d, c, *RANKS.iter().max().unwrap_or(&16)) else { continue };
            let rmax = up.len() / d.max(1);
            for r in RANKS {
                let r = r.min(rmax);
                // The leading r of the fitted directions.
                let down_r: Vec<f64> = down[..r * d].to_vec();
                let up_r: Vec<f64> = (0..d).flat_map(|m| up[m * rmax..m * rmax + r].to_vec()).collect();
                let lens = make(beta, &down_r, &up_r, r);
                let (mrr, first) = rank_pairs(data, &choose, pool, &|q| lens.apply(q));
                let _ = writeln!(log, "  beta {beta} ridge {c} rank {r}: choose MRR {mrr:.4} (base {base_choose:.4}), first {first} (base {base_first})");
                let key = (first >= base_first, mrr);
                if best.as_ref().is_none_or(|b| key.0 && !b.0.0 || key.0 == b.0.0 && key.1 > b.0.1) {
                    best = Some((key, lens, (beta, c, r)));
                }
            }
        }
    }
    let (_, lens, params) = best.ok_or_else(|| bad("no solvable fit"))?;
    // The shuffled-label control: the same numbers, items rotated among the training pairs.
    let shuffled: Vec<&[f32]> = (0..ty.len()).map(|k| ty[(k * 7 + 3) % ty.len()]).collect();
    let xf: Vec<Vec<f64>> = train.iter().map(|i| flatten(&data.q[*i], &dirs, params.0)).collect();
    let control = ridge_low_rank(&xf, &shuffled, d, params.1, params.2).map(|(dn, upv)| {
        let r = upv.len() / d.max(1);
        make(params.0, &dn, &upv, r)
    });
    let (b_mrr, b_first) = rank_pairs(data, &report, pool, &|q| q.to_vec());
    let (l_mrr, l_first) = rank_pairs(data, &report, pool, &|q| lens.apply(q));
    let (c_mrr, c_first) = control.as_ref().map_or((0.0, 0), |c| rank_pairs(data, &report, pool, &|q| c.apply(q)));
    // The flattening alone (rank 0), its beta chosen on fold 1 like the map's (roadmap Phase 0 item
    // 3, 2026-11-23): on embedder-base@4 the shuffled control gained nearly as much as the lens
    // (report MRR base 0.4354, lens 0.4416, control 0.4398), because both share the flattening of
    // the shared directions; that part helps without labels. The label map is held against the
    // flattening alone; when it adds too little over it, the flattening alone is the adapter.
    let mut flat_best: Option<(bool, f64, f64)> = None;
    for beta in BETAS.iter().copied().filter(|b| *b > 0.0) {
        let f = make(beta, &[], &[], 0);
        let (m, fi) = rank_pairs(data, &choose, pool, &|q| f.apply(q));
        let keeps = fi >= base_first;
        let _ = writeln!(log, "  beta {beta} flattening alone: choose MRR {m:.4} (base {base_choose:.4}), first {fi} (base {base_first})");
        if flat_best.is_none_or(|(k, bm, _)| keeps && !k || keeps == k && m > bm) {
            flat_best = Some((keeps, m, beta));
        }
    }
    let flat = flat_best.map(|(_, _, beta)| make(beta, &[], &[], 0));
    let (f_mrr, f_first) = flat.as_ref().map_or((b_mrr, b_first), |f| rank_pairs(data, &report, pool, &|q| f.apply(q)));
    let floor = f_mrr.max(b_mrr);
    let map_wins = l_mrr > floor && l_first >= b_first && c_mrr <= floor + 0.5 * (l_mrr - floor);
    let flat_wins = !map_wins && flat.is_some() && f_mrr > b_mrr && f_first >= b_first;
    let accepted = map_wins || flat_wins;
    let verdict = if map_wins {
        "accepted (the label map)"
    } else if flat_wins {
        "accepted (the flattening alone: the label map adds too little over it)"
    } else {
        "not accepted (no gain on the report half, or the control gains too)"
    };
    let report_text = format!(
        "lens fit: pairs fit {}, choose {}, report {}; pool {}; chosen beta {} ridge {} rank {}\n{log}  report half: MRR base {b_mrr:.4}, lens {l_mrr:.4}, flattening alone {f_mrr:.4}, shuffled control {c_mrr:.4}; first base {b_first}, lens {l_first}, flattening {f_first}, control {c_first} of {}; {verdict}\n",
        train.len(),
        choose.len(),
        report.len(),
        pool.len(),
        params.0,
        params.1,
        params.2,
        report.len(),
    );
    let (lens, params, l_mrr, l_first) = match (flat_wins, flat) {
        (true, Some(f)) => {
            let beta = f.beta.first().copied().map_or(0.0, f64::from);
            (f, (beta, 0.0, 0), f_mrr, f_first)
        }
        _ => (lens, params, l_mrr, l_first),
    };
    Ok(LensFit { lens, accepted, report: report_text, mrr: (b_mrr, l_mrr, c_mrr), first: (b_first, l_first, c_first), held: report.len(), params })
}

// ---------------------------------------------------------------- the judge read-out fit

/// What the judge fit gathers: running sums over the fit fold, and the held pairs in RAM.
#[derive(Default)]
pub struct JudgeData {
    pub d: usize,
    /// Sum of state-difference outer products, sum of differences, sum with random signs (the
    /// shuffled-label control), sum of states (for the bias), pairs.
    pub ss: Vec<f64>,
    pub s: Vec<f64>,
    pub s_ctrl: Vec<f64>,
    pub h_sum: Vec<f64>,
    pub n: usize,
    /// Sum and sum of squares of the judge's own log-odds differences on the fit fold.
    pub z_sum: f64,
    pub z_sq: f64,
    /// Held pairs: (fold, state difference, log-odds difference).
    pub held: Vec<(u8, Vec<f32>, f32)>,
}
impl JudgeData {
    #[must_use]
    pub fn new(d: usize) -> Self {
        Self { d, ss: vec![0.0; d * d], s: vec![0.0; d], s_ctrl: vec![0.0; d], h_sum: vec![0.0; d], ..Self::default() }
    }
    /// Adds one pair: the judge's states and log-odds for the true item and its rival.
    pub fn add(&mut self, fold: u8, h_true: &[f32], h_rival: &[f32], z_true: f32, z_rival: f32) {
        let d = self.d;
        if h_true.len() != d || h_rival.len() != d {
            return;
        }
        let delta: Vec<f32> = h_true.iter().zip(h_rival).map(|(a, b)| a - b).collect();
        let zd = z_true - z_rival;
        if fold != 0 {
            self.held.push((fold, delta, zd));
            return;
        }
        // A fixed pseudo-random sign per pair for the control.
        let sign = if nmhash::sha256_bytes(&self.n.to_le_bytes()).as_bytes().first().is_some_and(|b| b % 2 == 0) { 1.0 } else { -1.0 };
        for i in 0..d {
            let di = f64::from(delta[i]);
            self.s[i] += di;
            self.s_ctrl[i] += sign * di;
            self.h_sum[i] += f64::from(h_true[i]) + f64::from(h_rival[i]);
            if di == 0.0 {
                continue;
            }
            for (x, dj) in self.ss[i * d..i * d + d].iter_mut().zip(&delta) {
                *x += di * f64::from(*dj);
            }
        }
        self.z_sum += f64::from(zd);
        self.z_sq += f64::from(zd) * f64::from(zd);
        self.n += 1;
    }
}

/// One judge fit's result.
#[derive(Clone, Debug)]
pub struct JudgeFit {
    pub readout: Readout,
    pub accepted: bool,
    pub report: String,
    /// Report half: pairs ordered right (true above rival), base, adapter, control; pairs.
    pub right: (usize, usize, usize),
    pub held: usize,
}

const JUDGE_RIDGES: [f64; 3] = [0.1, 1.0, 10.0];
const ALPHAS: [f64; 4] = [0.25, 0.5, 1.0, 2.0];

fn solve_w(data: &JudgeData, s: &[f64], c: f64) -> Option<Vec<f64>> {
    let d = data.d;
    let trace: f64 = (0..d).map(|i| data.ss[i * d + i]).sum();
    let lambda = c * (trace / d as f64).max(1e-9);
    let mut a = data.ss.clone();
    for i in 0..d {
        a[i * d + i] += lambda;
    }
    if !cholesky(&mut a, d) {
        return None;
    }
    let mut w = s.to_vec();
    chol_solve(&a, d, &mut w, 1);
    // Scaled so the read-out's spread on the fit fold is one: var = (w'Sw)/n - (w's/n)^2.
    let nn = data.n.max(1) as f64;
    let sw = matvec(&data.ss, d, &w);
    let wsw: f64 = w.iter().zip(&sw).map(|(a, b)| a * b).sum();
    let ws: f64 = w.iter().zip(&data.s).map(|(a, b)| a * b).sum::<f64>() / nn;
    let var = (wsw / nn - ws * ws).max(1e-18);
    let sd = var.sqrt();
    Some(w.into_iter().map(|x| x / sd).collect())
}

fn right_count(held: &[(u8, Vec<f32>, f32)], fold: u8, w: &[f64], alpha: f64) -> (usize, usize) {
    let mut right = 0;
    let mut total = 0;
    for (f, delta, zd) in held.iter().filter(|h| h.0 == fold) {
        let _ = f;
        let m: f64 = w.iter().zip(delta).map(|(a, b)| a * f64::from(*b)).sum();
        total += 1;
        right += usize::from(f64::from(*zd) + alpha * m > 0.0);
    }
    (right, total)
}

/// Fits the judge's code read-out: ridge strength and scale chosen on fold 1 (pairs ordered right),
/// reported on fold 2 against the judge alone and the shuffled-sign control.
/// # Errors
/// Too few pairs, or no solvable fit.
pub fn fit_judge(data: &JudgeData) -> Result<JudgeFit, Fault> {
    let d = data.d;
    let choose = data.held.iter().filter(|h| h.0 == 1).count();
    let report = data.held.iter().filter(|h| h.0 == 2).count();
    if data.n < 32 || choose < 8 || report < 8 {
        return Err(bad(format!("too few judge pairs (fit {}, choose {choose}, report {report})", data.n)));
    }
    let nn = data.n as f64;
    // The scale of the judge's own log-odds differences, floored at one log-odds unit so a fold
    // where the judge barely varies does not shrink the read-out to nothing.
    let z_sd = (data.z_sq / nn - (data.z_sum / nn).powi(2)).max(1e-12).sqrt().max(1.0);
    let mut best: Option<(usize, Vec<f64>, f64, f64)> = None;
    let mut log = String::new();
    let (base_choose, _) = right_count(&data.held, 1, &vec![0.0; d], 0.0);
    for c in JUDGE_RIDGES {
        let Some(w) = solve_w(data, &data.s, c) else { continue };
        for a in ALPHAS {
            let alpha = a * z_sd;
            let (r, t) = right_count(&data.held, 1, &w, alpha);
            let _ = writeln!(log, "  ridge {c} alpha {a} x {z_sd:.3}: choose {r} of {t} right (judge alone {base_choose})");
            if best.as_ref().is_none_or(|b| r > b.0) {
                best = Some((r, w.clone(), alpha, c));
            }
        }
    }
    let (_, w, alpha, c) = best.ok_or_else(|| bad("no solvable judge fit"))?;
    let control = solve_w(data, &data.s_ctrl, c).unwrap_or_else(|| vec![0.0; d]);
    let (b_right, total) = right_count(&data.held, 2, &vec![0.0; d], 0.0);
    let (a_right, _) = right_count(&data.held, 2, &w, alpha);
    let (c_right, _) = right_count(&data.held, 2, &control, alpha);
    let accepted = a_right > b_right && c_right <= b_right + (a_right - b_right) / 2;
    // The bias centres the term over the fit fold's states.
    let mean_h: Vec<f64> = data.h_sum.iter().map(|x| x / (2.0 * nn)).collect();
    let bias = -w.iter().zip(&mean_h).map(|(a, b)| a * b).sum::<f64>();
    let report_text = format!(
        "judge code read-out fit: pairs fit {}, choose {choose}, report {report}; chosen ridge {c}, alpha {alpha:.4}\n{log}  report half: ordered right, judge alone {b_right}, with the read-out {a_right}, shuffled control {c_right} of {total}; {}\n",
        data.n,
        if accepted { "accepted" } else { "not accepted (no gain on the report half, or the control gains too)" }
    );
    Ok(JudgeFit {
        readout: Readout { key: String::new(), values: String::new(), w: to32(&w), bias: bias as f32, alpha: alpha as f32 },
        accepted,
        report: report_text,
        right: (b_right, a_right, c_right),
        held: total,
    })
}

// ---------------------------------------------------------------- the fit over the material

/// How much a fit reads.
#[derive(Clone, Debug)]
pub struct Opts {
    /// Pairs for the lens (each one or two embedder reads).
    pub pairs: usize,
    /// Pairs for the judge (two judge reads each), code items only.
    pub judge_pairs: usize,
    pub lens: bool,
    pub judge: bool,
    /// Write accepted adapters to the shelf (else report only).
    pub write: bool,
    /// Redo a fit whose labels have not changed.
    pub force: bool,
    /// Roadmap Phase 0 item 3 (2026-11-23): tuning questions with an answer place each (question,
    /// file, first line, last line), never the unseen verification set. A lens accepted on the
    /// material's own pairs must also not lower the MRR or the first places of these questions'
    /// answer pieces; empty: no such check. (The flattening alone passed its own control on
    /// embedder-base@4 and then cost large code first 27 to 23 and dev-large-2 code pack 90 to 84.)
    pub questions: Vec<(String, String, usize, usize)>,
    /// Only the question check of the lens on the shelf (no labels read, nothing fitted or
    /// written): the flattening at each strength on the question's side alone and on both sides.
    pub check_only: bool,
    /// The flattening strength chosen on the served quick board (roadmap Phase 0 item 3): the lens
    /// written is the flattening alone at this strength, still held to the report half and the
    /// question check.
    pub beta: Option<f32>,
}
impl Default for Opts {
    fn default() -> Self {
        Self { pairs: 2_000, judge_pairs: 400, lens: true, judge: true, write: true, force: false, questions: Vec::new(), check_only: false, beta: None }
    }
}

fn posed(question: &str) -> String {
    format!("Instruct: {}\nQuery:{question}", nmsubs::dense::task_for(question))
}

/// Roadmap Phase 0 item 3 (2026-11-23): the question check alone, on the newest lens of the shelf.
/// The tuning questions' answer places are ranked over the whole material, raw and on the drones'
/// calibrated scale, under the plain embedder and under the lens's flattening at each strength,
/// applied to the question alone (as served until now) and to both the question and the pieces
/// (all-but-the-top, Mu and Viswanath, ICLR 2018, takes the shared directions out of every vector
/// of the space; flattening the question alone left them in every piece, and every cosine fell by
/// about 0.1 with the order nearly kept). About two minutes: no labels are read, nothing written.
#[expect(clippy::too_many_arguments, reason = "the fit's own context, passed through")]
fn check_lens(
    home: &Path,
    sub: &nmsubs::Substrate,
    lane: &nmsubs::dense::DenseLane,
    e_values: &str,
    lens_id: &str,
    questions: &[(String, String, usize, usize)],
    reads: &dyn Fn(&str) -> Result<Vec<f32>, Fault>,
    mut out: String,
) -> Result<String, Fault> {
    let (_, file) = adapter::versions(home, lens_id).pop().ok_or_else(|| bad("no lens in the cartridge library"))?;
    let base_lens = adapter::lens_of(&file, e_values)?;
    let _ = writeln!(out, "lens check on {} ({} directions)", file.display(), base_lens.beta.len());
    let mut qd = LensData { q: Vec::new(), y: Vec::new(), fold: Vec::new(), own: Vec::new() };
    let mut qnulls: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut null_of: BTreeMap<&'static str, Vec<Vec<f32>>> = BTreeMap::new();
    for (question, file, a, b) in questions {
        let target = sub.chunks.iter().enumerate().find(|(_, c)| sub.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == *file) && c.start <= *b && c.end >= *a);
        let Some((t, c)) = target else { continue };
        let Some(y) = lane.vector(&c.key) else { continue };
        let task = nmsubs::dense::task_for(question);
        if !null_of.contains_key(task) {
            let mut ns = Vec::new();
            for cf in nmsubs::dense::CONTENT_FREE {
                ns.push(reads(&format!("Instruct: {task}\nQuery:{cf}"))?);
            }
            null_of.insert(task, ns);
        }
        qd.q.push(reads(&posed(question))?);
        qd.y.push(y.as_ref().clone());
        qd.fold.push(2);
        qd.own.push(t);
        qnulls.push(null_of.get(task).cloned().unwrap_or_default());
    }
    let held: Vec<(usize, Vec<f32>)> = sub.chunks.iter().enumerate().filter_map(|(i, c)| lane.vector(&c.key).map(|v| (i, v.as_ref().clone()))).collect();
    let all: Vec<usize> = (0..qd.q.len()).collect();
    let pool: Vec<(usize, &[f32])> = held.iter().map(|(i, v)| (*i, v.as_slice())).collect();
    let (bm, bf) = rank_pairs(&qd, &all, &pool, &|q| q.to_vec());
    let (cbm, cbf) = rank_calibrated(&qd, &qnulls, &pool, &|q| q.to_vec());
    let _ = writeln!(out, "  plain ({} answer places): raw MRR {bm:.4}, first {bf}; calibrated MRR {cbm:.4}, first {cbf}", qd.q.len());
    for beta in [0.1_f32, 0.2, 0.3, 0.5, 0.8, 1.0] {
        let mut f = base_lens.clone();
        f.beta = vec![beta; f.beta.len()];
        f.rank = 0;
        f.down.clear();
        f.up.clear();
        let (lm, lf) = rank_pairs(&qd, &all, &pool, &|q| f.apply(q));
        let (clm, clf) = rank_calibrated(&qd, &qnulls, &pool, &|q| f.apply(q));
        let _ = writeln!(out, "  flattening {beta}, question side: raw MRR {lm:.4}, first {lf}; calibrated MRR {clm:.4}, first {clf}");
        // Both sides: the pieces (and the answer pieces) flattened by the same directions.
        let fheld: Vec<(usize, Vec<f32>)> = held.iter().map(|(i, v)| (*i, f.apply(v))).collect();
        let fpool: Vec<(usize, &[f32])> = fheld.iter().map(|(i, v)| (*i, v.as_slice())).collect();
        let fqd = LensData { q: qd.q.clone(), y: qd.y.iter().map(|y| f.apply(y)).collect(), fold: qd.fold.clone(), own: qd.own.clone() };
        let (sm, sf) = rank_pairs(&fqd, &all, &fpool, &|q| f.apply(q));
        let (csm, csf) = rank_calibrated(&fqd, &qnulls, &fpool, &|q| f.apply(q));
        let _ = writeln!(out, "  flattening {beta}, both sides: raw MRR {sm:.4}, first {sf}; calibrated MRR {csm:.4}, first {csf}");
    }
    Ok(out)
}

/// The labels key the newest version of `id` was fitted on (`notes.labels`).
fn fitted_labels(home: &Path, id: &str) -> Option<String> {
    let (_, file) = adapter::versions(home, id).pop()?;
    let label = nmlearn::library::label_of(&file).ok()?;
    label.get("notes").ok()?.get("labels").ok()?.string().ok().map(str::to_owned)
}

/// Fits the project's adapters over the installed material: labels, the embedder's reads (the
/// lane's request form for questions, its document form for hidden-comment items, the lane's own
/// stored vectors for whole pieces), the lens fit, then the judge's reads and its fit. Every model
/// read takes the card as background work (it waits while a call is answered) and `stop` is
/// asked between reads: a call that arrives ends the fit, and what it gathered is dropped.
/// Returns the report.
/// # Errors
/// No meaning lane or embedder; the lane not ready; stopped; a fold too small; a file not written.
pub fn fit(org: &Organism, opts: &Opts, stop: &dyn Fn() -> bool) -> Result<String, Fault> {
    let home = org.home.clone();
    let sub = org.sub();
    let lane = sub.dense.read().ok().and_then(|d| d.clone()).ok_or_else(|| bad("no meaning lane"))?;
    if !lane.ready(&sub) {
        return Err(bad("the meaning lane is still filling"));
    }
    let embedder = crate::neural::bound_models().into_iter().find(|(r, _)| *r == "embedder").map(|(_, m)| m).ok_or_else(|| bad("the embedder is not bound"))?;
    let e_values = crate::neural::parameters_key(&embedder);
    let dev = if embedder.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
    let pairs = labels(&sub, opts.pairs);
    let key = labels_key(&sub, &pairs);
    let mut out = format!("adapter fit over {} ({} pieces): {} pairs ({} from commits, {} from doc comments); labels {}\n", sub.root.display(), sub.chunks.len(), pairs.len(), pairs.iter().filter(|p| p.source == "commit").count(), pairs.iter().filter(|p| p.source == "doc").count(), key.get(..12).unwrap_or(""));
    let lens_id = adapter::adapter_id(adapter::LENS_PREFIX, &sub.root);
    let judge_id = adapter::adapter_id(adapter::JUDGE_PREFIX, &sub.root);
    let lens_due = opts.lens && (opts.force || fitted_labels(&home, &lens_id).as_deref() != Some(&format!("{key} {e_values}")));
    let reads = |text: &str| -> Result<Vec<f32>, Fault> {
        if stop() {
            return Err(bad("stopped: a call arrived"));
        }
        let _bg = nmengn::residency::background();
        let _g = nmengn::placement::gpu_gate().enter("adapter fit (embedder)", false);
        nmengn::roles::card_embed_on(&embedder, dev, text)
    };
    if opts.check_only {
        return check_lens(&home, &sub, &lane, &e_values, &lens_id, &opts.questions, &reads, out);
    }
    // The embedder's reads: every pair's question, and its item.
    let mut data = LensData { q: Vec::new(), y: Vec::new(), fold: Vec::new(), own: Vec::new() };
    let mut kept: Vec<usize> = Vec::new();
    if lens_due || opts.judge {
        for (k, p) in pairs.iter().enumerate() {
            let Some(c) = sub.chunks.get(p.target) else { continue };
            let q = reads(&posed(&p.query))?;
            let y = match (p.source, lane.vector(&c.key)) {
                ("commit", Some(v)) => v.as_ref().clone(),
                _ => reads(&nmsubs::dense::doc_text_with(&p.file, &c.heading, &p.target_text))?,
            };
            data.q.push(q);
            data.y.push(y);
            data.fold.push(fold(&p.file));
            data.own.push(p.target);
            kept.push(k);
        }
    }
    // Roadmap Phase 0 item 3 (2026-11-23): the tuning questions' own answer places as lens pairs
    // (question to the answer piece's stored vector), half of the questions in the fit fold and half
    // in the choose fold (by the question's hash), each repeated so the questions weigh as much as
    // the material's pairs in their fold; the report fold stays the material's alone. The material's
    // pairs (doc comments, commit messages) stood for questions badly: every lens fitted on them
    // alone gained on the lane's ranking and lost served first places.
    if lens_due && !opts.questions.is_empty() {
        let mut qpairs: Vec<(Vec<f32>, Vec<f32>, u8, usize)> = Vec::new();
        for (question, file, a, b) in &opts.questions {
            let target = sub.chunks.iter().enumerate().find(|(_, c)| sub.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == *file) && c.start <= *b && c.end >= *a);
            let Some((t, c)) = target else { continue };
            let Some(y) = lane.vector(&c.key) else { continue };
            let f = u8::from(nmhash::sha256_bytes(question.as_bytes()).as_bytes().first().is_some_and(|x| x % 2 == 1));
            qpairs.push((reads(&posed(question))?, y.as_ref().clone(), f, t));
        }
        for f in [0_u8, 1] {
            let n_mat = data.fold.iter().filter(|x| **x == f).count();
            let n_q = qpairs.iter().filter(|p| p.2 == f).count();
            let times = n_mat.checked_div(n_q).map_or(0, |t| t.max(1));
            for (q, y, _, t) in qpairs.iter().filter(|p| p.2 == f) {
                for _ in 0..times {
                    data.q.push(q.clone());
                    data.y.push(y.clone());
                    data.fold.push(f);
                    data.own.push(*t);
                }
            }
        }
    }
    // The lane's stored vectors: the shared directions and the pool every pair is ranked in.
    let held_vecs: Vec<(usize, std::sync::Arc<Vec<f32>>)> = sub.chunks.iter().enumerate().filter_map(|(i, c)| lane.vector(&c.key).map(|v| (i, v))).collect();
    if lens_due {
        let material: Vec<&[f32]> = held_vecs.iter().map(|(_, v)| v.as_slice()).collect();
        let step = (held_vecs.len() / 2000).max(1);
        let mut pool: Vec<(usize, &[f32])> = held_vecs.iter().step_by(step).map(|(i, v)| (*i, v.as_slice())).collect();
        pool.extend(data.own.iter().zip(&data.y).zip(&data.fold).filter(|(_, f)| **f != 0).map(|((o, y), _)| (*o, y.as_slice())));
        match fit_lens(&data, &material, &pool, stop) {
            Ok(mut fit) => {
                out.push_str(&fit.report);
                // The question check: the tuning questions' first answer pieces ranked in the same
                // pool (with them added), base against the lens; a lens that lowers either refused.
                if fit.accepted && !opts.questions.is_empty() {
                    let mut qd = LensData { q: Vec::new(), y: Vec::new(), fold: Vec::new(), own: Vec::new() };
                    let mut qnulls: Vec<Vec<Vec<f32>>> = Vec::new();
                    let mut null_of: BTreeMap<&'static str, Vec<Vec<f32>>> = BTreeMap::new();
                    for (question, file, a, b) in &opts.questions {
                        let target = sub.chunks.iter().enumerate().find(|(_, c)| {
                            sub.files.get(c.file).is_some_and(|f| f.path.replace('\\', "/") == *file) && c.start <= *b && c.end >= *a
                        });
                        let Some((t, c)) = target else { continue };
                        let Some(y) = lane.vector(&c.key) else { continue };
                        let task = nmsubs::dense::task_for(question);
                        if !null_of.contains_key(task) {
                            let mut ns = Vec::new();
                            for cf in nmsubs::dense::CONTENT_FREE {
                                ns.push(reads(&format!("Instruct: {task}\nQuery:{cf}"))?);
                            }
                            null_of.insert(task, ns);
                        }
                        qd.q.push(reads(&posed(question))?);
                        qd.y.push(y.as_ref().clone());
                        qd.fold.push(2);
                        qd.own.push(t);
                        qnulls.push(null_of.get(task).cloned().unwrap_or_default());
                    }
                    // The whole material is the pool (the served lane ranks every piece): a sampled
                    // pool of about 3,000 hid the generic pieces the flattening raises (served:
                    // val08's answer left the lane's shortlist, drones.M 28 to 61).
                    let qpool: Vec<(usize, &[f32])> = held_vecs.iter().map(|(i, v)| (*i, v.as_slice())).collect();
                    let all: Vec<usize> = (0..qd.q.len()).collect();
                    let (bm, bf) = rank_pairs(&qd, &all, &qpool, &|q| q.to_vec());
                    let (cbm, cbf) = rank_calibrated(&qd, &qnulls, &qpool, &|q| q.to_vec());
                    // The hill-climb over the lens's strength on the tuning questions' answers (never
                    // the unseen set): the fitted lens and the flattening alone at each strength,
                    // ranked on both scales; the best on the calibrated scale that keeps first places
                    // on both is the lens. Each step reported.
                    let mut cands: Vec<(String, crate::adapter::Lens)> = vec![("fitted".to_owned(), fit.lens.clone())];
                    for beta in [0.05_f32, 0.1, 0.2, 0.3, 0.5, 0.8] {
                        let mut f = fit.lens.clone();
                        f.beta = vec![beta; f.beta.len()];
                        f.rank = 0;
                        f.down.clear();
                        f.up.clear();
                        cands.push((format!("flattening {beta}"), f));
                    }
                    let mut best: Option<QuestionPick> = None;
                    for (name, l) in cands {
                        let ((lm, lf), (clm, clf)) = both_sided(&l, &qd, &qnulls, &qpool);
                        let ok = lm >= bm && lf >= bf && clm >= cbm && clf >= cbf && clm > cbm;
                        let _ = writeln!(out, "  question hill-climb, {name}: raw MRR {lm:.4} (base {bm:.4}), first {lf} (base {bf}); calibrated MRR {clm:.4} (base {cbm:.4}), first {clf} (base {cbf}){}", if ok { "" } else { "; lowers the answers" });
                        if ok && best.as_ref().is_none_or(|b| clm > b.0) {
                            best = Some((clm, name, l, (lm, lf, clm, clf)));
                        }
                    }
                    // A strength chosen on the served quick board takes the place of the offline
                    // choice; it is still held to the question check (scored on both scales).
                    if let Some(b) = opts.beta {
                        let mut f = fit.lens.clone();
                        f.beta = vec![b; f.beta.len()];
                        f.rank = 0;
                        f.down.clear();
                        f.up.clear();
                        let ((lm2, lf2), (clm2, clf2)) = both_sided(&f, &qd, &qnulls, &qpool);
                        let ok = lm2 >= bm && lf2 >= bf && clm2 >= cbm && clf2 >= cbf;
                        let _ = writeln!(out, "  strength {b} from the served quick board: raw MRR {lm2:.4}, first {lf2}; calibrated MRR {clm2:.4}, first {clf2}{}", if ok { "" } else { "; lowers the answers" });
                        best = ok.then(|| (clm2, format!("flattening {b} (served quick board)"), f, (lm2, lf2, clm2, clf2)));
                    }
                    let keeps = best.is_some();
                    let (lm, lf, clm, clf) = best.as_ref().map_or((bm, bf, cbm, cbf), |b| b.3);
                    if let Some((_, name, l, _)) = best {
                        let _ = writeln!(out, "  chosen on the questions: {name}");
                        if name != "fitted" {
                            let beta = l.beta.first().copied().map_or(0.0, f64::from);
                            fit.lens = l;
                            fit.params = (beta, 0.0, 0);
                        }
                    }
                    let _ = writeln!(
                        out,
                        "  question check ({} answer places of the tuning questions): raw MRR base {bm:.4}, lens {lm:.4}, first {bf} to {lf}; calibrated MRR base {cbm:.4}, lens {clm:.4}, first {cbf} to {clf}; {}",
                        qd.q.len(),
                        if keeps { "kept" } else { "refused (the lens lowers the questions' answers)" }
                    );
                    if !keeps {
                        // Refused: nothing is written and the lens in force stays on, IN_TROUBLE,
                        // to be hill-climbed (part states: a part is never neutralised through its
                        // data, such as an identity file written in its place).
                        fit.accepted = false;
                        let _ = writeln!(out, "  the lens in force stays on (IN_TROUBLE); hill-climb its fit");
                    }
                }
                if fit.accepted && opts.write {
                    let notes = obj(vec![
                        ("values", s(&e_values)),
                        ("labels", s(&format!("{key} {e_values}"))),
                        ("project", s(&sub.root.display().to_string().replace('\\', "/"))),
                        ("fitted_on", s(&format!("{} pairs (fit fold), {} commit, {} doc", data.fold.iter().filter(|f| **f == 0).count(), pairs.iter().filter(|p| p.source == "commit").count(), pairs.iter().filter(|p| p.source == "doc").count()))),
                        ("measured", s(&format!("report half {} pairs: MRR {:.4} to {:.4}, first {} to {}; shuffled control MRR {:.4}", fit.held, fit.mrr.0, fit.mrr.1, fit.first.0, fit.first.1, fit.mrr.2))),
                        ("beta", f(fit.params.0)),
                        ("ridge", f(fit.params.1)),
                        ("rank", n(fit.params.2 as u64)),
                        ("acts_on", s("the request's vector only (item vectors unchanged)")),
                    ]);
                    let l = LabelIn {
                        id: &lens_id,
                        version: 0,
                        model: adapter::EMBEDDER,
                        weights: &embedder.pop.card.sha_main,
                        kind: adapter::LENS_KIND,
                        does: "the project's lens on the embedder's request vector: shared directions flattened, then a low-rank map toward this project's items",
                        made_from: "the project's own material: commit messages to the functions they changed, doc comments hidden from their definitions; split by file",
                        notes,
                    };
                    let d = fit.lens.dim;
                    let r = fit.lens.rank;
                    let k = fit.lens.beta.len();
                    let parts: Vec<Part> = vec![
                        ("flat".to_owned(), vec![k, d], fit.lens.flat.clone()),
                        ("beta".to_owned(), vec![k], fit.lens.beta.clone()),
                        ("down".to_owned(), vec![r, d], fit.lens.down.clone()),
                        ("up".to_owned(), vec![d, r], fit.lens.up.clone()),
                    ];
                    let (file, v) = write_next(&home, l, &parts)?;
                    let _ = writeln!(out, "  written {} (version {v})", file.display());
                    if let Ok(lens) = adapter::lens_of(&file, &e_values) {
                        adapter::seat_lens(lens, file);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(out, "lens fit: {e}");
            }
        }
    } else if opts.lens {
        let _ = writeln!(out, "lens: the newest version was fitted on these labels; not refitted");
    }
    if !opts.judge {
        return Ok(out);
    }
    // The judge: code items only, each against its hardest rival by the embedder.
    let judge = crate::neural::reranker_model(&home)?;
    let j_values = crate::neural::parameters_key(&judge);
    if !opts.force && fitted_labels(&home, &judge_id).as_deref() == Some(&format!("{key} {j_values}")) {
        let _ = writeln!(out, "judge: the newest version was fitted on these labels; not refitted");
        return Ok(out);
    }
    let code: Vec<(usize, &[f32])> = held_vecs.iter().filter(|(i, _)| sub.chunks.get(*i).and_then(|c| sub.files.get(c.file)).is_some_and(|f| f.is_code())).map(|(i, v)| (*i, v.as_slice())).collect();
    let place = |i: usize, text: &str| -> String {
        let c = sub.chunks.get(i);
        let path = c.and_then(|c| sub.files.get(c.file)).map_or_else(String::new, |f| f.path.replace('\\', "/"));
        crate::neural::close_doc(&judge, &format!("{path}:{}-{}", c.map_or(0, |x| x.start), c.map_or(0, |x| x.end)), text)
    };
    let mut jd = JudgeData::new(judge.pop.dims.hidden);
    let mut taken = 0;
    for (row, &k) in kept.iter().enumerate() {
        if taken >= opts.judge_pairs {
            break;
        }
        let Some(p) = pairs.get(k) else { continue };
        if !code.iter().any(|(i, _)| *i == p.target) {
            continue;
        }
        let q = &data.q[row];
        let rival = code.iter().filter(|(i, _)| *i != p.target).max_by(|a, b| adapter::dot(q, a.1).total_cmp(&adapter::dot(q, b.1))).map(|(i, _)| *i);
        let Some(rival) = rival else { continue };
        let rival_text = sub.chunks.get(rival).map_or("", |c| c.text.as_str());
        let read = |doc: String| -> Result<(Vec<f32>, f32), Fault> {
            if stop() {
                return Err(bad("stopped: a call arrived"));
            }
            let _bg = nmengn::residency::background();
            let _g = nmengn::placement::gpu_gate().enter("adapter fit (judge)", false);
            adapter::judge_state(&judge, &p.query, &doc)
        };
        let (ht, zt) = read(place(p.target, &p.target_text))?;
        let (hr, zr) = read(place(rival, rival_text))?;
        jd.add(data.fold[row], &ht, &hr, zt, zr);
        taken += 1;
    }
    match fit_judge(&jd) {
        Ok(fit) => {
            out.push_str(&fit.report);
            if fit.accepted && opts.write {
                let notes = obj(vec![
                    ("values", s(&j_values)),
                    ("labels", s(&format!("{key} {j_values}"))),
                    ("project", s(&sub.root.display().to_string().replace('\\', "/"))),
                    ("fitted_on", s(&format!("{} code pairs (fit fold), each true item against the embedder's hardest rival", jd.n))),
                    ("measured", s(&format!("report half {} pairs ordered right: judge alone {}, with the read-out {}, shuffled control {}", fit.held, fit.right.0, fit.right.1, fit.right.2))),
                    ("alpha", f(f64::from(fit.readout.alpha))),
                    ("bias", f(f64::from(fit.readout.bias))),
                    ("socket", s("final_norm, last position, the close-call prompt")),
                    ("scope", s("close calls whose candidates are all code")),
                ]);
                let l = LabelIn {
                    id: &judge_id,
                    version: 0,
                    model: adapter::JUDGE,
                    weights: &judge.pop.card.sha_main,
                    kind: adapter::READOUT_KIND,
                    does: "the judge's code read-out: one vector at its final norm added to its log-odds in code-against-code close calls",
                    made_from: "the project's own material: commit messages and doc comments to their code items, each against the embedder's hardest rival; split by file",
                    notes,
                };
                let parts: Vec<Part> = vec![("w".to_owned(), vec![fit.readout.w.len()], fit.readout.w.clone())];
                let (file, v) = write_next(&home, l, &parts)?;
                let _ = writeln!(out, "  written {} (version {v})", file.display());
                if let Ok(r) = adapter::readout_of(&file, &j_values) {
                    adapter::seat_readout(r, file);
                }
            }
        }
        Err(e) => {
            let _ = writeln!(out, "judge fit: {e}");
        }
    }
    Ok(out)
}

/// When the last idle fit ran (ms), so it runs at most once an hour.
static LAST_FIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Least time between idle fits (ms).
pub const FIT_EVERY_MS: u64 = 3_600_000;

/// The fit at idle (`writer_idle`'s loop, with `NANO_ADAPTER_FIT=on`): at most once an hour, only
/// over a full meaning lane, and only when the labels changed since the newest version (the fit
/// answers that itself). `stop` ends it when a call arrives.
/// # Errors
/// As `fit`.
pub fn fit_at_idle(org: &Organism, stop: &dyn Fn() -> bool) -> Result<Option<String>, Fault> {
    let now = nmtype::now_ms();
    let last = LAST_FIT.load(std::sync::atomic::Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < FIT_EVERY_MS {
        return Ok(None);
    }
    let sub = org.sub();
    if sub.dense.read().ok().and_then(|d| d.clone()).is_none_or(|l| !l.ready(&sub)) {
        return Ok(None);
    }
    LAST_FIT.store(now, std::sync::atomic::Ordering::Relaxed);
    fit(org, &Opts::default(), stop).map(Some)
}

/// Self-check of the arithmetic: Cholesky solves a known system; Jacobi and subspace iteration
/// find a known spectrum; the ridge read-out separates a planted direction; doc-comment pairs are
/// found and their comment hidden; definition names are read.
/// # Errors
/// The first of those that fails.
pub fn check() -> Result<(), String> {
    // [[4, 2], [2, 3]] x = [2, 1] -> x = [0.5, 0].
    let mut a = vec![4.0, 2.0, 2.0, 3.0];
    if !cholesky(&mut a, 2) {
        return Err("nmapi.adapter_fit: cholesky refused a positive definite matrix".into());
    }
    let mut b = vec![2.0, 1.0];
    chol_solve(&a, 2, &mut b, 1);
    if (b[0] - 0.5).abs() > 1e-12 || b[1].abs() > 1e-12 {
        return Err(format!("nmapi.adapter_fit: solve gave {b:?}"));
    }
    let (vals, _) = jacobi_eig(vec![2.0, 1.0, 1.0, 2.0], 2);
    if (vals[0] - 3.0).abs() > 1e-9 || (vals[1] - 1.0).abs() > 1e-9 {
        return Err(format!("nmapi.adapter_fit: jacobi gave {vals:?}"));
    }
    // diag(5, 3, 1, 0.5): the leading eigenvector is e0.
    let mut m = vec![0.0; 16];
    for (i, v) in [5.0, 3.0, 1.0, 0.5].into_iter().enumerate() {
        m[i * 4 + i] = v;
    }
    let (tv, tvec) = top_eig(&m, 4, 2, 30);
    if (tv[0] - 5.0).abs() > 1e-6 || tvec[0][0].abs() < 0.999 {
        return Err(format!("nmapi.adapter_fit: subspace iteration gave {tv:?}"));
    }
    // A planted direction: the true item's state is higher on channel 0.
    let mut jd = JudgeData::new(3);
    for k in 0..200 {
        let noise = (k as f32 * 0.37).sin();
        let fold = (k % 5) as u8 % 3;
        jd.add(fold, &[1.0 + 0.1 * noise, noise, 0.3], &[0.0, -noise, 0.3], -0.2, 0.0);
    }
    let fit = fit_judge(&jd).map_err(|e| e.to_string())?;
    if fit.right.1 <= fit.right.0 || fit.readout.w.first().is_none_or(|w| *w <= 0.0) {
        return Err(format!("nmapi.adapter_fit: planted read-out not found ({:?})", fit.right));
    }
    let mut pairs = Vec::new();
    chunk_doc_pairs("a.rs", "rust", 0, "/// Counts the open windows of the tide table.\npub fn tide_count(x: u32) -> u32 {\n    x\n}\n", &mut pairs);
    match pairs.first() {
        Some(p) if p.query == "Counts the open windows of the tide table." && !p.target_text.contains("Counts") && p.target_text.contains("tide_count") => {}
        other => return Err(format!("nmapi.adapter_fit: doc pair {other:?}")),
    }
    if def_name("pub(crate) async fn load_all(") != Some("load_all".into()) || def_name("let x = 1;").is_some() || def_name("class Tide:") != Some("Tide".into()) {
        return Err("nmapi.adapter_fit: definition names".into());
    }
    Ok(())
}
