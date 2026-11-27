//! `neuromorph-nano bench`: the served context path over a question set, one question after the
//! other in one process, as a server answers them. Per question and on average: seconds per stage
//! (from the whole path's own stage clock), card memory, weight uploads and releases, and whether
//! the right location is cited first, in the top three, or anywhere in the pack. Memory is off,
//! so nothing is recalled or written and runs repeat. Questions come from `tests/dev/questions.json`
//! (or `--questions <file>`); each answer is found at run time by its anchor line.
use nmjson::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

mod board;
mod evidence;
#[path = "bench/learn.rs"]
mod muscle;
mod pieces;

const USAGE: &str = "usage: neuromorph-nano bench [--mode plain|fast|full|stack|board|pieces|evidence|learn] [--paths plain,stack,fast,full] [--n <count>] [--root <repo>] [--questions <file>] [--dump <dir>] [--quiet] [-- <serve options>]";

/// One accepted answer: a file and a line range (or the whole file).
struct Place {
    file: String,
    first: usize,
    last: usize,
    whole_file: bool,
}

struct Question {
    id: String,
    kind: String,
    text: String,
    answers: Vec<Place>,
}

/// The questions of `file`, each answer's range moved to where its anchor line is now.
fn load(root: &Path, file: &Path) -> Result<Vec<Question>, String> {
    let bytes = std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let v = nmjson::parse(&bytes).map_err(|e| format!("{}: {e}", file.display()))?;
    let list = v.get("questions").and_then(Value::array).map_err(|e| format!("{}: {e}", file.display()))?;
    let str_of = |q: &Value, k: &str| q.get(k).and_then(Value::string).map(str::to_owned).map_err(|e| format!("{k}: {e}"));
    let mut out = Vec::new();
    for q in list {
        let id = str_of(q, "id")?;
        let mut answers = Vec::new();
        for a in q.get("answers").and_then(Value::array).map_err(|e| format!("{id}: answers: {e}"))? {
            let path = str_of(a, "file")?;
            let anchor = str_of(a, "anchor")?;
            let lines: Vec<usize> = a
                .get("lines")
                .and_then(Value::array)
                .map(|l| l.iter().filter_map(|x| x.integer().ok().and_then(|n| usize::try_from(n).ok())).collect())
                .map_err(|e| format!("{id}: lines: {e}"))?;
            let span = match lines.as_slice() {
                [a, b] if b >= a => b - a,
                _ => return Err(format!("{id}: lines must be [first, last]")),
            };
            // The file as the reader decodes it (any encoding), so line numbers are the reader's; a
            // container member (`file!member`) or a file read raw as the general reading derives it.
            let decoded = if path.contains('!') { None } else { std::fs::read(root.join(&path)).ok().and_then(|b| nmdoc::text::decode(&b).ok()).map(|t| t.0) };
            let text = match decoded {
                Some(t) => t,
                None => nmsubs::repo::fresh_text(root, &path).ok_or_else(|| format!("{id}: {path}: not found or not read"))?,
            };
            let hits: Vec<usize> = text.lines().enumerate().filter(|(_, l)| l.contains(anchor.as_str())).map(|(i, _)| i + 1).collect();
            let [first] = hits.as_slice() else {
                return Err(format!("{id}: anchor {anchor:?} occurs {} times in {path}", hits.len()));
            };
            let whole_file = a.get("match").and_then(Value::string).is_ok_and(|m| m == "file");
            answers.push(Place { file: path, first: *first, last: first + span, whole_file });
        }
        out.push(Question { id, kind: str_of(q, "kind")?, text: str_of(q, "question")?, answers });
    }
    Ok(out)
}

/// The places a text reply cites, in reply order: `@@ path:first-last` spans and `~~ path:line` notes.
fn cited(reply: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    for l in reply.lines() {
        let (head, span) = if let Some(r) = l.strip_prefix("@@ ") {
            (r, true)
        } else if let Some(r) = l.strip_prefix("~~ ") {
            (r, false)
        } else {
            continue;
        };
        // The span ends at the first space or `;` (a header's evidence may follow it directly).
        let Some((path, at)) = head.split_whitespace().next().map(|h| h.trim_end_matches(';')).and_then(|h| h.rsplit_once(':')) else { continue };
        let (a, b) = if span { at.split_once('-').unwrap_or((at, at)) } else { (at, at) };
        if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
            out.push((path.replace('\\', "/"), a, b));
        }
    }
    out
}

fn hits(q: &Question, c: &(String, usize, usize)) -> bool {
    q.answers.iter().any(|a| a.file == c.0 && (a.whole_file || (c.1 <= a.last && c.2 >= a.first)))
}

/// What one question measured.
struct Row {
    id: String,
    kind: String,
    secs: f64,
    stages: Vec<(String, f64)>,
    first: bool,
    top3: bool,
    in_pack: bool,
    withheld: bool,
    uploaded_mb: f64,
    released: u64,
    first_cited: String,
    /// Batched against one-at-a-time reranker scores (`--rerank-compare`): largest difference, same order.
    check: Option<(f64, bool)>,
    /// Reranker top-two log-odds gap, whether the second look ran, and right first before it.
    gap: f64,
    look_ran: bool,
    first_before_look: bool,
    dense_added: usize,
    rerank_docs: usize,
    rerank_dropped: usize,
    /// `--repeat`: seconds of the second identical call, whether the cache answered, same reply.
    again: Option<(f64, bool, bool)>,
    /// What the judge read (pieces, windowed, tokens) and the second look (pieces, tokens, passes).
    judge: (usize, usize, usize),
    look_read: (usize, usize, usize),
    /// Each candidate's reranker P(yes) and the second look's scores (`kind place score`).
    scores: Vec<String>,
    /// Reply tokens (characters / 4), every answer cited somewhere in the pack, low confidence.
    tokens: usize,
    all_in_pack: bool,
    low: bool,
    /// The reply's reason line when the pack is withheld or marked low.
    why: String,
}

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a String> {
    args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1))
}

#[expect(clippy::cast_precision_loss, reason = "byte and question counts for a report")]
fn mib(b: usize) -> f64 {
    b as f64 / f64::from(1 << 20)
}

fn uploaded_bytes() -> u64 {
    nmapi::neural::bound_models().iter().map(|(_, m)| m.pop.uploaded_bytes()).sum()
}

/// `neuromorph-nano bench ...`.
pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let code = match bench(home, args) {
        Ok(text) => {
            print!("{text}");
            0
        }
        Err(e) => {
            eprintln!("bench: {e}\n{USAGE}");
            1
        }
    };
    shutdown(code)
}

/// The bench's end (lane T1, 2026-11-19, T-086: the bench hung at exit after its board, one idle
/// thread holding about 5 GB, an hour on the Mini's tree): background card passes stop, no job
/// enters the device gate any more and the one holding it is waited for (30 s), what is written
/// is flushed, and the process ends without the libraries' detach handlers (a thread still inside
/// a card driver call, such as a background warm or a reading outside the gate, left the driver's
/// detach waiting for good). What still held the card is named on standard error.
pub(crate) fn shutdown(code: u32) -> ! {
    use std::io::Write;
    nmapi::key_scan::stop_passes(None, std::time::Duration::ZERO);
    if let Some(job) = nmengn::placement::gpu_gate().close(std::time::Duration::from_secs(30)) {
        eprintln!("shutdown: {job} still held the card after 30 s; ending");
    }
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    nmffi::dynlib::end_process(code)
}

#[expect(clippy::too_many_lines, reason = "one report, written top to bottom")]
#[expect(clippy::cast_precision_loss, reason = "counts and seconds for a report")]
fn bench(home: &Path, args: &[String]) -> Result<String, String> {
    use std::fmt::Write;
    let (own, extra) = match args.iter().position(|a| a == "--") {
        Some(i) => (args.get(..i).unwrap_or_default(), args.get(i + 1..).unwrap_or_default()),
        None => (args, &[][..]),
    };
    let saved = super::controls::current(home).get("mode").cloned().unwrap_or_else(|| "full".into());
    let mode = arg(own, "--mode").cloned().unwrap_or(saved);
    // Several modes run one after the other in one process (`--mode full,fast`), on the same warm
    // models and the same dense index.
    let modes: Vec<String> = mode.split(',').map(str::to_owned).collect();
    if mode == "stack" {
        return stack_bench(home, own);
    }
    if mode == "pieces" {
        return pieces::run(home, own);
    }
    if mode == "board" {
        return board::run(home, own, extra);
    }
    if mode == "evidence" {
        return evidence::run(home, own);
    }
    if mode == "muscle" {
        return muscle::run(home, own, extra);
    }
    if modes.iter().any(|m| !["plain", "fast", "full"].contains(&m.as_str())) || (modes.len() > 1 && modes.iter().any(|m| m == "plain")) {
        return Err(format!("--mode takes plain, fast or full, or a list of fast and full, got {mode}"));
    }
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let mut questions = load(&root, &qfile)?;
    if let Some(n) = arg(own, "--n") {
        questions.truncate(n.parse::<usize>().map_err(|_| format!("--n takes a count, got {n}"))?);
    }
    let quiet = own.iter().any(|a| a == "--quiet");
    let repeat = own.iter().any(|a| a == "--repeat");
    let t_open = Instant::now();
    // Anything the organism keeps on disk for this tree (its dense index) goes to a scratch folder,
    // never to the project's real memory; memory itself is off.
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, &root));
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    let mut serve_args: Vec<String> = if mode == "plain" { Vec::new() } else { vec!["--whole".into(), "--neural".into()] };
    serve_args.extend(extra.iter().cloned());
    super::configure(&org, &serve_args)?;
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    let open_s = t_open.elapsed().as_secs_f64();
    // A server has been up a while before its first question: the whole path's models are warm
    // (`--cold` leaves the first question to pay for loading them, as before).
    let mut warm_note = String::new();
    if !own.iter().any(|a| a == "--cold")
        && let Some(w) = nmapi::tools::installed().and_then(|o| o.whole.read().ok().and_then(|w| w.clone()))
    {
        let (secs, n) = w.warm().map_err(|e| format!("warm: {e}"))?;
        warm_note = format!("models warm before the first question: {:.1} s after open ({n} tensors staged)", secs);
    }
    // `--wait-dense`: the dense lane's index is complete before the first question.
    let mut dense_note = String::new();
    if own.iter().any(|a| a == "--wait-dense")
        && let Some(o) = nmapi::tools::installed()
    {
        let sub = o.sub();
        let lane = sub.dense.read().ok().and_then(|d| d.clone()).ok_or("--wait-dense: no dense lane (full or fast mode binds it)")?;
        let t = Instant::now();
        while !lane.ready(&sub) {
            std::thread::park_timeout(std::time::Duration::from_millis(250));
        }
        let st = lane.stats.lock().map(|s| s.clone()).unwrap_or_default();
        dense_note = format!(
            "dense lane: {} chunks embedded in {:.1} s of engine time, ready {:.1} s after the wait began",
            st.embedded_now,
            st.index_ms / 1000.0,
            t.elapsed().as_secs_f64()
        );
        eprintln!("{dense_note}");
    }
    // `--embed-probe <n>`: the embedder over the first n chunks, whole and batched as the dense
    // lane embeds them (its speed and card memory at the precision in use).
    if let Some(n) = arg(own, "--embed-probe").and_then(|v| v.parse::<usize>().ok())
        && let Some(o) = nmapi::tools::installed()
    {
        let m = nmapi::neural::embedder_model(home).map_err(|e| format!("embed probe: {e}"))?;
        let sub = o.sub();
        let texts: Vec<String> = sub
            .chunks
            .iter()
            .take(n)
            .map(|c| nmsubs::dense::doc_text(sub.files.get(c.file).map_or("", |f| f.path.as_str()), &c.text))
            .collect();
        let tokens: usize = texts.iter().map(|t| m.pop.tokenizer.encode(t).map_or(0, |v| v.len())).sum();
        let cap = nmengn::roles::input_cap(&m, nmengn::placement::Device::Gpu, 1);
        let limit = nmengn::roles::batch_tokens(&m, cap + 1);
        let t = Instant::now();
        let mut done = 0;
        for group in texts.chunks(nmapi::neural::DENSE_BATCH) {
            done += nmengn::roles::card_embed_batch(&m, group, cap, limit).map_err(|e| format!("embed probe: {e}"))?.len();
        }
        let secs = t.elapsed().as_secs_f64();
        let note = format!(
            "embed probe: {done} chunks ({tokens} tokens) in {secs:.1} s, {:.1} ms per chunk, {:.0} tokens per second; embedder on the card {:.0} MiB; whole tree ({} chunks) at this rate about {:.0} s",
            secs * 1000.0 / done.max(1) as f64,
            tokens as f64 / secs.max(1e-9),
            mib(m.pop.device_bytes()),
            sub.chunks.len(),
            secs / done.max(1) as f64 * sub.chunks.len() as f64
        );
        eprintln!("{note}");
        dense_note = if dense_note.is_empty() { note } else { format!("{dense_note}\n{note}") };
    }
    let api = nmapi::tools::api();

    // Card memory: the lowest free memory the driver reports, sampled while the bench runs.
    let stop = Arc::new(AtomicBool::new(false));
    let min_free = Arc::new(AtomicUsize::new(usize::MAX));
    let total = Arc::new(AtomicUsize::new(0));
    let sampler = {
        let (stop, min_free, total) = (Arc::clone(&stop), Arc::clone(&min_free), Arc::clone(&total));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Some((f, t)) = nmengn::residency::get().memory_now() {
                    min_free.fetch_min(f, Ordering::Relaxed);
                    total.store(t, Ordering::Relaxed);
                }
                std::thread::park_timeout(std::time::Duration::from_millis(20));
            }
        })
    };
    let used_at_start = nmengn::residency::get().memory_now().map(|(f, t)| t - f);

    // `--profile`: the engine's stage spans over the questions (they synchronise the card at each
    // node, so seconds grow; the shares are what they show).
    let profile = own.iter().any(|a| a == "--profile");
    if profile {
        nmffi::prof::reset();
        nmffi::prof::set(true);
    }
    let mut by_mode: Vec<(String, Vec<Row>)> = Vec::new();
    for mode in &modes {
        let mut rows: Vec<Row> = Vec::new();
        for q in &questions {
            let (up0, rel0) = (uploaded_bytes(), nmengn::residency::get().stats().weights_released);
            let mut input = vec![("question".to_owned(), Value::String(q.text.clone())), ("form".to_owned(), Value::String("full".into()))];
            if mode != "plain" {
                input.push(("mode".to_owned(), Value::String(mode.clone())));
            }
            let input = Value::Object(input.into_iter().collect());
            let as_text = |r: Result<Value, nmapi::ApiError>| match r {
                Ok(Value::String(s)) => s,
                Ok(v) => v.encode().unwrap_or_default(),
                Err(e) => format!("error {}: {}", e.code, e.message),
            };
            let t = Instant::now();
            let text = as_text(api.dispatch("nm_context_pack", &input));
            let secs = t.elapsed().as_secs_f64();
            let diag = nmapi::tools::take_diagnostics();
            // `--repeat`: the same call again at once (the answer cache's case).
            let again = repeat.then(|| {
                let t = Instant::now();
                let second = as_text(api.dispatch("nm_context_pack", &input));
                let s2 = t.elapsed().as_secs_f64();
                let hit = nmapi::tools::take_diagnostics().is_some_and(|d| d.get("answer_cache").is_ok());
                (s2, hit, second == text)
            });
            let stages: Vec<(String, f64)> = diag
                .as_ref()
                .and_then(|d| d.get("context_map").ok())
                .and_then(|c| c.get("stages").ok())
                .and_then(|s| s.object().ok())
                .map(|m| {
                    nmapi::whole::STAGES
                        .iter()
                        .map(|k| {
                            let v = m.get(*k).and_then(|x| if let Value::Number(n) = x { n.parse::<f64>().ok() } else { None });
                            ((*k).to_owned(), v.unwrap_or(0.0))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let cmap = diag.as_ref().and_then(|d| d.get("context_map").ok());
            let check = cmap.and_then(|c| c.get("rerank_check").ok()).and_then(|r| {
                let d = match r.get("max_diff").ok()? {
                    Value::Number(n) => n.parse::<f64>().ok()?,
                    _ => return None,
                };
                Some((d, matches!(r.get("same_order"), Ok(Value::Bool(true)))))
            });
            let num = |k: &str| {
                cmap.and_then(|c| c.get(k).ok()).and_then(|x| if let Value::Number(n) = x { n.parse::<f64>().ok() } else { None })
            };
            let gap = num("top_gap").unwrap_or(f64::INFINITY);
            let look_ran = cmap.is_some_and(|c| matches!(c.get("look_ran"), Ok(Value::Bool(true))));
            let pre = cmap.and_then(|c| c.get("pre_look_first").ok()).and_then(|x| x.string().ok()).map(|p| cited(&format!("@@ {p}\n")));
            let first_before_look = pre.as_ref().and_then(|p| p.first()).is_some_and(|x| hits(q, x));
            let faults: Vec<String> = cmap
                .and_then(|c| c.get("faults").ok())
                .and_then(|f| f.array().ok())
                .map(|f| f.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
                .unwrap_or_default();
            if let Some(cap) = num("rerank_cap").filter(|_| !quiet) {
                eprintln!("{} reranker document cap {cap} tokens", q.id);
            }
            if !faults.is_empty() {
                eprintln!("{} faults: {}", q.id, faults.join(" | "));
            }
            // `--dump <dir>`: each reply as served, one file per question, for reading the misses.
            if let Some(dir) = arg(own, "--dump") {
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(Path::new(dir).join(format!("{}-{mode}.txt", q.id)), &text);
            }
            let c = cited(&text);
            let all_in_pack = !q.answers.is_empty()
                && q.answers.iter().all(|a| c.iter().any(|x| a.file == x.0 && (a.whole_file || (x.1 <= a.last && x.2 >= a.first))));
            let row = Row {
                tokens: text.chars().count() / 4,
                all_in_pack,
                low: text.lines().any(|l| l.starts_with("confidence: low")),
                why: text
                    .lines()
                    .take_while(|l| !l.starts_with("@@ "))
                    .find(|l| (l.starts_with("reason: ") || l.starts_with("answer: ")) && (l.contains("withheld") || l.contains("low confidence")))
                    .map(str::to_owned)
                    .unwrap_or_default(),
                id: q.id.clone(),
                kind: q.kind.clone(),
                secs,
                stages,
                first: c.first().is_some_and(|x| hits(q, x)),
                top3: c.iter().take(3).any(|x| hits(q, x)),
                in_pack: c.iter().any(|x| hits(q, x)),
                withheld: text.lines().next().is_some_and(|l| l.contains(" withheld")),
                uploaded_mb: (uploaded_bytes().saturating_sub(up0)) as f64 / f64::from(1 << 20),
                released: nmengn::residency::get().stats().weights_released.saturating_sub(rel0),
                first_cited: c.first().map(|x| format!("{}:{}-{}", x.0, x.1, x.2)).unwrap_or_default(),
                check,
                gap,
                look_ran,
                first_before_look,
                again,
                rerank_docs: num("rerank_docs").map_or(0, |x| x as usize),
                rerank_dropped: num("rerank_dropped").map_or(0, |x| x as usize),
                dense_added: cmap.and_then(|c| c.get("dense_added").ok()).and_then(|a| a.array().ok()).map_or(0, <[Value]>::len),
                judge: (
                    num("judge_pieces").map_or(0, |x| x as usize),
                    num("judge_windowed").map_or(0, |x| x as usize),
                    num("judge_tokens").map_or(0, |x| x as usize),
                ),
                look_read: (
                    num("look_pieces").map_or(0, |x| x as usize),
                    num("look_tokens").map_or(0, |x| x as usize),
                    num("look_passes").map_or(0, |x| x as usize),
                ),
                scores: cmap
                    .and_then(|c| c.get("scores").ok())
                    .and_then(|a| a.array().ok())
                    .map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect())
                    .unwrap_or_default(),
            };
            if !quiet {
                eprintln!(
                    "{} {:<9} {:6.2} s  first {}  top3 {}  pack {}  {}",
                    row.id,
                    row.kind,
                    row.secs,
                    u8::from(row.first),
                    u8::from(row.top3),
                    u8::from(row.in_pack),
                    row.first_cited
                );
            }
            rows.push(row);
        }
        by_mode.push((mode.clone(), rows));
    }
    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    nmffi::prof::set(false);

    let mut out = String::new();
    let _ = writeln!(
        out,
        "neuromorph-nano bench: mode {mode}, {} questions from {}, root {}",
        questions.len(),
        qfile.strip_prefix(home).unwrap_or(&qfile).display(),
        root.display()
    );
    let _ = writeln!(out, "serve options: {}", if serve_args.is_empty() { "(plain path)".to_owned() } else { serve_args.join(" ") });
    let _ = writeln!(out, "precision: {}", nmengn::model::PRECISION.read().map(|p| format!("{p:?}")).unwrap_or_default());
    let _ = writeln!(out, "open and configure: {open_s:.2} s");
    if !warm_note.is_empty() {
        let _ = writeln!(out, "{warm_note}");
    }
    if !dense_note.is_empty() {
        let _ = writeln!(out, "{dense_note}");
    }
    let mut tail = String::new();
    for (mode, rows) in &by_mode {
        let n = rows.len().max(1) as f64;
        let out = &mut tail;
        let _ = writeln!(out);
        let _ = writeln!(out, "== mode {mode}");
        let _ = writeln!(out, "id   kind       seconds  first top3 pack  uploaded MB released  first cited");
        for r in rows {
            let _ = writeln!(
                out,
                "{:<4} {:<9} {:8.2}  {:>5} {:>4} {:>4}  {:11.1} {:8}  {}{}",
                r.id,
                r.kind,
                r.secs,
                u8::from(r.first),
                u8::from(r.top3),
                u8::from(r.in_pack),
                r.uploaded_mb,
                r.released,
                r.first_cited,
                if r.withheld { " (withheld)" } else { "" }
            );
        }
        for r in rows.iter().filter(|r| !r.why.is_empty()) {
            let _ = writeln!(out, "why {} {}", r.id, r.why);
        }
        let mean =
            |rs: &[&Row], f: &dyn Fn(&Row) -> f64| if rs.is_empty() { 0.0 } else { rs.iter().map(|r| f(r)).sum::<f64>() / rs.len() as f64 };
        let all: Vec<&Row> = rows.iter().collect();
        let warm: Vec<&Row> = rows.iter().skip(1).collect();
        let _ = writeln!(out);
        let _ = writeln!(out, "seconds per question     all    after the first");
        let _ = writeln!(out, "{:<22} {:6.3} {:8.3}", "total", mean(&all, &|r| r.secs), mean(&warm, &|r| r.secs));
        if rows.iter().any(|r| !r.stages.is_empty()) {
            for k in nmapi::whole::STAGES {
                let f = |r: &Row| r.stages.iter().find(|s| s.0 == k).map_or(0.0, |s| s.1);
                let _ = writeln!(out, "{:<22} {:6.3} {:8.3}", k, mean(&all, &f), mean(&warm, &f));
            }
            let rest = |r: &Row| r.secs - r.stages.iter().map(|s| s.1).sum::<f64>();
            let _ = writeln!(out, "{:<22} {:6.3} {:8.3}", "other (reply, tool)", mean(&all, &rest), mean(&warm, &rest));
        }
        let count = |f: &dyn Fn(&Row) -> bool, kind: Option<&str>| rows.iter().filter(|r| kind.is_none_or(|k| r.kind == k) && f(r)).count();
        let _ = writeln!(out);
        let _ = writeln!(out, "kind        n  first  top3  pack  withheld  every");
        let mut kinds: Vec<&str> = rows.iter().map(|r| r.kind.as_str()).collect();
        kinds.sort_unstable();
        kinds.dedup();
        for k in kinds.iter().map(|k| Some(*k)).chain(std::iter::once(None)) {
            let _ = writeln!(
                out,
                "{:<9} {:>3}  {:>5} {:>5} {:>5} {:>9} {:>6}",
                k.unwrap_or("all"),
                count(&|_| true, k),
                count(&|r| r.first, k),
                count(&|r| r.top3, k),
                count(&|r| r.in_pack, k),
                count(&|r| r.withheld, k),
                count(&|r| r.all_in_pack, k)
            );
        }
        let right3: Vec<&Row> = rows.iter().filter(|r| r.top3).collect();
        let answered: Vec<&Row> = rows.iter().filter(|r| r.kind != "decoy").collect();
        let decoys: Vec<&Row> = rows.iter().filter(|r| r.kind == "decoy").collect();
        let _ = writeln!(
            out,
            "reply tokens (characters / 4): {:.0} per question, {:.0} per right answer in the top 3; every answer in the pack on {} of {} questions with answers",
            mean(&all, &|r| r.tokens as f64),
            rows.iter().map(|r| r.tokens).sum::<usize>() as f64 / right3.len().max(1) as f64,
            answered.iter().filter(|r| r.all_in_pack).count(),
            answered.len()
        );
        if !decoys.is_empty() {
            let _ = writeln!(
                out,
                "decoys: {} of {} withheld or marked low confidence; questions with answers withheld {}, marked low {}",
                decoys.iter().filter(|r| r.withheld || r.low).count(),
                decoys.len(),
                answered.iter().filter(|r| r.withheld).count(),
                answered.iter().filter(|r| r.low).count()
            );
        }
        let _ = writeln!(
            out,
            "weights uploaded after the first question: {:.1} MB; weight releases: {} (after the first question {})",
            warm.iter().map(|r| r.uploaded_mb).sum::<f64>(),
            rows.iter().map(|r| r.released).sum::<u64>(),
            warm.iter().map(|r| r.released).sum::<u64>()
        );
        if rows.iter().any(|r| r.look_ran) {
            let _ = writeln!(out, "second look: ran on {} of {} questions", rows.iter().filter(|r| r.look_ran).count(), rows.len());
            let _ = writeln!(out, "id   gap (log-odds)  look  first before  first after");
            for r in rows.iter().filter(|r| r.gap.is_finite()) {
                let _ = writeln!(
                    out,
                    "{:<4} {:14.3}  {:>4}  {:>12}  {:>11}",
                    r.id,
                    r.gap,
                    u8::from(r.look_ran),
                    u8::from(r.first_before_look),
                    u8::from(r.first)
                );
            }
        }
        let agains: Vec<(f64, bool, bool)> = rows.iter().filter_map(|r| r.again).collect();
        if !agains.is_empty() {
            let _ = writeln!(
                out,
                "repeated call: {:.4} s per question on average (first call {:.3} s); answered by the cache {} of {}, reply identical {} of {}",
                agains.iter().map(|a| a.0).sum::<f64>() / agains.len() as f64,
                mean(&all, &|r| r.secs),
                agains.iter().filter(|a| a.1).count(),
                agains.len(),
                agains.iter().filter(|a| a.2).count(),
                agains.len()
            );
        }
        if rows.iter().any(|r| r.rerank_docs > 0) {
            let _ = writeln!(
                out,
                "reranker documents: {:.1} per question in batches; early drop stopped {} of {}",
                rows.iter().map(|r| r.rerank_docs).sum::<usize>() as f64 / n,
                rows.iter().map(|r| r.rerank_dropped).sum::<usize>(),
                rows.iter().map(|r| r.rerank_docs).sum::<usize>()
            );
        }
        if rows.iter().any(|r| r.judge.0 > 0) {
            let _ = writeln!(
                out,
                "judge read: {:.1} pieces per question, {:.1} as a window, {:.0} tokens per question ({:.0} per piece)",
                rows.iter().map(|r| r.judge.0).sum::<usize>() as f64 / n,
                rows.iter().map(|r| r.judge.1).sum::<usize>() as f64 / n,
                rows.iter().map(|r| r.judge.2).sum::<usize>() as f64 / n,
                rows.iter().map(|r| r.judge.2).sum::<usize>() as f64 / rows.iter().map(|r| r.judge.0).sum::<usize>().max(1) as f64
            );
        }
        let looked: Vec<&Row> = rows.iter().filter(|r| r.look_read.0 > 0).collect();
        if !looked.is_empty() {
            let _ = writeln!(
                out,
                "second look read: {:.1} pieces and {:.0} tokens over {:.1} passes per question it ran on ({} questions)",
                looked.iter().map(|r| r.look_read.0).sum::<usize>() as f64 / looked.len() as f64,
                looked.iter().map(|r| r.look_read.1).sum::<usize>() as f64 / looked.len() as f64,
                looked.iter().map(|r| r.look_read.2).sum::<usize>() as f64 / looked.len() as f64,
                looked.len()
            );
        }
        if rows.iter().any(|r| r.dense_added > 0) {
            let _ = writeln!(
                out,
                "dense lane: {} candidates added over {} questions",
                rows.iter().map(|r| r.dense_added).sum::<usize>(),
                rows.iter().filter(|r| r.dense_added > 0).count()
            );
        }
        let checks: Vec<(f64, bool)> = rows.iter().filter_map(|r| r.check).collect();
        if !checks.is_empty() {
            let _ = writeln!(
                out,
                "reranker batched against one at a time: largest score difference {:.6} over {} questions; order identical on {} of them",
                checks.iter().map(|c| c.0).fold(0.0_f64, f64::max),
                checks.len(),
                checks.iter().filter(|c| c.1).count()
            );
        }
        let _ = n;
    }
    out.push_str(&tail);
    let _ = writeln!(out);
    let tot = total.load(Ordering::Relaxed);
    // Scores per question for paired comparison of runs, at the end of the report.
    let mut scores = String::new();
    for (mode, rows) in &by_mode {
        for r in rows {
            let _ = writeln!(scores, "score {mode} {} first {}", r.id, r.first_cited);
            for x in &r.scores {
                let _ = writeln!(scores, "score {mode} {} {x}", r.id);
            }
        }
    }
    let mf = min_free.load(Ordering::Relaxed);
    if tot > 0 && mf != usize::MAX {
        let _ = writeln!(
            out,
            "card memory: peak used {:.0} of {:.0} MiB (driver view, every process); {:.0} MiB used before the first question",
            mib(tot - mf),
            mib(tot),
            used_at_start.map_or(0.0, mib)
        );
        if let Some(p) = nmengn::residency::get().pool() {
            let _ = writeln!(
                out,
                "  engine pool: long-lived (weights) {:.0} MiB, working peak {:.0} MiB, idle cached {:.0} MiB, out-of-memory retries {}",
                mib(p.live_kept),
                mib(p.peak_work),
                mib(p.idle_bytes),
                p.oom_retries
            );
        }
    } else {
        let _ = writeln!(out, "card memory: no graphics card in use");
    }
    let rs = nmengn::residency::get().stats();
    let _ = writeln!(
        out,
        "residency: weights released {} in all; room asked by background work stopped short of releasing weights {} times",
        rs.weights_released, rs.weights_kept_for_background
    );
    if let Some(o) = nmapi::tools::installed() {
        let sub = o.sub();
        if let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) {
            let st = lane.stats.lock().map(|s| s.clone()).unwrap_or_default();
            let (waits, wait_s) = nmengn::placement::gpu_gate().take_background_waits();
            let _ = writeln!(
                out,
                "dense lane at the end: {} of {} chunks embedded (ready {}), {:.1} s of engine time; background waits at the device gate {waits} ({wait_s:.1} s)",
                lane.coverage(&sub).0,
                sub.chunks.len(),
                lane.ready(&sub),
                st.index_ms / 1000.0
            );
        }
    }
    for (role, m) in nmapi::neural::bound_models() {
        let _ = writeln!(
            out,
            "  {role}: {} on the card now {:.0} MiB, uploaded in all {:.0} MiB",
            m.pop.card.id,
            mib(m.pop.device_bytes()),
            mib(usize::try_from(m.pop.uploaded_bytes()).unwrap_or(usize::MAX))
        );
    }
    if own.iter().any(|a| a == "--memory-map")
        && let Some(o) = nmapi::tools::installed()
    {
        memory_map(&mut out, &o.sub(), &questions);
    }
    if profile {
        let mut t = nmffi::prof::table();
        t.sort_by(|a, b| b.1.total_cmp(&a.1));
        let _ = writeln!(out, "profile (exclusive seconds, calls), top 25:");
        for (path, secs, count) in t.iter().take(25) {
            let _ = writeln!(out, "  {secs:9.3} {count:8}  {path}");
        }
    }
    out.push_str(&scores);
    let runs = home.join("runs");
    let _ = std::fs::create_dir_all(&runs);
    let file = runs.join(format!("bench-{}-{}.txt", mode.replace(',', "-"), nmtype::now_ms()));
    if std::fs::write(&file, &out).is_ok() {
        let _ = writeln!(out, "written: {}", file.strip_prefix(home).unwrap_or(&file).display());
    }
    Ok(out)
}

/// Where one question's answer sits at each level of the Globalized Context Stack (0-based places;
/// `None`: not there).
struct Placed {
    id: String,
    /// Place in the drones' intake order (flag strength, ties by chunk), and whether it was flagged.
    intake: Option<usize>,
    flagged: bool,
    /// The tentacle holding it and its place in that `g0_tx`'s order; whether the `g0_tx` reported it.
    window: Option<(usize, usize)>,
    reported: bool,
    /// Place in G1's order (hand-over, then queue), and whether it is in the hand-over.
    g1: Option<usize>,
    handover: bool,
    /// Place in what G2 sends up, and whether every answer (deciding line) has a piece there.
    up: Option<usize>,
    every_up: bool,
}

impl Placed {
    fn line(&self) -> String {
        let p = |x: Option<usize>| x.map_or_else(|| "-".to_owned(), |v| (v + 1).to_string());
        let w = self.window.map_or_else(|| "-".to_owned(), |(t, r)| format!("t{}#{}{}", t + 1, r + 1, if self.reported { "*" } else { "" }));
        format!(
            "{:<4} {:>7}{} {:>8} {:>5}{} {:>4} {:>3} {:>5}",
            self.id,
            p(self.intake),
            if self.flagged { " " } else { "!" },
            w,
            p(self.g1),
            if self.handover { "*" } else { " " },
            p(self.up),
            u8::from(self.up.is_some_and(|x| x < nmapi::whole::FAST_K)),
            u8::from(self.every_up)
        )
    }
}

/// `bench --mode stack`: the Globalized Context Stack alone (processor only, no models), the same
/// call the whole path makes, and per question where the answer sits after the drones' intake, in
/// the tentacles' windows and `g0_tx` reports, in G1's order and in what G2 sends up. The
/// placements carry no times, so two runs of the same build give the same placement hash.
#[expect(clippy::too_many_lines, reason = "one report, written top to bottom")]
#[expect(clippy::cast_precision_loss, reason = "counts and seconds for a report")]
fn stack_bench(home: &Path, own: &[String]) -> Result<String, String> {
    use std::fmt::Write;
    let root = match arg(own, "--root") {
        Some(r) if Path::new(r).is_dir() => nmsubs::repo::canon(Path::new(r)),
        Some(r) => return Err(format!("--root {r} is not a folder")),
        None => home.to_path_buf(),
    };
    let qfile = arg(own, "--questions").map_or_else(|| home.join("tests/dev/questions.json"), PathBuf::from);
    let mut questions = load(&root, &qfile)?;
    if let Some(n) = arg(own, "--n") {
        questions.truncate(n.parse::<usize>().map_err(|_| format!("--n takes a count, got {n}"))?);
    }
    let t_open = Instant::now();
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, &root));
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    let sub = org.sub();
    let mut keep = nmapi::drone_windows::Keep::open(&nmtype::models_dir(home).join("qwen3-0.6b/files")).map_err(|e| format!("intake: {e}"))?;
    keep.sync(&sub).map_err(|e| format!("intake: {e}"))?;
    let open_s = t_open.elapsed().as_secs_f64();
    // `--lenses on`: each tentacle's window composed by its lens, as `serve --lenses on`.
    let lenses = arg(own, "--lenses").is_some_and(|v| v == "on");
    let mut rows: Vec<Placed> = Vec::new();
    let mut times: Vec<[f64; 5]> = Vec::new();
    let mut per_t = vec![0_usize; nmsubs::lens::NAMES.len()];
    let mut up_n = 0;
    // `--why`: for every answer missing from the up set, where its chunks were lost and how they
    // relate to what went up.
    let why = own.iter().any(|a| a == "--why");
    let mut whys: Vec<String> = Vec::new();
    // How many chunks hold each identifier (the count the specific relations' rarity limit is
    // compared with), read only for `--why`.
    let mut holders: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    if why {
        for ch in &sub.chunks {
            for id in nmsubs::refgraph::identifiers(&ch.text) {
                *holders.entry(id).or_default() += 1;
            }
        }
        let g = sub.refs();
        whys.push(format!("specific relations: rarity limit {} chunks per name ({})", g.specific.limit, g.specific.rule));
    }
    for q in &questions {
        let run = nmapi::drone_path::stack_run(&sub, &keep, &q.text, None, true, lenses).map_err(|e| format!("{}: {e}", q.id))?;
        times.push([run.sense_s, run.compose_s, run.g1_s, run.total_s, f64::from(run.rounds)]);
        up_n = up_n.max(run.up.len());
        let ans: Vec<usize> = (0..sub.chunks.len())
            .filter(|c| {
                sub.chunks.get(*c).is_some_and(|ch| {
                    let path = sub.files.get(ch.file).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
                    q.answers.iter().any(|a| a.file == path && (a.whole_file || (ch.start <= a.last && ch.end >= a.first)))
                })
            })
            .collect();
        let mut order: Vec<usize> = (0..run.flags.len()).collect();
        order.sort_by(|a, b| run.flags[*b].total_cmp(&run.flags[*a]).then(a.cmp(b)));
        let best = |list: &[usize]| list.iter().position(|c| ans.contains(c));
        let window = run
            .orders
            .iter()
            .enumerate()
            .filter_map(|(t, o)| best(o).map(|r| (t, r)))
            .min_by_key(|(t, r)| (*r, *t));
        let reported = window.is_some_and(|(t, r)| r < run.depths.get(t).copied().unwrap_or(0));
        if let (Some((t, _)), true) = (window, reported)
            && let Some(x) = per_t.get_mut(t)
        {
            *x += 1;
        }
        let mut g1_order = run.handover.clone();
        g1_order.extend(run.up.iter().chain(&run.queue).filter(|c| !run.handover.contains(c)));
        let g1 = best(&g1_order);
        rows.push(Placed {
            id: q.id.clone(),
            intake: best(&order),
            flagged: ans.iter().any(|c| run.flags.get(*c).is_some_and(|f| *f > 0.0)),
            window,
            reported,
            g1,
            handover: g1.is_some_and(|p| p < run.handover.len()),
            up: best(&run.up),
            every_up: !q.answers.is_empty()
                && q.answers.iter().all(|a| {
                    run.up.iter().any(|c| {
                        sub.chunks.get(*c).is_some_and(|ch| {
                            sub.files.get(ch.file).is_some_and(|f| f.path.replace('\\', "/") == a.file)
                                && (a.whole_file || (ch.start <= a.last && ch.end >= a.first))
                        })
                    })
                }),
        });
        if !own.iter().any(|a| a == "--quiet") {
            eprintln!("{}", rows.last().map(Placed::line).unwrap_or_default());
        }
        if why && rows.last().is_some_and(|r| !r.every_up) {
            whys.extend(why_missing(&sub, &keep, &run, q, &order, &g1_order, &holders));
            whys.extend(run.g2.iter().map(|l| format!("  {} G2: {l}", q.id)));
            whys.push(format!("  {} up: {}", q.id, run.up.iter().map(|c| sub.chunks.get(*c).map_or_else(String::new, |ch| format!("{c}:{}:{}", sub.files.get(ch.file).map(|f| f.path.as_str()).unwrap_or(""), ch.start))).collect::<Vec<_>>().join(" ")));
        }
    }
    let lines: Vec<String> = rows.iter().map(Placed::line).collect();
    let mut out = String::new();
    let _ = writeln!(
        out,
        "neuromorph-nano bench: mode stack (Globalized Context Stack only, processor), {} questions from {}, root {}",
        questions.len(),
        qfile.strip_prefix(home).unwrap_or(&qfile).display(),
        root.display()
    );
    let _ = writeln!(out, "open and intake: {open_s:.2} s ({} chunks); windows by {}", sub.chunks.len(), if lenses { "lens (--lenses on)" } else { "BM25F flags" });
    let _ = writeln!(out, "places are 1-based; ! not flagged by any drone; window t<tentacle>#<place in its g0_tx order>, * reported; G1 * in the hand-over");
    let _ = writeln!(out, "id    intake    window    G1     G2  f4 every  (f4: in G2's first {}, what fast mode reads; every: every answer in the up set)", nmapi::whole::FAST_K);
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    let answered: Vec<&Placed> = rows.iter().filter(|r| questions.iter().any(|q| q.id == r.id && !q.answers.is_empty())).collect();
    let n = answered.len();
    let count = |f: &dyn Fn(&Placed) -> bool| answered.iter().filter(|r| f(r)).count();
    let up = up_n.max(1);
    let _ = writeln!(out);
    let _ = writeln!(out, "of {n} questions with answers (up set {up} pieces):");
    let _ = writeln!(out, "  intake: flagged {}, in the drones' first {up} (the whole GCX bypassed) {}", count(&|r| r.flagged), count(&|r| r.intake.is_some_and(|p| p < up)));
    let _ = writeln!(out, "  tentacles: in a window {}, reported by its g0_tx {}", count(&|r| r.window.is_some()), count(&|r| r.reported));
    let _ = writeln!(out, "  G1: in the hand-over {}, in G1's first {up} (G2 bypassed) {}", count(&|r| r.handover), count(&|r| r.g1.is_some_and(|p| p < up)));
    let _ = writeln!(
        out,
        "  G2: in the up set {}, first {}, top 3 {}, in the first {} (what fast mode reads) {}",
        count(&|r| r.up.is_some()),
        count(&|r| r.up == Some(0)),
        count(&|r| r.up.is_some_and(|p| p < 3)),
        nmapi::whole::FAST_K,
        count(&|r| r.up.is_some_and(|p| p < nmapi::whole::FAST_K))
    );
    let _ = writeln!(
        out,
        "  every answer (each deciding line) in the up set: {}
  reported by tentacle: {}",
        count(&|r| r.every_up),
        per_t.iter().enumerate().map(|(t, c)| format!("t{} {c}", t + 1)).collect::<Vec<_>>().join(", ")
    );
    let mean = |k: usize, skip: usize| {
        let v: Vec<f64> = times.iter().skip(skip).filter_map(|t| t.get(k).copied()).collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };
    let _ = writeln!(out);
    let _ = writeln!(out, "seconds per question     all    after the first");
    for (k, name) in ["drones' intake", "windows and g0_tx", "G1", "GCX in all"].iter().enumerate() {
        let _ = writeln!(out, "{:<22} {:6.3} {:8.3}", name, mean(k, 0), mean(k, 1));
    }
    let _ = writeln!(out, "G1 rounds per question: {:.2}", mean(4, 0));
    let _ = writeln!(out, "placements sha256: {}", nmhash::sha256_bytes(lines.join("\n").as_bytes()));
    if why {
        let _ = writeln!(out);
        let _ = writeln!(out, "answers missing from the up set (--why):");
        for w in &whys {
            let _ = writeln!(out, "{w}");
        }
    }
    let runs = home.join("runs");
    let _ = std::fs::create_dir_all(&runs);
    let file = runs.join(format!("bench-stack-{}.txt", nmtype::now_ms()));
    if std::fs::write(&file, &out).is_ok() {
        let _ = writeln!(out, "written: {}", file.strip_prefix(home).unwrap_or(&file).display());
    }
    Ok(out)
}

/// For each answer of `q` with no piece in the up set, one line per chunk holding it: its lines,
/// tokens (and twice the mean held chunk), the definitions it holds (name, kind, chunks holding the
/// name), its place in the intake order, the tentacle holding it with its place in that `g0_tx`'s
/// order and gate depth, its place in G1's order, its relations to the pieces that went up
/// (specific graph: direction and kind; first form: direction and weight; `u` the place in the up
/// set) and the up pieces of its own file (`next`: the adjacent chunk).
#[expect(clippy::too_many_arguments, reason = "the question, the stack's run, its two orders and the name counts")]
fn why_missing(
    sub: &nmsubs::Substrate,
    keep: &nmapi::drone_windows::Keep,
    run: &nmapi::drone_path::StackRun,
    q: &Question,
    order: &[usize],
    g1_order: &[usize],
    holders: &std::collections::BTreeMap<String, usize>,
) -> Vec<String> {
    let path_of = |c: usize| sub.chunks.get(c).and_then(|ch| sub.files.get(ch.file)).map(|f| f.path.replace('\\', "/")).unwrap_or_default();
    let covers = |c: usize, a: &Place| sub.chunks.get(c).is_some_and(|ch| path_of(c) == a.file && (a.whole_file || (ch.start <= a.last && ch.end >= a.first)));
    let held: Vec<usize> = run.windows.iter().flatten().map(|c| keep.tokens(*c).max(1)).collect();
    let mean = held.iter().sum::<usize>().checked_div(held.len()).unwrap_or(1).max(1);
    let graph = sub.refs();
    let at = |list: &[usize], c: usize| list.iter().position(|x| *x == c).map_or_else(|| "-".to_owned(), |p| (p + 1).to_string());
    let mut out = Vec::new();
    for a in q.answers.iter().filter(|a| !run.up.iter().any(|c| covers(*c, a))) {
        let cs: Vec<usize> = (0..sub.chunks.len()).filter(|c| covers(*c, a)).collect();
        // The innermost definition enclosing the answer's first line, the chunks it spans and
        // which of them went up.
        let enclosing = sub
            .symbols
            .iter()
            .filter(|s| s.file.replace('\\', "/") == a.file && s.line <= a.first && a.first <= s.end)
            .min_by_key(|s| s.end - s.line)
            .map_or_else(
                || "none".to_owned(),
                |s| {
                    let spans: Vec<String> = (0..sub.chunks.len())
                        .filter(|c| sub.chunks.get(*c).is_some_and(|ch| path_of(*c) == a.file && ch.start <= s.end && ch.end >= s.line))
                        .map(|c| format!("{c}{}", run.up.iter().position(|x| *x == c).map_or_else(String::new, |p| format!("=u{}", p + 1))))
                        .collect();
                    format!("{}:{} lines {}-{} in chunks [{}]", s.name, s.kind, s.line, s.end, spans.join(" "))
                },
            );
        let head = format!("{} {}:{}-{} (in {enclosing})", q.id, a.file, a.first, a.last);
        if cs.is_empty() {
            out.push(format!("{head}: no chunk holds it"));
            continue;
        }
        for &c in &cs {
            let Some(ch) = sub.chunks.get(c) else { continue };
            let defs: Vec<String> = sub
                .symbols
                .iter()
                .filter(|s| s.file.replace('\\', "/") == a.file && s.line >= ch.start && s.line <= ch.end)
                .map(|s| format!("{}:{}:{}", s.name, s.kind, holders.get(&s.name).copied().unwrap_or(0)))
                .collect();
            let win = run.windows.iter().position(|w| w.contains(&c)).map_or_else(
                || "-".to_owned(),
                |t| {
                    let place = run.orders.get(t).and_then(|o| o.iter().position(|x| *x == c)).map_or_else(|| "?".to_owned(), |p| (p + 1).to_string());
                    format!("t{}#{place}/d{}", t + 1, run.depths.get(t).copied().unwrap_or(0))
                },
            );
            let spec: Vec<String> = graph
                .specific
                .edges_of(c)
                .iter()
                .filter_map(|e| {
                    let u = run.up.iter().position(|x| *x == e.to as usize)?;
                    Some(format!("{} {} u{}", e.dir.name(), e.kind.name(), u + 1))
                })
                .collect();
            let first: Vec<String> = graph
                .neighbours(c)
                .into_iter()
                .filter_map(|(j, rel, w)| {
                    let u = run.up.iter().position(|x| *x == j)?;
                    Some(format!("{} u{} {w:.2}", rel.name(), u + 1))
                })
                .collect();
            let same: Vec<String> = run
                .up
                .iter()
                .enumerate()
                .filter(|(_, u)| path_of(**u) == a.file)
                .map(|(p, u)| format!("u{}{}", p + 1, if u.abs_diff(c) == 1 { "(next)" } else { "" }))
                .collect();
            out.push(format!(
                "{head}: chunk {c} lines {}-{} of {} chunks, {} tokens (2x mean {}), defines [{}], intake {} flag {:.2}, window {win}, G1 {}; specific [{}]; first form [{}]; same file up [{}]",
                ch.start,
                ch.end,
                cs.len(),
                keep.tokens(c),
                2 * mean,
                defs.join(" "),
                at(order, c),
                run.flags.get(c).copied().unwrap_or(0.0),
                at(g1_order, c),
                spec.join(", "),
                first.join(", "),
                same.join(" ")
            ));
        }
    }
    out
}

/// The part of a model a weight tensor belongs to: token table, one layer, final norm, output head.
fn part_of(tensor: &str) -> String {
    if tensor.starts_with("model.embed_tokens") {
        "token table".into()
    } else if let Some(rest) = tensor.strip_prefix("model.layers.") {
        format!("layer {:02}", rest.split('.').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(0))
    } else if tensor.starts_with("model.norm") {
        "final norm".into()
    } else if tensor.starts_with("lm_head") {
        "output head".into()
    } else {
        "other".into()
    }
}

/// `--memory-map`: per bound model, card and RAM bytes by part (after the run), the token ids the
/// studied material and the questions use, and tensors byte-identical across the models.
#[expect(clippy::too_many_lines, reason = "one report section")]
fn memory_map(out: &mut String, sub: &nmsubs::Substrate, questions: &[Question]) {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fmt::Write;
    let models = nmapi::neural::bound_models();
    let _ = writeln!(out, "\nmemory map (bytes held now, by part; MiB):");
    for (role, m) in &models {
        let mut by: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for (id, st, ram) in m.pop.prov.residency() {
            let tensor = id.split('/').nth(1).unwrap_or("");
            let dev = if let nmnpcs::provision::Storage::Device { bytes } = st { bytes } else { 0 };
            let e = by.entry(part_of(tensor)).or_default();
            e.0 += dev;
            e.1 += ram;
        }
        let tied = !m.pop.tensors.iter().any(|t| t.name == "lm_head.weight");
        let layers: Vec<(&String, &(u64, u64))> = by.iter().filter(|(k, _)| k.starts_with("layer ")).collect();
        let on_card: Vec<usize> =
            layers.iter().filter(|(_, v)| v.0 > 0).filter_map(|(k, _)| k.get(6..).and_then(|n| n.parse().ok())).collect();
        let lmb = |b: u64| mib(usize::try_from(b).unwrap_or(usize::MAX));
        let _ = writeln!(
            out,
            "  {role} ({}, {} layers{}): card {:.0}, RAM {:.0}",
            m.pop.card.id,
            m.pop.dims.layers,
            if tied { ", output head tied to the token table" } else { "" },
            lmb(by.values().map(|v| v.0).sum()),
            lmb(by.values().map(|v| v.1).sum())
        );
        for k in ["token table", "final norm", "output head", "other"] {
            if let Some(v) = by.get(k) {
                let _ = writeln!(out, "    {k:<12} card {:8.1}  RAM {:8.1}", lmb(v.0), lmb(v.1));
            }
        }
        let card_layers: u64 = layers.iter().map(|(_, v)| v.0).sum();
        let ram_layers: u64 = layers.iter().map(|(_, v)| v.1).sum();
        let _ = writeln!(
            out,
            "    layers       card {:8.1}  RAM {:8.1}  ({} of {} layers on the card: {}; {:.1} card MiB per layer)",
            lmb(card_layers),
            lmb(ram_layers),
            on_card.len(),
            layers.len(),
            match (on_card.first(), on_card.last()) {
                (Some(a), Some(b)) => format!("{a}..={b}"),
                _ => "none".into(),
            },
            lmb(card_layers) / on_card.len().max(1) as f64
        );
    }
    // Token ids the studied material and the questions use (the token table's rows ever read).
    if let Some((_, m)) = models.first() {
        let tok = &m.pop.tokenizer;
        let mut material: BTreeSet<i32> = BTreeSet::new();
        let mut asked: BTreeSet<i32> = BTreeSet::new();
        for c in &sub.chunks {
            let path = sub.files.get(c.file).map_or("", |f| f.path.as_str());
            material.extend(tok.encode(&nmsubs::dense::doc_text(path, &c.text)).unwrap_or_default());
        }
        for q in questions {
            asked.extend(tok.encode(&q.text).unwrap_or_default());
        }
        let prompt: BTreeSet<i32> =
            tok.encode(&nmengn::roles::card_prompt(nmengn::roles::CODE_INSTRUCTION, "", "")).unwrap_or_default().into_iter().collect();
        let union: BTreeSet<i32> = material.iter().chain(&asked).chain(&prompt).copied().collect();
        let _ = writeln!(
            out,
            "  token ids: {} distinct in the studied material ({} chunks), {} in the questions, {} in the fixed prompts; {} in all, of {} rows in the token table ({:.1}%)",
            material.len(),
            sub.chunks.len(),
            asked.len(),
            prompt.len(),
            union.len(),
            m.pop.dims.vocab,
            100.0 * union.len() as f64 / m.pop.dims.vocab.max(1) as f64
        );
    }
    // Output-head rows each model reads.
    for (role, m) in &models {
        let rows = match *role {
            "reranker" => "2 (the yes and no label rows only; the rest are not loaded)".to_owned(),
            "embedder" => "none (the job ends at the final norm and a last-token pool; no head is loaded)".to_owned(),
            _ => format!("{} (the full vocabulary, for the second look's yes or no tails, which read 2 of them)", m.pop.dims.vocab),
        };
        let _ = writeln!(out, "  output head rows read by the {role}: {rows}");
    }
    // Tensors byte-identical across the models (each population holds its own copy).
    let hash = |m: &nmengn::model::Model, t: &nmnpcs::decompose::TensorDesc| -> Option<String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(m.pop.card.dir.join(&t.file)).ok()?;
        f.seek(SeekFrom::Start(t.start)).ok()?;
        let mut buf = vec![0_u8; usize::try_from(t.end.saturating_sub(t.start)).ok()?];
        f.read_exact(&mut buf).ok()?;
        Some(nmhash::sha256_bytes(&buf))
    };
    let mut seen: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for (role, m) in &models {
        for t in &m.pop.tensors {
            if let Some(h) = hash(m, t) {
                seen.entry(h).or_default().push(((*role).to_owned(), t.name.clone()));
            }
        }
    }
    let shared: Vec<&Vec<(String, String)>> = seen.values().filter(|v| v.iter().map(|x| &x.0).collect::<BTreeSet<_>>().len() > 1).collect();
    let _ = writeln!(out, "  tensors byte-identical across the models: {} (each model holds its own copy)", shared.len());
    for v in shared.iter().take(12) {
        let _ = writeln!(out, "    {}", v.iter().map(|(r, n)| format!("{r}:{n}")).collect::<Vec<_>>().join(" = "));
    }
}

/// Section check (processor only): reply parsing and answer matching.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let reply = "nm-pack v3 full 2 of 2 packets\nanswer: x\n@@ a/b.rs:10-20 (matched 12)\nfn x() {}\n~~ c.rs:7 symbol\nfn y\n";
    let c = cited(reply);
    if c != vec![("a/b.rs".to_owned(), 10, 20), ("c.rs".to_owned(), 7, 7)] {
        return Err(format!("nmcore.bench: cited {c:?}"));
    }
    let q = Question {
        id: "t".into(),
        kind: "name".into(),
        text: String::new(),
        answers: vec![Place { file: "a/b.rs".into(), first: 18, last: 30, whole_file: false }],
    };
    let far = ("a/b.rs".to_owned(), 31, 40);
    if !hits(&q, &c[0]) || hits(&q, &c[1]) || hits(&q, &far) {
        return Err("nmcore.bench: answer matching".into());
    }
    Ok(())
}
