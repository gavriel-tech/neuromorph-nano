//! `writer-idle --root <repo> --questions <file> [--notes <n>] [-- <serve options>]`: the trial of the
//! writer at idle (`nmapi::writer_idle`). One organism as `serve` opens it; every question is a served
//! call (marked as `mcp::call` marks it). Measured: call seconds before any writer session (second
//! pass, warm); one session of `--notes` notes (load, note times, release, restore, card bytes); call
//! seconds after it; then a session left to write while a call arrives mid-write (the call's seconds,
//! the time until the writer let go and the finding MicroModels were restored). Nothing is written
//! to disk.
use nmjson::Value;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn questions(file: &Path) -> Result<Vec<String>, String> {
    let v = nmjson::parse(&std::fs::read(file).map_err(|e| format!("{}: {e}", file.display()))?).map_err(|e| e.to_string())?;
    let list = v.get("questions").and_then(|q| q.array().map(<[Value]>::to_vec)).map_err(|e| e.to_string())?;
    Ok(list.iter().filter_map(|q| q.get("question").and_then(Value::string).ok().map(str::to_owned)).collect())
}

/// Calls asked so far: each call's token budget differs by one from the last, so no call is an
/// answer-cache hit (the cache keys on the budget) and every timing is a real served call.
static ASKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn ask(q: &str) -> f64 {
    let _call = nmapi::writer_idle::call();
    let n = ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tokens = Value::Number((nmapi::drone_path::PACK_TOKENS + n % 64).to_string());
    let input = Value::Object(
        [("question".to_owned(), Value::String(q.to_owned())), ("form".to_owned(), Value::String("cited".to_owned())), ("tokens".to_owned(), tokens)].into_iter().collect(),
    );
    let t = Instant::now();
    if let Err(e) = nmapi::tools::api().dispatch("nm_context_pack", &input) {
        eprintln!("writer-idle: call refused: {}: {}", e.code, e.message);
    }
    let _ = nmapi::tools::take_diagnostics();
    t.elapsed().as_secs_f64()
}

fn summary(label: &str, secs: &[f64]) -> String {
    let mut s = secs.to_vec();
    s.sort_by(f64::total_cmp);
    let med = s.get(s.len() / 2).copied().unwrap_or(0.0);
    let worst = s.last().copied().unwrap_or(0.0);
    format!("{label}: {} calls, median {med:.2} s, worst {worst:.2} s", s.len())
}

fn mib(b: usize) -> usize {
    b >> 20
}

fn stats_line(st: &nmapi::writer_idle::Stats) -> String {
    let mut n = st.note_ms.clone();
    n.sort_by(f64::total_cmp);
    format!(
        "load {:.1} s; notes {} (median {:.2} s, worst {:.2} s); release {:.0} ms; restore {:.0} ms; card peak {} of {} MiB; finders before {:?} MiB, after {:?} MiB; preempted {}; embedder check {} checked, {} dropped; summaries {}, file notes {}; faults {:?}",
        st.load_ms / 1000.0,
        n.len(),
        n.get(n.len() / 2).copied().unwrap_or(0.0) / 1000.0,
        n.last().copied().unwrap_or(0.0) / 1000.0,
        st.release_ms,
        st.restore_ms,
        mib(st.card_peak),
        mib(st.card_total),
        st.before.iter().map(|(k, b)| (k.clone(), mib(*b))).collect::<Vec<_>>(),
        st.after.iter().map(|(k, b)| (k.clone(), mib(*b))).collect::<Vec<_>>(),
        st.preempted,
        st.checked,
        st.dropped,
        st.summaries,
        st.file_notes,
        st.faults
    )
}

pub fn run(home: &Path, args: &[String]) -> ExitCode {
    let code = match trial(home, args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("writer-idle: {e}");
            1
        }
    };
    super::bench::shutdown(code)
}

fn trial(home: &Path, args: &[String]) -> Result<(), String> {
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1).cloned());
    let root = std::path::PathBuf::from(opt("--root").ok_or("--root")?);
    let qs = questions(Path::new(&opt("--questions").ok_or("--questions")?))?;
    let n_notes: usize = opt("--notes").and_then(|n| n.parse().ok()).unwrap_or(8);
    let serve_args: Vec<String> = args.iter().skip_while(|a| *a != "--").skip(1).cloned().collect();
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, &root));
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    super::configure(&org, &serve_args)?;
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    let pieces = nmapi::tools::installed().map_or(0, |o| o.sub().chunks.len());
    println!("writer-idle: {} questions, {pieces} pieces, notes per session {n_notes}", qs.len());
    // Stage 5 WP-C4: `--prefill <n>`: every question asked once (its pieces become served), then
    // sessions until `n` results are written or nothing is pending; the results go to the note store
    // (`NANO_WRITER_CACHE=on`, `NANO_WRITER_DIR`). Nothing else is measured.
    if let Some(target) = opt("--prefill").and_then(|n| n.parse::<usize>().ok()) {
        match nmapi::writer_idle::open_store() {
            Some((file, n)) => println!("prefill: note store {} ({n} results read)", file.display()),
            None => return Err("--prefill needs a note store: NANO_WRITER_CACHE=on and NANO_WRITER_DIR".to_owned()),
        }
        for q in &qs {
            let _ = ask(q);
        }
        let t = Instant::now();
        let mut total = 0;
        while total < target {
            let n = nmapi::writer_idle::session(home, (target - total).min(256)).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            total += n;
            println!("prefill: {total} results in {:.0} s; {}", t.elapsed().as_secs_f64(), stats_line(&nmapi::writer_idle::stats()));
        }
        let st = nmapi::writer_idle::stats();
        println!("prefill done: {total} results; notes {}, likely names {}, module notes {}, file notes {}, checked {}, dropped {}, kept resident {}, card peak {} MiB", st.notes, st.likely_names, st.module_notes, st.file_notes, st.checked, st.dropped, st.kept_resident, mib(st.card_peak));
        return Ok(());
    }
    // Warm pass, then the baseline pass.
    let _: Vec<f64> = qs.iter().map(|q| ask(q)).collect();
    let base: Vec<f64> = qs.iter().map(|q| ask(q)).collect();
    println!("{}", summary("calls before the writer", &base));
    // One session of n notes.
    let t = Instant::now();
    let n = nmapi::writer_idle::session(home, n_notes).map_err(|e| e.to_string())?;
    println!("session: {n} notes in {:.1} s; {}", t.elapsed().as_secs_f64(), stats_line(&nmapi::writer_idle::stats()));
    let after: Vec<f64> = qs.iter().map(|q| ask(q)).collect();
    println!("{}", summary("calls after the writer let go", &after));
    // A call that arrives mid-write.
    let held = nmapi::writer_idle::notes_held();
    let home2 = home.to_path_buf();
    let writer = std::thread::spawn(move || {
        let t = Instant::now();
        let r = nmapi::writer_idle::session(&home2, usize::MAX);
        (r.map_err(|e| e.to_string()), t)
    });
    let waited = Instant::now();
    while nmapi::writer_idle::notes_held() < held + 2 && !writer.is_finished() {
        std::thread::park_timeout(std::time::Duration::from_millis(50));
    }
    if writer.is_finished() {
        let (r, _) = writer.join().map_err(|_| "writer thread")?;
        println!("preempt: the session ended before a call could arrive ({:.1} s): {r:?}", waited.elapsed().as_secs_f64());
    } else {
        let q = qs.first().cloned().unwrap_or_default();
        let t_call = Instant::now();
        let secs = ask(&q);
        let (r, _) = writer.join().map_err(|_| "writer thread")?;
        let gone = t_call.elapsed().as_secs_f64();
        println!(
            "preempt: call mid-write {secs:.2} s (baseline for that question {:.2} s); writer let go and finders restored {gone:.2} s after the call began; {:?} notes; {}",
            base.first().copied().unwrap_or(0.0),
            r,
            stats_line(&nmapi::writer_idle::stats())
        );
    }
    let last: Vec<f64> = qs.iter().map(|q| ask(q)).collect();
    println!("{}", summary("calls after the preemption", &last));
    let sample: Vec<String> = nmapi::tools::installed()
        .map(|o| o.sub().chunks.iter().filter_map(|c| nmapi::writer_idle::note(&c.key).map(|n| format!("{}: {n}", c.start))).take(8).collect())
        .unwrap_or_default();
    println!("notes held {}; first notes: {sample:?}", nmapi::writer_idle::notes_held());
    Ok(())
}

/// `adapter-fit --root <repo> [--pairs N] [--judge-pairs N] [--no-lens] [--no-judge] [--write]
/// [--force] [--wait S] [-- <serve options>]` (lane B, 2026-11-19): the project's adapters fitted
/// (`nmapi::adapter_fit::fit`) on an organism configured exactly as `serve` configures it (so the
/// embedder, its precision and base cartridge, and the stored vectors' header are the served ones),
/// once its meaning lane is full. Report on stdout and in `runs/ADAPTER-fit-<ms>.txt`.
pub fn fit_run(home: &Path, args: &[String]) -> ExitCode {
    let code = match fit_trial(home, args) {
        Ok(t) => {
            let p = home.join(format!("runs/ADAPTER-fit-{}.txt", nmtype::now_ms()));
            let _ = std::fs::write(&p, &t);
            print!("{t}");
            println!("report: {}", p.display());
            0
        }
        Err(e) => {
            eprintln!("adapter-fit: {e}");
            1
        }
    };
    super::bench::shutdown(code)
}

fn fit_trial(home: &Path, args: &[String]) -> Result<String, String> {
    let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1).cloned());
    let flag = |k: &str| args.iter().take_while(|a| *a != "--").any(|a| a == k);
    let root = std::path::PathBuf::from(opt("--root").ok_or("--root")?);
    let serve_args: Vec<String> = args.iter().skip_while(|a| *a != "--").skip(1).cloned().collect();
    let memory = home.join("scratch").join("bench-memory").join(nmtype::project::label(home, &root));
    let org = nmapi::forage::Organism::open_with(home, &root, &["neuromorph-nano".to_owned()], false, &memory).map_err(|e| format!("organism: {e}"))?;
    super::configure(&org, &serve_args)?;
    nmapi::tools::install(org).map_err(|e| format!("{}: {}", e.code, e.message))?;
    let o = nmapi::tools::installed().ok_or("not installed")?;
    let wait: u64 = opt("--wait").and_then(|v| v.parse().ok()).unwrap_or(3600);
    let t = Instant::now();
    loop {
        let sub = o.sub();
        if let Some(lane) = sub.dense.read().ok().and_then(|d| d.clone()) {
            if lane.ready(&sub) {
                break;
            }
            if t.elapsed().as_secs() > wait {
                let (c, n) = lane.coverage(&sub);
                return Err(format!("the meaning lane is not full after {wait} s ({c} of {n})"));
            }
        } else if t.elapsed().as_secs() > wait {
            return Err(format!("no meaning lane after {wait} s (serve options {serve_args:?})"));
        }
        std::thread::park_timeout(std::time::Duration::from_secs(2));
    }
    let d = nmapi::adapter_fit::Opts::default();
    let opts = nmapi::adapter_fit::Opts {
        pairs: opt("--pairs").and_then(|v| v.parse().ok()).unwrap_or(d.pairs),
        judge_pairs: opt("--judge-pairs").and_then(|v| v.parse().ok()).unwrap_or(d.judge_pairs),
        lens: !flag("--no-lens"),
        judge: !flag("--no-judge"),
        write: flag("--write"),
        force: flag("--force"),
        ..d
    };
    let mut out = format!("adapter-fit {}\nmeaning lane full after {:.0} s\n", args.join(" "), t.elapsed().as_secs_f64());
    out.push_str(&nmapi::adapter_fit::fit(o, &opts, &|| false).map_err(|e| e.to_string())?);
    Ok(out)
}
