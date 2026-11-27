//! The Neuromorph Nano lab: research harnesses, stage runs, generators of design data, developer
//! checks, and the builders of parameter maps, base cartridges and job filters. Every command
//! writes its report under `runs/`.
//!
//! 2026-11-26 (Luis Jake Gabriel III: "Nano lab should be a part of Neuromorph nano"; "Don't remove or
//! confuse anything just make it all work right together as one product"): a library, so the
//! product carries it inside one program (`neuromorph-nano lab <command>`), and the `nano-lab`
//! program keeps running the same commands as before. Harnesses whose data sets were not copied into the Nano
//! report the missing file when run.
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

/// `--keep` on a command that builds a base cartridge (2026-11-26): the new version is added beside
/// the versions on the shelf, nothing removed and no stack pointed at it, as a build started from
/// the interface, MCP or the API asks (`library.upgrade` keep). Without it the lab replaces the
/// base cartridge in place, as it always has.
pub(crate) static KEEP_BESIDE: AtomicBool = AtomicBool::new(false);
/// Whether the run applies parameter overlays, as its reports record it
/// (`nmengn::overlay::report_line`), taken once as the run starts.
static OVERLAY_LINE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The line a lab report records about parameter overlays: `overlay: applied (<n> objects)`,
/// `overlay: not applied (--no-overlay)` or `overlay: none present`.
#[must_use]
pub fn overlay_line() -> String {
    OVERLAY_LINE
        .get()
        .cloned()
        .unwrap_or_else(|| nmengn::overlay::report_line(&nano_root(), nmengn::overlay::AT_OPEN.load(Ordering::Relaxed)))
}

/// `text` as a report file holds it: the overlay line (`overlay_line`) first.
#[must_use]
pub fn with_overlay(text: &str) -> String {
    format!("{}\n{text}", overlay_line())
}

mod adapt;
mod arb;
mod bcart;
mod cart;
mod distil;
mod dx;
mod e2;
mod e3;
mod e3j;
mod ecart;
mod emap;
mod g1;
mod kept;
mod keys;
mod p1;
mod pmap_cli;
mod q8;
mod s10;
mod s11;
mod s6;
mod s7;
mod s8;
mod s9;
mod soak;
mod stage_bench;
mod surgery;
mod tc1;
mod x1;

/// The anchor of the earlier tentacle pass on the lab's own tree (E3.3), kept byte-identical so
/// the lab harnesses reproduce earlier runs. The served pass uses `nmapi::tentacle_pass::SELF_ANCHOR`.
pub const MINI_ANCHOR: &str = "Repository neuromorph-mini: a Rust workspace of 18 crates (nmsubs substrate, nmpod tentacles, nmbrain Core Brain, nmengn engine, nmapi tools).\n";

/// The Nano folder, found from the running program (`nmtype::nano_root`), canonical.
fn nano_root() -> PathBuf {
    nmsubs::repo::canon(&nmtype::nano_root())
}

const USAGE: &str = "The Neuromorph Nano lab (parameter maps, base cartridges, job filters, research harnesses, stage runs, generators, developer checks)

usage: neuromorph-nano lab <command> [args]   (the same commands: nano-lab <command> [args])

developer checks and generators:
  conform                         conformance rules over the crates and designs
  scoreboard                      worst-first part table
  anatomy tentacles [design]      tentacle and drone bodies, then the design table
  compose <kind>                  write a composition assembly
  run tentacle assemblies         write designs/assemblies/tentacle-t<n>.json
  atlas compare [--base <id>] [--with <a,b>]
                                  parameter atlas comparison

stage runs:
  run s2 --root <repo>            three questions through the organism on the NTE
  run s5 --root <repo>            live anatomy change while tasks run, fault drill
  run s6 <part> [cpu|gpu|both]    assemblies, parity, split, place, record, roles, roles-eval,
                                  roles-record, roles-parity, writer-parity, kv, fingerprint, blocks
  run s7 <part>                   decode [cpu|gpu|both], writer <id>, router
  run s8 <part> ...               learning-unit isolation, overlap, fit, check, rollback
  run s9 <part> ...               breakage preflight and build-help items
  run s10 delta ...               scratch tree kept in step by the poll thread
  run s11 <part> ...              outcomes, features, readout learner

distillation (learned adapter cartridges from curated teacher pairs):
  run distil build <role> [--set <name>] [--technique lens|teacher-loop] [--base <file>|--open] [--rounds <n>]
                                  fit, tune and gate on the held-out split; writes <role>-distil-<set>
  run distil teach --prompts <file> --out <file> [--teacher <model>] [--instruction <text>]
                                  a local open-weights teacher writes teacher pairs (JSON lines)

research harnesses (most need the CX1 question set in tests/capability-gates/cx1-context/):
  run e2 ... | run e3 ... | run g1 ... | run kept [layout] | run x1 | run p1 s1|s2|s3
  run tc1 [n] | run tc1 dump [n] | run tc1 replay [folder] | run keys [dump folder] [n]
  run dx <base|d1..d9|t1a|t1|t1b|t4|t5>... | run arb ... | run cart <step> ... | run bcart <step> ...
  run msg route | run musc drift [seeds]

  --help, --version";

/// Runs one lab command; `args` are the words after the program name (`nano-lab <args>`) or after
/// `neuromorph-nano lab`.
#[must_use]
pub fn run(args: Vec<String>) -> ExitCode {
    KEEP_BESIDE.store(args.iter().any(|a| a == "--keep"), Ordering::Relaxed);
    // Parameter overlays (`memory/overlays/`): lab measurements and builds open every population
    // with them applied, unless `--no-overlay` asks for the released files alone. The run says which.
    let no_overlay = args.iter().any(|a| a == "--no-overlay");
    let args: Vec<String> = args.into_iter().filter(|a| a != "--no-overlay").collect();
    // The Nano folder must be found before any command reads from it.
    let info = matches!(args.first().map(String::as_str), Some("--help" | "-h" | "help" | "--version" | "-V"));
    if !info && let Err(e) = nmtype::find_root() {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    if !info {
        nmengn::overlay::AT_OPEN.store(!no_overlay, Ordering::Relaxed);
        eprintln!("lab: {}", nmengn::overlay::lab_line(&nmtype::nano_root(), !no_overlay));
        let _ = OVERLAY_LINE.set(nmengn::overlay::report_line(&nmtype::nano_root(), !no_overlay));
    }
    // Precision and kernel switches apply to every command, before anything binds.
    if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--precision")).and_then(|w| w.get(1)) {
        match nmengn::model::Precision::parse(v) {
            Ok(p) => {
                if let Ok(mut g) = nmengn::model::PRECISION.write() {
                    *g = p;
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if let Some(v) = args.windows(2).find(|w| w.first().map(String::as_str) == Some("--precision-map")).and_then(|w| w.get(1)) {
        match nmengn::model::read_precision_map(Path::new(v)) {
            Ok(m) => {
                if let Ok(mut g) = nmengn::model::PRECISION_MAP.write() {
                    *g = Some(m);
                }
            }
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    if args.iter().any(|a| a == "--half-math") {
        nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if args.iter().any(|a| a == "--dp4a") {
        nmengn::gpu::DP4A.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let sub = args.get(1).map(String::as_str);
    match args.first().map(String::as_str) {
        Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("--version" | "-V") => {
            println!("nano-lab {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("conform") => nmtool::conformance::cli(&nano_root()),
        // 2026-11-10 (tentacle core A10, first form): which components each tentacle and drone holds.
        // 2026-11-10 (tentacle bodies): the bound bodies (modules and clusters each tentacle and its
        // drone tip hold), then the design table; `anatomy tentacles design` prints the design only.
        Some("anatomy") if args.get(1).map(String::as_str) == Some("tentacles") => {
            match nmapi::tentacle_core::anatomy_table(&nano_root()) {
                Ok(t) => {
                    if args.get(2).map(String::as_str) != Some("design") {
                        match nmapi::tentacle_body::anatomy_table(&nano_root()) {
                            Ok(b) => println!(
                                "{b}
"
                            ),
                            Err(e) => println!(
                                "tentacle bodies not bound: {}
",
                                e.message
                            ),
                        }
                    }
                    println!("{t}");
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("anatomy tentacles: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("scoreboard") => nmtool::scoreboard::cli(&nano_root()),
        // 2026-11-13: the internal communication route measure (`run msg route`), processor.
        Some("run") if args.get(1).map(String::as_str) == Some("msg") && args.get(2).map(String::as_str) == Some("route") => {
            let home = nano_root();
            let result = (|| -> Result<String, nmtype::Fault> {
                let mem = home.join(format!("tests/memory-system/msg-route-{}", nmtype::now_ms()));
                let org = nmapi::forage::Organism::open_with(&home, &home, &[], false, &mem)?;
                let sub = org.sub();
                let mut keep = nmapi::drone_windows::Keep::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files"))?;
                keep.sync(&sub)?;
                let tok = nmtokz::Tokenizer::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files/tokenizer.json"))
                    .map_err(|e| nmtype::Fault::new("msg", nmtype::FaultKind::Config, e.to_string()))?;
                let mut qs = Vec::new();
                for area in ["context", "engine", "tools"] {
                    let file = home.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"));
                    let v = nmjson::parse(&std::fs::read(&file).map_err(|e| {
                        nmtype::Fault::new("msg", nmtype::FaultKind::Config, format!("needs the CX1 question set: {}: {e}", file.display()))
                    })?)
                    .map_err(|e| nmtype::Fault::new("msg", nmtype::FaultKind::Config, e.to_string()))?;
                    for x in v
                        .get("questions")
                        .and_then(|x| x.array())
                        .map_err(|e| nmtype::Fault::new("msg", nmtype::FaultKind::Config, e.to_string()))?
                    {
                        qs.push(x.get("q").and_then(|x| x.string()).unwrap_or("").to_owned());
                    }
                }
                nmapi::whole::route_report(&sub, &keep, &tok, &qs)
            })();
            match result {
                Ok(text) => {
                    let p = home.join(format!("runs/MSG-route-{}.txt", nmtype::now_ms()));
                    let _ = std::fs::write(&p, with_overlay(&text));
                    print!("{text}");
                    println!("report: {}", p.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("msg route: {}", e.message);
                    ExitCode::FAILURE
                }
            }
        }
        // 2026-11-13: the G3 drift test (`run musc drift [seeds]`), processor, seconds.
        Some("run") if args.get(1).map(String::as_str) == Some("musc") && args.get(2).map(String::as_str) == Some("drift") => {
            let seeds = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
            let text = nmmusc::measure::drift_report(seeds);
            let p = nano_root().join(format!("runs/MUSC-drift-{}.txt", nmtype::now_ms()));
            let _ = std::fs::write(&p, with_overlay(&text));
            print!("{text}");
            println!("report: {}", p.display());
            ExitCode::SUCCESS
        }
        // 2026-11-10: generate every tentacle's own assembly (designs/assemblies/tentacle-t<n>.json).
        Some("run") if args.get(1).map(String::as_str) == Some("tentacle") && args.get(2).map(String::as_str) == Some("assemblies") => {
            match nmapi::tentacle_body::write(&nano_root()) {
                Ok(v) => {
                    for (p, n) in v {
                        println!("written {p} ({n} components)");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("tentacle assemblies: {}", e.message);
                    ExitCode::FAILURE
                }
            }
        }
        // 2026-11-11: atlas step 0, the comparative map (`atlas compare [--base id] [--with a,b]`).
        Some("atlas") if args.get(1).map(String::as_str) == Some("compare") => {
            let opt = |k: &str| args.windows(2).find(|w| w.first().map(String::as_str) == Some(k)).and_then(|w| w.get(1)).cloned();
            let base = opt("--base").unwrap_or_else(|| "qwen3-0.6b".into());
            let with: Vec<String> =
                opt("--with").unwrap_or_else(|| "qwen3-reranker-0.6b,qwen3-embedding-0.6b".into()).split(',').map(str::to_owned).collect();
            let date = nmlearn::cartridge::date_of(nmtype::now_ms());
            match nmnpcs::compare::run(&nano_root(), &base, &with, nmtype::now_ms(), &date) {
                Ok((atlas, report, text)) => {
                    println!("{text}\nwritten {}\nwritten {}", atlas.display(), report.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("atlas compare: {}", e.message);
                    ExitCode::FAILURE
                }
            }
        }
        Some("compose") => match nmapi::anatomy::compose_file(&nano_root(), args.get(1).map_or("context", String::as_str)) {
            Ok(p) => {
                println!("composed {}", p.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("compose: {e}");
                ExitCode::FAILURE
            }
        },
        Some("stage-bench") => stage_bench::run(&args, &nano_root()),
        Some("run") => match sub {
            Some("s2") => run_s2(&args),
            Some("s5") => run_s5(&args),
            Some("s6") => s6::run(&args, &nano_root()),
            Some("s7") => s7::run(&args, &nano_root()),
            Some("s8") => s8::run(&args, &nano_root()),
            Some("s9") => s9::run(&args, &nano_root()),
            Some("s10") => s10::run(&args, &nano_root()),
            Some("s11") => s11::run(&args, &nano_root()),
            Some("e2") => e2::run(&args, &nano_root()),
            Some("e3") => e3::run(&args, &nano_root()),
            Some("g1") => g1::run(&args, &nano_root()),
            Some("kept") => kept::run(&args, &nano_root()),
            Some("x1") => x1::run(&args, &nano_root()),
            Some("p1") => p1::run(&args, &nano_root()),
            Some("tc1") => tc1::run(&args, &nano_root()),
            Some("keys") => keys::run(&args, &nano_root()),
            Some("dx") => dx::run(&args, &nano_root()),
            Some("cart") => cart::run(&args, &nano_root()),
            Some("bcart") => bcart::run(&args, &nano_root()),
            Some("arb") => arb::run(&args, &nano_root()),
            Some("q8") => q8::run(&args, &nano_root()),
            Some("emap") => emap::run(&args, &nano_root()),
            Some("ecart") => ecart::run(&args, &nano_root()),
            Some("adapter") => adapt::run(&args, &nano_root()),
            Some("distil") => distil::run(&args, &nano_root()),
            Some("pmap") => pmap_cli::run(&args, &nano_root()),
            Some("soak") => soak::run(&args, &nano_root()),
            _ => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

/// `nano-lab run s5 --root <repo>`: the anatomy changes while tasks run (add a
/// tentacle, rewire, swap, a refused bypass, remove), a fault drill with quarantine and
/// restart, composition per objective kind, the Parameter Map count. Writes a new
/// `runs/S5-run-<ms>.txt`.
#[allow(clippy::too_many_lines, reason = "one scenario, read top to bottom")]
fn run_s5(args: &[String]) -> ExitCode {
    use std::fmt::Write as _;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    let Some(root) = args.iter().position(|a| a == "--root").and_then(|i| args.get(i + 1)) else {
        eprintln!("usage: nano-lab run s5 --root <repository to study>");
        return ExitCode::FAILURE;
    };
    let stamp = nmtype::now_ms();
    let mem = nmtype::tests_dir("stage-gates/s5").join(format!("s5-{stamp}"));
    // One worker makes each task's frame long enough (its 26 nodes run in turn) for every
    // change below to land while a frame is running; the result says so per change.
    let org = match nmapi::forage::Organism::open_with(&nano_root(), Path::new(root), &["neuromorph-nano".to_owned()], true, &mem) {
        Ok(mut o) => {
            o.workers = 1;
            Arc::new(o)
        }
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Ok(mut p) = org.parmap.lock() {
        p.recording = true;
    }
    let mut out = String::new();
    let snap = org.nte.snapshot();
    let tentacles = |s: &nmnte::Snapshot| s.nodes().iter().filter(|n| n.kind == nmtype::ComponentType::Tentacle).count();
    let _ = writeln!(out, "S5 run: live anatomy change on the composed assembly (substrate digest {})", org.sub().digest());
    let _ = writeln!(
        out,
        "start: v{} with {} tentacles, {} nodes, {} edges, waves {:?}",
        snap.version,
        tentacles(&snap),
        snap.nodes().len(),
        snap.edges().len(),
        snap.plan.waves.iter().map(Vec::len).collect::<Vec<_>>()
    );
    // Hold no snapshot while changes run, or the Drain phase would wait for this reference.
    drop(snap);
    // Background: questions keep running on whatever snapshot is current when each starts.
    let done = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    // (ran on version, version at end, faults, withheld, faulting nodes) per task.
    type TaskRow = (u64, u64, usize, bool, String);
    let tasks: Arc<Mutex<Vec<TaskRow>>> = Arc::new(Mutex::new(Vec::new()));
    let worker = {
        let (org, done, stop, tasks) = (Arc::clone(&org), Arc::clone(&done), Arc::clone(&stop), Arc::clone(&tasks));
        std::thread::spawn(move || {
            let qs = [
                "Where is the struct Coordinator defined?",
                "What value is the constant MAX_FINDINGS set to?",
                "What value is the constant DRONE_FANOUT_CAP set to?",
            ];
            let mut i = 0;
            while !stop.load(Ordering::SeqCst) {
                let q = qs.get(i % 3).copied().unwrap_or_default();
                if let Ok(a) = org.ask(q, 2000) {
                    let faults = a.faults.iter().map(|f| f.component.clone()).collect::<Vec<_>>().join(",");
                    if let Ok(mut t) = tasks.lock() {
                        t.push((a.ran_version, a.version, a.faults.len(), a.pack.withheld, faults));
                    }
                }
                done.fetch_add(1, Ordering::SeqCst);
                i += 1;
            }
        })
    };
    // Wait for n more tasks, then until a task's frame is running on the current snapshot (it
    // holds a reference besides the executor's), so the next change lands mid-task.
    let wait = |n: usize| {
        let target = done.load(Ordering::SeqCst) + n;
        let limit = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while done.load(Ordering::SeqCst) < target && std::time::Instant::now() < limit {
            std::thread::yield_now();
        }
        while std::time::Instant::now() < limit {
            let s = org.nte.snapshot();
            let running = Arc::strong_count(&s) >= 3;
            drop(s);
            if running {
                break;
            }
            std::thread::yield_now();
        }
    };
    let mut changes = 0;
    let mut log_change = |out: &mut String, title: &str, r: Result<nmnte::PhaseLog, (nmtype::Fault, nmnte::PhaseLog)>| {
        changes += 1;
        let (ok, log) = match r {
            Ok(l) => (format!("committed v{}", l.committed.unwrap_or(0)), l),
            Err((f, l)) => (format!("refused: {f}"), l),
        };
        let _ = writeln!(out, "\n{title}: {ok}");
        for line in log.lines() {
            let _ = writeln!(out, "  {line}");
        }
    };
    wait(3);
    match nmapi::anatomy::tentacle_change(&org, 8) {
        Ok(c) => log_change(&mut out, "ADD tentacle 8 (six nodes from the template)", nmapi::anatomy::change(&org, &c, "s5")),
        Err(e) => {
            let _ = writeln!(out, "ADD: {e}");
        }
    }
    wait(2);
    let note_edge =
        |b: usize| nmnte::Edge::new("nmpod.pipeline#t1", "packets", "nmmem.working#t1", "packets", nmnte::ConnectionMode::Buffered(b));
    log_change(
        &mut out,
        "REWIRE t1 pipeline to note, buffer 8 to 16",
        nmapi::anatomy::change(&org, &nmnte::Change::Rewire { add: vec![note_edge(16)], remove: vec![note_edge(8)] }, "s5"),
    );
    wait(2);
    let d0 = org.nte.snapshot().graph.node("nmpod.unit#d0").cloned();
    if let Some(d0) = d0 {
        log_change(&mut out, "SWAP drone tip d0 for a fresh instance", nmapi::anatomy::change(&org, &nmnte::Change::Swap(d0), "s5"));
    }
    wait(2);
    let bypass = nmnte::Edge::new("nmpod.pipeline#t0-emit", "packets", "nmmem.working#t1", "packets", nmnte::ConnectionMode::Direct);
    log_change(
        &mut out,
        "REWIRE refused: t0 emit straight into t1 note (a tentacle-to-tentacle link)",
        nmapi::anatomy::change(&org, &nmnte::Change::Rewire { add: vec![bypass], remove: vec![] }, "s5"),
    );
    wait(2);
    log_change(
        &mut out,
        "REMOVE tentacle 8 with its five nodes",
        nmapi::anatomy::change(&org, &nmnte::Change::Remove("nmpod.unit#t8".into()), "s5"),
    );
    wait(2);
    // Fault drill on d2: inject, let homeostasis quarantine it, then restart it.
    let d2 = org.nte.snapshot().graph.node("nmpod.unit#d2").cloned();
    if let Some(mut d2) = d2 {
        if let nmjson::Value::Object(cfg) = &mut d2.config {
            cfg.insert("inject".into(), nmjson::Value::String("fault".into()));
        }
        log_change(
            &mut out,
            "FAULT DRILL: d2 bound to a component that faults",
            nmapi::anatomy::change(&org, &nmnte::Change::Swap(d2), "s5"),
        );
    }
    let limit = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !org.quarantined.lock().is_ok_and(|q| q.contains("nmpod.unit#d2")) && std::time::Instant::now() < limit {
        std::thread::yield_now();
    }
    let quarantined = org.quarantined.lock().is_ok_and(|q| q.contains("nmpod.unit#d2"));
    let _ = writeln!(
        out,
        "\nhomeostasis quarantined d2 after its fault streak: {quarantined}; actions so far {:?}",
        org.homeostasis.lock().map(|h| h.actions.clone()).unwrap_or_default()
    );
    wait(2);
    match nmapi::anatomy::restart(&org, "nmpod.unit#d2") {
        Ok(l) => log_change(&mut out, "RESTART d2 from its description", Ok(l)),
        Err(e) => {
            let _ = writeln!(out, "RESTART d2: {e}");
        }
    }
    let restarted_at = org.nte.snapshot().version;
    wait(3);
    stop.store(true, Ordering::SeqCst);
    let _ = worker.join();
    let t = tasks.lock().map(|t| t.clone()).unwrap_or_default();
    let spanning = t.iter().filter(|x| x.0 < x.1).count();
    let faulted: Vec<String> = t.iter().filter(|x| x.2 > 0).map(|x| format!("v{}:{}", x.0, x.4)).collect();
    let after_restart_faults = t.iter().filter(|x| x.0 >= restarted_at && x.2 > 0).count();
    let snap = org.nte.snapshot();
    let _ = writeln!(
        out,
        "\ntasks: {} completed while the anatomy changed; {spanning} of them ran on a snapshot older than the one current when they ended (finished on their own version); tasks with faults: {faulted:?}; faults after the restart: {after_restart_faults}",
        t.len()
    );
    let _ = writeln!(
        out,
        "end: v{} with {} tentacles, {} nodes; quarantined now {:?}",
        snap.version,
        tentacles(&snap),
        snap.nodes().len(),
        org.quarantined.lock().map(|q| q.clone()).unwrap_or_default()
    );
    for objective in [
        "Where is the struct Coordinator defined?",
        "Fix the build error in emma-backnd",
        "Verify that the drone crates still pass their tests",
        "Rewire tentacle 3 to a larger buffer",
    ] {
        if let Ok(r) = nmapi::anatomy::composition_for(&org, objective) {
            let _ = writeln!(
                out,
                "composition for {objective:?}: row {} ({} tentacles, slots {:?}, views {:?}, budget {} tokens)",
                r.objective_kind, r.tentacles, r.slots, r.views, r.budget.tokens
            );
        }
    }
    let (seen, written) = org.parmap.lock().map(|p| (p.seen, p.written)).unwrap_or((0, 0));
    let _ = writeln!(
        out,
        "Parameter Map: {changes} changes reported here, {seen} observed, {written} written to {}",
        mem.join("parmap/edits.jsonl").display()
    );
    print!("{out}");
    let path = nano_root().join(format!("runs/S5-run-{stamp}.txt"));
    if path.exists() || std::fs::write(&path, with_overlay(&out)).is_err() {
        eprintln!("report not written");
        return ExitCode::FAILURE;
    }
    println!("report written to {}", path.display());
    ExitCode::SUCCESS
}

/// `nano-lab run s2 --root <repo>`: three questions through the organism on the NTE
/// (a cited answer, a name with two values, a decoy), the anatomy printed, the report text
/// written to a new `runs/S2-run-<epoch ms>.txt` (never overwriting an earlier run). The repository is an argument, never a path in
/// the code.
fn run_s2(args: &[String]) -> ExitCode {
    use std::fmt::Write as _;
    let Some(root) = args.iter().position(|a| a == "--root").and_then(|i| args.get(i + 1)) else {
        eprintln!("usage: nano-lab run s2 --root <repository to study>");
        return ExitCode::FAILURE;
    };
    let mut out = String::new();
    let t0 = std::time::Instant::now();
    let org = match nmapi::forage::Organism::open(&nano_root(), Path::new(root), &["neuromorph-nano".to_owned()]) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };
    let open_ms = t0.elapsed().as_millis();
    let snap = org.nte.snapshot();
    let count = |k: nmtype::ComponentType| snap.nodes().iter().filter(|n| n.kind == k).count();
    let lateral = snap.edges().iter().filter(|e| nmnte::topology::bypasses_brain(&snap.graph, &e.from, &e.to)).count();
    let _ = writeln!(out, "S2 run, organism loop on the NTE, exact controls only (no model loaded)");
    let sub = org.sub();
    let _ = writeln!(
        out,
        "substrate: {} files, {} symbols, {} crates, version {} (scan {open_ms} ms)",
        sub.files.len(),
        sub.symbols.len(),
        sub.crates.len(),
        sub.version
    );
    let _ = writeln!(
        out,
        "anatomy v{}: {} nodes ({} tentacles, {} drone tips, {} brain, {} other), {} edges, {} edges bypassing the Core Brain, waves {:?}",
        snap.version,
        snap.nodes().len(),
        count(nmtype::ComponentType::Tentacle),
        count(nmtype::ComponentType::Drone),
        count(nmtype::ComponentType::Brain),
        count(nmtype::ComponentType::Other),
        snap.edges().len(),
        lateral,
        snap.plan.waves.iter().map(Vec::len).collect::<Vec<_>>()
    );
    let unit = nmpod::Unit::new("nmpod.unit#t0", nmpod::Level::Tentacle, nmpod::Job::Retrieval, std::sync::Arc::clone(&org.reflexes), 1);
    let fields: Vec<String> = unit.to_value().object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    let _ = writeln!(out, "unit fields (no handle to another unit): {fields:?}");
    let budget = nmtype::Budget { tokens: 4000, ms: 50_000, depth: 1, units: 16 };
    let questions = [
        "Where is the struct Coordinator defined?",
        "What value is the constant MAX_FINDINGS set to?",
        "What value is the constant DRONE_FANOUT_CAP set to?",
    ];
    for q in questions {
        let t = std::time::Instant::now();
        let a = match org.ask(q, budget.tokens) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ask: {e}");
                return ExitCode::FAILURE;
            }
        };
        let _ = writeln!(
            out,
            "\nQ: {q}\n  time {} ms; {} raw packets from the tentacles; nodes run {}; skipped {}; faults {}",
            t.elapsed().as_millis(),
            a.raw_packets,
            a.ran.len(),
            a.skipped.len(),
            a.faults.len()
        );
        for f in &a.faults {
            let _ = writeln!(out, "  fault: {f}");
        }
        let _ = writeln!(out, "  pack: withheld={} tokens={} reason: {}", a.pack.withheld, a.pack.tokens, a.pack.reason);
        for p in &a.pack.packets {
            let _ = writeln!(
                out,
                "  [{}] {} | {} | source {} | evidence {:?}",
                p.get_status().name(),
                p.key(),
                p.text(),
                p.provenance().source,
                p.get_evidence()
            );
        }
        for qp in a.questions.iter().filter(|x| x.key().starts_with("conflict:")) {
            let _ = writeln!(out, "  question to resolve: {}", qp.text());
        }
        let _ = writeln!(out, "  emitted [{}]: {}", a.emitted.get_status().name(), a.emitted.text().replace('\n', " / "));
    }
    let _ = writeln!(out, "\ntotal {} ms", t0.elapsed().as_millis());
    print!("{out}");
    // Run logs are never overwritten: each run writes a new file named by its start time.
    let path = nano_root().join(format!("runs/S2-run-{}.txt", nmtype::now_ms()));
    if path.exists() {
        eprintln!("report: {} already exists", path.display());
        return ExitCode::FAILURE;
    }
    println!("report written to {}", path.display());
    if let Err(e) = std::fs::write(&path, with_overlay(&out)) {
        eprintln!("report: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
