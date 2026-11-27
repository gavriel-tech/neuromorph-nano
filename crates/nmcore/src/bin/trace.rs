//! `neuromorph-nano trace --root <repo> [--memory-dir <category>/<name>] [--neural] [--tentacles] <question>...`
//! (2026-11-09): one question at a time through the live organism, without Claude Code, and
//! what every tentacle, drone tip and pipeline did: job, territory, readers, findings by
//! reader, packets handed to the Core Brain, and which of them reached the final pack.
//! Memory is off unless `--memory-dir` is given; `--refine off|uniform|nbv` sets the second
//! coordination round. Prints the GCX context map and the Map Coordinator's decision.
//! `--neural` (2026-11-10) binds the embedder and reranker and starts the dense lane, as
//! `serve --neural` does; `--tentacles` adds the tentacle pass (eight lens windows read by the
//! drone heads; `--tentacle-judge`, `--tentacle-pose`), and the trace prints each window.
//! 2026-11-10: `--tentacle-core` runs the tentacle core first form (G1 loop over the `g0_tx`
//! reports); `--brain-chains` runs the Core Brain side (G2, a MicroModel chain, G3 log, cartridges,
//! preferences) and the trace prints its behavioural steps after each answer.
use nmjson::Value;
use nmtype::Payload;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1))
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let Some(root) = arg(args, "--root") else {
        eprintln!(
            "usage: neuromorph-nano trace --root <repo> [--memory-dir <category>/<name>] [--refine off|uniform|nbv] [--neural] [--tentacles] [--tentacle-judge heads|reranker|rrf|sum|place] [--tentacle-pose plain|focus] [--tentacle-score mass|density|evidence|peak] [--tentacle-shortlist <n>] [--tentacle-core] [--brain-chains] <question>..."
        );
        return ExitCode::FAILURE;
    };
    let mem = arg(args, "--memory-dir");
    let flags: Vec<&String> = [
        "--root",
        "--memory-dir",
        "--refine",
        "--tentacle-judge",
        "--tentacle-pose",
        "--tentacle-score",
        "--tentacle-shortlist",
        "--dense",
        "--recall-meaning",
    ]
    .iter()
    .filter_map(|k| arg(args, k))
    .collect();
    let questions: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--") && !flags.contains(a)).collect();
    let dir = mem.map_or_else(|| nmtype::tests_dir("mcp-launcher/trace-memory-off"), |m| nmtype::tests_dir(m));
    let org = match nmapi::forage::Organism::open_with(home, Path::new(root), &[], mem.is_some(), &dir) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("open: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let (Some(m), Ok(mut r)) = (arg(args, "--refine"), org.refine.write()) {
        r.clone_from(m);
    }
    if args.iter().any(|a| a == "--neural") {
        // 2026-11-12: `--dense off` binds the engines without building the dense lane.
        let dense =
            !args.windows(2).any(|w| w.first().map(String::as_str) == Some("--dense") && w.get(1).map(String::as_str) == Some("off"));
        match nmapi::neural::bind(&org).and_then(|m| {
            eprintln!("neural: {m}");
            if dense { nmapi::neural::start_dense(&org) } else { Ok("dense lane off".to_owned()) }
        }) {
            Ok(m) => eprintln!("dense: {m}"),
            Err(e) => {
                eprintln!("neural: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    match nmapi::tentacle_pass::from_args(args) {
        Ok(Some(s)) => eprintln!("{}", nmapi::tentacle_pass::enable(&org, s)),
        Ok(None) => {}
        Err(e) => {
            eprintln!("trace: {e}");
            return ExitCode::FAILURE;
        }
    }
    if nmapi::brain_side::from_args(args) {
        match nmapi::brain_side::enable(&org) {
            Ok(m) => eprintln!("{m}"),
            Err(e) => {
                eprintln!("trace: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // `--whole` (2026-11-12): answer through the whole path, as `serve --whole` does.
    let whole = args.iter().any(|a| a == "--whole");
    if whole {
        if let Ok(mut w) = org.whole.write() {
            *w = Some(std::sync::Arc::new(nmapi::whole::WholePath::new(&org.home)));
        }
        eprintln!("whole path: on");
    }
    let mut status = ExitCode::SUCCESS;
    for q in questions {
        let a = match org.ask(q, 4000) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ask: {e}");
                status = ExitCode::FAILURE;
                continue;
            }
        };
        if whole {
            println!("== {q}");
            println!("whole path: {}", a.context_map.encode().unwrap_or_default());
            println!(
                "pack: {} packets, {} tokens, {}",
                a.pack.packets.len(),
                a.pack.tokens,
                a.pack.reason.chars().take(160).collect::<String>()
            );
            for p in a.pack.packets.iter().take(5) {
                println!("  {}", p.key());
            }
            println!("recalled {}, episode {:?}", a.recalled, a.episode);
            continue;
        }
        let frame = org.last_frame.lock().map(|f| f.clone()).unwrap_or_default();
        println!("== {q}");
        println!("answer: {}", a.emitted.text().lines().collect::<Vec<_>>().join(" | "));
        println!(
            "ran {} nodes, skipped {:?}, faults {:?}, raw packets {}, recalled {}",
            a.ran.len(),
            a.skipped,
            a.faults.iter().map(ToString::to_string).collect::<Vec<_>>(),
            a.raw_packets,
            a.recalled
        );
        let in_pack: Vec<String> = a.pack.packets.iter().map(|p| p.provenance().origin.clone()).collect();
        for i in 0..org.tentacles {
            let t = format!("nmpod.unit#t{i}");
            let assign = match frame.get(&t) {
                Some(Payload::Json(v)) if !matches!(v, Value::Null) => v.clone(),
                _ => {
                    println!("t{i}: no assignment");
                    continue;
                }
            };
            let job = assign.get("job").and_then(Value::string).unwrap_or("?").to_owned();
            let regions: Vec<String> = assign
                .get("regions")
                .and_then(Value::array)
                .map(|r| r.iter().filter_map(|x| x.get("path").and_then(Value::string).ok().map(str::to_owned)).collect())
                .unwrap_or_default();
            let mut by_reader: BTreeMap<String, usize> = BTreeMap::new();
            let mut readers = 0;
            if let Some(Payload::Json(v)) = frame.get(&format!("nmpod.unit#d{i}")) {
                readers = v.get("readers").and_then(Value::integer).unwrap_or(0);
                for f in v.get("findings").and_then(Value::array).map(<[Value]>::to_vec).unwrap_or_default() {
                    let r = f.get("reader").and_then(Value::string).unwrap_or("?");
                    let k = f.get("kind").and_then(Value::string).unwrap_or("?");
                    *by_reader.entry(format!("{r}/{k}")).or_default() += 1;
                }
            }
            let emitted = match frame.get(&format!("nmpod.pipeline#t{i}-emit")) {
                Some(Payload::Packets(v)) => v.len(),
                _ => 0,
            };
            let unit = format!("nmpod.unit#t{i}");
            let kept = in_pack.iter().filter(|o| **o == unit).count();
            println!(
                "t{i} {job:<12} readers {readers} findings {by_reader:?} -> {emitted} to brain, {kept} in pack | territory {}",
                regions.join(" ")
            );
        }
        let others: Vec<&String> = in_pack.iter().filter(|o| !o.starts_with("nmpod.unit#t")).collect();
        println!("pack: {} packets ({} from tentacles, others {:?})", a.pack.packets.len(), in_pack.len() - others.len(), others);
        let m = &a.context_map;
        let g = |k: &str| m.get(k).ok().and_then(|v| v.encode().ok()).unwrap_or_default();
        println!(
            "context map: segments {} scope_files {} coverage {} gates {} positions {} merged {} conflicts {} rounds {} refine {} refined {}",
            g("segments"),
            g("scope_files"),
            g("coverage"),
            g("gates"),
            g("positions"),
            g("merged"),
            g("conflicts"),
            g("rounds"),
            g("refine_mode"),
            g("refined")
        );
        println!("coordinator: {}", g("coordinator"));
        if let Some(Value::Array(ws)) = m.get("windows").ok().filter(|v| matches!(v, Value::Array(a) if !a.is_empty())) {
            println!(
                "tentacle windows (no chunk in two windows: crossover {}):",
                m.get("gates").ok().and_then(|x| x.get("crossover").ok()).and_then(|v| v.encode().ok()).unwrap_or_default()
            );
            for w in ws {
                let f = |k: &str| w.get(k).ok().and_then(|v| v.encode().ok()).unwrap_or_default();
                println!(
                    "  t{} {:<10} pose {} chunks {} tokens {} files {} level {} forwarded {} best {} no_information {}",
                    f("tentacle"),
                    f("lens").trim_matches('"'),
                    f("pose").trim_matches('"'),
                    f("chunks"),
                    f("tokens"),
                    f("files"),
                    f("level"),
                    f("findings"),
                    f("best"),
                    f("no_information")
                );
            }
            println!("{}", org.last_pass.lock().map(|n| n.clone()).unwrap_or_default());
        }
        if let (Some(s), Some(r)) = (nmapi::brain_side::side(&org), a.brain.as_ref()) {
            for l in s.trace_lines(&org.sub(), r) {
                println!("{l}");
            }
        }
    }
    status
}
