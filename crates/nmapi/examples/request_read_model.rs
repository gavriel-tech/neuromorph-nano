//! Lane R (2026-11-18): the request's drone in its model form on a question file, for test REQ-1.
//! usage: `request_read_model <home> <questions> <material dir> [--words | --embed]`
//! Questions: `id<TAB>question` or `id<TAB>kind<TAB>question` lines. With `--words` the reader is
//! not used (the finer word rules and the tags only). Prints each read, then the kind agreement
//! with the given kinds, trace read as trace, and the reader's milliseconds per request.
use nmpod::intake::{self, Bf16Table, Drone, Kind, Role, Source, SymbolIndex, Tokens};
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Chars;
impl Tokens for Chars {
    fn tokens(&self, text: &str) -> Result<Vec<i32>, nmtype::Fault> {
        Ok(text.bytes().map(|b| i32::from(b % 4)).collect())
    }
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| matches!(x, "rs" | "md" | "ts" | "tsx" | "js" | "py" | "toml" | "json")) {
            out.push(p);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (Some(home), Some(qfile), Some(mat)) = (a.get(1), a.get(2), a.get(3)) else {
        eprintln!("usage: request_read_model <home> <questions> <material dir> [--words]");
        return;
    };
    let words_only = a.iter().any(|x| x == "--words");
    let embed = a.iter().any(|x| x == "--embed");
    // The material's symbol index.
    let drone = Drone::new(Arc::new(Chars), Bf16Table::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![-1.0, 0.0]]));
    let mut files = Vec::new();
    walk(Path::new(mat), &mut files);
    let mut recs = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        let rel = f.strip_prefix(mat).unwrap_or(f).to_string_lossy().replace('\\', "/");
        if let Ok(s) = drone.take_in(Kind::Code, Source::Lines { path: rel, start: 1, end: text.lines().count() }, i, &text) {
            recs.push(s.into_inner());
        }
    }
    let ix = SymbolIndex::from_records(recs.iter());
    eprintln!("material: {} files, symbol index {} entries", recs.len(), ix.len());
    let embedder: Option<Arc<dyn intake::TextEmbedder>> = if embed {
        match nmapi::neural::embedder_model(Path::new(home)) {
            Ok(m) => {
                eprintln!("embedder on {}", if m.pop.gpu.is_some() { "gpu" } else { "cpu" });
                Some(Arc::new(nmapi::request_read::EmbedderText::new(m)))
            }
            Err(e) => {
                eprintln!("embedder not loaded: {e}");
                return;
            }
        }
    } else {
        None
    };
    let model: Option<Arc<dyn intake::RequestModel>> = if words_only || embed {
        None
    } else {
        intake::set_model_form(true);
        match nmapi::neural::model(Path::new(home)) {
            Ok(m) => {
                let dev = if m.pop.gpu.is_some() { nmengn::placement::Device::Gpu } else { nmengn::placement::Device::Cpu };
                eprintln!("reader on {}", dev.name());
                Some(Arc::new(nmapi::request_read::ReaderLabels::new(m, dev)))
            }
            Err(e) => {
                eprintln!("reader not loaded: {e}");
                return;
            }
        }
    };
    let Ok(text) = std::fs::read_to_string(qfile) else { return };
    let (mut n, mut agree, mut traces, mut trace_ok, mut loc_trace) = (0, 0, 0, 0, 0);
    let mut ms: Vec<f64> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (id, given, q) = match f.as_slice() {
            [id, q] => (*id, "", *q),
            [id, k, q] => (*id, *k, *q),
            _ => continue,
        };
        let r = intake::read_with_parts(q, &[], model.as_deref(), embedder.as_deref(), Some(&ix), true);
        let req = &r.request;
        println!("{id} | {q}");
        let probs: Vec<String> = r.kind_probs.iter().map(|(k, p)| format!("{k} {p:.2}")).collect();
        let yes: Vec<String> = r.relation_yes.iter().map(|(a, p)| format!("{} {p:.2}", a.name())).collect();
        println!("  kind {} [{}] | relations {} [{}] | {:.0} ms reader, {:.0} ms read", req.kind, probs.join(", "), req.relations.iter().map(|x| x.name()).collect::<Vec<_>>().join(", "), yes.join(", "), r.model_ms, r.read_ms);
        for asp in &req.aspects {
            let parts: Vec<String> = asp.parts.iter().map(|p| format!("{}{}", if p.role == Role::Asked { "?" } else { "" }, p.text)).collect();
            println!("  aspect [{}]", parts.join(" | "));
        }
        if !r.tags.is_empty() {
            println!("  tags {}", r.tags.iter().map(|t| format!("{} -> {} ({:?})", t.words.join(" "), t.symbol, t.how)).collect::<Vec<_>>().join("; "));
        }
        if !given.is_empty() {
            n += 1;
            let want = match given {
                "documents" => "prose",
                k => k,
            };
            if req.kind == want {
                agree += 1;
            }
            if given == "locate" && req.kind == "trace" {
                loc_trace += 1;
            }
            if given == "trace" {
                traces += 1;
                if req.kind == "trace" {
                    trace_ok += 1;
                }
            }
        }
        if r.model {
            ms.push(r.model_ms);
        }
    }
    ms.sort_by(f64::total_cmp);
    let pick = |q: f64| {
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss, reason = "an index")]
        let i = ((ms.len().saturating_sub(1)) as f64 * q).round() as usize;
        ms.get(i).copied().unwrap_or(0.0)
    };
    println!("summary: kind as given {agree} of {n}; trace read as trace {trace_ok} of {traces}; locate read as trace {loc_trace}; model ms per request median {:.0}, p95 {:.0}, max {:.0} over {}", pick(0.5), pick(0.95), pick(1.0), ms.len());
}
