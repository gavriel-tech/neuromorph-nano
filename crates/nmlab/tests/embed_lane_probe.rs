//! Embedding lane probe (2026-11-10): how well the released Qwen3-Embedding-0.6B, run through
//! the shared Qwen3 parts, finds the CX1 development targets among the substrate's chunks.
//! A measurement only; nothing in the live path changes. Development half only (runbook
//! `runs/2026-11-10-CX1-runbook.md`). Vectors are kept in `scratch/embed-lane/` (disposable) so a
//! rerun does not embed again; the ranks go to `scratch/embed-lane/probe-<ms>.json` (positions only).
//! Run: `cargo test --release --test embed_lane_probe -- --ignored --nocapture`.
#![expect(clippy::unwrap_used, reason = "probe harness")]
use nmengn::model::{Model, Population};
use nmengn::placement::Device;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

const TASK: &str = "Given a question about a code repository, retrieve the code that answers it";
const TYPES: [&str; 5] = ["behaviour", "name", "value", "explain", "relation"];

/// Development questions and decoys of CX1: (id, text).
/// (id, text) pairs.
type Pairs = Vec<(String, String)>;

fn dev_half(mini: &std::path::Path) -> (Pairs, Pairs) {
    let (mut qs, mut ds) = (Vec::new(), Vec::new());
    for area in ["context", "engine", "tools"] {
        let v =
            nmjson::parse(&std::fs::read(mini.join(format!("tests/capability-gates/cx1-context/questions-{area}.json"))).unwrap()).unwrap();
        let s = |x: &nmjson::Value, k: &str| x.get(k).unwrap().string().unwrap().to_owned();
        let mut order: Vec<(usize, String, String)> = v
            .get("questions")
            .unwrap()
            .array()
            .unwrap()
            .iter()
            .map(|x| (TYPES.iter().position(|t| *t == s(x, "type")).unwrap(), s(x, "id"), s(x, "q")))
            .collect();
        order.sort();
        qs.extend(order.into_iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, (_, id, q))| (id, q)));
        let d = v.get("decoys").unwrap().array().unwrap();
        ds.push((s(&d[0], "id"), s(&d[0], "q")));
    }
    (qs, ds)
}

#[test]
#[ignore = "probe: needs the GPU and about 20 minutes on first run"]
fn embed_lane_probe() {
    let mini = nmtype::nano_root();
    let sub = nmsubs::Substrate::scan(&mini, &[], Arc::new(nmbrain::slot::embed));
    let docs: Vec<(String, usize, usize, String)> = sub
        .chunks
        .iter()
        .map(|c| (sub.files[c.file].path.clone(), c.start, c.end, format!("{}\n{}", sub.files[c.file].path, c.text)))
        .collect();
    let pop = Population::open(&mini, "qwen3-embedding-0.6b", true, true).unwrap();
    let asm = nmengn::Assembly::load(&mini.join("designs/assemblies"), "qwen3-embedding").unwrap();
    let m = Model::bind(&pop, asm).unwrap();
    let dev = if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    // Cache: one line per chunk text hash, its vector as hex of the f32 bits.
    let dir = mini.join("scratch/embed-lane");
    std::fs::create_dir_all(&dir).unwrap();
    let cache_path = dir.join("vectors.txt");
    let mut cache: BTreeMap<String, Vec<f32>> = std::fs::read_to_string(&cache_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (h, v) = l.split_once(' ')?;
            Some((h.to_owned(), v.split(',').filter_map(|x| u32::from_str_radix(x, 16).ok().map(f32::from_bits)).collect()))
        })
        .collect();
    let t = Instant::now();
    let mut fresh = 0;
    let mut vecs = Vec::with_capacity(docs.len());
    for (_, _, _, text) in &docs {
        let h = nmhash::sha256_bytes(text.as_bytes());
        let v = if let Some(v) = cache.get(&h) {
            v.clone()
        } else {
            let v = nmengn::roles::card_embed_on(&m, dev, text).unwrap();
            let mut line = format!("{h} ");
            line.push_str(&v.iter().map(|x| format!("{:x}", x.to_bits())).collect::<Vec<_>>().join(","));
            line.push('\n');
            std::fs::OpenOptions::new().create(true).append(true).open(&cache_path).unwrap().write_all_str(&line);
            cache.insert(h, v.clone());
            fresh += 1;
            if fresh % 100 == 0 {
                println!("embedded {fresh} chunks in {:.0} s", t.elapsed().as_secs_f64());
            }
            v
        };
        vecs.push(v);
    }
    let index_s = t.elapsed().as_secs_f64();
    let (qs, ds) = dev_half(&mini);
    let mut out = String::from("{\"chunks\":");
    write!(out, "{},\"embedded_now\":{fresh},\"index_s\":{index_s:.1},\"device\":\"{}\",\"questions\":{{", docs.len(), dev.name()).unwrap();
    let mut qms = Vec::new();
    for (i, (id, q)) in qs.iter().chain(ds.iter()).enumerate() {
        let t = Instant::now();
        let qv = nmengn::roles::card_embed_on(&m, dev, &nmengn::roles::card_query(TASK, q)).unwrap();
        qms.push(t.elapsed().as_secs_f64() * 1000.0);
        let mut sc: Vec<(f32, usize)> =
            vecs.iter().enumerate().map(|(j, v)| (v.iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>(), j)).collect();
        sc.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let top: Vec<String> =
            sc.iter().take(50).map(|(s, j)| format!("[\"{}\",{},{},{s:.4}]", docs[*j].0, docs[*j].1, docs[*j].2)).collect();
        write!(out, "{}\"{id}\":[{}]", if i == 0 { "" } else { "," }, top.join(",")).unwrap();
    }
    qms.sort_by(f64::total_cmp);
    write!(out, "}},\"query_ms_median\":{:.1}}}", qms[qms.len() / 2]).unwrap();
    let p = dir.join(format!("probe-{}.json", nmtype::now_ms()));
    std::fs::write(&p, out).unwrap();
    println!(
        "chunks {}, embedded now {fresh}, index {index_s:.0} s on {}, query median {:.0} ms; ranks in {}",
        docs.len(),
        dev.name(),
        qms[qms.len() / 2],
        p.display()
    );
}

trait WriteStr {
    fn write_all_str(self, s: &str);
}
impl WriteStr for std::fs::File {
    fn write_all_str(mut self, s: &str) {
        use std::io::Write as _;
        self.write_all(s.as_bytes()).unwrap();
    }
}
