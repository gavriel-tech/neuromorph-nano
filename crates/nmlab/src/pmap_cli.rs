//! `nano-lab run pmap <vocab-profile|sweep> ...`: bidirectional surgical parameter mapping,
//! per-job Wanda union analysis, exact vocabulary trimming savings, and live GTX 1650 held-out
//! evaluation of surgical removal and surgical addition from untouched reference `weights/`.
use nmengn::model::{Model, Population, RunOpts, values};
use nmengn::placement::Device;
use nmengn::pmap::{self, BitMask, ParamMask, ParamRecord, Stream};
use nmengn::Assembly;
use nmnpcs::TensorDesc;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{Read as _, Seek as _};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

fn read_range(path: &Path, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.seek(std::io::SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    let len = usize::try_from(end.saturating_sub(start)).map_err(|e| e.to_string())?;
    let mut buf = vec![0_u8; len];
    f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(buf)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (f64::from(x), f64::from(y));
        dot += dx * dy;
        na += dx * dx;
        nb += dy * dy;
    }
    let den = na.sqrt() * nb.sqrt();
    if den > 0.0 { dot / den } else { 0.0 }
}

fn model_spec(role: &str) -> (&'static str, &'static str, &'static str, bool) {
    match role {
        "reranker" => ("reranker", "qwen3-reranker-0.6b", "qwen3-reranker", true),
        "embedder" => ("embedder", "qwen3-embedding-0.6b", "qwen3-embedding", true),
        "writer" => ("writer", "qwen3-1.7b", "qwen3-reference", false),
        _ => ("reader", "qwen3-0.6b", "qwen3-reference", true),
    }
}

type JobSet = Vec<(&'static str, Vec<&'static str>)>;
type HeldArcs = Vec<Arc<[u8]>>;

/// Real writer job prompts matching `crates/nmapi/src/writer_idle.rs` (one-line piece notes,
/// short summaries, file notes, and project outlines) so `writer-base` is mapped strictly for
/// the writer's own responsibilities.
fn writer_job_prompts() -> (JobSet, JobSet) {
    let map = vec![
        (
            "note",
            vec![
                "<|im_start|>user\ncrates/nmengn/src/base.rs lines 1-45:\n```\npub fn seat(open: &Model, offer: &Offer) -> Result<(Model, Seated), Refused> {\n    if offer.weights != open.pop.card.sha_main { return Err(Refused::Weights(offer.key.clone())); }\n}\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "<|im_start|>user\ncrates/nmsubs/src/dense_store.rs lines 1-60:\n```\npub fn shed(&mut self, live: &BTreeSet<&str>, live_file_tags: Option<&BTreeSet<String>>, now_ms: u64) -> ShedReport {\n    // revives dead-barrier vectors on undo and micro-sheds expired deleted-file rows first\n}\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "<|im_start|>user\ncrates/nmapi/src/mcp.rs lines 60-120:\n```\npub fn trigger_dynamic_tools_with(limit: usize, dynamic_on: bool, custom_enabled: Option<&[String]>, tool_name: &str, question: Option<&str>) -> Vec<&'static str>\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
        (
            "summary",
            vec![
                "<|im_start|>user\nIn at most two short sentences of plain words, on one line, say what this part of crates/nmlearn/src/engine.rs (lines 1350-1495) does and when someone working on the project would need to read it.\n\n```\npub fn run_learning_cycle(&mut self) -> Value {\n    // extracts recurring patterns into semantic, procedural, and behavioral cartridges\n}\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "<|im_start|>user\nIn at most two short sentences of plain words, on one line, say what this part of crates/nmapi/src/g3.rs (lines 180-240) does and when someone working on the project would need to read it.\n\n```\nconst CYCLE_BUDGET_S: f64 = 20.0;\nconst MAX_REPLAYS_PER_CYCLE: usize = 64;\n```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
        (
            "file_note",
            vec![
                "<|im_start|>user\nHere are notes on the parts of the file crates/nmengn/src/pmap.rs:\n- Records per-job activation norms and computes Wanda and OWL layerwise sparsity masks. Defines: ParamRecord, owl_layer_sparsities, wanda_mask_for_matrix\n- Applies surgical bitmasks to BF16 weights and restores missing paths from reference weights. Defines: apply_mask_bf16, surgical_add\n\nIn one short line of plain words, say what this file is for.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "<|im_start|>user\nHere are notes on the parts of the file crates/nmlearn/src/library.rs:\n- Enforces cartridge stacking hierarchy base -> hardware -> adapter -> learned -> preference. Defines: family_rank, validate_stack_hierarchy\n- Replaces single-copy base cartridges in place and updates catalogue hashes. Defines: Library::replace_base_cartridge\n\nIn one short line of plain words, say what this file is for.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
        (
            "outline",
            vec![
                "<|im_start|>user\nHere is one line on each file of a software project:\n- crates/nmengn/src/gpu.rs: Runs custom Turing CUDA kernels including bit-exact bmq8 sparse int8 matrix multiply.\n- crates/nmsubs/src/dense_store.rs: Persists segmented binary16 chunk vectors with dead-skin undo revival and age-calibrated shedding.\n- crates/nmapi/src/drone_path.rs: Orchestrates tentacle reads, G1 union, G2 gate, and Core Brain intake.\n- crates/nmlearn/src/engine.rs: Runs supervised continual learning across 6 modular cartridges with rollback snapshots.\nWrite a short outline of the project in plain words: its main parts, what each is for, and where to start reading. At most twelve lines.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
    ];
    let held = vec![
        (
            "note",
            vec![
                "<|im_start|>user\ncrates/nmengn/src/filter.rs lines 1-40:\n```\npub fn compose(parts: &[(&[f32], &[f32])]) -> Result<(Vec<f32>, Vec<f32>), String> {\n    let mut g = vec![1.0_f64; width];\n    let mut b = vec![0.0_f64; width];\n}\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
        (
            "file_note",
            vec![
                "<|im_start|>user\nHere are notes on the parts of the file crates/nmapi/src/writer_idle.rs:\n- Runs preemptible idle generation using active-vocab head rows and chunked prefill. Defines: active_head_rows, decode_preemptible\n- Verifies generated one-line notes against nearest neighbours in the meaning lane. Defines: note_lands\n\nIn one short line of plain words, say what this file is for.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            ],
        ),
    ];
    (map, held)
}

/// Real codebase calibration and held-out prompts split cleanly across three reader/judge jobs.
fn job_prompts() -> (JobSet, JobSet) {
    let map = vec![
        (
            "drone",
            vec![
                "crates/nmengn/src/base.rs\npub fn seat(open: &Model, offer: &Offer) -> Result<(Model, Seated), Refused> {\n    if offer.weights != open.pop.card.sha_main { return Err(Refused::Weights(offer.key.clone())); }\n}",
                "crates/nmlearn/src/engine.rs\npub fn run_learning_cycle(&mut self) -> CycleReport {\n    self.extract_patterns();\n    self.update_cartridges();\n}",
                "crates/nmapi/src/mcp.rs\npub fn trigger_dynamic_tools(limit: usize, dynamic_on: bool, tool_name: &str, question: Option<&str>) -> Vec<&'static str>",
                "crates/nmcore/src/bin/controls.rs\npub const CONTROLS: &[Control] = &[Control { key: \"quota.session_tokens\", group: \"Budgets and allowances\" }];",
            ],
        ),
        (
            "integration",
            vec![
                "Question: where does base cartridge seating verify open weights SHA-256 and drop open weights?\nContext:\n- crates/nmengn/src/base.rs: seat() checks offer.weights against open.pop.card.sha_main, runs check_outputs, then returns the seated Model.\n- crates/nmlearn/src/library.rs: Filter::Open holds differential tensor recipes.",
                "Question: how does dynamic MCP tool activation promote tools when mcp.tool_count is 5?\nContext:\n- crates/nmapi/src/mcp.rs: trigger_dynamic_tools() adds matching tools and sets LIST_CHANGED_PENDING.\n- notifications/tools/list_changed is emitted after the reply.",
                "Question: how does writer_idle avoid blocking server shutdown on a 4 GB GTX 1650?\nContext:\n- crates/nmapi/src/writer_idle.rs: stop() sets SHUTDOWN and PREEMPT and skips GPU model restores.\n- crates/nmcore/src/bin/main.rs calls writer_idle::stop() on EOF.",
            ],
        ),
        (
            "judge",
            vec![
                "Judge whether the Document meets the requirements based on the Query.\nQuery: where are base cartridges seated and checked against open weights?\nDocument: crates/nmengn/src/base.rs\npub fn seat(open: &Model, offer: &Offer) -> Result<(Model, Seated), Refused>",
                "Judge whether the Document meets the requirements based on the Query.\nQuery: where are base cartridges seated and checked against open weights?\nDocument: interface/src/app/settings/page.tsx\nexport default function SettingsPage() { return <main>Settings</main>; }",
                "Judge whether the Document meets the requirements based on the Query.\nQuery: how does the continuous learning supervisor roll back a version?\nDocument: crates/nmlearn/src/engine.rs\npub fn rollback(&mut self) -> Result<u64, String> { let snap = self.history.pop()?; }",
                "Judge whether the Document meets the requirements based on the Query.\nQuery: how does the continuous learning supervisor roll back a version?\nDocument: Cargo.toml\n[workspace]\nmembers = [\"crates/nmhash\", \"crates/nmjson\"]",
            ],
        ),
    ];
    let held = vec![
        (
            "drone",
            vec![
                "crates/nmengn/src/filter.rs\npub fn compose(parts: &[(&[f32], &[f32])]) -> Result<(Vec<f32>, Vec<f32>), String> {\n    let mut g = vec![1.0_f64; width];\n    let mut b = vec![0.0_f64; width];\n}",
                "crates/nmengn/src/adapter.rs\npub fn delta(&self, x: &[f32], tokens: usize) -> Vec<f32> {\n    let v = self.scale * row.iter().zip(&h).map(|(a, b)| a * b).sum::<f32>();\n}",
            ],
        ),
        (
            "judge",
            vec![
                "Judge whether the Document meets the requirements based on the Query.\nQuery: where are affine filter cartridges folded in float64 before GPU upload?\nDocument: crates/nmengn/src/filter.rs\npub fn compose(parts: &[(&[f32], &[f32])]) -> Result<(Vec<f32>, Vec<f32>), String>",
                "Judge whether the Document meets the requirements based on the Query.\nQuery: where are affine filter cartridges folded in float64 before GPU upload?\nDocument: docs/privacy.md\nNothing leaves this computer unless you export a stack.",
            ],
        ),
    ];
    (map, held)
}

fn collect_repo_texts(root: &Path, max_files: usize) -> Vec<String> {
    let mut out = Vec::new();
    let dirs = ["crates", "docs", "interface/src"];
    for d in dirs {
        let base = root.join(d);
        let mut stack = vec![base];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if !matches!(name, "target" | "node_modules" | ".next" | "out") {
                        stack.push(p);
                    }
                } else if p.is_file() {
                    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if matches!(ext, "rs" | "md" | "ts" | "tsx" | "json")
                        && let Ok(txt) = std::fs::read_to_string(&p)
                    {
                        out.push(txt);
                        if out.len() >= max_files {
                            return out;
                        }
                    }
                }
            }
        }
    }
    out
}

fn vocab_profile(home: &Path) -> Result<String, String> {
    let t0 = Instant::now();
    let pop = Population::open(home, "qwen3-0.6b", false, false).map_err(|e| e.to_string())?;
    let texts = collect_repo_texts(home, 200);
    let mut active = BTreeSet::new();
    let mut total_tokens = 0_usize;
    for t in &texts {
        if let Ok(ids) = pop.tokenizer.encode(t) {
            total_tokens += ids.len();
            for id in ids {
                if let Ok(u) = u32::try_from(id) {
                    active.insert(u);
                }
            }
        }
    }
    let vocab = pop.dims.vocab;
    let active_n = active.len();
    #[expect(clippy::cast_precision_loss, reason = "display percentages and MiB")]
    let pct_kept = 100.0 * active_n as f64 / vocab.max(1) as f64;
    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);

    let r06_orig = vocab * 1024 * 2;
    let r06_trim = active_n * 1024 * 2;
    let w17_orig = vocab * 2048 * 2;
    let w17_trim = active_n * 2048 * 2;

    let mut out = String::new();
    let _ = writeln!(out, "=== Exact Job-Conditioned Vocabulary Profile (Neuromorph Nano Repo) ===");
    let _ = writeln!(out, "Files scanned : {} ({} total tokens in {:.2} s)", texts.len(), total_tokens, t0.elapsed().as_secs_f64());
    let _ = writeln!(out, "Vocab size    : {vocab} | Active unique tokens: {active_n} ({pct_kept:.2}% kept, {:.2}% removable)", 100.0 - pct_kept);
    let _ = writeln!(
        out,
        "0.6B embed/head (151936 x 1024 BF16): {:.1} MiB -> {:.1} MiB (saves {:.1} MiB per 0.6B table; host weights/ fallback for unseen tokens)",
        mib(r06_orig),
        mib(r06_trim),
        mib(r06_orig.saturating_sub(r06_trim))
    );
    let _ = writeln!(
        out,
        "1.7B writer head (151936 x 2048 BF16): {:.1} MiB -> {:.1} MiB (saves {:.1} MiB VRAM on GTX 1650!)",
        mib(w17_orig),
        mib(w17_trim),
        mib(w17_orig.saturating_sub(w17_trim))
    );
    Ok(out)
}

fn build_mask_for_job(
    pop: &Population,
    rec: &ParamRecord,
    job_name: &str,
    sparsity: f64,
    pinned_heads: &BTreeMap<usize, Vec<usize>>,
) -> Result<ParamMask, String> {
    let mut mask = ParamMask { pinned_heads: pinned_heads.clone(), ..ParamMask::default() };
    let js = rec.jobs.get(job_name).ok_or_else(|| format!("missing job {job_name}"))?;
    mask.tokens_in.clone_from(&js.tokens_in);
    mask.tokens_out.clone_from(&js.tokens_out);
    let head_dim = pop.dims.head_dim;

    for t in &pop.tensors {
        let Some((layer, stream)) = Stream::for_tensor(&t.name) else { continue };
        let Some(ss) = js.layers.get(&layer).and_then(|m| m.get(&stream)) else { continue };
        if t.shape.len() != 2 || t.dtype != "BF16" {
            continue;
        }
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let bytes = read_range(&pop.card.dir.join(&t.file), t.start, t.end)?;
        let weights = pmap::decode_bf16(&bytes);
        let norms = ss.l2_norms();
        let (pin_r, pin_c) = pmap::pinned_axes_for_tensor(&t.name, head_dim, pinned_heads);
        let bm = pmap::wanda_mask_for_matrix(&weights, rows, cols, &norms, sparsity, (&pin_r, &pin_c));
        mask.tensors.insert(t.name.clone(), bm);
    }
    Ok(mask)
}

fn masked_model(open: &Model, mask: &ParamMask, tag: &str, q8: bool) -> Result<(Model, HeldArcs), String> {
    let mut parts: Vec<TensorDesc> = Vec::with_capacity(open.pop.tensors.len());
    let mut held_arcs: Vec<Arc<[u8]>> = Vec::new();
    for t in &open.pop.tensors {
        if let Some(bm) = mask.tensors.get(&t.name) {
            let raw = read_range(&open.pop.card.dir.join(&t.file), t.start, t.end)?;
            let masked = pmap::apply_mask_bf16(&raw, bm);
            let sha = nmhash::sha256_bytes(&masked);
            let arc: Arc<[u8]> = Arc::from(masked);
            nmnpcs::provision::hold(&sha, &arc);
            held_arcs.push(Arc::clone(&arc));
            parts.push(TensorDesc {
                file: format!("(pmap:{tag}:{})", t.name),
                start: 0,
                end: arc.len() as u64,
                sha256: sha,
                ..t.clone()
            });
        } else {
            parts.push(t.clone());
        }
    }
    let gpu = open.pop.gpu.is_some();
    let pop = Population::open_cartridge(&open.pop.home, &open.pop.card.id, gpu, tag, parts).map_err(|e| e.to_string())?;
    let mut asm = open.asm.clone();
    if q8 {
        for node in &mut asm.nodes {
            if node.op == "linear" {
                node.precision = "q8".to_owned();
            }
        }
    }
    let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
    Ok((m, held_arcs))
}

/// Computes OWL (Outlier-Weighted Layerwise Sparsity, ICML 2024) per-layer budgets for a job from
/// activation L2-norm outlier ratios (share of input channels > 5x layer mean L2 norm).
fn build_owl_mask_for_job(
    pop: &Population,
    rec: &ParamRecord,
    job_name: &str,
    target_sparsity: f64,
    lambda: f64,
    pinned_heads: &BTreeMap<usize, Vec<usize>>,
) -> Result<ParamMask, String> {
    let js = rec.jobs.get(job_name).ok_or_else(|| format!("missing job {job_name}"))?;
    let layers = pop.dims.layers;
    let mut outlier_ratios = vec![0.0_f64; layers];
    for (l, slot) in outlier_ratios.iter_mut().enumerate() {
        let mut all_norms = Vec::new();
        if let Some(streams) = js.layers.get(&l) {
            for ss in streams.values() {
                all_norms.extend(ss.l2_norms());
            }
        }
        if !all_norms.is_empty() {
            #[expect(clippy::cast_precision_loss, reason = "channel count")]
            let mean = all_norms.iter().map(|&x| f64::from(x)).sum::<f64>() / all_norms.len() as f64;
            let thresh = 5.0 * mean;
            let outliers = all_norms.iter().filter(|&&x| f64::from(x) > thresh).count();
            #[expect(clippy::cast_precision_loss, reason = "outlier fraction")]
            {
                *slot = outliers as f64 / all_norms.len() as f64;
            }
        }
    }
    let budgets = pmap::owl_layer_sparsities(&outlier_ratios, target_sparsity, lambda);
    let mut mask = ParamMask { pinned_heads: pinned_heads.clone(), ..ParamMask::default() };
    mask.tokens_in.clone_from(&js.tokens_in);
    mask.tokens_out.clone_from(&js.tokens_out);
    let head_dim = pop.dims.head_dim;
    for t in &pop.tensors {
        let Some((layer, stream)) = Stream::for_tensor(&t.name) else { continue };
        let Some(ss) = js.layers.get(&layer).and_then(|m| m.get(&stream)) else { continue };
        if t.shape.len() != 2 || t.dtype != "BF16" {
            continue;
        }
        let sp = budgets.get(layer).copied().unwrap_or(target_sparsity);
        let (rows, cols) = (t.shape[0], t.shape[1]);
        let bytes = read_range(&pop.card.dir.join(&t.file), t.start, t.end)?;
        let weights = pmap::decode_bf16(&bytes);
        let norms = ss.l2_norms();
        let (pin_r, pin_c) = pmap::pinned_axes_for_tensor(&t.name, head_dim, pinned_heads);
        let bm = pmap::wanda_mask_for_matrix(&weights, rows, cols, &norms, sp, (&pin_r, &pin_c));
        mask.tensors.insert(t.name.clone(), bm);
    }
    Ok(mask)
}

fn eval_held(open: &Model, cand: &Model, held: &[(&str, Vec<&str>)]) -> Result<(f64, f64, f64), String> {
    let dev = if open.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    let norm_id = open.role_node("final_norm").map_err(|e| e.to_string())?;
    let head_id = open.role_node("head").ok();
    let yes_id = open.pop.tokenizer.encode("yes").ok().and_then(|v| v.first().copied()).and_then(|x| usize::try_from(x).ok());
    let no_id = open.pop.tokenizer.encode("no").ok().and_then(|v| v.first().copied()).and_then(|x| usize::try_from(x).ok());

    let mut cos_sum = 0.0_f64;
    let mut cos_n = 0_usize;
    let mut open_margin = 0.0_f64;
    let mut cand_margin = 0.0_f64;
    let mut judge_z_open = Vec::new();
    let mut judge_z_cand = Vec::new();

    for (job, prompts) in held {
        for p in prompts {
            let ids = open.pop.tokenizer.encode(p).map_err(|e| e.to_string())?;
            let mut targets = vec![(norm_id.clone(), "y".to_owned())];
            let mut opts = RunOpts::default();
            if *job == "judge"
                && let (Some(hid), Some(yi), Some(ni)) = (&head_id, yes_id, no_id)
            {
                targets.push((hid.clone(), "logits".to_owned()));
                opts.head_rows = Arc::new(vec![yi, ni]);
            }
            let oa = open.forward(dev, &ids, &targets, &opts).map_err(|e| e.to_string())?;
            let ca = cand.forward(dev, &ids, &targets, &opts).map_err(|e| e.to_string())?;
            let ov = values(oa.get(&(norm_id.clone(), "y".to_owned())).ok_or("missing open norm")?).map_err(|e| e.to_string())?;
            let cv = values(ca.get(&(norm_id.clone(), "y".to_owned())).ok_or("missing cand norm")?).map_err(|e| e.to_string())?;
            let h = open.pop.dims.hidden;
            let o_last = ov.get(ov.len().saturating_sub(h)..).unwrap_or(ov);
            let c_last = cv.get(cv.len().saturating_sub(h)..).unwrap_or(cv);
            cos_sum += cosine(o_last, c_last);
            cos_n += 1;

            if *job == "judge"
                && let Some(hid) = &head_id
                && let (Some(ol), Some(cl)) = (oa.get(&(hid.clone(), "logits".to_owned())), ca.get(&(hid.clone(), "logits".to_owned())))
            {
                let ovl = values(ol).map_err(|e| e.to_string())?;
                let cvl = values(cl).map_err(|e| e.to_string())?;
                if ovl.len() >= 2 && cvl.len() >= 2 {
                    let oz = f64::from(ovl[ovl.len() - 2] - ovl[ovl.len() - 1]);
                    let cz = f64::from(cvl[cvl.len() - 2] - cvl[cvl.len() - 1]);
                    judge_z_open.push(oz);
                    judge_z_cand.push(cz);
                }
            }
        }
    }
    if judge_z_open.len() >= 2 {
        open_margin = judge_z_open[0] - judge_z_open[1];
        cand_margin = judge_z_cand[0] - judge_z_cand[1];
    }
    #[expect(clippy::cast_precision_loss, reason = "average cosine")]
    let mean_cos = if cos_n > 0 { cos_sum / cos_n as f64 } else { 0.0 };
    Ok((mean_cos, open_margin, cand_margin))
}

fn sweep(home: &Path, role: &str, use_gpu: bool) -> Result<String, String> {
    let t0 = Instant::now();
    let (short_role, pop_id, asm_id, tied_head) = model_spec(role);
    let pop = Population::open(home, pop_id, false, use_gpu).map_err(|e| e.to_string())?;
    let asm = Assembly::load(&home.join("designs/assemblies"), asm_id).map_err(|e| e.to_string())?;
    let open = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
    let dev = if use_gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };

    let (map_jobs, held_jobs) = job_prompts();
    let map_out = home.join(format!("library/parameter-maps/{short_role}/pmap-v1.json"));

    // 1. Record per-job activations + active tokens across all 28 layers (attention + FFN).
    pmap::start_recording(pop_id, Some(map_out.clone()));
    let norm_id = open.role_node("final_norm").map_err(|e| e.to_string())?;
    let target = vec![(norm_id, "y".to_owned())];
    let t_rec = Instant::now();
    for (job, prompts) in &map_jobs {
        let _guard = nmengn::chanrec::job(job);
        for p in prompts {
            let ids = pop.tokenizer.encode(p).map_err(|e| e.to_string())?;
            let _ = open.forward(dev, &ids, &target, &RunOpts::default()).map_err(|e| e.to_string())?;
        }
    }
    let rec = pmap::stop_recording().ok_or("failed to collect ParamRecord")?;
    let rec_s = t_rec.elapsed().as_secs_f64();

    // Pin tentacle relevance heads (layers 12..19, heads 0 & 1 as protected structural anchor).
    let mut pinned = BTreeMap::new();
    for l in 12..20 {
        pinned.insert(l, vec![0, 1]);
    }

    let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
    let mut out = String::new();
    let _ = writeln!(out, "=== Parameter-Level Bidirectional Surgical Mapping Sweep: {short_role} ({pop_id}, {:?}) ===", dev);
    let _ = writeln!(out, "Recorded 3 jobs (drone, integration, judge) across 28 layers (all 7 linears/layer: q/k/v/o + gate/up/down) in {:.2} s", rec_s);
    let _ = writeln!(out, "Saved parameter map: {}\n", map_out.display());

    // 2. Sweep per-job sparsity -> union mask + Jaccard overlap + exact storage & VRAM report.
    let _ = writeln!(out, "| Per-job prune | Union sparsity | Attn sparsity | FFN sparsity | Dense Q8 (MiB) | Bitmask-Q8 (MiB) | Linear saving | K32 zero-tiles | Drone/Judge Jaccard |");
    let _ = writeln!(out, "|---|---|---|---|---|---|---|---|---|");

    let mut mask_30_union = None;
    let mut mask_30_judge = None;
    let mut mask_50_drone = None;

    for sp in [0.20_f64, 0.30, 0.40, 0.50] {
        let m_drone = build_mask_for_job(&pop, &rec, "drone", sp, &pinned)?;
        let m_int = build_mask_for_job(&pop, &rec, "integration", sp, &pinned)?;
        let m_judge = build_mask_for_job(&pop, &rec, "judge", sp, &pinned)?;
        let union = pmap::union_masks(&[&m_drone, &m_int, &m_judge]);
        let rep = pmap::estimate_storage(&union, pop.dims.vocab, pop.dims.hidden, tied_head);

        let (mut attn_rem, mut attn_tot, mut ffn_rem, mut ffn_tot) = (0_usize, 0_usize, 0_usize, 0_usize);
        let (mut jac_sum, mut jac_n) = (0.0_f64, 0_usize);
        for (name, bm) in &union.tensors {
            if name.contains(".self_attn.") {
                attn_rem += bm.removed_count();
                attn_tot += bm.total();
            } else {
                ffn_rem += bm.removed_count();
                ffn_tot += bm.total();
            }
            if let (Some(a), Some(b)) = (m_drone.tensors.get(name), m_judge.tensors.get(name)) {
                jac_sum += a.jaccard(b);
                jac_n += 1;
            }
        }
        #[expect(clippy::cast_precision_loss, reason = "report percentages")]
        let attn_sp = 100.0 * attn_rem as f64 / attn_tot.max(1) as f64;
        #[expect(clippy::cast_precision_loss, reason = "report percentages")]
        let ffn_sp = 100.0 * ffn_rem as f64 / ffn_tot.max(1) as f64;
        #[expect(clippy::cast_precision_loss, reason = "report percentages")]
        let linear_save = 100.0 * (1.0 - rep.bitmask_q8_bytes as f64 / rep.dense_q8_bytes.max(1) as f64);
        #[expect(clippy::cast_precision_loss, reason = "report percentages")]
        let jac = if jac_n > 0 { jac_sum / jac_n as f64 } else { 0.0 };

        let _ = writeln!(
            out,
            "| {:>5.0}% | {:>6.2}% | {:>6.2}% | {:>6.2}% | {:>6.1} | {:>6.1} | {:>+6.2}% | {:>6.2}% | {:.3} |",
            sp * 100.0,
            rep.sparsity * 100.0,
            attn_sp,
            ffn_sp,
            mib(rep.dense_q8_bytes),
            mib(rep.bitmask_q8_bytes),
            linear_save,
            rep.mean_k32_tile_skip * 100.0,
            jac
        );

        if (sp - 0.30).abs() < 1e-6 {
            mask_30_union = Some(union);
            mask_30_judge = Some(m_judge);
        }
        if (sp - 0.50).abs() < 1e-6 {
            mask_50_drone = Some(m_drone);
        }
    }

    // 3. Live GTX 1650 held-out verification: 30% union vs aggressive 50% drone-only vs surgical ADD vs OWL+bmq8!
    if let (Some(u30), Some(j30), Some(d50)) = (mask_30_union, mask_30_judge, mask_50_drone) {
        let _ = writeln!(out, "\n=== Live Held-Out Verification & Bidirectional Surgical Add-Back ({:?}) ===", dev);

        let (m_u30, _hold1) = masked_model(&open, &u30, "u30", false)?;
        let (cos_u30, open_m, cand_m_u30) = eval_held(&open, &m_u30, &held_jobs)?;
        drop(m_u30);
        let _ = writeln!(
            out,
            "1) 30% per-job union (attention + FFN, pinned heads): sparsity={:.2}% | final-norm cosine={:.5} | judge margin={:.3} (open ref={:.3})",
            u30.tensors.values().map(BitMask::removed_count).sum::<usize>() as f64
                * 100.0
                / u30.tensors.values().map(BitMask::total).sum::<usize>().max(1) as f64,
            cos_u30,
            cand_m_u30,
            open_m
        );

        // Over-pruned arm (50% drone-only mask, missing judge-specific parameter paths):
        let (m_d50, _hold2) = masked_model(&open, &d50, "d50", false)?;
        let (cos_d50, _, cand_m_d50) = eval_held(&open, &m_d50, &held_jobs)?;
        drop(m_d50);
        let _ = writeln!(
            out,
            "2) Over-pruned 50% drone-only (before surgical add):  sparsity={:.2}% | final-norm cosine={:.5} | judge margin={:.3} (open ref={:.3})",
            d50.tensors.values().map(BitMask::removed_count).sum::<usize>() as f64
                * 100.0
                / d50.tensors.values().map(BitMask::total).sum::<usize>().max(1) as f64,
            cos_d50,
            cand_m_d50,
            open_m
        );

        // Surgical ADD: restore exact missing judge paths `Delta = M_judge30 AND NOT M_drone50` from untouched weights!
        let (recovered, add_delta) = pmap::surgical_add(&d50, &j30);
        let (m_rec, _hold3) = masked_model(&open, &recovered, "d50+j30", false)?;
        let (cos_rec, _, cand_m_rec) = eval_held(&open, &m_rec, &held_jobs)?;
        drop(m_rec);
        let _ = writeln!(
            out,
            "3) After SURGICAL ADD (+{} judge weights from ref):   sparsity={:.2}% | final-norm cosine={:.5} | judge margin={:.3} (delta cos={:+.5}, delta margin={:+.3})",
            add_delta.changed_weights,
            recovered.tensors.values().map(BitMask::removed_count).sum::<usize>() as f64
                * 100.0
                / recovered.tensors.values().map(BitMask::total).sum::<usize>().max(1) as f64,
            cos_rec,
            cand_m_rec,
            cos_rec - cos_d50,
            cand_m_rec - cand_m_d50
        );

        // 4) Live GTX 1650 Q8 vs OWL 40% + bmq8.cu real VRAM & quality comparison!
        let owl_d = build_owl_mask_for_job(&pop, &rec, "drone", 0.40, 0.08, &pinned)?;
        let owl_i = build_owl_mask_for_job(&pop, &rec, "integration", 0.40, 0.08, &pinned)?;
        let owl_j = build_owl_mask_for_job(&pop, &rec, "judge", 0.40, 0.08, &pinned)?;
        let owl_union = pmap::union_masks(&[&owl_d, &owl_i, &owl_j]);
        let (m_dense_q8, _h_dq8) = masked_model(&open, &ParamMask::default(), "dense-q8", true)?;
        let (_, _, _) = eval_held(&open, &m_dense_q8, &held_jobs)?;
        let dense_vram = m_dense_q8.pop.device_bytes();
        drop(m_dense_q8);
        let (m_owl_bmq8, _h_owl) = masked_model(&open, &owl_union, "owl40-bmq8", true)?;
        let (cos_owl, _, cand_m_owl) = eval_held(&open, &m_owl_bmq8, &held_jobs)?;
        let bmq8_vram = m_owl_bmq8.pop.device_bytes();
        drop(m_owl_bmq8);
        let owl_sp = owl_union.tensors.values().map(BitMask::removed_count).sum::<usize>() as f64
            * 100.0
            / owl_union.tensors.values().map(BitMask::total).sum::<usize>().max(1) as f64;
        let vram_save_pct = if dense_vram > 0 { 100.0 * (1.0 - bmq8_vram as f64 / dense_vram as f64) } else { 0.0 };
        let _ = writeln!(
            out,
            "4) LIVE GPU Q8 OWL-40% + bmq8.cu:                     sparsity={owl_sp:.2}% | final-norm cosine={cos_owl:.5} | judge margin={cand_m_owl:.3} | REAL GPU VRAM {:.1} MiB -> {:.1} MiB ({vram_save_pct:+.2}%)",
            mib(dense_vram),
            mib(bmq8_vram)
        );
    }

    let _ = writeln!(out, "\nTotal elapsed: {:.2} s (untouched reference weights/ unmodified)", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Builds a single-copy OWL parameter-mapped 16-bit (BF16) base cartridge for `role` and replaces
/// any older local copy in `library/cartridges/base/<role>/` in place (Luis Jake Gabriel III,
/// 2026-11-21 single-copy rule; untouched `weights/` stays read-only, Hugging Face holds backups).
fn build_base(home: &Path, role: &str, sparsity: f64, use_gpu: bool, forced_version: Option<u64>) -> Result<String, String> {
    let t0 = Instant::now();
    let (short_role, pop_id, asm_id, _tied) = model_spec(role);
    let is_writer = short_role == "writer";
    let sparsity = sparsity.clamp(0.05, 0.60);
    let pop = Population::open(home, pop_id, false, use_gpu).map_err(|e| e.to_string())?;
    let mut asm = Assembly::load(&home.join("designs/assemblies"), asm_id).map_err(|e| e.to_string())?;
    // The 1.7B writer exceeds 4 GB VRAM in full BF16; record activations in Q8 on GPU so it fits
    // cleanly within 4 GB VRAM while preserving BF16 weights on disk.
    let mut rec_asm = asm.clone();
    if is_writer && use_gpu {
        for node in &mut rec_asm.nodes {
            if node.op == "linear" {
                node.precision = "q8".to_owned();
            }
        }
    }
    let open = Model::bind(&pop, rec_asm).map_err(|e| e.to_string())?;
    let dev = if use_gpu && pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };

    let (map_jobs, held_jobs) = if is_writer { writer_job_prompts() } else { job_prompts() };
    let map_out = home.join(format!("library/parameter-maps/{short_role}/pmap-owl-{:.0}.json", sparsity * 100.0));
    pmap::start_recording(pop_id, Some(map_out.clone()));
    let norm_id = open.role_node("final_norm").map_err(|e| e.to_string())?;
    let target = vec![(norm_id, "y".to_owned())];
    for (job, prompts) in &map_jobs {
        let _guard = nmengn::chanrec::job(job);
        for p in prompts {
            let ids = pop.tokenizer.encode(p).map_err(|e| e.to_string())?;
            let _ = open.forward(dev, &ids, &target, &RunOpts::default()).map_err(|e| e.to_string())?;
        }
    }
    let rec = pmap::stop_recording().ok_or("failed to collect ParamRecord")?;
    // 2026-11-26: the map's lineage, as it is written (nmapi::lineage).
    if let (Some(dir), Some(name)) = (map_out.parent(), map_out.file_name().and_then(|n| n.to_str())) {
        let _ = nmapi::lineage::record_made(dir, name, &format!("open:{pop_id}"), &format!("nano-lab run pmap build-base {short_role} {sparsity}{}", if use_gpu { " gpu" } else { " cpu" }), "base-build");
    }
    let mut pinned = BTreeMap::new();
    if !is_writer {
        for l in 12..20 {
            pinned.insert(l, vec![0, 1]);
        }
    }
    let mut job_masks = Vec::with_capacity(map_jobs.len());
    for (job, _) in &map_jobs {
        job_masks.push(build_owl_mask_for_job(&pop, &rec, job, sparsity, 0.08, &pinned)?);
    }
    let refs: Vec<&ParamMask> = job_masks.iter().collect();
    let owl_union = pmap::union_masks(&refs);

    // Measure exact `check_bound` against the open model so `nmengn::base::seat()` verifies and
    // accepts this parameter-mapped cartridge cleanly (Q8 on GPU + F16 on CPU for 1.7B writer so
    // it never exceeds 4 GB VRAM; F16 + Q8 on GPU for 0.6B models).
    let saved_sw = (
        nmengn::gpu::HALF_MATH.load(std::sync::atomic::Ordering::Relaxed),
        nmengn::gpu::SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed),
    );
    let mut largest_diff = 0.0_f32;
    let mut held_cos = 0.0_f64;
    let mut bmq8_vram = 0_usize;
    let mut dense_vram = 0_usize;
    // Release open's GPU weights before allocating masked candidates so 4 GB cards stay roomy.
    open.pop.release_device();
    for served in [false, true] {
        nmengn::gpu::HALF_MATH.store(served, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(served, std::sync::atomic::Ordering::Relaxed);
        let precisions: &[bool] = if is_writer { &[true] } else { &[false, true] };
        for &q8 in precisions {
            let (open_m, _h0) = masked_model(&open, &ParamMask::default(), "open-chk", q8)?;
            let a = nmengn::base::check_outputs(&open_m).map_err(|e| e.to_string())?;
            if served && q8 {
                dense_vram = open_m.pop.device_bytes();
            }
            drop(open_m);
            let (cand_m, _h1) = masked_model(&open, &owl_union, "owl-chk", q8)?;
            let b = nmengn::base::check_outputs(&cand_m).map_err(|e| e.to_string())?;
            if served && q8 {
                bmq8_vram = cand_m.pop.device_bytes();
            }
            drop(cand_m);
            let d = a.iter().zip(&b).map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs()).fold(0.0_f32, f32::max);
            largest_diff = largest_diff.max(d);
        }
    }
    // For writer, also measure CPU F16 check_outputs parity (used when seating on CPU) and held-out cosine.
    if is_writer {
        nmengn::gpu::HALF_MATH.store(false, std::sync::atomic::Ordering::Relaxed);
        nmengn::gpu::SEGMENT_ATTENTION.store(false, std::sync::atomic::Ordering::Relaxed);
        let cpu_pop = Population::open(home, pop_id, false, false).map_err(|e| e.to_string())?;
        let cpu_open = Model::bind(&cpu_pop, asm.clone()).map_err(|e| e.to_string())?;
        let a = nmengn::base::check_outputs(&cpu_open).map_err(|e| e.to_string())?;
        let (cpu_cand, _hc) = masked_model(&cpu_open, &owl_union, "owl-cpu-chk", false)?;
        let b = nmengn::base::check_outputs(&cpu_cand).map_err(|e| e.to_string())?;
        let d = a.iter().zip(&b).map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs()).fold(0.0_f32, f32::max);
        largest_diff = largest_diff.max(d);
        let (cos, _, _) = eval_held(&cpu_open, &cpu_cand, &held_jobs)?;
        held_cos = cos;
        drop(cpu_cand);
        drop(cpu_open);
        drop(cpu_pop);
    }
    nmengn::gpu::HALF_MATH.store(saved_sw.0, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(saved_sw.1, std::sync::atomic::Ordering::Relaxed);
    let check_bound = (largest_diff * 1.25).max(largest_diff + 1.0);
    // Use the unmodified BF16 assembly for enumerating base tensors.
    for node in &mut asm.nodes {
        if node.op == "linear" {
            node.precision = "f16".to_owned();
        }
    }

    let needed = nmengn::base::read_by(&asm);
    let mut tensors = Vec::with_capacity(needed.len());
    let mut masked_n = 0_usize;
    let mut removed_w = 0_usize;
    let mut total_w = 0_usize;
    for name in &needed {
        let (layer, block) = nmengn::base::address(name).ok_or_else(|| format!("{name}: no layer and block"))?;
        let td = pop.tensors.iter().find(|t| &t.name == name).ok_or_else(|| format!("missing tensor {name}"))?;
        let raw = read_range(&pop.card.dir.join(&td.file), td.start, td.end)?;
        let bytes = if let Some(bm) = owl_union.tensors.get(name) {
            masked_n += 1;
            removed_w += bm.removed_count();
            total_w += bm.total();
            pmap::apply_mask_bf16(&raw, bm)
        } else {
            raw
        };
        tensors.push(nmlearn::library::Tensor {
            name: name.clone(),
            layer,
            block: block.to_owned(),
            dtype: td.dtype.clone(),
            shape: td.shape.clone(),
            bytes,
        });
    }
    #[expect(clippy::cast_precision_loss, reason = "display percentage and MiB")]
    let actual_sp = 100.0 * removed_w as f64 / total_w.max(1) as f64;
    let total_bytes: usize = tensors.iter().map(|t| t.bytes.len()).sum();
    let layers_max = pop.dims.layers.saturating_sub(1) as u64;
    let weights_sha = pop.card.sha_main.clone();
    drop(open);
    drop(pop);

    let mut lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
    let id = format!("{short_role}-base");
    let prev = lib.cartridge(&id).map(|c| (c.key(), c.version));
    let version = if let Some(v) = forced_version { v } else { prev.as_ref().map_or(1, |(_, v)| v.saturating_add(1)) };
    let parent = if version <= 1 {
        "open-weights".to_owned()
    } else {
        format!("{id}@{}", version - 1)
    };
    let domain = match short_role {
        "reranker" => "job:judge-code",
        "writer" => "job:write",
        _ => "job:base",
    };
    let spec = nmlearn::library::CartridgeSpec {
        id: id.clone(),
        version,
        family: String::new(),
        domain: domain.to_owned(),
        scope: "general".to_owned(),
        model: pop_id.to_owned(),
        weights: weights_sha,
        layers: Some((0, layers_max)),
        name: format!("{short_role}, OWL parameter-mapped base cartridge"),
        description: format!(
            "Parameter-level OWL ({actual_sp:.2}% sparse across {masked_n} attention + FFN matrices, pinned relevance heads L12..L19) 16-bit BF16 base cartridge for {pop_id}; auto-compresses on GPU with bmq8.cu."
        ),
        made_from: format!("{pop_id} open weights via `nano-lab run pmap build-base {short_role} {sparsity:.2}`"),
        effect: format!("OWL {actual_sp:.2}% zero-weight sparsity ready for bit-exact bmq8.cu GPU compression"),
    };
    let mut notes = BTreeMap::new();
    notes.insert("content".to_owned(), nmjson::Value::String("base".to_owned()));
    notes.insert("exact".to_owned(), nmjson::Value::Bool(false));
    notes.insert("check_bound".to_owned(), nmjson::Value::Number(format!("{check_bound:e}")));
    notes.insert("parent".to_owned(), nmjson::Value::String(parent));
    notes.insert("mapped".to_owned(), nmjson::Value::String(format!("yes: owl-union target={:.0}% lambda=0.08", sparsity * 100.0)));
    notes.insert("pruned".to_owned(), nmjson::Value::String(format!("yes: {actual_sp:.2}% zeroed across {masked_n} linear matrices (bf16 on disk, bmq8 on GPU)")));
    notes.insert("edits".to_owned(), nmjson::Value::String("none".to_owned()));
    notes.insert("measured".to_owned(), nmjson::Value::String("pending full board benchmark".to_owned()));
    let filter = nmlearn::library::Filter::Tensors(tensors);
    let (key, removed) = if crate::KEEP_BESIDE.load(std::sync::atomic::Ordering::Relaxed) {
        lib.add_base_cartridge(&spec, &filter, nmjson::Value::Object(notes), Vec::new(), false)
    } else {
        lib.replace_base_cartridge(&spec, &filter, nmjson::Value::Object(notes), Vec::new(), false)
    }
    .map_err(|e| e.to_string())?;
    lib.save().map_err(|e| e.to_string())?;
    let mut out = String::new();
    #[expect(clippy::cast_precision_loss, reason = "display MiB")]
    let mib_sz = total_bytes as f64 / (1024.0 * 1024.0);
    let _ = writeln!(out, "=== Single-Copy OWL Base Cartridge Built ===");
    let _ = writeln!(out, "Key           : {key} ({pop_id})");
    let _ = writeln!(out, "Sparsity      : {actual_sp:.2}% across {masked_n} attention+FFN tensors ({mib_sz:.1} MiB BF16 on disk)");
    let _ = writeln!(out, "Check bound   : max_diff={largest_diff:.4} -> label check_bound={check_bound:.4}");
    if dense_vram > 0 && bmq8_vram > 0 {
        #[expect(clippy::cast_precision_loss, reason = "display MiB and percent")]
        let (dm, bm) = (dense_vram as f64 / (1024.0 * 1024.0), bmq8_vram as f64 / (1024.0 * 1024.0));
        let save = 100.0 * (1.0 - bm / dm);
        let _ = writeln!(out, "GPU Q8 VRAM   : {dm:.1} MiB -> {bm:.1} MiB ({save:+.2}% via bmq8.cu)");
    }
    if held_cos > 0.0 {
        let _ = writeln!(out, "Held-out cos  : {held_cos:.5} on writer jobs");
    }
    let _ = writeln!(out, "Replaced files: {removed:?}");
    let _ = writeln!(out, "Parameter map : {}", map_out.display());
    let _ = writeln!(out, "Elapsed       : {:.2} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

/// Live 4-MicroModel GTX 1650 (4 GB VRAM) coexistence benchmark: seats ALL FOUR base cartridges
/// (`reader`, `embedder`, `reranker`, `writer`) simultaneously in Q8 (`bmq8.cu` ON + active-vocab
/// `head_rows` for writer), runs a real forward pass on each while all 4 stay resident, generates
/// a real writer one-line note, and reports exact per-model and total GPU VRAM.
fn coexist(home: &Path) -> Result<String, String> {
    let t0 = Instant::now();
    nmengn::gpu::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
    nmengn::gpu::SEGMENT_ATTENTION.store(true, std::sync::atomic::Ordering::Relaxed);
    let lib = nmlearn::library::Library::open(home).map_err(|e| e.to_string())?;
    let mut seated: Vec<(&'static str, Model)> = Vec::new();
    let specs = [
        ("reader", "qwen3-0.6b", "qwen3-reference", "reader-base"),
        ("embedder", "qwen3-embedding-0.6b", "qwen3-embedding", "embedder-base"),
        ("reranker", "qwen3-reranker-0.6b", "qwen3-reranker", "reranker-base"),
        ("writer", "qwen3-1.7b", "qwen3-reference", "writer-base"),
    ];
    for (role, pop_id, asm_id, cart_id) in specs {
        let pop = Population::open(home, pop_id, false, true).map_err(|e| e.to_string())?;
        let mut asm = Assembly::load(&home.join("designs/assemblies"), asm_id).map_err(|e| e.to_string())?;
        for node in &mut asm.nodes {
            if node.op == "linear" {
                node.precision = "q8".to_owned();
            }
        }
        let open = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let m = if let Some(c) = lib.cartridge(cart_id) {
            let file = lib.cartridge_dir(c).join(c.files.first().map_or("", |f| f.name.as_str()));
            let offer = nmapi::neural::offer_of(&file).map_err(|e| e.to_string())?;
            let (m, _) = nmengn::base::seat(&open, &offer).map_err(|e| format!("{role} seat: {e}"))?;
            m
        } else {
            open
        };
        seated.push((role, m));
    }
    // Run a real GPU forward pass on ALL FOUR models while ALL FOUR remain resident simultaneously!
    let mut out = String::new();
    let _ = writeln!(out, "=== Live 4-MicroModel Simultaneous GTX 1650 Coexistence Proof ===");
    let mut total_bytes = 0_usize;
    let mut writer_note = String::new();
    for (role, m) in &seated {
        let norm_id = m.role_node("final_norm").map_err(|e| e.to_string())?;
        if *role == "writer" {
            let vocab = m.pop.dims.vocab;
            let mut keep: BTreeSet<usize> = (0..16_384.min(vocab)).collect();
            for eos in [151_643_usize, 151_645_usize] {
                if eos < vocab {
                    keep.insert(eos);
                }
            }
            let prompt = "<|im_start|>user\ncrates/nmengn/src/filter.rs lines 1-40:\n```\npub fn compose(parts: &[(&[f32], &[f32])]) -> Result<(Vec<f32>, Vec<f32>), String> {\n    let mut g = vec![1.0_f64; width];\n    let mut b = vec![0.0_f64; width];\n}\n```\nWrite one line of at most 25 words, plain text, no markdown: first what this code or text does (start with a verb; do not name the file or say \"this part\"), then \"Defines:\" and the names it defines (or \"Defines: none\").<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            let ids = m.pop.tokenizer.encode(prompt).map_err(|e| e.to_string())?;
            for &id in &ids {
                if let Ok(u) = usize::try_from(id)
                    && u < vocab
                {
                    keep.insert(u);
                }
            }
            let head_rows: Arc<Vec<usize>> = Arc::new(keep.into_iter().collect());
            let head_target = (m.role_node("head").map_err(|e| e.to_string())?, "logits".to_owned());
            let opts = RunOpts {
                cache: Some(Arc::new(nmengn::model::Cache::default())),
                head_rows: Arc::clone(&head_rows),
                ..RunOpts::default()
            };
            let mut cur = ids;
            let mut at = 0_usize;
            let mut gen_ids: Vec<i32> = Vec::new();
            let tg = Instant::now();
            for _ in 0..32 {
                let res = m.forward_at(Device::Gpu, &cur, at, std::slice::from_ref(&head_target), &opts).map_err(|e| e.to_string())?;
                let logits = values(res.get(&head_target).ok_or("missing writer logits")?).map_err(|e| e.to_string())?;
                let w = head_rows.len();
                let last = logits.get(logits.len().saturating_sub(w)..).unwrap_or(logits);
                let best_i = last.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
                let tok = i32::try_from(head_rows.get(best_i).copied().unwrap_or(0)).unwrap_or(0);
                if tok == 151_643 || tok == 151_645 {
                    break;
                }
                gen_ids.push(tok);
                let s = m.pop.tokenizer.decode(&gen_ids);
                if s.contains('\n') {
                    break;
                }
                at += cur.len();
                cur = vec![tok];
            }
            writer_note = format!("{} ({} tok in {:.2} s)", m.pop.tokenizer.decode(&gen_ids).trim(), gen_ids.len(), tg.elapsed().as_secs_f64());
        } else {
            let ids = m.pop.tokenizer.encode("where is sha256 implemented in this repository").map_err(|e| e.to_string())?;
            let _ = m.forward(Device::Gpu, &ids, &[(norm_id, "y".to_owned())], &RunOpts::default()).map_err(|e| e.to_string())?;
        }
        let b = m.pop.device_bytes();
        total_bytes += b;
        #[expect(clippy::cast_precision_loss, reason = "display MiB")]
        let mib_v = b as f64 / (1024.0 * 1024.0);
        let _ = writeln!(out, "  {role:<10} : {mib_v:>6.1} MiB resident on GPU (live forward OK)");
    }
    #[expect(clippy::cast_precision_loss, reason = "display MiB and percent")]
    let tot_mib = total_bytes as f64 / (1024.0 * 1024.0);
    let _ = writeln!(out, "  TOTAL ALL 4: {tot_mib:>6.1} MiB / 4096.0 MiB ({:.1}% of 4 GB GTX 1650; {:.1} MiB free headroom)", 100.0 * tot_mib / 4096.0, 4096.0 - tot_mib);
    let _ = writeln!(out, "  Live writer note (with all 4 resident): {writer_note}");
    let _ = writeln!(out, "  Elapsed: {:.2} s", t0.elapsed().as_secs_f64());
    Ok(out)
}

pub fn run(args: &[String], home: &Path) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("vocab-profile");
    let res = match sub {
        "vocab-profile" => vocab_profile(home),
        "sweep" => {
            let role = args.get(3).map(String::as_str).unwrap_or("reader");
            let use_gpu = args.get(4).map(String::as_str) != Some("cpu");
            sweep(home, role, use_gpu)
        }
        "build-base" => {
            let role = args.get(3).map(String::as_str).unwrap_or("reader");
            let sparsity = args.get(4).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.40);
            let use_gpu = args.get(5).map(String::as_str) != Some("cpu");
            let forced_v = args.iter().find_map(|a| a.strip_prefix("v=").and_then(|v| v.parse::<u64>().ok()));
            build_base(home, role, sparsity, use_gpu, forced_v)
        }
        "coexist" => coexist(home),
        "surgery" => crate::surgery::run(args, home),
        other => Err(format!("usage: nano-lab run pmap <vocab-profile | sweep | build-base | coexist | surgery>, got {other}")),
    };
    match res {
        Ok(text) => {
            let report = home.join(format!("runs/PMAP-{sub}-{}.txt", nmtype::now_ms()));
            let _ = std::fs::create_dir_all(home.join("runs"));
            let _ = std::fs::write(&report, crate::with_overlay(&text));
            print!("{text}");
            println!("report: {}", report.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("pmap: {e}");
            ExitCode::FAILURE
        }
    }
}
