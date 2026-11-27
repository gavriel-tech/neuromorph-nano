//! Roles over the one base population (law 9; Luis Jake Gabriel III, 23:39 Finnish time, 2026-11-07: nothing
//! fine-tuned): `embed` and `rerank` are our components over the base assembly, placed per job
//! (research note sections 7 and 8).
//! - embed: position-weighted mean pooling of the final normalised hidden states (SGPT,
//!   Muennighoff 2022, arXiv 2202.08904), L2-normalised.
//! - rerank: logit(" yes") minus logit(" no") after a fixed relevance prompt; orders only.
use crate::model::{Model, RunOpts, values};
use crate::placement::{Choice, Device, shape};
use nmtype::{Fault, FaultKind};
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.roles", FaultKind::Execution, m)
}

/// Token cap for one input of the base-model roles `embed` and `rerank` (keeps every call
/// short; longer text is cut at the end). The released role models' inputs (`card_*`) take the
/// derived cap of `input_cap` instead (2026-11-10, teardown patch 3).
pub const MAX_TOKENS: usize = 192;

/// The reference ceiling for one role model input (2026-11-10, teardown patch 3): the
/// `max_length` of 8,192 tokens in the Qwen3-Embedding and Qwen3-Reranker model card code.
pub const REFERENCE_MAX_TOKENS: usize = 8192;

/// The earlier fixed document cap of the released role models (384 tokens), kept for the
/// comparison tests and as the floor of the derived cap (the context a document gets is never
/// below what it had before).
pub const EARLIER_DOC_CAP: usize = MAX_TOKENS * 2;

/// Device (or RAM) bytes one more input token needs while a role model runs: one layer's
/// activations at binary16 (hidden states, queries, keys, values, the gated feed-forward) plus
/// that token's keys and values over every layer (a prefilled head's cache holds them).
fn bytes_per_token(m: &Model) -> u64 {
    let d = m.pop.dims;
    let act = (d.hidden * 6 + d.heads * d.head_dim + 2 * d.kv_heads * d.head_dim + d.inter * 3) * 2;
    let kv = d.layers * 2 * d.kv_heads * d.head_dim * 2;
    (act + kv).max(1) as u64
}

/// The document token cap for one role model input on `device`, derived at run time
/// (2026-11-10, teardown patch 3): the reference ceiling (`REFERENCE_MAX_TOKENS`) bounded by the
/// model's own context (`max_position_embeddings` of its config), minus the `fixed` tokens the
/// prompt adds around the document, and bounded by the memory free now: on the graphics card the
/// free device memory less what this model's weights still need there, on the processor half of
/// the free physical memory (as `kvstore::derive_ram_limit`), divided by `bytes_per_token`. Never
/// below `EARLIER_DOC_CAP` (or the ceiling, if that is smaller). When memory cannot be read the
/// ceiling is used. Roadmap Phase 4.5 (2026-11-25): both figures are the effective ones under the
/// user's limits (`Population::gpu_free` on the card, `limits::host_free_now` on the processor).
#[must_use]
pub fn input_cap(m: &Model, device: Device, fixed: usize) -> usize {
    let ctx = m.pop.card.int("max_position_embeddings").unwrap_or(REFERENCE_MAX_TOKENS);
    let ceiling = REFERENCE_MAX_TOKENS.min(ctx).saturating_sub(fixed);
    let free = match device {
        Device::Gpu => m.pop.gpu_free().map(|f| f.saturating_sub(m.device_need().saturating_sub(m.pop.device_bytes() as u64))),
        Device::Cpu => crate::limits::host_free_now().map(|f| f / 2),
    };
    let by_memory = free.map_or(ceiling, |f| usize::try_from(f / bytes_per_token(m)).unwrap_or(usize::MAX));
    ceiling.min(by_memory).max(EARLIER_DOC_CAP.min(ceiling))
}

/// The device a role model runs on when no placement chooses: the graphics card when one is open.
fn own_device(m: &Model) -> Device {
    if m.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu }
}

/// Run one job where placement says, record its time for that shape, and say where it ran.
fn placed<T>(m: &Model, op: &str, tokens: usize, job: impl Fn(Device) -> Result<T, Fault>) -> Result<(T, Choice, f64), Fault> {
    let s = shape(op, tokens);
    let c = m.pop.placement.choose(&s, m.device_need().saturating_sub(m.pop.device_bytes() as u64), m.pop.gpu_free());
    let t = Instant::now();
    let out = match job(c.device) {
        Ok(v) => v,
        Err(e) if c.device == Device::Gpu => {
            let v = job(Device::Cpu)?;
            return Ok((
                v,
                Choice { device: Device::Cpu, why: format!("GPU job failed ({e}); ran on the CPU") },
                t.elapsed().as_secs_f64() * 1000.0,
            ));
        }
        Err(e) => return Err(e),
    };
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    m.pop.placement.record(&s, c.device, ms);
    Ok((out, c, ms))
}

fn ids(m: &Model, text: &str) -> Result<Vec<i32>, Fault> {
    let mut ids = m.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    ids.truncate(MAX_TOKENS);
    if ids.is_empty() {
        ids.push(0);
    }
    Ok(ids)
}

/// Embed `text` on `device` (no placement): position-weighted mean of the final hidden states.
/// # Errors
/// Tokenizer or engine failure.
pub fn embed_on(m: &Model, device: Device, text: &str) -> Result<Vec<f32>, Fault> {
    let ids = ids(m, text)?;
    let target = (m.role_node("final_norm")?, "y".to_owned());
    let out = m.forward(device, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
    let h = values(out.get(&target).ok_or_else(|| bad("no hidden state"))?)?;
    let d = m.pop.dims.hidden;
    let t = ids.len();
    let mut v = vec![0.0_f32; d];
    #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
    let total = (t * (t + 1) / 2) as f32;
    for (i, row) in h.chunks_exact(d).enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
        let w = (i + 1) as f32 / total;
        for (a, x) in v.iter_mut().zip(row) {
            *a += w * x;
        }
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    Ok(v)
}

/// Embed `text` where placement says.
/// # Errors
/// As `embed_on`.
pub fn embed(m: &Model, text: &str) -> Result<(Vec<f32>, Choice, f64), Fault> {
    let n = ids(m, text)?.len();
    placed(m, "embed", n, |d| embed_on(m, d, text))
}

/// The relevance prompt of the reranker.
#[must_use]
pub fn prompt(query: &str, doc: &str) -> String {
    format!("Query: {query}\nDocument: {doc}\nIs the document relevant to the query? Answer yes or no.\nAnswer:")
}

/// Rerank score on `device`: logit(" yes") - logit(" no") at the last position.
/// # Errors
/// Tokenizer or engine failure.
pub fn rerank_on(m: &Model, device: Device, query: &str, doc: &str) -> Result<f32, Fault> {
    let tok = |s: &str| -> Result<usize, Fault> {
        let v = m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()))?;
        v.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| bad(format!("no token for {s:?}")))
    };
    let (yes, no) = (tok(" yes")?, tok(" no")?);
    let mut d_ids = m.pop.tokenizer.encode(doc).map_err(|e| bad(e.to_string()))?;
    d_ids.truncate(MAX_TOKENS);
    let doc = m.pop.tokenizer.decode(&d_ids);
    let ids = m.pop.tokenizer.encode(&prompt(query, &doc)).map_err(|e| bad(e.to_string()))?;
    let target = (m.role_node("head")?, "logits".to_owned());
    let out = m.forward(device, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
    let l = values(out.get(&target).ok_or_else(|| bad("no logits"))?)?;
    Ok(l.get(yes).copied().unwrap_or(0.0) - l.get(no).copied().unwrap_or(0.0))
}

/// Rerank where placement says.
/// # Errors
/// As `rerank_on`.
pub fn rerank(m: &Model, query: &str, doc: &str) -> Result<(f32, Choice, f64), Fault> {
    let n = ids(m, &prompt(query, doc))?.len();
    placed(m, "rerank", n, |d| rerank_on(m, d, query, doc))
}

/// The reranker model card's default instruction (Qwen3-Reranker-0.6B, 2026-11-09).
pub const CARD_INSTRUCTION: &str = "Given a web search query, retrieve relevant passages that answer the query";
/// Code instructions for the reranker (2026-11-10, teardown patch 1), in the card's
/// "Given ..., ..." form; the card reports 1% to 5% from a task-specific instruction. Test T-E1
/// chooses among `CARD_INSTRUCTION` and these; `CODE_INSTRUCTION` is the one served.
pub const CODE_INSTRUCTIONS: [&str; 2] = [
    "Given a question about a code repository, judge whether the code answers it",
    "Given a question about a code repository, retrieve the code that answers it",
];
/// The reranker instruction the served roles pass (see `CODE_INSTRUCTIONS`).
pub const CODE_INSTRUCTION: &str = CARD_INSTRUCTION;
const CARD_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const CARD_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The reranker card's prompt for one query and document.
#[must_use]
pub fn card_prompt(instruction: &str, query: &str, doc: &str) -> String {
    format!("{CARD_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{CARD_SUFFIX}")
}

/// P(yes) of `doc` for `query` on `device`, read from the assembly's `labelsoftmax` node (the
/// released Qwen3-Reranker-0.6B with the `qwen3-reranker` assembly).
/// # Errors
/// Tokenizer or engine failure, or an assembly without the `label` node.
pub fn card_rerank_on(m: &Model, device: Device, instruction: &str, query: &str, doc: &str) -> Result<f32, Fault> {
    let mut d_ids = m.pop.tokenizer.encode(doc).map_err(|e| bad(e.to_string()))?;
    let fixed = m.pop.tokenizer.encode(&card_prompt(instruction, query, "")).map_err(|e| bad(e.to_string()))?.len();
    d_ids.truncate(input_cap(m, device, fixed));
    let doc = m.pop.tokenizer.decode(&d_ids);
    let ids = m.pop.tokenizer.encode(&card_prompt(instruction, query, &doc)).map_err(|e| bad(e.to_string()))?;
    let target = (m.role_node("label")?, "y".to_owned());
    let out = m.forward(device, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
    values(out.get(&target).ok_or_else(|| bad("no label output"))?)?.first().copied().ok_or_else(|| bad("empty label output"))
}

/// P(yes) where placement says.
/// # Errors
/// As `card_rerank_on`.
pub fn card_rerank(m: &Model, instruction: &str, query: &str, doc: &str) -> Result<(f32, Choice, f64), Fault> {
    let n = ids(m, &card_prompt(instruction, query, doc))?.len();
    placed(m, "rerank", n, |d| card_rerank_on(m, d, instruction, query, doc))
}

/// P(yes) for many documents of one query (2026-11-10): the shared prompt head (system text,
/// instruction, query) is prefilled once into a key and value cache, and each document runs
/// only its own tokens from a fork of that cache. The head is reused only when its tokens are
/// an exact prefix of the full prompt's tokens; otherwise that document runs whole.
/// # Errors
/// Tokenizer or engine failure.
pub fn card_rerank_set(m: &Model, device: Device, instruction: &str, query: &str, docs: &[String]) -> Result<Vec<f32>, Fault> {
    card_rerank_set_capped(m, device, instruction, query, docs, None)
}

/// SN6 (2026-11-13; research file Solution 3, Sensory: "the processor takes 2 to 3 reads at the
/// same time"): score a set on the card and the processor at once. The last `on_cpu` documents
/// are scored on the processor on a second thread while the rest are scored on the card; the
/// scores come back in the set's order. The document cap is derived once for the card's side.
/// # Errors
/// As `card_rerank_set`.
pub fn card_rerank_split(m: &Model, instruction: &str, query: &str, docs: &[String], on_cpu: usize) -> Result<Vec<f32>, Fault> {
    let k = docs.len().saturating_sub(on_cpu.min(docs.len()));
    let (gpu_docs, cpu_docs) = docs.split_at(k);
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let cap = input_cap(m, Device::Gpu, tok(&card_prompt(instruction, query, ""))?.len());
    let (g, c) = std::thread::scope(|s| {
        let h = s.spawn(|| card_rerank_set_capped(m, Device::Cpu, instruction, query, cpu_docs, Some(cap)));
        let g = card_rerank_set_capped(m, Device::Gpu, instruction, query, gpu_docs, Some(cap));
        (g, h.join())
    });
    let mut out = g?;
    out.extend(c.map_err(|_| bad("processor side panicked"))??);
    Ok(out)
}

/// Cached GPU KV prefix for the constant system + instruction header of Qwen3-Reranker (`J1`,
/// WP-A1): ~55 constant tokens prefilled once per process (~6 MiB VRAM) and forked O(1) across
/// queries, invalidating automatically if the population, base source, or filter generation
/// changes. Disable with `NANO_PREFIX_CACHE=off` for paired control comparisons.
struct PrefixKvEntry {
    pop: std::sync::Weak<crate::model::Population>,
    source: String,
    filter_gen: u64,
    tokens: Vec<i32>,
    cache: std::sync::Arc<crate::model::Cache>,
}

static PREFIX_KV: std::sync::Mutex<Option<PrefixKvEntry>> = std::sync::Mutex::new(None);

fn prefix_cache_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("NANO_PREFIX_CACHE").as_deref(), Ok("off" | "0" | "false")))
}

/// Clears any cached reranker system-prefix KV state (e.g., if VRAM must be reclaimed).
pub fn clear_prefix_cache() {
    if let Ok(mut g) = PREFIX_KV.lock() {
        *g = None;
    }
}

/// Target node needed to populate every layer's KV cache without wasting final-layer FFN/norm work.
fn kv_prefill_target(m: &Model) -> Result<(String, String), Fault> {
    if m.pop.dims.layers > 0 {
        let last_score = crate::assembly::nid(&format!("L{}.score", m.pop.dims.layers - 1));
        if m.asm.node(&last_score).is_some() {
            return Ok((last_score, "y".to_owned()));
        }
    }
    Ok((m.role_node("final_norm")?, "y".to_owned()))
}

/// Prefills the shared reranker prompt head (`CARD_PREFIX + instruction + query + <Document>:`)
/// into a `Cache`, reusing the cached constant system+instruction KV prefix on GPU when valid.
fn prefill_rerank_head(m: &Model, device: Device, instruction: &str, head: &[i32]) -> Result<std::sync::Arc<crate::model::Cache>, Fault> {
    let stop = kv_prefill_target(m)?;
    // 2026-11-24: also with cartridge filters plugged. The stored prefix is keyed on the filter
    // generation (rack switches and job-filter sets both move it) and was computed under the same
    // filters; skipping it under filters gave a second numeric path, so a zero-effect cartridge
    // moved the reranker's P(yes) (0.115 to 0.117; nmcore tests/library.rs:431).
    if device == Device::Gpu && prefix_cache_on() {
        let sys_text = format!("{CARD_PREFIX}<Instruct>: {instruction}\n<Query>:");
        if let Ok(sys_ids) = m.pop.tokenizer.encode(&sys_text)
            && !sys_ids.is_empty()
            && head.len() > sys_ids.len()
            && head.starts_with(&sys_ids)
        {
            let fgen = crate::filter::generation();
            let pop_weak = std::sync::Arc::downgrade(&m.pop);
            let cached = PREFIX_KV.lock().ok().and_then(|g| {
                g.as_ref()
                    .filter(|e| {
                        e.pop.strong_count() > 0
                            && std::sync::Weak::ptr_eq(&e.pop, &pop_weak)
                            && e.source == m.pop.source
                            && e.filter_gen == fgen
                            && e.tokens == sys_ids
                    })
                    .map(|e| std::sync::Arc::clone(&e.cache))
            });
            let sys_cache = match cached {
                Some(c) => c,
                None => {
                    let c = std::sync::Arc::new(crate::model::Cache::default());
                    let pre = RunOpts { cache: Some(std::sync::Arc::clone(&c)), ..RunOpts::default() };
                    m.forward_at(device, &sys_ids, 0, std::slice::from_ref(&stop), &pre)?;
                    if let Ok(mut g) = PREFIX_KV.lock() {
                        *g = Some(PrefixKvEntry {
                            pop: pop_weak,
                            source: m.pop.source.clone(),
                            filter_gen: fgen,
                            tokens: sys_ids.clone(),
                            cache: std::sync::Arc::clone(&c),
                        });
                    }
                    c
                }
            };
            let base = std::sync::Arc::new(sys_cache.fork());
            let rest = head.get(sys_ids.len()..).unwrap_or(&[]);
            let pre = RunOpts { cache: Some(std::sync::Arc::clone(&base)), ..RunOpts::default() };
            m.forward_at(device, rest, sys_ids.len(), std::slice::from_ref(&stop), &pre)?;
            return Ok(base);
        }
    }
    let base = std::sync::Arc::new(crate::model::Cache::default());
    let pre = RunOpts { cache: Some(std::sync::Arc::clone(&base)), ..RunOpts::default() };
    m.forward_at(device, head, 0, std::slice::from_ref(&stop), &pre)?;
    Ok(base)
}

/// `card_rerank_set` with an explicit document cap (`None`: derived once for the set by
/// `input_cap`, the prompt's head and suffix counted as fixed). For comparison tests.
/// # Errors
/// Tokenizer or engine failure.
pub fn card_rerank_set_capped(
    m: &Model,
    device: Device,
    instruction: &str,
    query: &str,
    docs: &[String],
    cap: Option<usize>,
) -> Result<Vec<f32>, Fault> {
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let head = tok(&format!("{CARD_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>:"))?;
    let cap = match cap {
        Some(c) => c,
        None => input_cap(m, device, tok(&card_prompt(instruction, query, ""))?.len()),
    };
    let label = (m.role_node("label")?, "y".to_owned());
    let base = prefill_rerank_head(m, device, instruction, &head)?;
    let mut out = Vec::with_capacity(docs.len());
    for d in docs {
        let mut d_ids = tok(d)?;
        d_ids.truncate(cap);
        let doc = m.pop.tokenizer.decode(&d_ids);
        let full = tok(&card_prompt(instruction, query, &doc))?;
        let p = if full.len() > head.len() && full.starts_with(&head) {
            let fork = RunOpts { cache: Some(std::sync::Arc::new(base.fork())), ..RunOpts::default() };
            let r = m.forward_at(device, full.get(head.len()..).unwrap_or(&[]), head.len(), std::slice::from_ref(&label), &fork)?;
            values(r.get(&label).ok_or_else(|| bad("no label output"))?)?.first().copied()
        } else {
            let r = m.forward(device, &full, std::slice::from_ref(&label), &RunOpts::default())?;
            values(r.get(&label).ok_or_else(|| bad("no label output"))?)?.first().copied()
        };
        out.push(p.ok_or_else(|| bad("empty label output"))?);
    }
    Ok(out)
}

/// The document cap `card_rerank_set_capped` derives for `query` on `device` now (`input_cap` with
/// the card prompt's own tokens as fixed): it follows the memory free at this moment.
/// # Errors
/// Tokenizer failure.
pub fn card_doc_cap(m: &Model, device: Device, instruction: &str, query: &str) -> Result<usize, Fault> {
    let fixed = m.pop.tokenizer.encode(&card_prompt(instruction, query, "")).map_err(|e| bad(e.to_string()))?.len();
    Ok(input_cap(m, device, fixed))
}

/// Score sets in batches on the card (`card_rerank_batch`); off gives the one-at-a-time path for comparison.
pub static RERANK_BATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Most tokens of one batched reranker forward when card memory cannot be read.
pub const BATCH_TOKENS_FALLBACK: usize = 4096;

/// Tokens one batched forward of `m` may hold on the card now: free memory less what the model's
/// weights still need there, half of it kept back, divided by `bytes_per_token`; at least one
/// document's worth (`floor`), at most twice the reference context.
#[must_use]
pub fn batch_tokens(m: &Model, floor: usize) -> usize {
    let free = m.pop.gpu_free().map(|f| f.saturating_sub(m.device_need().saturating_sub(m.pop.device_bytes() as u64)));
    let by_memory = free.map_or(BATCH_TOKENS_FALLBACK, |f| usize::try_from(f / 2 / bytes_per_token(m)).unwrap_or(usize::MAX));
    by_memory.min(REFERENCE_MAX_TOKENS * 2).max(floor)
}

/// The judge's batched read gives its int8 linears of 64 tokens or more the half products of
/// `wide_h2q8_kernel` (when half math is on), as the second look does; off keeps the widening
/// int8 kernels.
pub static JUDGE_Q8_HALF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// P(yes) for many documents of one query in a few forwards on the card: the shared prompt
/// head is prefilled once, then the documents' own tokens are packed side by side
/// (`RunOpts::segments`), each reading the head as its prefix and attending only to itself, in
/// batches of at most `batch_tokens` tokens. Each score equals the one-at-a-time score of
/// `card_rerank_set_capped` within binary16 rounding. A document whose prompt tokens do not start
/// with the head's runs whole, alone. Off the card, or with `RERANK_BATCH` off, the
/// one-at-a-time path runs.
/// # Errors
/// Tokenizer or engine failure.
pub fn card_rerank_batch(
    m: &Model,
    device: Device,
    instruction: &str,
    query: &str,
    docs: &[String],
    cap: Option<usize>,
) -> Result<Vec<f32>, Fault> {
    if device != Device::Gpu || !RERANK_BATCH.load(std::sync::atomic::Ordering::Relaxed) || docs.len() < 2 {
        return card_rerank_set_capped(m, device, instruction, query, docs, cap);
    }
    rerank_packed(m, device, instruction, query, docs, cap, false)
}

/// P(yes) for pairs of (query, document) with different queries, packed side by side on the card
/// in as few forwards as `batch_tokens` allows: each pair's whole card prompt is its own sequence
/// (`RunOpts::segments` with no shared prefix), so the posed question and the content-free forms
/// of every contested piece go in one read. Each score equals the pair's one-at-a-time score
/// within binary16 rounding. Off the card, one pair at a time.
/// # Errors
/// Tokenizer or engine failure.
pub fn card_rerank_pairs(m: &Model, device: Device, instruction: &str, pairs: &[(&str, &str)]) -> Result<Vec<f32>, Fault> {
    let _q8 = JUDGE_Q8_HALF.load(std::sync::atomic::Ordering::Relaxed).then(crate::gpu::Q8HalfScope::new);
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let label = (m.role_node("label")?, "y".to_owned());
    let mut prompts: Vec<(usize, Vec<i32>)> = Vec::with_capacity(pairs.len());
    for (i, (q, d)) in pairs.iter().enumerate() {
        let cap = input_cap(m, device, tok(&card_prompt(instruction, q, ""))?.len());
        let mut d_ids = tok(d)?;
        d_ids.truncate(cap);
        prompts.push((i, tok(&card_prompt(instruction, q, &m.pop.tokenizer.decode(&d_ids)))?));
    }
    let mut out = vec![f32::NAN; pairs.len()];
    if device != Device::Gpu {
        for (i, ids) in &prompts {
            let r = m.forward(device, ids, std::slice::from_ref(&label), &RunOpts::default())?;
            let p = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?.first().copied();
            if let Some(x) = out.get_mut(*i) {
                *x = p.ok_or_else(|| bad("empty label output"))?;
            }
        }
        return Ok(out);
    }
    prompts.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    let limit = batch_tokens(m, prompts.first().map_or(1, |t| t.1.len()));
    SCORED.fetch_add(prompts.len(), std::sync::atomic::Ordering::Relaxed);
    let mut at = 0;
    while at < prompts.len() {
        let (mut segs, mut packed, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        while let Some((i, t)) = prompts.get(at) {
            if !segs.is_empty() && packed.len() + t.len() > limit {
                break;
            }
            segs.push(t.len());
            packed.extend_from_slice(t);
            idx.push(*i);
            at += 1;
        }
        let opts = RunOpts { segments: Some(std::sync::Arc::new(segs)), ..RunOpts::default() };
        let r = m.forward_at(device, &packed, 0, std::slice::from_ref(&label), &opts)?;
        let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
        let per = v.len() / idx.len().max(1);
        for (k, i) in idx.iter().enumerate() {
            if let Some(x) = out.get_mut(*i) {
                *x = v.get(k * per).copied().ok_or_else(|| bad("short label output"))?;
            }
        }
    }
    Ok(out)
}

/// The judge's z (the head's yes logit minus its no logit, which P(yes) is the logistic of) for
/// many documents of one query, packed on the card exactly as `card_rerank_batch` packs them: the
/// same arithmetic read before the label softmax, so it never saturates at 1.0. For measurement.
/// # Errors
/// Tokenizer or engine failure, or a device other than the card.
pub fn card_rerank_batch_z(m: &Model, instruction: &str, query: &str, docs: &[String], cap: Option<usize>) -> Result<Vec<f32>, Fault> {
    rerank_packed(m, Device::Gpu, instruction, query, docs, cap, true)
}

/// A document's last hidden state from the judge's packed run, when the run gives it.
pub type LastState = Option<Vec<f32>>;

/// `card_rerank_batch_z` that also returns each document's last hidden state (the final norm's
/// row at its last token) from the same packed run (lane T1, 2026-11-19, speed: lane B's code
/// read-out needs that state, and reading it by a second unpacked forward per document cost about
/// as much as the judge's whole read). A state the run cannot give (an output of another shape) is
/// `None`; the caller reads it alone then.
/// # Errors
/// Tokenizer or engine failure.
#[expect(clippy::too_many_lines, reason = "the prefill, the documents and the packed batches, in order")]
pub fn card_rerank_batch_z_states(m: &Model, instruction: &str, query: &str, docs: &[String], cap: Option<usize>) -> Result<(Vec<f32>, Vec<LastState>), Fault> {
    let device = Device::Gpu;
    let _q8 = JUDGE_Q8_HALF.load(std::sync::atomic::Ordering::Relaxed).then(crate::gpu::Q8HalfScope::new);
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let head = tok(&format!("{CARD_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>:"))?;
    let cap = match cap {
        Some(c) => c,
        None => input_cap(m, device, tok(&card_prompt(instruction, query, ""))?.len()),
    };
    let hidden = m.pop.dims.hidden;
    let label = (m.role_node("head")?, "logits".to_owned());
    let fin = (m.role_node("final_norm")?, "y".to_owned());
    let targets = [label.clone(), fin.clone()];
    let base = prefill_rerank_head(m, device, instruction, &head)?;
    let mut out = vec![f32::NAN; docs.len()];
    let mut states: Vec<Option<Vec<f32>>> = vec![None; docs.len()];
    let mut tails: Vec<(usize, Vec<i32>)> = Vec::with_capacity(docs.len());
    for (i, d) in docs.iter().enumerate() {
        let mut d_ids = tok(d)?;
        d_ids.truncate(cap);
        let doc = m.pop.tokenizer.decode(&d_ids);
        let full = tok(&card_prompt(instruction, query, &doc))?;
        match full.get(head.len()..).filter(|t| !t.is_empty() && full.starts_with(&head)) {
            Some(t) => tails.push((i, t.to_vec())),
            None => {
                let r = m.forward(device, &full, &targets, &RunOpts { last_rows_only: true, ..RunOpts::default() })?;
                let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
                if let Some(x) = out.get_mut(i) {
                    *x = match (v.first(), v.get(1)) {
                        (Some(a), Some(b)) => a - b,
                        _ => return Err(bad("empty label output")),
                    };
                }
                let y = r.get(&fin).and_then(|a| values(a).ok()).unwrap_or(&[]);
                if let Some(st) = states.get_mut(i) {
                    *st = y.get(y.len().saturating_sub(hidden)..).filter(|h| h.len() == hidden && hidden > 0).map(<[f32]>::to_vec);
                }
            }
        }
    }
    tails.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    let limit = batch_tokens(m, tails.first().map_or(1, |t| t.1.len()));
    SCORED.fetch_add(tails.len(), std::sync::atomic::Ordering::Relaxed);
    let mut at = 0;
    while at < tails.len() {
        let (mut segs, mut packed, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        while let Some((i, t)) = tails.get(at) {
            if !segs.is_empty() && packed.len() + t.len() > limit {
                break;
            }
            segs.push(t.len());
            packed.extend_from_slice(t);
            idx.push(*i);
            at += 1;
        }
        let ends: Vec<usize> = segs
            .iter()
            .scan(0, |e, n| {
                *e += n;
                Some(*e)
            })
            .collect();
        let opts = RunOpts {
            cache: Some(std::sync::Arc::clone(&base)),
            segments: Some(std::sync::Arc::new(segs)),
            last_rows_only: true,
            ..RunOpts::default()
        };
        let r = m.forward_at(device, &packed, head.len(), &targets, &opts)?;
        let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
        let per = v.len() / idx.len().max(1);
        let y = r.get(&fin).and_then(|a| values(a).ok()).unwrap_or(&[]);
        for (k, i) in idx.iter().enumerate() {
            if let Some(x) = out.get_mut(*i) {
                *x = match (v.get(k * per), v.get(k * per + 1)) {
                    (Some(a), Some(b)) if per >= 2 => a - b,
                    _ => return Err(bad("short label output")),
                };
            }
            // The final norm's rows: every packed token (each document's last is taken), or one
            // row per document; any other shape gives no state.
            let row = if hidden > 0 && y.len() == packed.len() * hidden {
                ends.get(k).and_then(|e| y.get((e - 1) * hidden..e * hidden))
            } else if hidden > 0 && y.len() == idx.len() * hidden {
                y.get(k * hidden..(k + 1) * hidden)
            } else {
                None
            };
            if let Some(st) = states.get_mut(*i) {
                *st = row.map(<[f32]>::to_vec);
            }
        }
    }
    Ok((out, states))
}

#[expect(clippy::too_many_lines, reason = "the prefill, the documents and the packed batches, in order")]
#[expect(clippy::too_many_arguments, reason = "card_rerank_batch's arguments and which output to read")]
fn rerank_packed(m: &Model, device: Device, instruction: &str, query: &str, docs: &[String], cap: Option<usize>, z: bool) -> Result<Vec<f32>, Fault> {
    // A judge bound at 8 bits takes half products over its int8 weights; 16-bit linears are untouched.
    let _q8 = JUDGE_Q8_HALF.load(std::sync::atomic::Ordering::Relaxed).then(crate::gpu::Q8HalfScope::new);
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
    let head = tok(&format!("{CARD_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>:"))?;
    let cap = match cap {
        Some(c) => c,
        None => input_cap(m, device, tok(&card_prompt(instruction, query, ""))?.len()),
    };
    let label = if z { (m.role_node("head")?, "logits".to_owned()) } else { (m.role_node("label")?, "y".to_owned()) };
    // P(yes) is the label's first value; z is the head's first logit less its second.
    let read = |v: &[f32], k: usize, per: usize| -> Option<f32> {
        if z { Some(v.get(k * per)? - v.get(k * per + 1)?) } else { v.get(k * per).copied() }
    };
    let base = prefill_rerank_head(m, device, instruction, &head)?;
    let mut out = vec![f32::NAN; docs.len()];
    let mut tails: Vec<(usize, Vec<i32>)> = Vec::with_capacity(docs.len());
    for (i, d) in docs.iter().enumerate() {
        let mut d_ids = tok(d)?;
        d_ids.truncate(cap);
        let doc = m.pop.tokenizer.decode(&d_ids);
        let full = tok(&card_prompt(instruction, query, &doc))?;
        match full.get(head.len()..).filter(|t| !t.is_empty() && full.starts_with(&head)) {
            Some(t) => tails.push((i, t.to_vec())),
            None => {
                let r = m.forward(device, &full, std::slice::from_ref(&label), &RunOpts::default())?;
                let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
                let per = if z { v.len() } else { 1 };
                let p = read(v, 0, per).filter(|_| !z || per >= 2);
                if let Some(x) = out.get_mut(i) {
                    *x = p.ok_or_else(|| bad("empty label output"))?;
                }
            }
        }
    }
    // Longest first, so each batch packs documents of similar length.
    tails.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    let limit = batch_tokens(m, tails.first().map_or(1, |t| t.1.len()));
    SCORED.fetch_add(tails.len(), std::sync::atomic::Ordering::Relaxed);
    let drop = RERANK_DROP.lock().ok().and_then(|d| *d).filter(|d| d.layer + 1 < m.pop.dims.layers && tails.len() > d.keep);
    if let Some(d) = drop {
        return drop_weak(m, &head, &base, &tails, limit, d, z, out);
    }
    let mut at = 0;
    while at < tails.len() {
        let (mut segs, mut packed, mut idx) = (Vec::new(), Vec::new(), Vec::new());
        while let Some((i, t)) = tails.get(at) {
            if !segs.is_empty() && packed.len() + t.len() > limit {
                break;
            }
            segs.push(t.len());
            packed.extend_from_slice(t);
            idx.push(*i);
            at += 1;
        }
        let opts = RunOpts {
            cache: Some(std::sync::Arc::clone(&base)),
            segments: Some(std::sync::Arc::new(segs)),
            last_rows_only: true,
            ..RunOpts::default()
        };
        let r = m.forward_at(device, &packed, head.len(), std::slice::from_ref(&label), &opts)?;
        let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
        let per = v.len() / idx.len().max(1);
        for (k, i) in idx.iter().enumerate() {
            if let Some(x) = out.get_mut(*i) {
                *x = read(v, k, per).ok_or_else(|| bad("short label output"))?;
            }
        }
    }
    Ok(out)
}

/// Early drop inside the batched reranker (training-free, after GRATING, arXiv 2510.15620):
/// after `layer`, the output head reads each document's last token as it stands (final norm
/// and label rows over that layer's residual), and documents whose P(yes) there is more than
/// `margin` log-odds below the best are not run through the remaining layers; at least `keep`
/// documents always finish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyDrop {
    pub layer: usize,
    pub margin: f32,
    pub keep: usize,
}

/// Early drop setting of the batched reranker; `None` (the default) finishes every document.
pub static RERANK_DROP: std::sync::Mutex<Option<EarlyDrop>> = std::sync::Mutex::new(None);
/// True when `--rerank-drop` was explicitly given on the command line (so the control in
/// `settings.json` does not overwrite it).
pub static RERANK_DROP_CLI: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Documents the batched reranker took in, and those early drop stopped at the middle layer.
pub static SCORED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static DROPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn log_odds(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// `card_rerank_batch` with early drop: every document runs to `d.layer` (in the same batches),
/// the head reads it there, the weak are dropped and the rest are packed again and finish from
/// that layer. A dropped document's score is its middle-layer P(yes), kept strictly below every
/// finished document's, so it never outranks one.
#[expect(clippy::too_many_lines, reason = "the two phases of one batched run")]
#[expect(clippy::too_many_arguments, reason = "the batch as card_rerank_batch prepared it, the setting and the scores so far")]
fn drop_weak(
    m: &Model,
    head: &[i32],
    base: &std::sync::Arc<crate::model::Cache>,
    tails: &[(usize, Vec<i32>)],
    limit: usize,
    d: EarlyDrop,
    z: bool,
    mut out: Vec<f32>,
) -> Result<Vec<f32>, Fault> {
    use crate::model::Act;
    let o = m.pop.gpu.as_ref().ok_or_else(|| bad("early drop: no GPU"))?;
    let mid = (crate::assembly::nid(&format!("L{}.res2", d.layer)), "y".to_owned());
    let last = (crate::assembly::nid(&format!("L{}.res2", m.pop.dims.layers - 1)), "y".to_owned());
    let pos_id = crate::assembly::nid("pos");
    let label = if z { (m.role_node("head")?, "logits".to_owned()) } else { (m.role_node("label")?, "y".to_owned()) };
    let hidden = m.pop.dims.hidden;
    let start = head.len();
    let segs_of = |idx: &[usize]| std::sync::Arc::new(idx.iter().filter_map(|k| tails.get(*k).map(|t| t.1.len())).collect::<Vec<_>>());
    let pos_of = |idx: &[usize]| -> Vec<i32> {
        idx.iter().filter_map(|k| tails.get(*k)).flat_map(|t| (start..start + t.1.len()).map(|p| i32::try_from(p).unwrap_or(0))).collect()
    };
    let probs = |r: &std::collections::BTreeMap<(String, String), Act>, n: usize| -> Result<Vec<f32>, Fault> {
        let v = values(r.get(&label).ok_or_else(|| bad("no label output"))?)?;
        let per = v.len() / n.max(1);
        (0..n)
            .map(|k| {
                let s = if z {
                    v.get(k * per).and_then(|a| v.get(k * per + 1).map(|b| *a - *b))
                } else {
                    v.get(k * per).copied()
                };
                s.ok_or_else(|| bad("short label output"))
            })
            .collect()
    };
    let to_logit = |s: f32| if z { s } else { log_odds(s) };
    // Phase one: every document to the middle layer, the head's reading there.
    let mut batches: Vec<(Vec<usize>, crate::gpu::DevT)> = Vec::new();
    let mut probe = vec![0.0_f32; tails.len()];
    let mut at = 0;
    while at < tails.len() {
        let (mut idx, mut packed) = (Vec::new(), Vec::new());
        while let Some((_, t)) = tails.get(at) {
            if !idx.is_empty() && packed.len() + t.len() > limit {
                break;
            }
            packed.extend_from_slice(t);
            idx.push(at);
            at += 1;
        }
        let opts =
            RunOpts { cache: Some(std::sync::Arc::clone(base)), segments: Some(segs_of(&idx)), device_targets: true, ..RunOpts::default() };
        let r = m.forward_at(Device::Gpu, &packed, start, std::slice::from_ref(&mid), &opts)?;
        let Some(Act::Gpu(x)) = r.get(&mid).cloned() else { return Err(bad("early drop: no middle-layer output")) };
        let last_x = o.alloc(vec![idx.len(), hidden], 2).map_err(bad)?;
        let mut end = 0_usize;
        for (r, k) in idx.iter().enumerate() {
            end += tails.get(*k).map_or(0, |t| t.1.len());
            o.gpu.copy(&x.buf, (end - 1) * hidden * 2, &last_x.buf, r * hidden * 2, hidden * 2).map_err(bad)?;
        }
        let lens = RunOpts { segments: Some(std::sync::Arc::new(vec![1; idx.len()])), ..RunOpts::default() };
        let p = m.run_from(
            Device::Gpu,
            vec![(last.0.clone(), last.1.clone(), Act::Gpu(last_x))],
            std::slice::from_ref(&label),
            idx.len(),
            &lens,
        )?;
        for (k, v) in idx.iter().zip(probs(&p, idx.len())?) {
            if let Some(slot) = probe.get_mut(*k) {
                *slot = v;
            }
        }
        batches.push((idx, x));
    }
    // Which finish: within the margin of the best, and always the `keep` best.
    let best = probe.iter().copied().map(to_logit).fold(f32::NEG_INFINITY, f32::max);
    let mut by_probe: Vec<usize> = (0..tails.len()).collect();
    by_probe.sort_by(|a, b| probe[*b].total_cmp(&probe[*a]).then(a.cmp(b)));
    let finish: std::collections::BTreeSet<usize> = by_probe
        .iter()
        .enumerate()
        .filter(|(rank, k)| *rank < d.keep || to_logit(probe[**k]) >= best - d.margin)
        .map(|(_, k)| *k)
        .collect();
    DROPPED.fetch_add(tails.len() - finish.len(), std::sync::atomic::Ordering::Relaxed);
    // Phase two: the finishing documents' rows of the middle layer, packed again.
    let mut rows: Vec<(usize, crate::gpu::DevT, usize)> = Vec::new();
    for (idx, x) in &batches {
        let mut off = 0;
        for k in idx {
            let n = tails.get(*k).map_or(0, |t| t.1.len());
            if finish.contains(k) {
                rows.push((*k, x.clone(), off));
            }
            off += n;
        }
    }
    let mut final_score: Vec<(usize, f32)> = Vec::new();
    let mut i = 0;
    while i < rows.len() {
        let (mut idx, mut n) = (Vec::new(), 0_usize);
        while let Some((k, _, _)) = rows.get(i + idx.len()) {
            let len = tails.get(*k).map_or(0, |t| t.1.len());
            if !idx.is_empty() && n + len > limit {
                break;
            }
            idx.push(*k);
            n += len;
        }
        let x = o.alloc(vec![n, hidden], 2).map_err(bad)?;
        let mut at = 0;
        for (k, src, off) in rows.get(i..i + idx.len()).unwrap_or(&[]) {
            let take = tails.get(*k).map_or(0, |t| t.1.len());
            o.gpu.copy(&src.buf, off * hidden * 2, &x.buf, at * hidden * 2, take * hidden * 2).map_err(bad)?;
            at += take;
        }
        let seeds = vec![(mid.0.clone(), mid.1.clone(), Act::Gpu(x)), (pos_id.clone(), "y".to_owned(), Act::Ids(pos_of(&idx)))];
        let opts = RunOpts { cache: Some(std::sync::Arc::clone(base)), segments: Some(segs_of(&idx)), ..RunOpts::default() };
        let r = m.run_from(Device::Gpu, seeds, std::slice::from_ref(&label), n, &opts)?;
        final_score.extend(idx.iter().copied().zip(probs(&r, idx.len())?));
        i += idx.len();
    }
    let floor = final_score.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
    for (k, v) in &final_score {
        if let Some(slot) = tails.get(*k).and_then(|t| out.get_mut(t.0)) {
            *slot = *v;
        }
    }
    let probe_max = probe.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probe_min = probe.iter().copied().fold(f32::INFINITY, f32::min);
    let probe_span = (probe_max - probe_min).max(1.0);
    for (k, p) in probe.iter().enumerate().filter(|(k, _)| !finish.contains(k)) {
        if let Some(slot) = tails.get(k).and_then(|t| out.get_mut(t.0)) {
            // Below every finished document, in the order of the middle-layer reading.
            *slot = if z {
                floor - 1.0 - (probe_max - *p) / probe_span
            } else {
                (floor * 0.999).min(*p * floor * 0.999)
            };
        }
    }
    Ok(out)
}

/// The embedding model card's query text (`Instruct: {task}\nQuery:{query}`).
#[must_use]
pub fn card_query(task: &str, query: &str) -> String {
    format!("Instruct: {task}\nQuery:{query}")
}

/// The token ids the embedding model reads for `text`: its tokens, then `<|endoftext|>`.
/// # Errors
/// Tokenizer failure, or `<|endoftext|>` not being one token.
pub fn card_embed_ids(m: &Model, text: &str) -> Result<Vec<i32>, Fault> {
    card_embed_ids_capped(m, text, None)
}

/// `card_embed_ids` with an explicit cap on the text's tokens (`None`: `input_cap` on the
/// model's own device, the end token counted as fixed). For comparison tests.
/// # Errors
/// As `card_embed_ids`.
pub fn card_embed_ids_capped(m: &Model, text: &str, cap: Option<usize>) -> Result<Vec<i32>, Fault> {
    let mut ids = m.pop.tokenizer.encode(text).map_err(|e| bad(e.to_string()))?;
    ids.truncate(cap.unwrap_or_else(|| input_cap(m, own_device(m), 1)));
    match m.pop.tokenizer.encode("<|endoftext|>").map_err(|e| bad(e.to_string()))?.as_slice() {
        [end] => ids.push(*end),
        _ => return Err(bad("<|endoftext|> is not one token")),
    }
    Ok(ids)
}

/// The L2-normalised embedding of `text` on `device`, read from the assembly's `lastpool` node
/// (the released Qwen3-Embedding-0.6B with the `qwen3-embedding` assembly).
/// # Errors
/// Tokenizer or engine failure, or an assembly without the `pool` node.
pub fn card_embed_on(m: &Model, device: Device, text: &str) -> Result<Vec<f32>, Fault> {
    card_embed_on_capped(m, device, text, None)
}

/// `card_embed_on` with an explicit cap (`None`: derived on `device`). For comparison tests.
/// # Errors
/// As `card_embed_on`.
pub fn card_embed_on_capped(m: &Model, device: Device, text: &str, cap: Option<usize>) -> Result<Vec<f32>, Fault> {
    let ids = card_embed_ids_capped(m, text, Some(cap.unwrap_or_else(|| input_cap(m, device, 1))))?;
    let target = (m.role_node("pool")?, "y".to_owned());
    let out = m.forward(device, &ids, std::slice::from_ref(&target), &RunOpts::default())?;
    values(out.get(&target).ok_or_else(|| bad("no pool output"))?).map(<[f32]>::to_vec)
}

/// Batched embedding (2026-11-13): the L2-normalised embeddings of `texts` (each capped at
/// `cap` tokens plus the end token), packed into forwards of at most `max_tokens` tokens on the
/// GPU (`RunOpts::segments`: positions restart per text, attention within each text, one pooled
/// vector per text). The same vectors as `card_embed_on_capped` one at a time, within binary16
/// rounding (checked by `run cart eb`).
/// # Errors
/// Tokenizer or engine failure, or no GPU.
pub fn card_embed_batch(m: &Model, texts: &[String], cap: usize, max_tokens: usize) -> Result<Vec<Vec<f32>>, Fault> {
    let ids: Vec<Vec<i32>> = texts.iter().map(|t| card_embed_ids_capped(m, t, Some(cap))).collect::<Result<_, _>>()?;
    let target = (m.role_node("pool")?, "y".to_owned());
    let hidden = m.pop.dims.hidden;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    let mut i = 0;
    while i < ids.len() {
        let mut segs = Vec::new();
        let mut packed: Vec<i32> = Vec::new();
        while i < ids.len() && (segs.is_empty() || packed.len() + ids[i].len() <= max_tokens) {
            segs.push(ids[i].len());
            packed.extend_from_slice(&ids[i]);
            i += 1;
        }
        let opts = RunOpts { segments: Some(std::sync::Arc::new(segs.clone())), ..RunOpts::default() };
        let r = m.forward(Device::Gpu, &packed, std::slice::from_ref(&target), &opts)?;
        let v = values(r.get(&target).ok_or_else(|| bad("no pool output"))?)?;
        for k in 0..segs.len() {
            out.push(v.get(k * hidden..(k + 1) * hidden).map(<[f32]>::to_vec).unwrap_or_default());
        }
    }
    Ok(out)
}

/// The embedding where placement says.
/// # Errors
/// As `card_embed_on`.
pub fn card_embed(m: &Model, text: &str) -> Result<(Vec<f32>, Choice, f64), Fault> {
    let n = card_embed_ids(m, text)?.len();
    placed(m, "embed", n, |d| card_embed_on(m, d, text))
}

/// Write by the card's sampling settings (2026-11-12): as `write`, decoded with `s`
/// (`decode::Sampling::from_card`) in place of greedy.
/// # Errors
/// Tokenizer or engine failure.
pub fn write_sampled(m: &Model, prompt: &str, max_new: usize, s: &crate::decode::Sampling) -> Result<(String, Choice, f64), Fault> {
    let ids = m.pop.tokenizer.encode(prompt).map_err(|e| bad(e.to_string()))?;
    let tok = &m.pop.tokenizer;
    let stop = |t: i32| t == 151_643 || t == 151_645 || tok.decode(&[t]).contains('\n');
    let (g, c, ms) = placed(m, "write", ids.len(), |d| crate::decode::sample_until(m, d, &ids, max_new, &stop, s))?;
    let text = tok.decode(&g.tokens);
    let first = text.split('\n').next().unwrap_or("").to_owned();
    Ok((first, c, ms))
}

/// The code line of a chat reply: the first non-empty line inside its first code block, or, with no
/// code block, its first non-empty line.
#[must_use]
pub fn first_code_line(reply: &str) -> String {
    let lines: Vec<&str> = reply.lines().collect();
    if let Some(open) = lines.iter().position(|l| l.trim_start().starts_with("```")) {
        return lines
            .iter()
            .skip(open + 1)
            .take_while(|l| !l.trim_start().starts_with("```"))
            .find(|l| !l.trim().is_empty())
            .map_or_else(String::new, |l| (*l).to_owned());
    }
    lines.iter().find(|l| !l.trim().is_empty()).map_or_else(String::new, |l| (*l).to_owned())
}

/// Write in the model card's chat form (2026-11-12): `prompt` as the user turn, the assistant turn
/// opened with an empty think block (non-thinking mode, as the card gives it), decoded by `s` (or
/// greedy with `None`); generation stops at an end id or once a code block closes, and the reply's
/// code line is returned (`first_code_line`).
/// # Errors
/// Tokenizer or engine failure.
pub fn write_chat(m: &Model, prompt: &str, max_new: usize, s: Option<&crate::decode::Sampling>) -> Result<(String, Choice, f64), Fault> {
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = m.pop.tokenizer.encode(&text).map_err(|e| bad(e.to_string()))?;
    let tok = &m.pop.tokenizer;
    // Stop at an end id, or once a code block has closed.
    let seen = std::cell::RefCell::new(String::new());
    let stop = |t: i32| {
        if t == 151_643 || t == 151_645 {
            return true;
        }
        let mut so_far = seen.borrow_mut();
        so_far.push_str(&tok.decode(&[t]));
        so_far.matches("```").count() >= 2
    };
    let greedy = crate::decode::Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0, seed: 0, source: "greedy".into() };
    let s = s.unwrap_or(&greedy);
    let (g, c, ms) = placed(m, "write", ids.len(), |d| {
        seen.borrow_mut().clear();
        crate::decode::sample_until(m, d, &ids, max_new, &stop, s)
    })?;
    Ok((first_code_line(&tok.decode(&g.tokens)), c, ms))
}

/// Write: greedy continuation of `prompt` up to `max_new` tokens, stopping at the first line
/// break (the writer answers one line) or the end-of-text id; placed per job.
/// # Errors
/// Tokenizer or engine failure.
pub fn write(m: &Model, prompt: &str, max_new: usize) -> Result<(String, Choice, f64), Fault> {
    let ids = m.pop.tokenizer.encode(prompt).map_err(|e| bad(e.to_string()))?;
    // Stop at the first token whose text holds a line break (Qwen merges "\n" into tokens
    // such as ";\n\n"), or at an end-of-text id; the answer is cut at the break.
    let tok = &m.pop.tokenizer;
    let stop = |t: i32| t == 151_643 || t == 151_645 || tok.decode(&[t]).contains('\n');
    let (g, c, ms) = placed(m, "write", ids.len(), |d| crate::decode::greedy_until(m, d, &ids, max_new, &stop))?;
    let text = tok.decode(&g.tokens);
    let first = text.split('\n').next().unwrap_or("").to_owned();
    Ok((first, c, ms))
}

/// Section check (2026-11-13, review: the Core Brain's scoring had no check of its own). Processor:
/// the card prompt holds instruction, query and document in that order inside the card's frame;
/// the code line reader takes a code block's first line, else the first line. With the released
/// reranker and embedder (skipped with a note when their weights are absent): P(yes) lies in 0 to
/// 1, a document that answers outscores an unrelated one by at least 0.3, and the set call equals
/// the single call; embeddings have unit length, the related text is nearer, and the batched
/// embedding equals the single one (cosine at least 0.999).
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let p = card_prompt("INS", "QUE", "DOC");
    let (i, q, d) = (p.find("INS"), p.find("QUE"), p.find("DOC"));
    if !(p.starts_with(CARD_PREFIX) && p.ends_with(CARD_SUFFIX) && i < q && q < d && i.is_some()) {
        return Err("nmengn.roles: card prompt layout".into());
    }
    if first_code_line("Sure.\n```rust\n\nlet x = 1;\n```\n") != "let x = 1;"
        || first_code_line("\n  plain line\nmore") != "  plain line"
        || !first_code_line("").is_empty()
    {
        return Err("nmengn.roles: code line reader".into());
    }
    let home = nmtype::nano_root();
    let open = |pop: &str, asm: &str| -> Result<Option<Model>, String> {
        if !nmtype::models_dir(&home).join(pop).exists() {
            return Ok(None);
        }
        let pop = crate::model::Population::open(&home, pop, true, true).map_err(|e| e.to_string())?;
        let asm = crate::assembly::Assembly::load(&home.join("designs/assemblies"), asm).map_err(|e| e.to_string())?;
        Model::bind(&pop, asm).map(Some).map_err(|e| e.to_string())
    };
    let query = "Which function adds two numbers?";
    let docs = vec![
        "fn add(a: u32, b: u32) -> u32 {\n    a + b\n}".to_owned(),
        "The weather in spring is mild and the days grow longer.".to_owned(),
    ];
    if let Some(r) = open("qwen3-reranker-0.6b", "qwen3-reranker")? {
        let dev = if r.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let set = card_rerank_set(&r, dev, CODE_INSTRUCTION, query, &docs).map_err(|e| e.to_string())?;
        let one = card_rerank_on(&r, dev, CODE_INSTRUCTION, query, &docs[0]).map_err(|e| e.to_string())?;
        let (a, b) = (set.first().copied().unwrap_or(-1.0), set.get(1).copied().unwrap_or(-1.0));
        if !(0.0..=1.0).contains(&a) || !(0.0..=1.0).contains(&b) || a < b + 0.3 || (a - one).abs() > 1e-3 {
            return Err(format!("nmengn.roles: reranker answer {a:.4}, unrelated {b:.4}, single {one:.4}"));
        }
        // The batched set (documents packed side by side on the card) equals the one-at-a-time set.
        let three: Vec<String> =
            docs.iter().cloned().chain(std::iter::once("fn sub(a: u32, b: u32) -> u32 { a - b }".to_owned())).collect();
        let batch = card_rerank_batch(&r, dev, CODE_INSTRUCTION, query, &three, None).map_err(|e| e.to_string())?;
        let serial = card_rerank_set(&r, dev, CODE_INSTRUCTION, query, &three).map_err(|e| e.to_string())?;
        let worst = batch.iter().zip(&serial).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        if batch.len() != serial.len() || worst > 1e-3 {
            return Err(format!("nmengn.roles: batched reranker differs from one at a time by {worst:.5} ({batch:?} against {serial:?})"));
        }
    }
    if let Some(e) = open("qwen3-embedding-0.6b", "qwen3-embedding")? {
        let dev = if e.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
        let qv = card_embed_on(&e, dev, &card_query(CARD_INSTRUCTION, query)).map_err(|x| x.to_string())?;
        let dv: Vec<Vec<f32>> = docs.iter().map(|d| card_embed_on(&e, dev, d)).collect::<Result<_, _>>().map_err(|x| x.to_string())?;
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let norm_ok = std::iter::once(&qv).chain(&dv).all(|v| (dot(v, v).sqrt() - 1.0).abs() < 1e-3);
        let (near, far) = (dot(&qv, &dv[0]), dot(&qv, &dv[1]));
        if !norm_ok || near <= far {
            return Err(format!("nmengn.roles: embedder unit length {norm_ok}, related {near:.4}, unrelated {far:.4}"));
        }
        if dev == Device::Gpu {
            let batch = card_embed_batch(&e, &docs, MAX_TOKENS, 4096).map_err(|x| x.to_string())?;
            let worst = batch
                .iter()
                .zip(&docs)
                .map(|(b, d)| card_embed_on_capped(&e, dev, d, Some(MAX_TOKENS)).map(|s| dot(b, &s)))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|x| x.to_string())?
                .into_iter()
                .fold(1.0_f32, f32::min);
            if worst < 0.999 {
                return Err(format!("nmengn.roles: batched embedding cosine {worst:.5}"));
            }
        }
    }
    Ok(())
}
