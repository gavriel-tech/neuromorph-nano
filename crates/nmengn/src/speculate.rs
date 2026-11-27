//! Zero-VRAM lossless greedy speculative decoding (`reader-base@3 -> writer-base@4`).
//!
//! When a smaller draft MicroModel (`draft`, e.g. `Qwen3-0.6B`) and a larger target MicroModel
//! (`target`, e.g. `Qwen3-1.7B`) share a tokenizer and are both already resident on the card,
//! `draft` proposes `lookahead` candidate tokens and `target` verifies all of them in a single
//! parallel forward pass (`RunOpts::all_positions = true`). Every accepted token matches
//! `argmax(target_logits)` exactly, and the first mismatch is replaced by the target's exact
//! greedy token while both KV caches rewind via `Cache::truncate`, giving mathematically identical
//! greedy output at 0 MiB extra VRAM.
use crate::decode::{Generation, argmax};
use crate::model::{Cache, Model, RunOpts, values};
use crate::placement::Device;
use nmtype::{Fault, FaultKind};
use std::sync::Arc;
use std::time::Instant;

/// Result of one speculative greedy generation with acceptance telemetry.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecGeneration {
    pub generation: Generation,
    pub drafted: usize,
    pub accepted: usize,
    pub target_passes: usize,
}

/// Resolve the argmax token id from a logit slice, mapping through `head_rows` when an active
/// vocabulary subset was used.
fn pick_token(row: &[f32], head_rows: &[usize]) -> i32 {
    let idx = argmax(row);
    let tok = if head_rows.is_empty() { idx } else { head_rows.get(idx).copied().unwrap_or(0) };
    i32::try_from(tok).unwrap_or(0)
}

/// Core lossless greedy verification step: given `drafted` candidate tokens proposed after the
/// last accepted token and `target_picks` (`drafted.len() + 1` exact greedy picks from the
/// target's parallel verification pass), returns `(emitted_tokens, matched_count)`.
#[must_use]
pub fn verify_greedy(drafted: &[i32], target_picks: &[i32]) -> (Vec<i32>, usize) {
    let mut out = Vec::with_capacity(drafted.len() + 1);
    let mut matched = 0_usize;
    for (i, &d) in drafted.iter().enumerate() {
        let Some(&t) = target_picks.get(i) else { break };
        out.push(t);
        if t == d {
            matched += 1;
        } else {
            return (out, matched);
        }
    }
    if let Some(&bonus) = target_picks.get(drafted.len()) {
        out.push(bonus);
    }
    (out, matched)
}

/// Whether speculative drafting is enabled (`on` unless `NANO_SPECULATE=off`).
#[must_use]
pub fn enabled() -> bool {
    !std::env::var("NANO_SPECULATE").is_ok_and(|v| v == "off" || v == "0")
}

/// Lossless greedy speculative decoding with `draft` proposing up to `lookahead` tokens per
/// `target` verification pass. Supports optional `head_rows` on `target` (active vocabulary).
/// # Errors
/// An engine fault.
#[expect(clippy::too_many_arguments, reason = "draft and target models, device, prompt, budget, lookahead, head rows, and stop predicate")]
pub fn greedy_until(
    draft: &Model,
    target: &Model,
    device: Device,
    prompt: &[i32],
    max_new: usize,
    lookahead: usize,
    head_rows: &[usize],
    stop: &dyn Fn(i32) -> bool,
) -> Result<SpecGeneration, Fault> {
    if max_new == 0 || lookahead == 0 || prompt.is_empty() {
        let g = crate::decode::greedy_until(target, device, prompt, max_new, stop)?;
        let passes = g.tokens.len();
        return Ok(SpecGeneration { generation: g, drafted: 0, accepted: 0, target_passes: passes });
    }
    let draft_head = (draft.role_node("head")?, "logits".to_owned());
    let target_head = (target.role_node("head")?, "logits".to_owned());
    let draft_cache = Arc::new(Cache::default());
    let target_cache = Arc::new(Cache::default());
    let draft_kv_w = draft.pop.dims.kv_heads * draft.pop.dims.head_dim;
    let target_kv_w = target.pop.dims.kv_heads * target.pop.dims.head_dim;
    let head_rows_arc: Arc<Vec<usize>> = Arc::new(head_rows.to_vec());
    let target_vocab = if head_rows.is_empty() { target.pop.dims.vocab } else { head_rows.len() };

    let draft_opts = RunOpts { cache: Some(Arc::clone(&draft_cache)), ..RunOpts::default() };
    let target_prefill_opts = RunOpts {
        cache: Some(Arc::clone(&target_cache)),
        head_rows: Arc::clone(&head_rows_arc),
        ..RunOpts::default()
    };
    let target_verify_opts = RunOpts {
        cache: Some(Arc::clone(&target_cache)),
        head_rows: Arc::clone(&head_rows_arc),
        all_positions: true,
        ..RunOpts::default()
    };

    let t0 = Instant::now();
    // Prefill both caches on the prompt; first emitted token comes directly from target.
    let _ = draft.forward_at(device, prompt, 0, std::slice::from_ref(&draft_head), &draft_opts)?;
    let out0 = target.forward_at(device, prompt, 0, std::slice::from_ref(&target_head), &target_prefill_opts)?;
    let logits0 = values(out0.get(&target_head).ok_or_else(|| Fault::new("nmengn.speculate", FaultKind::Execution, "no target logits"))?)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut tokens = Vec::with_capacity(max_new);
    let mut step_ms = Vec::new();
    let mut stopped = false;
    let mut drafted_total = 0_usize;
    let mut accepted_total = 0_usize;
    let mut target_passes = 1_usize;

    let first = pick_token(logits0, head_rows);
    tokens.push(first);
    if stop(first) {
        return Ok(SpecGeneration {
            generation: Generation { tokens, prefill_ms, step_ms, stopped: true },
            drafted: 0,
            accepted: 0,
            target_passes,
        });
    }

    let mut draft_synced = prompt.len();

    while tokens.len() < max_new {
        let remaining = max_new - tokens.len();
        let k = lookahead.min(remaining);
        let base_pos = prompt.len() + tokens.len() - 1;
        let last = tokens.last().copied().unwrap_or(0);
        let step_t0 = Instant::now();

        // 1. Draft up to `k` candidate tokens on the smaller resident model.
        // When the previous round accepted all drafted tokens plus a target bonus token,
        // `draft_synced == base_pos - 1`; feeding `&tokens[draft_synced - prompt.len()..]`
        // catches the draft KV cache up in the same forward pass as predicting the first draft token.
        let mut proposed = Vec::with_capacity(k);
        let mut cur = last;
        for j in 0..k {
            let dout = if j == 0 {
                let start = draft_synced.saturating_sub(prompt.len());
                draft.forward_at(device, &tokens[start..], draft_synced, std::slice::from_ref(&draft_head), &draft_opts)?
            } else {
                draft.forward_at(device, &[cur], base_pos + j, std::slice::from_ref(&draft_head), &draft_opts)?
            };
            let dlogits = values(dout.get(&draft_head).ok_or_else(|| Fault::new("nmengn.speculate", FaultKind::Execution, "no draft logits"))?)?;
            let cand = pick_token(dlogits, &[]);
            proposed.push(cand);
            cur = cand;
            if stop(cand) {
                break;
            }
        }
        drafted_total += proposed.len();

        // 2. Verify `[last, proposed...]` in ONE parallel target pass (`all_positions = true`).
        let mut verify_in = Vec::with_capacity(1 + proposed.len());
        verify_in.push(last);
        verify_in.extend_from_slice(&proposed);
        let tout = target.forward_at(device, &verify_in, base_pos, std::slice::from_ref(&target_head), &target_verify_opts)?;
        target_passes += 1;
        let tlogits = values(tout.get(&target_head).ok_or_else(|| Fault::new("nmengn.speculate", FaultKind::Execution, "no target logits"))?)?;
        let target_picks: Vec<i32> = tlogits
            .chunks_exact(target_vocab.max(1))
            .take(verify_in.len())
            .map(|row| pick_token(row, head_rows))
            .collect();

        let (emitted, matched) = verify_greedy(&proposed, &target_picks);
        accepted_total += matched;
        let elapsed = step_t0.elapsed().as_secs_f64() * 1000.0;
        let per_tok = elapsed / (emitted.len().max(1) as f64);

        for tok in emitted {
            if tokens.len() >= max_new {
                break;
            }
            tokens.push(tok);
            step_ms.push(per_tok);
            if stop(tok) {
                stopped = true;
                break;
            }
        }
        if stopped || tokens.len() >= max_new {
            break;
        }
        // Rewind both KV caches to `prompt.len() + tokens.len() - 1` so the newest accepted token
        // is fed cleanly as `last` on the next round without duplicating KV positions.
        let keep = prompt.len() + tokens.len() - 1;
        target_cache.truncate(keep, target_kv_w);
        draft_synced = (base_pos + proposed.len()).min(keep);
        draft_cache.truncate(draft_synced, draft_kv_w);
    }

    Ok(SpecGeneration {
        generation: Generation { tokens, prefill_ms, step_ms, stopped },
        drafted: drafted_total,
        accepted: accepted_total,
        target_passes,
    })
}

/// Section check (no GPU required): verifies lossless greedy verification across full match,
/// mid-sequence mismatch, first-token mismatch, and active-vocab `head_rows` mapping.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    // Full match emits all drafted tokens plus the target's bonus token.
    let (e1, m1) = verify_greedy(&[10, 20, 30], &[10, 20, 30, 40]);
    if e1 != vec![10, 20, 30, 40] || m1 != 3 {
        return Err(format!("nmengn.speculate: full match gave {e1:?} matched={m1}"));
    }
    // Mid-sequence mismatch emits matching prefix + exact target correction and stops.
    let (e2, m2) = verify_greedy(&[10, 20, 30], &[10, 99, 30, 40]);
    if e2 != vec![10, 99] || m2 != 1 {
        return Err(format!("nmengn.speculate: mid mismatch gave {e2:?} matched={m2}"));
    }
    // First-token mismatch emits only the exact target token.
    let (e3, m3) = verify_greedy(&[10, 20], &[77, 88, 99]);
    if e3 != vec![77] || m3 != 0 {
        return Err(format!("nmengn.speculate: first mismatch gave {e3:?} matched={m3}"));
    }
    // Active vocabulary row mapping.
    let tok = pick_token(&[0.1, 2.5, 1.0], &[100, 420, 900]);
    if tok != 420 {
        return Err(format!("nmengn.speculate: head_rows pick gave {tok}, wanted 420"));
    }
    Ok(())
}
