//! Drone keys and the keys-only scan (2026-11-10; tentacle core design
//! `docs/2026-11-10-neumor-design-tentacle-core-v1.md` section 2.2 and engine additions A6 and A7,
//! architecture by Luis Jake Gabriel III). Three separable parts, one check:
//! - `RelevanceKeys` (drone keys until 2026-11-18): per chunk, only what the drone readers read: the key rows (before rope) of the
//!   key-value heads the drone set uses, at the drone layers. The bytes per token follow from the
//!   set (`pairs`): distinct (layer, key-value head) pairs x head width x 2 bytes; 3,328 for the
//!   16-head default set of Qwen3-0.6B (13 pairs), against 73,728 for every key and value of
//!   layers 0 to 17. `KeyStore` holds them by chunk content key, least recently used out past a
//!   byte limit the caller derives.
//! - `question_path` (A7): the question tokens run through the layers up to the deepest drone
//!   layer behind a short prefix (the anchor) at the positions they would have after a full
//!   window; the drone heads' queries (after the head norm and rope, as the attention scoring
//!   receives them) are taken at each drone layer, found in the assembly by operation and layer.
//! - `scan_keys` (A6): the raw attention scores of those queries against a window's drone keys,
//!   each key given its window position's rope on read (`cpu::Rows`, the engine's own read), the
//!   score `q.k / sqrt(d)` with the engine's own dot product. With queries taken from the exact
//!   read of the same window the scores equal the probe's scores bit for bit (the check). With
//!   queries from the question path they are an approximation: the question no longer attends to
//!   the window in the early layers. No weight is changed and no parameter set is written.
use crate::cpu;
use crate::kvstore::{ChunkKv, Window};
use crate::model::{Act, Model, RunOpts};
use crate::placement::Device;
use nmtype::{Fault, FaultKind};
use std::collections::BTreeMap;
use std::sync::Arc;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmengn.drone", FaultKind::Execution, m)
}

/// The distinct (layer, key-value head) pairs a drone set reads, in order; `group` query heads
/// share one key-value head.
#[must_use]
pub fn pairs(heads: &[(usize, usize)], group: usize) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = heads.iter().map(|&(l, h)| (l, h / group.max(1))).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Bytes one token holds in a drone-key packet: pairs x head width x 2 (binary16).
#[must_use]
pub fn bytes_per_token(pairs: &[(usize, usize)], head_dim: usize) -> usize {
    pairs.len() * head_dim * 2
}

/// One piece's relevance keys (the tentacle's periphery data, master plan 3.2; named drone keys
/// until 2026-11-18): per pair of `pairs`, the piece's key rows before rope, `[tokens, head
/// width]` binary16 bits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelevanceKeys {
    pub tokens: usize,
    pub pairs: Vec<(usize, usize)>,
    pub rows: Vec<Vec<u16>>,
}
impl RelevanceKeys {
    /// The packet of an encoded chunk (keys of every key-value head per layer, `ChunkKv`).
    /// # Errors
    /// A pair past the layers the chunk holds, or rows that do not fit the token count.
    pub fn from_chunk(c: &ChunkKv, pairs: &[(usize, usize)], kv_heads: usize, head_dim: usize) -> Result<Self, Fault> {
        let width = kv_heads * head_dim;
        let mut rows = Vec::with_capacity(pairs.len());
        for &(l, g) in pairs {
            let k =
                c.k_pre.get(l).ok_or_else(|| bad(format!("the chunk holds {} layers; the drone set reads layer {l}", c.k_pre.len())))?;
            if k.len() != c.tokens * width || g >= kv_heads {
                return Err(bad(format!(
                    "layer {l}: {} key values for {} tokens of {width}, or key-value head {g} of {kv_heads}",
                    k.len(),
                    c.tokens
                )));
            }
            let mut r = Vec::with_capacity(c.tokens * head_dim);
            for t in 0..c.tokens {
                r.extend_from_slice(k.get(t * width + g * head_dim..t * width + (g + 1) * head_dim).unwrap_or(&[]));
            }
            rows.push(r);
        }
        Ok(Self { tokens: c.tokens, pairs: pairs.to_vec(), rows })
    }
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.iter().map(|r| r.len() * 2).sum()
    }
}

/// The former name of `RelevanceKeys`, kept for the lab's tools until they move.
pub type DroneKeys = RelevanceKeys;

/// Relevance-key packets by chunk content key, least recently used out past `cap` bytes.
#[derive(Debug, Default)]
pub struct KeyStore {
    map: BTreeMap<String, (Arc<RelevanceKeys>, u64)>,
    bytes: usize,
    cap: usize,
    tick: u64,
}
impl KeyStore {
    /// An empty store holding at most `cap` bytes.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self { cap, ..Self::default() }
    }
    /// Bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// Tokens held (over every packet).
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.map.values().map(|e| e.0.tokens).sum()
    }
    /// The byte limit.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }
    /// Packets held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }
    /// No packet held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// True when `key` is held (not marked as used).
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }
    /// The packet under `key`, marked as just used.
    pub fn get(&mut self, key: &str) -> Option<Arc<RelevanceKeys>> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(key).map(|e| {
            e.1 = t;
            Arc::clone(&e.0)
        })
    }
    /// Hold `p` under `key`, dropping the least recently used past the cap. Returns the packets
    /// dropped.
    pub fn put(&mut self, key: &str, p: Arc<RelevanceKeys>) -> usize {
        self.tick += 1;
        self.bytes += p.bytes();
        if let Some((old, _)) = self.map.insert(key.to_owned(), (p, self.tick)) {
            self.bytes -= old.bytes();
        }
        let mut dropped = 0;
        // 2026-11-27: the ceiling is live (`limits::host_cache_cap`), so the store gives packets
        // back when the machine or this process runs short of memory.
        let cap = crate::limits::host_cache_cap(self.cap, self.bytes);
        while self.bytes > cap && self.map.len() > 1 {
            let Some(k) = self.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| k.clone()) else { break };
            if let Some((old, _)) = self.map.remove(&k) {
                self.bytes -= old.bytes();
                dropped += 1;
            }
        }
        dropped
    }
    /// Change the byte limit, dropping the least recently used past it. Returns the packets
    /// dropped.
    pub fn set_cap(&mut self, cap: usize) -> usize {
        self.cap = cap;
        let mut dropped = 0;
        while self.bytes > self.cap && !self.map.is_empty() {
            let Some(k) = self.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| k.clone()) else { break };
            if let Some((old, _)) = self.map.remove(&k) {
                self.bytes -= old.bytes();
                dropped += 1;
            }
        }
        dropped
    }
    /// Drop every packet whose key is not in `live` (chunks that changed or left the tree).
    /// Returns the packets dropped.
    pub fn retain(&mut self, live: &std::collections::BTreeSet<String>) -> usize {
        let before = self.map.len();
        let mut freed = 0;
        self.map.retain(|k, e| {
            let keep = live.contains(k);
            if !keep {
                freed += e.0.bytes();
            }
            keep
        });
        self.bytes -= freed;
        before - self.map.len()
    }
}

/// The drone heads' queries of one question: per head of `heads` (in that order),
/// `[tokens, head width]` after the head norm and rope.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Queries {
    pub tokens: usize,
    pub heads: Vec<(usize, usize)>,
    pub q: Vec<Vec<f32>>,
}

/// One query output: (layer, node id, port, global query head range).
pub type QueryTarget = (usize, String, String, (usize, usize));

/// Per drone layer, the query outputs the attention scoring reads: (layer, node id, port, global
/// query head range), found in the assembly by operation and layer (the edge into each `score`
/// component's `q` port).
/// # Errors
/// A drone layer with no `score` component, or a `score` whose `q` input has no source.
pub fn query_targets(m: &Model, heads: &[(usize, usize)]) -> Result<Vec<QueryTarget>, Fault> {
    let d = m.pop.dims;
    let mut layers: Vec<usize> = heads.iter().map(|p| p.0).collect();
    layers.sort_unstable();
    layers.dedup();
    let mut out = Vec::new();
    for l in layers {
        let scores: Vec<_> = m.asm.nodes.iter().filter(|n| n.op == "score" && n.layer == Some(l)).collect();
        if scores.is_empty() {
            return Err(bad(format!("assembly {} has no score component on layer {l}", m.asm.name)));
        }
        for s in scores {
            let e = m.asm.edges.iter().find(|e| e.to == s.id && e.to_port == "q").ok_or_else(|| bad(format!("{}: no q input", s.id)))?;
            out.push((l, e.from.clone(), e.from_port.clone(), s.heads.unwrap_or((0, d.heads))));
        }
    }
    Ok(out)
}

/// The question path (A7): `ids` run behind `prefix` at positions `start..` through the layers
/// up to the deepest drone layer; returns the drone heads' queries. With `prefix` a full window
/// and `start` its length this is exactly the query side of the exact read; with the anchor as
/// prefix and `start` a window's length it is the keys-only scan's question path.
/// # Errors
/// Engine or device faults, or a head no query output covers.
pub fn question_path(
    m: &Model,
    dev: Device,
    prefix: &Arc<Window>,
    ids: &[i32],
    start: usize,
    heads: &[(usize, usize)],
) -> Result<Queries, Fault> {
    let d = m.pop.dims;
    let targets = query_targets(m, heads)?;
    let want: Vec<(String, String)> = targets.iter().map(|t| (t.1.clone(), t.2.clone())).collect();
    let acts = m.forward_at(dev, ids, start, &want, &RunOpts { window: Some(Arc::clone(prefix)), ..RunOpts::default() })?;
    let mut host: BTreeMap<(String, String), Vec<f32>> = BTreeMap::new();
    for (k, a) in acts {
        if !want.contains(&k) {
            continue;
        }
        let v = match a {
            Act::Cpu(x) => x,
            Act::Gpu(t) => m.pop.gpu.as_ref().ok_or_else(|| bad("a device result without a device"))?.download(&t).map_err(bad)?,
            Act::Ids(_) => return Err(bad("query output is not a tensor")),
        };
        host.insert(k, v);
    }
    let tokens = ids.len();
    let mut q = Vec::with_capacity(heads.len());
    for &(l, h) in heads {
        let (_, id, port, range) = targets
            .iter()
            .find(|t| t.0 == l && h >= t.3.0 && h < t.3.1)
            .ok_or_else(|| bad(format!("no query output covers layer {l} head {h}")))?;
        let v = host.get(&(id.clone(), port.clone())).ok_or_else(|| bad(format!("{id}.{port} not returned")))?;
        let n = range.1 - range.0;
        if v.len() != tokens * n * d.head_dim {
            return Err(bad(format!("{id}.{port}: {} values for {tokens} tokens of {n} heads", v.len())));
        }
        let local = h - range.0;
        let mut rows = Vec::with_capacity(tokens * d.head_dim);
        for t in 0..tokens {
            rows.extend_from_slice(v.get((t * n + local) * d.head_dim..(t * n + local + 1) * d.head_dim).unwrap_or(&[]));
        }
        q.push(rows);
    }
    Ok(Queries { tokens, heads: heads.to_vec(), q })
}

/// The keys-only scan (A6): per head of `q.heads` (in that order), the raw attention scores
/// `[tokens, len]` of its queries against the window made of `parts` placed side by side
/// (anchor first, `len` positions), each key roped at its window position from `table`
/// (`cpu::rope_table(len, head width, theta)`). Single-threaded; callers run one window per
/// thread.
/// # Errors
/// A part without a pair a head needs, or a table shorter than the window.
pub fn scan_keys(
    q: &Queries,
    parts: &[Arc<RelevanceKeys>],
    len: usize,
    table: &[(f32, f32)],
    head_dim: usize,
    group: usize,
) -> Result<Vec<Vec<f32>>, Fault> {
    let total: usize = parts.iter().map(|p| p.tokens).sum();
    if total != len || table.len() < len * (head_dim / 2) {
        return Err(bad(format!(
            "window of {len} positions: parts hold {total}, rope table {} pairs",
            table.len() / (head_dim / 2).max(1)
        )));
    }
    #[expect(clippy::cast_precision_loss, reason = "a head width is small")]
    let scale = (head_dim as f32).sqrt();
    let mut out: Vec<Vec<f32>> = q.heads.iter().map(|_| vec![f32::NEG_INFINITY; q.tokens * len]).collect();
    let mut by_pair: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, &(l, h)) in q.heads.iter().enumerate() {
        by_pair.entry((l, h / group.max(1))).or_default().push(i);
    }
    let mut row = vec![0.0_f32; head_dim];
    for (pair, idx) in by_pair {
        let mut segs: Vec<&[u16]> = Vec::with_capacity(parts.len());
        for p in parts {
            let k = p
                .pairs
                .iter()
                .position(|x| *x == pair)
                .ok_or_else(|| bad(format!("a packet lacks layer {} key-value head {}", pair.0, pair.1)))?;
            segs.push(p.rows.get(k).map_or(&[][..], Vec::as_slice));
        }
        let rows = cpu::Rows { pre: segs.len(), cached: segs, new: &[], width: head_dim, table, head: head_dim, ..cpu::Rows::default() };
        for pos in 0..len {
            rows.read(pos, 0, &mut row);
            for &i in &idx {
                let (qs, o) = (q.q.get(i).map_or(&[][..], Vec::as_slice), out.get_mut(i));
                let Some(o) = o else { continue };
                for t in 0..q.tokens {
                    if let (Some(qr), Some(x)) = (qs.get(t * head_dim..(t + 1) * head_dim), o.get_mut(t * len + pos)) {
                        *x = cpu::dot(qr, &row) / scale;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// How `scan_pieces` reduces a piece's positions: log-sum-exp (the old table, `scan_keys` then
/// `part_lse`), or log-mean-exp (log-sum minus the log of the position count, so a long piece
/// gains nothing by its length; master plan 3.3 step 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduce {
    Sum,
    Mean,
}

/// Positions scored together: one block of roped keys (64 positions x head width 128 x 4 bytes,
/// 32 KiB) stays in the first-level data cache (48 KiB on this processor's performance cores)
/// while every query of the pair reads it.
const BLOCK: usize = 64;

/// Scores of four query rows against eight keys laid out dimension by dimension (`kt[d]` holds
/// dimension `d` of the eight keys): per dimension one fused multiply-add per query row over the
/// eight keys' lanes, so the keys are loaded once for four queries (register blocking, the
/// micro-kernel form of BLIS and of `llamafile`'s `tinyBLAS`). The sum over dimensions runs in
/// dimension order with fused rounding, so a score differs from `cpu::dot` within float
/// resolution (the D2 test reports by how much).
#[inline]
fn dot4x8(q: [&[f32]; 4], kt: &[[f32; 8]]) -> [[f32; 8]; 4] {
    let mut s = [[0.0_f32; 8]; 4];
    let n = kt.len().min(q[0].len()).min(q[1].len()).min(q[2].len()).min(q[3].len());
    let (q0, q1, q2, q3) = (&q[0][..n], &q[1][..n], &q[2][..n], &q[3][..n]);
    for ((((y, a), b), c), d) in kt[..n].iter().zip(q0).zip(q1).zip(q2).zip(q3) {
        for i in 0..8 {
            s[0][i] = a.mul_add(y[i], s[0][i]);
            s[1][i] = b.mul_add(y[i], s[1][i]);
            s[2][i] = c.mul_add(y[i], s[2][i]);
            s[3][i] = d.mul_add(y[i], s[3][i]);
        }
    }
    s
}

/// `exp(x)` for `x` at most 0 (a score minus the running maximum) in eight lanes: `2^n * p(r)`
/// with `n = floor(x / ln 2 + 1/2)`, `r = x - n ln 2` (ln 2 split in two for exactness) and the degree-6
/// Taylor polynomial of `exp(r)` on `|r| <= ln 2 / 2` (relative error under 2e-7, the float's own
/// resolution; Cephes `expf` form). Below -87 the result is 0, where `exp` underflows anyway.
#[inline]
fn exp8(x: [f32; 8]) -> [f32; 8] {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    let mut out = [0.0_f32; 8];
    for i in 0..8 {
        let v = x[i].max(-87.0);
        let n = (v * LOG2E + 0.5).floor();
        let r = (v - n * LN2_HI) - n * LN2_LO;
        let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "n is in -126..=0, so n + 127 is in 1..=127")]
        let bits = ((n as i32 + 127) as u32) << 23;
        out[i] = if x[i] < -87.0 { 0.0 } else { p * f32::from_bits(bits) };
    }
    out
}

/// The largest of `xs` (negative infinity for none), eight lanes at a time.
#[inline]
fn max8(xs: &[f32]) -> f32 {
    let (c, r) = xs.as_chunks::<8>();
    let mut m = [f32::NEG_INFINITY; 8];
    for x in c {
        for i in 0..8 {
            m[i] = if x[i] > m[i] { x[i] } else { m[i] };
        }
    }
    let mut t = m.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in r {
        t = t.max(*v);
    }
    t
}

/// `sum(exp(x - m))` over `xs` (every `x` at most `m`), eight lanes at a time.
#[inline]
fn exp_sum(xs: &[f32], m: f32) -> f32 {
    let (c, r) = xs.as_chunks::<8>();
    let mut acc = [0.0_f32; 8];
    for x in c {
        let e = exp8(x.map(|v| v - m));
        for i in 0..8 {
            acc[i] += e[i];
        }
    }
    let mut t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for v in r {
        t += (v - m).exp();
    }
    t
}

/// `half::r16` on eight lanes: the mantissa rounding where every lane is in binary16's normal
/// range (the common case, which the compiler vectorises), `half::r16` lane by lane otherwise.
/// Equal to `half::r16` bit for bit.
#[inline]
fn r16x8(x: [f32; 8]) -> [f32; 8] {
    let mut out = [0.0_f32; 8];
    let mut ok = true;
    for i in 0..8 {
        let b = x[i].to_bits();
        let e = (b >> 23) & 0xff;
        ok &= (113..=141).contains(&e);
        let lsb = (b >> 13) & 1;
        out[i] = f32::from_bits(b.wrapping_add(0x0fff + lsb) & !0x1fff);
    }
    if ok { out } else { r16_lanes(x) }
}

/// `half::r16` lane by lane, out of the vectorised path (rare: a lane outside binary16's normal
/// range).
#[cold]
#[inline(never)]
fn r16_lanes(x: [f32; 8]) -> [f32; 8] {
    x.map(crate::half::r16)
}

/// Binary16 bits to f32 on eight lanes: bit arithmetic where every lane is a normal number,
/// `cpu::wide` lane by lane otherwise (zero, subnormal, infinite). Equal to `cpu::wide`.
#[inline]
fn wide8(h: &[u16; 8]) -> [f32; 8] {
    let mut out = [0.0_f32; 8];
    let mut ok = true;
    for i in 0..8 {
        let v = u32::from(h[i]);
        let e = (v >> 10) & 0x1f;
        ok &= e != 0 && e != 31;
        out[i] = f32::from_bits(((v & 0x8000) << 16) | ((e + 112) << 23) | ((v & 0x3ff) << 13));
    }
    if ok { out } else { wide_lanes(h) }
}

/// `cpu::wide` lane by lane, out of the vectorised path (rare: zero, subnormal or infinite).
#[cold]
#[inline(never)]
fn wide_lanes(h: &[u16; 8]) -> [f32; 8] {
    h.map(cpu::wide)
}

/// One key row decoded and roped at one position (`cpu::Rows::read` on a key before rope, the
/// arithmetic of `cpu::rotate` in eight lanes): `dst` and `src` are a head's width, `c` and `s`
/// the position's cosines and sines (half the width each). Equal to `cpu::Rows::read` bit for bit.
#[inline]
fn rope_row(dst: &mut [f32], src: &[u16], c: &[f32], s: &[f32]) {
    let half = c.len();
    if !half.is_multiple_of(8) || src.len() != 2 * half || dst.len() != 2 * half || s.len() != half {
        for (o, b) in dst.iter_mut().zip(src) {
            *o = cpu::wide(*b);
        }
        let cs: Vec<(f32, f32)> = c.iter().copied().zip(s.iter().copied()).collect();
        cpu::rotate(dst, &cs);
        return;
    }
    let (sc, _) = src.as_chunks::<8>();
    let (dc, _) = dst.as_chunks_mut::<8>();
    for (o, h) in dc.iter_mut().zip(sc) {
        *o = wide8(h);
    }
    let (lo, hi) = dc.split_at_mut(half / 8);
    let (cc, _) = c.as_chunks::<8>();
    let (ss, _) = s.as_chunks::<8>();
    for (((l, r), c8), s8) in lo.iter_mut().zip(hi.iter_mut()).zip(cc).zip(ss) {
        let (lv, rv) = (*l, *r);
        let mut a = [0.0_f32; 8];
        let mut b = [0.0_f32; 8];
        let mut e = [0.0_f32; 8];
        let mut f = [0.0_f32; 8];
        for i in 0..8 {
            a[i] = lv[i] * c8[i];
            b[i] = -rv[i] * s8[i];
            e[i] = rv[i] * c8[i];
            f[i] = lv[i] * s8[i];
        }
        let (a, b, e, f) = (r16x8(a), r16x8(b), r16x8(e), r16x8(f));
        let mut nl = [0.0_f32; 8];
        let mut nr = [0.0_f32; 8];
        for i in 0..8 {
            nl[i] = a[i] + b[i];
            nr[i] = e[i] + f[i];
        }
        *l = r16x8(nl);
        *r = r16x8(nr);
    }
}

/// The periphery kernel (master plan 3.3 step 3, Phase D2, 2026-11-18): per head of `q.heads`
/// (rows) and per piece of `parts_at` (columns, `(window position, positions)`), the piece's
/// log-sum-exp (or log-mean-exp, `reduce`) of the raw scores over its positions, averaged over the
/// query tokens `use_tokens` (the question's content tokens; `None` for every token). The same
/// table as `scan_keys` followed by `part_lse` (within float tolerance: the per-piece sum runs
/// block by block with a running maximum, as flash attention does), but in one pass and without a
/// per-position buffer: per block of positions the rope's cosines and sines are laid out once,
/// each pair's keys are decoded and roped once (eight lanes, the engine's binary16 rounding kept
/// bit for bit), the block is laid out dimension by dimension and read by four queries at once (`dot4x8`), and the
/// block is folded into the piece's running maximum and sum at once. Positions outside every
/// piece are not read. Packets are uncompressed (the compressed tier is Phase C5); single-threaded,
/// one window per thread.
/// # Errors
/// As `scan_keys`; a piece past the window.
#[expect(clippy::too_many_arguments, reason = "the window's geometry and the reduction")]
#[expect(clippy::too_many_lines, reason = "one kernel, its loops kept together")]
pub fn scan_pieces(
    q: &Queries,
    parts: &[Arc<RelevanceKeys>],
    parts_at: &[(usize, usize)],
    len: usize,
    table: &[(f32, f32)],
    geometry: (usize, usize),
    use_tokens: Option<&[usize]>,
    reduce: Reduce,
) -> Result<Vec<f32>, Fault> {
    let (head_dim, group) = geometry;
    let half = head_dim / 2;
    let total: usize = parts.iter().map(|p| p.tokens).sum();
    if total != len || table.len() < len * half {
        return Err(bad(format!("window of {len} positions: parts hold {total}, rope table {} pairs", table.len() / half.max(1))));
    }
    if let Some(&(at, n)) = parts_at.iter().find(|(at, n)| at + n > len) {
        return Err(bad(format!("piece at {at} of {n} positions past the window of {len}")));
    }
    let toks: Vec<usize> = use_tokens.map_or_else(|| (0..q.tokens).collect(), |t| t.iter().copied().filter(|x| *x < q.tokens).collect());
    let nt = toks.len();
    #[expect(clippy::cast_precision_loss, reason = "a head width and token counts are small")]
    let (scale, per) = ((head_dim as f32).sqrt(), 1.0 / nt.max(1) as f32);
    // Window position where each packet starts.
    let mut starts = Vec::with_capacity(parts.len());
    let mut at = 0;
    for p in parts {
        starts.push(at);
        at += p.tokens;
    }
    // Per pair: its heads (indices into `q.heads`), every packet's rows of it, and the heads'
    // query rows for the tokens used.
    let mut by_pair: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, &(l, h)) in q.heads.iter().enumerate() {
        by_pair.entry((l, h / group.max(1))).or_default().push(i);
    }
    type PairPlan<'a> = (Vec<usize>, Vec<&'a [u16]>, Vec<Vec<&'a [f32]>>);
    let mut plan: Vec<PairPlan<'_>> = Vec::with_capacity(by_pair.len());
    for (pair, idx) in by_pair {
        let mut rows: Vec<&[u16]> = Vec::with_capacity(parts.len());
        for p in parts {
            let k = p
                .pairs
                .iter()
                .position(|x| *x == pair)
                .ok_or_else(|| bad(format!("a packet lacks layer {} key-value head {}", pair.0, pair.1)))?;
            rows.push(p.rows.get(k).map_or(&[][..], Vec::as_slice));
        }
        let qs: Vec<Vec<&[f32]>> = idx
            .iter()
            .map(|&i| {
                let all = q.q.get(i).map_or(&[][..], Vec::as_slice);
                toks.iter().map(|t| all.get(t * head_dim..(t + 1) * head_dim).unwrap_or(&[])).collect()
            })
            .collect();
        plan.push((idx, rows, qs));
    }
    let mut out = vec![0.0_f32; q.heads.len() * parts_at.len()];
    let mut keys = vec![0.0_f32; BLOCK * head_dim];
    // The block's roped keys dimension by dimension, eight keys a group: `kt[g * head_dim + d][i]`
    // is dimension `d` of key `g * 8 + i`.
    let mut kt = vec![[0.0_f32; 8]; BLOCK / 8 * head_dim];
    let mut tile = [[0.0_f32; BLOCK]; 4];
    let mut cos = vec![0.0_f32; BLOCK * half];
    let mut sin = vec![0.0_f32; BLOCK * half];
    let mut place: Vec<(usize, usize)> = vec![(0, 0); BLOCK];
    // Running maximum and sum per (head, token) of the current piece.
    let mut m = vec![f32::NEG_INFINITY; q.heads.len() * nt];
    let mut s = vec![0.0_f32; q.heads.len() * nt];
    for (c, &(p0, n)) in parts_at.iter().enumerate() {
        m.fill(f32::NEG_INFINITY);
        s.fill(0.0);
        let mut b0 = p0;
        while b0 < p0 + n {
            let bn = BLOCK.min(p0 + n - b0);
            // The block's rope and where each position's key sits, once for every pair.
            for j in 0..bn {
                let pos = b0 + j;
                let pk = starts.partition_point(|x| *x <= pos).saturating_sub(1);
                if let Some(x) = place.get_mut(j) {
                    *x = (pk, pos - starts.get(pk).copied().unwrap_or(0));
                }
                let cs = table.get(pos * half..(pos + 1) * half).unwrap_or(&[]);
                for (k, &(cv, sv)) in cs.iter().enumerate() {
                    if let (Some(a), Some(b)) = (cos.get_mut(j * half + k), sin.get_mut(j * half + k)) {
                        *a = cv;
                        *b = sv;
                    }
                }
            }
            for (idx, rows, qs) in &plan {
                // Decode and rope the block's keys of this pair once for its heads.
                for j in 0..bn {
                    let (pk, local) = place.get(j).copied().unwrap_or((0, 0));
                    let src = rows.get(pk).and_then(|r| r.get(local * head_dim..(local + 1) * head_dim)).unwrap_or(&[]);
                    let (Some(dst), Some(cj), Some(sj)) =
                        (keys.get_mut(j * head_dim..(j + 1) * head_dim), cos.get(j * half..(j + 1) * half), sin.get(j * half..(j + 1) * half))
                    else {
                        continue;
                    };
                    rope_row(dst, src, cj, sj);
                }
                // Lay the block out dimension by dimension; positions past the block stay zero and
                // are not read below.
                let groups = bn.div_ceil(8);
                for g in 0..groups {
                    let Some(dst) = kt.get_mut(g * head_dim..(g + 1) * head_dim) else { continue };
                    for i in 0..8 {
                        let j = g * 8 + i;
                        let row = if j < bn { keys.get(j * head_dim..(j + 1) * head_dim).unwrap_or(&[]) } else { &[] };
                        for (d, lane) in dst.iter_mut().enumerate() {
                            lane[i] = row.get(d).copied().unwrap_or(0.0);
                        }
                    }
                }
                for (&i, qh) in idx.iter().zip(qs) {
                    for (t0, four) in qh.chunks(4).enumerate() {
                        let last = four.last().copied().unwrap_or(&[]);
                        let qr = [0, 1, 2, 3].map(|k| four.get(k).copied().unwrap_or(last));
                        for g in 0..groups {
                            let sc = dot4x8(qr, kt.get(g * head_dim..(g + 1) * head_dim).unwrap_or(&[]));
                            for (row, v) in tile.iter_mut().zip(sc) {
                                if let Some(x) = row.get_mut(g * 8..g * 8 + 8) {
                                    for (o, y) in x.iter_mut().zip(v) {
                                        *o = y / scale;
                                    }
                                }
                            }
                        }
                        for (k, row) in tile.iter().enumerate().take(four.len()) {
                            let block = row.get(..bn).unwrap_or(&[]);
                            let bmax = max8(block);
                            let at = i * nt + t0 * 4 + k;
                            let (Some(mm), Some(ss)) = (m.get(at).copied(), s.get(at).copied()) else { continue };
                            let nm = mm.max(bmax);
                            if !nm.is_finite() {
                                continue;
                            }
                            let carried = if mm.is_finite() { ss * (mm - nm).exp() } else { 0.0 };
                            let add = exp_sum(block, nm);
                            if let (Some(a), Some(b)) = (m.get_mut(at), s.get_mut(at)) {
                                *a = nm;
                                *b = carried + add;
                            }
                        }
                    }
                }
            }
            b0 += bn;
        }
        #[expect(clippy::cast_precision_loss, reason = "a piece's positions are few")]
        let shift = if reduce == Reduce::Mean && n > 0 { (n as f32).ln() } else { 0.0 };
        for i in 0..q.heads.len() {
            let mut acc = 0.0_f32;
            for ti in 0..nt {
                let mm = m.get(i * nt + ti).copied().unwrap_or(f32::NEG_INFINITY);
                let ss = s.get(i * nt + ti).copied().unwrap_or(0.0);
                if mm.is_finite() {
                    acc += mm + ss.ln() - shift;
                }
            }
            if let Some(o) = out.get_mut(i * parts_at.len() + c) {
                *o = acc * per;
            }
        }
    }
    Ok(out)
}

/// Section check (2026-11-10), on the released Qwen3-0.6B on the processor (no device used, so it
/// runs beside a card job): (a) the pair rule and bytes per token for the drone set in
/// `designs/drones.json` (3,328 for the default set); (b) packets cut from encoded chunks hold
/// exactly the keys of their pairs; (c) with queries from the question path over the full window
/// (the exact read's query side), the keys-only scores equal the exact probe's raw scores bit for
/// bit for every drone head, token and position; (d) the question path over the anchor only gives
/// finite scores and their largest difference from the exact scores is reported (the
/// approximation, measured on data by `run keys`); (e) the store's limit and retain rule.
/// Values: `tests/section-checks/nmengn-drone/report.txt`.
/// # Errors
/// One fault line.
pub(crate) fn check() -> Result<(), String> {
    crate::kvstore::p2_report("nmengn-drone", "nmengn.drone (keys-only scan A6, question path A7)", drone_check)
}

fn drone_check(rep: &mut String) -> Result<(), String> {
    use std::fmt::Write as _;
    let home = nmtype::nano_root();
    let set = crate::kvstore::drone_set(&home, crate::kvstore::DRONE_FALLBACK_POPULATION).map_err(|e| e.to_string())?;
    let pop = crate::model::Population::open(&home, crate::kvstore::DRONE_FALLBACK_POPULATION, true, false).map_err(|e| e.to_string())?;
    let asm = crate::assembly::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
    let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
    let d = m.pop.dims;
    let group = d.heads / d.kv_heads.max(1);
    let pr = pairs(&set.heads, group);
    let bpt = bytes_per_token(&pr, d.head_dim);
    let _ = writeln!(
        rep,
        "(a) drone set {} heads ({}); {} distinct (layer, key-value head) pairs; {bpt} bytes per token (full keys and values of layers 0..={}: {})",
        set.heads.len(),
        set.source,
        pr.len(),
        set.last_layer(),
        crate::kvstore::bytes_per_token(&m, set.last_layer() + 1)
    );
    if pairs(&[(3, 0), (3, 1), (3, 2), (4, 1)], 2) != vec![(3, 0), (3, 1), (4, 0)] {
        return Err("pair rule".into());
    }
    // Encode the anchor and three chunks on the processor.
    let last = set.last_layer();
    let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
    let ak =
        Arc::new(crate::kvstore::encode_through(&m, Device::Cpu, None, &tok(crate::kvstore::P2_ANCHOR)?, last).map_err(|e| e.to_string())?);
    let aw = Arc::new(Window::build(&m, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
    let mut parts = vec![Arc::clone(&ak)];
    for c in crate::kvstore::P2_CHUNKS {
        parts.push(Arc::new(crate::kvstore::encode_through(&m, Device::Cpu, Some(&aw), &tok(c)?, last).map_err(|e| e.to_string())?));
    }
    let w = Arc::new(Window::build(&m, &parts).map_err(|e| e.to_string())?);
    let packets: Vec<Arc<RelevanceKeys>> = parts
        .iter()
        .map(|c| RelevanceKeys::from_chunk(c, &pr, d.kv_heads, d.head_dim).map(Arc::new))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    let held: usize = packets.iter().map(|p| p.bytes()).sum();
    let toks: usize = packets.iter().map(|p| p.tokens).sum();
    // (b) Each packet row equals the chunk's key columns of its pair.
    let mut cut_ok = held == toks * bpt;
    for (p, c) in packets.iter().zip(&parts) {
        for (k, &(l, g)) in pr.iter().enumerate() {
            for t in 0..c.tokens {
                let a = p.rows.get(k).and_then(|r| r.get(t * d.head_dim..(t + 1) * d.head_dim));
                let b = c
                    .k_pre
                    .get(l)
                    .and_then(|r| r.get(t * d.kv_heads * d.head_dim + g * d.head_dim..t * d.kv_heads * d.head_dim + (g + 1) * d.head_dim));
                cut_ok &= a.is_some() && a == b;
            }
        }
    }
    let _ = writeln!(
        rep,
        "(b) {} packets over {toks} tokens hold {held} bytes ({bpt} per token); rows equal the chunks' key columns: {cut_ok}",
        packets.len()
    );
    if !cut_ok {
        return Err("packets do not hold the chunks' keys".into());
    }
    // (c) Exact queries: the keys-only scores equal the probe's scores.
    let text = "\nQuestion: where is the ring buffer pushed?";
    let ids = tok(text)?;
    let probe = crate::kvstore::logit_probe(&m, Device::Cpu, &w, &set.heads, last, text, true)?;
    let exact_q = question_path(&m, Device::Cpu, &w, &ids, w.len, &set.heads).map_err(|e| e.to_string())?;
    let scores = scan_keys(&exact_q, &packets, w.len, w.table(), d.head_dim, group).map_err(|e| e.to_string())?;
    let (mut equal, mut values) = (true, 0_usize);
    for (i, &(l, h)) in set.heads.iter().enumerate() {
        let lg = probe.logits(l, h).ok_or_else(|| format!("no probe logits for layer {l} head {h}"))?;
        let s = scores.get(i).ok_or("missing head")?;
        equal &= crate::kvstore::same_bits(&lg.logits, s);
        values += s.len();
    }
    let _ = writeln!(
        rep,
        "(c) exact queries (question path over the full window, {} tokens at position {}): keys-only scores equal the probe's raw scores bit for bit over {values} values: {equal}",
        ids.len(),
        w.len
    );
    if !equal {
        return Err("keys-only scores with exact queries differ from the probe".into());
    }
    // Repeat: deterministic.
    let again = scan_keys(&exact_q, &packets, w.len, w.table(), d.head_dim, group).map_err(|e| e.to_string())?;
    if again != scores {
        return Err("keys-only scan is not repeatable".into());
    }
    // (d) The question path over the anchor only, at the window's end.
    let approx_q = question_path(&m, Device::Cpu, &aw, &ids, w.len, &set.heads).map_err(|e| e.to_string())?;
    let approx = scan_keys(&approx_q, &packets, w.len, w.table(), d.head_dim, group).map_err(|e| e.to_string())?;
    let mut worst = 0.0_f32;
    let mut finite = true;
    for (a, b) in approx.iter().zip(&scores) {
        for (x, y) in a.iter().zip(b) {
            finite &= x.is_finite();
            worst = worst.max((x - y).abs());
        }
    }
    let _ = writeln!(
        rep,
        "(d) question path over the anchor only ({} positions) at position {}: scores finite {finite}; largest difference from the exact scores {worst:.3} (the approximation; agreement is measured on data by `run keys`)",
        aw.len, w.len
    );
    if !finite {
        return Err("question path scores not finite".into());
    }
    // (e) The store.
    let one = packets.first().cloned().ok_or("no packet")?;
    let mut st = KeyStore::new(one.bytes() * 2);
    st.put("a", Arc::clone(&one));
    st.put("b", Arc::clone(&one));
    let _ = st.get("a");
    let dropped = st.put("c", Arc::clone(&one));
    let order_ok = dropped == 1 && st.contains("a") && st.contains("c") && !st.contains("b") && st.bytes() == one.bytes() * 2;
    let live: std::collections::BTreeSet<String> = ["c".to_owned()].into_iter().collect();
    let gone = st.retain(&live);
    let _ = writeln!(
        rep,
        "(e) store: least recently used dropped at the limit {order_ok}; retain dropped {gone}, {} left, {} bytes",
        st.len(),
        st.bytes()
    );
    if !order_ok || gone != 1 || st.len() != 1 || st.bytes() != one.bytes() {
        return Err("key store limit or retain rule".into());
    }
    Ok(())
}

