//! CPU kernels with the float16 rounding points of the cukern kernels and PyTorch float16
//! (research note section 3; reference: emma-labs `native/rscore/backnd` cpu and attention,
//! cukern element, linear, attention). Activations are f32 values exactly representable in
//! binary16; sums run in f32; outputs are rounded with `r16`. Weights are read from the
//! cluster views (bfloat16 bytes, shared, never copied whole) and widened row by row.
use crate::half::{r16, w16};
use nmnpcs::View;
use std::sync::Arc;

/// The processor's thread count (the one budget, `nmtype::budget`, read once). The kernels cut
/// their work into this many partitions at most; the workers that run them are lent by the budget
/// (2026-11-18, Phase C1), so a kernel inside another parallel stage adds no threads.
/// Roadmap Phase 4.5 (2026-11-25): the partition count follows the hardware
/// (`budget::hardware`), not the user's live thread limit, so a kernel cuts its work the same way
/// and gives the same sums bit for bit whatever the limit; the limit caps the workers the budget
/// lends to run those partitions.
#[must_use]
pub fn threads() -> usize {
    nmtype::budget::hardware()
}

/// Weight-only int8 precision of one whole tensor (research note section 3; the recording's
/// oracle and blkstk): per output row, scale = absmax / 127 (stored binary16), values =
/// round(w / scale) clamped to 127; the weight used is `r16(value * scale)`. Quantized from the
/// bfloat16 values once per tensor and shared by every cluster of the tensor (an output row's
/// scale spans all its columns, so column clusters share their rows' scales).
#[derive(Debug)]
pub struct Q8 {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub k: usize,
}
impl Q8 {
    /// Quantize a whole bfloat16 tensor `[n, k]`.
    /// # Errors
    /// A malformed tensor.
    pub fn from_bf16(bytes: &[u8], k: usize) -> Result<Self, String> {
        let w: Vec<f32> = bytes.as_chunks::<2>().0.iter().map(|s| nmsafe::bfloat16_to_f32(u16::from_le_bytes(*s))).collect();
        let q = nmsafe::quantize(&w, k).map_err(|e| e.to_string())?;
        Ok(Self {
            values: q.values.iter().map(|b| i8::from_ne_bytes([*b])).collect(),
            scales: q.scales.iter().map(|s| crate::half::from16(*s)).collect(),
            k,
        })
    }
    /// The 255 weights an output row can hold (`r16(value * scale)` for every int8 value),
    /// indexed by `value + 127`; built once per row so dequantizing a row is one table read
    /// per element instead of a multiply and a rounding (the S7 CPU decode lever).
    #[inline]
    fn row_table(&self, row: usize) -> [f32; 255] {
        let s = self.scales.get(row).copied().unwrap_or(0.0);
        let mut t = [0.0_f32; 255];
        for (i, x) in t.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "i is below 255")]
            let v = i as f32 - 127.0;
            *x = r16(v * s);
        }
        t
    }
    /// Dequantize `dst.len()` weights of `row` from column `col0` into `dst`.
    #[inline]
    fn row_into(&self, row: usize, col0: usize, dst: &mut [f32]) {
        let t = self.row_table(row);
        let start = row * self.k + col0;
        let vals = self.values.get(start..start + dst.len()).unwrap_or(&[]);
        for (d, v) in dst.iter_mut().zip(vals) {
            *d = t.get(usize::try_from(i16::from(*v) + 127).unwrap_or(0)).copied().unwrap_or(0.0);
        }
    }
}

/// A linear map over cluster views: `rows` stacks the blocks' output rows (query, key, value,
/// gate, up, down, head), `cols` stacks their input columns (the output projection over a
/// head range). With `q8` set (one per block, the block's whole-tensor quantization) the
/// weights are the int8 values; otherwise the binary16 values.
///
/// Weight placement (2026-11-27): a bound model's map holds no host weights of its own; a held
/// copy (`Model::held`) carries the blocks' 8-bit packets (`q8`, when `int8`) or their tensor
/// bytes (`pins`) for as long as one reader uses it, read again on demand when the host copies
/// were dropped after the weights went to the card.
#[derive(Clone, Debug)]
pub struct Linear {
    pub blocks: Vec<Arc<View>>,
    pub q8: Vec<Arc<Q8>>,
    /// The map is an 8-bit map (its node's precision is `q8`), whether or not `q8` is filled.
    pub int8: bool,
    /// The blocks' tensor bytes, one per block, on a held copy (empty: read from the views).
    pub pins: Vec<Arc<[u8]>>,
    /// 4-bit precision (2026-11-14): the map's weights rounded to 4 bits (groups of `Q4_GROUP`
    /// along the input, one binary16 scale each, values -8 to 7), packed two to a byte; `row`
    /// unpacks them, and the card unpacks them per call (`q4_unpack_kernel`).
    pub q4: Option<Arc<Q4>>,
    pub cols: bool,
    /// Output width and input width.
    pub n: usize,
    pub k: usize,
}
impl Linear {
    /// Build from views (each a 2-D region), checking that the blocks fit together.
    /// # Errors
    /// Blocks of different widths.
    pub fn new(blocks: Vec<Arc<View>>, cols: bool) -> Result<Self, String> {
        let dims: Vec<(usize, usize)> = blocks.iter().map(|b| b.cluster.dims()).collect();
        let (n, k) = if cols {
            let r = dims.first().map_or(0, |d| d.0);
            if dims.iter().any(|d| d.0 != r) {
                return Err("column blocks of different heights".into());
            }
            (r, dims.iter().map(|d| d.1).sum())
        } else {
            let c = dims.first().map_or(0, |d| d.1);
            if dims.iter().any(|d| d.1 != c) {
                return Err("row blocks of different widths".into());
            }
            (dims.iter().map(|d| d.0).sum(), c)
        };
        Ok(Self { blocks, q8: Vec::new(), int8: false, pins: Vec::new(), q4: None, cols, n, k })
    }
    /// The tensor bytes of block `i`: its pin on a held copy, else the view's copy in RAM
    /// (empty when that is dropped and no reader holds it).
    fn block_bytes(&self, i: usize) -> Arc<[u8]> {
        self.pins.get(i).cloned().or_else(|| self.blocks.get(i).and_then(|b| b.bytes())).unwrap_or_else(|| Arc::from(Vec::new()))
    }
    /// Whether every block's bytes are reachable now (pinned, or in RAM).
    #[must_use]
    pub fn bytes_held(&self) -> bool {
        self.pins.len() == self.blocks.len() || self.blocks.iter().all(|b| b.bytes().is_some())
    }
    /// Output row `row` as f32 weights (binary16 values), into `buf` (length `k`).
    pub fn row(&self, row: usize, buf: &mut [f32]) {
        if let Some(q) = &self.q4 {
            q.row_into(row, self.k, buf);
            return;
        }
        if !self.q8.is_empty() {
            self.row_q8(row, buf);
            return;
        }
        let fill = |dst: &mut [f32], src: &[u8]| {
            for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = w16(u16::from_le_bytes(*s));
            }
        };
        // A pinned block is read in place (no reference count touched per row); otherwise the
        // view's copy in RAM is taken once for this row.
        if self.cols {
            let mut at = 0;
            for (i, b) in self.blocks.iter().enumerate() {
                let c = b.cluster.dims().1;
                if let Some(d) = buf.get_mut(at..at + c) {
                    let taken;
                    let bytes: &[u8] = if let Some(p) = self.pins.get(i) {
                        p
                    } else {
                        taken = self.block_bytes(i);
                        &taken
                    };
                    fill(d, b.row_in(bytes, row, 2));
                }
                at += c;
            }
        } else {
            let mut base = 0;
            for (i, b) in self.blocks.iter().enumerate() {
                let r = b.cluster.dims().0;
                if row < base + r {
                    let taken;
                    let bytes: &[u8] = if let Some(p) = self.pins.get(i) {
                        p
                    } else {
                        taken = self.block_bytes(i);
                        &taken
                    };
                    fill(buf, b.row_in(bytes, row - base, 2));
                    return;
                }
                base += r;
            }
        }
    }
    fn row_q8(&self, row: usize, buf: &mut [f32]) {
        use nmnpcs::Region;
        let mut at = 0;
        let mut base = 0;
        for (b, q) in self.blocks.iter().zip(&self.q8) {
            let (r, c) = b.cluster.dims();
            match &b.cluster.region {
                Region::Cols(cr) => {
                    if let Some(d) = buf.get_mut(at..at + c) {
                        q.row_into(row, cr.start, d);
                    }
                    at += c;
                }
                region => {
                    if row < base + r {
                        let gr = match region {
                            Region::Rows(rr) => rr.start + row - base,
                            _ => row - base,
                        };
                        let end = c.min(buf.len());
                        if let Some(d) = buf.get_mut(..end) {
                            q.row_into(gr, 0, d);
                        }
                        return;
                    }
                    base += r;
                }
            }
        }
    }
    /// The binary16 row scales (`[n]`) of a q8 map.
    #[must_use]
    pub fn scales16(&self) -> Vec<u8> {
        use nmnpcs::Region;
        let mut out = Vec::with_capacity(self.n * 2);
        for row in 0..self.n {
            let mut base = 0;
            let mut scale = 0.0_f32;
            for (b, q) in self.blocks.iter().zip(&self.q8) {
                let r = b.cluster.dims().0;
                match &b.cluster.region {
                    Region::Cols(_) => scale = q.scales.get(row).copied().unwrap_or(0.0),
                    Region::Rows(rr) if row >= base && row < base + r => {
                        scale = q.scales.get(rr.start + row - base).copied().unwrap_or(0.0)
                    }
                    Region::Whole if row >= base && row < base + r => scale = q.scales.get(row - base).copied().unwrap_or(0.0),
                    _ => {}
                }
                base += r;
            }
            out.extend(crate::half::bits16(scale).to_le_bytes());
        }
        out
    }
    /// Packed int8 values (`[n, k]`) and binary16 row scales (`[n]`), for a device upload.
    #[must_use]
    pub fn packed_q8(&self) -> (Vec<u8>, Vec<u8>) {
        use nmnpcs::Region;
        let (mut vals, mut scales) = (Vec::with_capacity(self.n * self.k), Vec::with_capacity(self.n * 2));
        for row in 0..self.n {
            let mut base = 0;
            let mut scale = 0.0_f32;
            for (b, q) in self.blocks.iter().zip(&self.q8) {
                let (r, c) = b.cluster.dims();
                match &b.cluster.region {
                    Region::Cols(cr) => {
                        vals.extend(
                            q.values.get(row * q.k + cr.start..row * q.k + cr.start + c).unwrap_or(&[]).iter().map(|v| v.to_ne_bytes()[0]),
                        );
                        scale = q.scales.get(row).copied().unwrap_or(0.0);
                    }
                    region => {
                        if row >= base && row < base + r {
                            let gr = match region {
                                Region::Rows(rr) => rr.start + row - base,
                                _ => row - base,
                            };
                            vals.extend(q.values.get(gr * q.k..gr * q.k + c).unwrap_or(&[]).iter().map(|v| v.to_ne_bytes()[0]));
                            scale = q.scales.get(gr).copied().unwrap_or(0.0);
                        }
                        base += r;
                    }
                }
            }
            scales.extend(crate::half::bits16(scale).to_le_bytes());
        }
        (vals, scales)
    }
    /// The packed binary16 bytes of the rows `rows` only, in that order (`[rows.len(), k]`): the
    /// same bytes `packed16` holds for those rows. For a table lookup done on the host
    /// (2026-11-10, parameter loader step 2): the device then holds the rows a job uses, never
    /// the whole table. `None` when a row is outside the map.
    #[must_use]
    pub fn packed16_rows(&self, rows: &[usize]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(rows.len() * self.k * 2);
        let mut buf = vec![0.0_f32; self.k];
        for &r in rows {
            if r >= self.n {
                return None;
            }
            self.row(r, &mut buf);
            out.extend(buf.iter().flat_map(|v| crate::half::bits16(*v).to_le_bytes()));
        }
        Some(out)
    }
    /// The packed binary16 bytes of the whole map (`[n, k]`), for a device upload.
    #[must_use]
    pub fn packed16(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n * self.k * 2);
        let mut buf = vec![0.0_f32; self.k];
        for r in 0..self.n {
            self.row(r, &mut buf);
            out.extend(buf.iter().flat_map(|v| crate::half::bits16(*v).to_le_bytes()));
        }
        out
    }
}

/// Input columns per 4-bit scale group.
pub const Q4_GROUP: usize = 32;

/// A map's 4-bit weights: `packed` `[n, k / 2]` (low nibble first, value + 8), `scales`
/// `[n, k / Q4_GROUP]` binary16 bits; the weight used is `r16(value * scale)`.
#[derive(Debug)]
pub struct Q4 {
    pub packed: Vec<u8>,
    pub scales: Vec<u16>,
}
impl Q4 {
    fn row_into(&self, row: usize, k: usize, dst: &mut [f32]) {
        let groups = k / Q4_GROUP;
        for (j, d) in dst.iter_mut().enumerate().take(k) {
            let e = row * k + j;
            let b = self.packed.get(e / 2).copied().unwrap_or(0x88);
            let nib = if e.is_multiple_of(2) { b & 15 } else { b >> 4 };
            let s = crate::half::from16(self.scales.get(row * groups + j / Q4_GROUP).copied().unwrap_or(0));
            *d = r16(f32::from(i16::from(nib) as i8 - 8) * s);
        }
    }
}

impl Linear {
    /// Round the map's binary16 weights to 4 bits (groups of `Q4_GROUP` along each row, scale =
    /// absmax / 7 rounded to binary16, values -8 to 7) and keep them packed (`q4`). A map whose
    /// input width is not a multiple of `Q4_GROUP` stays at binary16.
    #[must_use]
    pub fn with_q4(mut self) -> Self {
        if !self.k.is_multiple_of(Q4_GROUP) || self.n == 0 {
            return self;
        }
        let groups = self.k / Q4_GROUP;
        let (mut packed, mut scales) = (vec![0_u8; self.n * self.k / 2], Vec::with_capacity(self.n * groups));
        let mut buf = vec![0.0_f32; self.k];
        for r in 0..self.n {
            self.row(r, &mut buf);
            for (gi, g) in buf.chunks(Q4_GROUP).enumerate() {
                let s = r16(g.iter().fold(0.0_f32, |m, x| m.max(x.abs())).max(1e-8) / 7.0);
                scales.push(crate::half::bits16(s));
                for (j, x) in g.iter().enumerate() {
                    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "value in -8..=7 plus 8")]
                    let v = ((x / s).round().clamp(-8.0, 7.0) + 8.0) as u8;
                    let e = r * self.k + gi * Q4_GROUP + j;
                    if let Some(b) = packed.get_mut(e / 2) {
                        *b |= if e.is_multiple_of(2) { v } else { v << 4 };
                    }
                }
            }
        }
        self.q4 = Some(Arc::new(Q4 { packed, scales }));
        self
    }
}

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0.0_f32; 8];
    let (ac, ar) = a.as_chunks::<8>();
    let (bc, br) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..8 {
            s[i] += x[i] * y[i];
        }
    }
    let mut t = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
    for (x, y) in ar.iter().zip(br) {
        t += x * y;
    }
    t
}

/// `y[t, n] = r16(sum_k x[t, k] * w[n, k])` for `tokens` rows of `x` (row-parallel).
#[must_use]
pub fn linear(x: &[f32], tokens: usize, w: &Linear) -> Vec<f32> {
    let (n, k) = (w.n, w.k);
    let work = n * k * tokens;
    let parts = if work < 2_000_000 { 1 } else { threads().min(n) };
    let chunk = n.div_ceil(parts.max(1));
    let run = |r0: usize, r1: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; (r1 - r0) * tokens];
        let mut buf = vec![0.0_f32; k];
        for r in r0..r1 {
            w.row(r, &mut buf);
            for t in 0..tokens {
                let xs = x.get(t * k..(t + 1) * k).unwrap_or(&[]);
                if let Some(o) = out.get_mut(t * (r1 - r0) + (r - r0)) {
                    *o = r16(dot(xs, &buf));
                }
            }
        }
        out
    };
    let pieces: Vec<(usize, usize, Vec<f32>)> = if parts <= 1 {
        vec![(0, n, run(0, n))]
    } else {
        let bounds: Vec<(usize, usize)> = (0..parts).map(|p| (p * chunk, ((p + 1) * chunk).min(n))).filter(|(a, b)| a < b).collect();
        let outs = nmtype::budget::run(bounds.len(), |p| bounds.get(p).map_or_else(Vec::new, |&(a, b)| run(a, b)));
        bounds.into_iter().zip(outs).map(|((a, b), o)| (a, b, o.unwrap_or_default())).collect()
    };
    let mut y = vec![0.0_f32; n * tokens];
    for (a, b, out) in pieces {
        let width = b - a;
        for t in 0..tokens {
            if let (Some(dst), Some(src)) = (y.get_mut(t * n + a..t * n + b), out.get(t * width..(t + 1) * width)) {
                dst.copy_from_slice(src);
            }
        }
    }
    y
}

/// RMSNorm over rows of `width`: `r16(r16(x * f) * w)`, `f = 1/sqrt(mean(x^2) + eps)`.
#[must_use]
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let width = w.len().max(1);
    let mut y = vec![0.0_f32; x.len()];
    for (row, out) in x.chunks_exact(width).zip(y.chunks_exact_mut(width)) {
        let mut sum = 0.0_f32;
        for v in row {
            sum += v * v;
        }
        #[expect(clippy::cast_precision_loss, reason = "a norm width is small")]
        let f = 1.0 / (sum / width as f32 + eps).sqrt();
        for ((o, v), g) in out.iter_mut().zip(row).zip(w) {
            *o = r16(r16(v * f) * g);
        }
    }
    y
}

/// Rope frequencies for head width `d`.
fn rope_freq(d: usize, theta: f32) -> Vec<f32> {
    #[expect(clippy::cast_precision_loss, reason = "dims are small")]
    (0..d / 2).map(|j| 1.0 / theta.powf((2 * j) as f32 / d as f32)).collect()
}
/// One position's cosines and sines, rounded as the engine uses them.
fn rope_cs(p: i32, freq: &[f32]) -> impl Iterator<Item = (f32, f32)> + '_ {
    #[expect(clippy::cast_precision_loss, reason = "positions are small")]
    freq.iter().map(move |f| {
        let angle = p as f32 * f;
        (r16(angle.cos()), r16(angle.sin()))
    })
}
/// Rotate one head's `d` values (rotate-half layout) by one position's cosines and sines.
#[inline]
pub(crate) fn rotate(x: &mut [f32], cs: &[(f32, f32)]) {
    let half = cs.len();
    let (a, b) = x.split_at_mut(half.min(x.len()));
    for ((l, r), &(c, s)) in a.iter_mut().zip(b.iter_mut()).zip(cs) {
        let (lv, rv) = (*l, *r);
        *l = r16(r16(lv * c) + r16(-rv * s));
        *r = r16(r16(rv * c) + r16(lv * s));
    }
}

/// Every position `0..len`'s rope cosines and sines (`[len, d / 2]`), the values `rope` uses:
/// a reference-only window's whole position state (E3.1).
#[must_use]
pub fn rope_table(len: usize, d: usize, theta: f32) -> Vec<(f32, f32)> {
    let freq = rope_freq(d, theta);
    (0..len).flat_map(|p| rope_cs(i32::try_from(p).unwrap_or(i32::MAX), &freq).collect::<Vec<_>>()).collect()
}

/// Rotary position encoding in place, rotate-half layout, on `[tokens, heads, d]`.
pub fn rope(x: &mut [f32], heads: usize, d: usize, pos: &[i32], theta: f32) {
    // Frequencies once, and each position's cosines and sines once for all heads (E1,
    // 2026-11-10: the same values as before, computed heads times fewer).
    let freq = rope_freq(d, theta);
    for (t, p) in pos.iter().enumerate() {
        let cs: Vec<(f32, f32)> = rope_cs(*p, &freq).collect();
        for h in 0..heads {
            if let Some(head) = x.get_mut((t * heads + h) * d..(t * heads + h + 1) * d) {
                rotate(head, &cs);
            }
        }
    }
}

/// Causal grouped-query attention. `q` is `[tokens, heads, d]`, `k` and `v` are
/// `[cached, kv, d]` with the new tokens last; masked heads give exact zeros. Output
/// `[tokens, heads * d]`.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the attention geometry is seven numbers")]
pub fn attention(q: &[f32], k: &[f32], v: &[f32], tokens: usize, heads: usize, kv: usize, d: usize, off: &[bool]) -> Vec<f32> {
    let cached = k.len() / (kv * d).max(1);
    let group = (heads / kv.max(1)).max(1);
    #[expect(clippy::cast_precision_loss, reason = "a head width is small")]
    let scale = (d as f32).sqrt();
    let per_head = |h: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; tokens * d];
        if off.get(h).copied().unwrap_or(false) {
            return out;
        }
        let g = h / group;
        let mut s = vec![0.0_f32; cached];
        for t in 0..tokens {
            let visible = cached - tokens + t + 1;
            let qr = q.get((t * heads + h) * d..(t * heads + h + 1) * d).unwrap_or(&[]);
            let mut max = f32::NEG_INFINITY;
            for (p, sp) in s.iter_mut().enumerate().take(visible) {
                let kr = k.get((p * kv + g) * d..(p * kv + g + 1) * d).unwrap_or(&[]);
                let mut sum = 0.0_f32;
                for (a, b) in qr.iter().zip(kr) {
                    sum += a * b;
                }
                *sp = sum / scale;
                max = max.max(*sp);
            }
            let mut total = 0.0_f32;
            for sp in s.iter_mut().take(visible) {
                *sp = (*sp - max).exp();
                total += *sp;
            }
            for sp in s.iter_mut().take(visible) {
                *sp = r16(*sp / total);
            }
            for j in 0..d {
                let mut acc = 0.0_f32;
                for (p, sp) in s.iter().enumerate().take(visible) {
                    acc += sp * v.get((p * kv + g) * d + j).copied().unwrap_or(0.0);
                }
                if let Some(o) = out.get_mut(t * d + j) {
                    *o = r16(acc);
                }
            }
        }
        out
    };
    let per: Vec<Vec<f32>> = if tokens * cached * heads < 50_000 {
        (0..heads).map(per_head).collect()
    } else {
        nmtype::budget::run(heads, per_head).into_iter().map(Option::unwrap_or_default).collect()
    };
    let mut y = vec![0.0_f32; tokens * heads * d];
    for (h, out) in per.iter().enumerate() {
        for t in 0..tokens {
            if let (Some(dst), Some(src)) = (y.get_mut((t * heads + h) * d..(t * heads + h + 1) * d), out.get(t * d..(t + 1) * d)) {
                dst.copy_from_slice(src);
            }
        }
    }
    y
}

/// Binary16 bits to f32 by table (65,536 entries, built once): cached keys and values are held
/// as binary16 bits (2026-11-10, E1), which is lossless because every activation is already a
/// binary16 value, and widened row by row as attention streams them.
#[inline]
#[must_use]
pub fn wide(b: u16) -> f32 {
    static T: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| (0..=u16::MAX).map(crate::half::from16).collect());
    t.get(usize::from(b)).copied().unwrap_or(0.0)
}

/// Keys or values in segments: cached binary16 rows first (a shared window, then a decode
/// cache), the new tokens' f32 rows last; `row(p)` is position `p` of the whole sequence.
/// The first `pre` cached segments hold keys before rope (a reference-only window, E3.1): each
/// of their rows gets its position's rope from `table` as it is read, with the arithmetic of
/// `rope`, so the values equal keys placed by `rope` ahead of time.
/// 2026-11-10 (step P0): the first `win` cached segments may be laid out wider than this
/// attention component reads (a shared window holds every key-value head of its layer): their
/// rows are `win_width` values apart and this component's columns start at `win_off`. With
/// `win_width` 0 (or equal to `width`) and `win_off` 0 the reads are exactly as before.
#[derive(Default)]
pub struct Rows<'a> {
    pub cached: Vec<&'a [u16]>,
    pub new: &'a [f32],
    pub width: usize,
    pub pre: usize,
    pub table: &'a [(f32, f32)],
    pub head: usize,
    pub win: usize,
    pub win_width: usize,
    pub win_off: usize,
}
impl Rows<'_> {
    /// Row stride of the window segments.
    #[inline]
    fn stride(&self) -> usize {
        if self.win > 0 && self.win_width > 0 { self.win_width } else { self.width }
    }
    /// Positions held.
    #[must_use]
    pub fn len(&self) -> usize {
        let (ws, w) = (self.stride().max(1), self.width.max(1));
        let mut n = 0;
        for (i, s) in self.cached.iter().enumerate() {
            n += if i < self.win { s.len() / ws } else { s.len() / w };
        }
        n + self.new.len() / w
    }
    /// No positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Values `off..off + out.len()` of position `p`, widened into `out`.
    #[inline]
    pub fn read(&self, p: usize, off: usize, out: &mut [f32]) {
        let ws = self.stride();
        // Offset into the window segments (their stride), then into this component's own rows.
        let own = |at: usize| if ws == self.width { at } else { at / ws.max(1) * self.width };
        let mut at = p * ws;
        for (i, s) in self.cached.iter().enumerate() {
            if i == self.win && i > 0 {
                at = own(at);
            }
            let base = if i < self.win { self.win_off } else { 0 };
            if at < s.len() {
                for (o, b) in out.iter_mut().zip(s.get(at + base + off..).unwrap_or(&[])) {
                    *o = wide(*b);
                }
                let half = self.head / 2;
                if i < self.pre && half > 0 && off.is_multiple_of(self.head) {
                    let cs = self.table.get(p * half..(p + 1) * half).unwrap_or(&[]);
                    for head in out.chunks_exact_mut(self.head) {
                        rotate(head, cs);
                    }
                }
                return;
            }
            at -= s.len();
        }
        if self.win > 0 && self.win >= self.cached.len() {
            at = own(at);
        }
        if let Some(src) = self.new.get(at + off..at + off + out.len()) {
            out.copy_from_slice(src);
        }
    }
}

/// Causal grouped-query attention over cached binary16 rows and new f32 rows (E1, 2026-11-10).
/// The arithmetic of `attention`, except that each score is an eight-lane vectorised sum (the
/// rounding points are unchanged; results agree with `attention` within binary16 resolution).
/// Each key and value row is widened once per head and used by every query token, instead of
/// once per token. `q` is `[tokens, heads, d]`; output `[tokens, heads * d]`.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the attention geometry is seven numbers")]
pub fn attention_rows(q: &[f32], k: &Rows<'_>, v: &Rows<'_>, tokens: usize, heads: usize, kv: usize, d: usize, off: &[bool]) -> Vec<f32> {
    let cached = k.len();
    let group = (heads / kv.max(1)).max(1);
    #[expect(clippy::cast_precision_loss, reason = "a head width is small")]
    let scale = (d as f32).sqrt();
    let per_head = |h: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; tokens * d];
        if off.get(h).copied().unwrap_or(false) || tokens == 0 {
            return out;
        }
        let g = h / group;
        let first = cached - tokens;
        // Scores: position-major, so a key row is widened once and serves every token.
        let mut s = vec![0.0_f32; tokens * cached];
        let mut row = vec![0.0_f32; d];
        for p in 0..cached {
            k.read(p, g * d, &mut row);
            for t in p.saturating_sub(first)..tokens {
                let qr = q.get((t * heads + h) * d..(t * heads + h + 1) * d).unwrap_or(&[]);
                // Eight-lane sum (as `linear` uses), vectorised; the order differs from the
                // plain kernel's sequential sum only below binary16 resolution.
                if let Some(x) = s.get_mut(t * cached + p) {
                    *x = dot(qr, &row) / scale;
                }
            }
        }
        for t in 0..tokens {
            let visible = first + t + 1;
            let st = s.get_mut(t * cached..t * cached + visible).unwrap_or(&mut []);
            let max = st.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0_f32;
            for sp in st.iter_mut() {
                *sp = (*sp - max).exp();
                total += *sp;
            }
            for sp in st.iter_mut() {
                *sp = r16(*sp / total);
            }
        }
        // Values: position-major again; per (token, dim) the sum still runs in position order.
        let mut acc = vec![0.0_f32; tokens * d];
        for p in 0..cached {
            v.read(p, g * d, &mut row);
            for t in p.saturating_sub(first)..tokens {
                let w = s.get(t * cached + p).copied().unwrap_or(0.0);
                for (a, x) in acc.get_mut(t * d..(t + 1) * d).unwrap_or(&mut []).iter_mut().zip(&row) {
                    *a += w * x;
                }
            }
        }
        for (o, a) in out.iter_mut().zip(&acc) {
            *o = r16(*a);
        }
        out
    };
    let per: Vec<Vec<f32>> = if tokens * cached * heads < 50_000 {
        (0..heads).map(per_head).collect()
    } else {
        nmtype::budget::run(heads, per_head).into_iter().map(Option::unwrap_or_default).collect()
    };
    let mut y = vec![0.0_f32; tokens * heads * d];
    for (h, out) in per.iter().enumerate() {
        for t in 0..tokens {
            if let (Some(dst), Some(src)) = (y.get_mut((t * heads + h) * d..(t * heads + h + 1) * d), out.get(t * d..(t + 1) * d)) {
                dst.copy_from_slice(src);
            }
        }
    }
    y
}

/// Attention mass per head on the first `span` positions (a shared window), summed over the
/// query tokens: `[heads, span]`, each token's weights normalised over every position it sees.
/// The drone reading of E1 (QRHead-style scoring, arXiv 2506.09944): how much each head's
/// queries land on each window position. `only`: the heads to read (the rest stay zero).
/// Since 2026-11-10 (tentacle core A5) a thin call of `attention_read` without logits: the
/// mass is computed by the same arithmetic in the same order, so it is unchanged bit for bit.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the attention geometry is seven numbers and a head choice")]
pub fn attention_mass(
    q: &[f32],
    k: &Rows<'_>,
    tokens: usize,
    heads: usize,
    kv: usize,
    d: usize,
    span: usize,
    only: Option<&[usize]>,
) -> Vec<f32> {
    attention_read(q, k, tokens, heads, kv, d, span, only, false).0
}

/// One head's raw attention scores on the window (tentacle core A5, 2026-11-10): the scaled
/// dot products before softmax, `logits[t * span + p]` for question token `t` and window
/// position `p`, and per token the log of the softmax normaliser over every position the token
/// sees (`lse[t]`), so `exp(logits[t * span + p] - lse[t])` is the token's attention weight on
/// `p` and its sum over tokens is the recorded mass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeadLogits {
    /// The head, local to its attention component.
    pub head: usize,
    pub logits: Vec<f32>,
    pub lse: Vec<f32>,
}

/// `attention_mass`, and with `logits` also the raw scores of every read head
/// (`HeadLogits`, local head order). The mass arithmetic is the one `attention_mass` always
/// used; the logits are the same `s` values it normalises, copied out before softmax.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the attention geometry is seven numbers, a head choice and the logits switch")]
pub fn attention_read(
    q: &[f32],
    k: &Rows<'_>,
    tokens: usize,
    heads: usize,
    kv: usize,
    d: usize,
    span: usize,
    only: Option<&[usize]>,
    logits: bool,
) -> (Vec<f32>, Vec<HeadLogits>) {
    let cached = k.len();
    let group = (heads / kv.max(1)).max(1);
    #[expect(clippy::cast_precision_loss, reason = "a head width is small")]
    let scale = (d as f32).sqrt();
    let first = cached.saturating_sub(tokens);
    let per_head = |h: usize| -> (Vec<f32>, Option<HeadLogits>) {
        let g = h / group;
        let mut s = vec![0.0_f32; tokens * cached];
        let mut row = vec![0.0_f32; d];
        for p in 0..cached {
            k.read(p, g * d, &mut row);
            for t in p.saturating_sub(first)..tokens {
                let qr = q.get((t * heads + h) * d..(t * heads + h + 1) * d).unwrap_or(&[]);
                if let Some(x) = s.get_mut(t * cached + p) {
                    *x = dot(qr, &row) / scale;
                }
            }
        }
        let mut mass = vec![0.0_f32; span];
        let mut lse = if logits { Vec::with_capacity(tokens) } else { Vec::new() };
        for t in 0..tokens {
            let st = s.get(t * cached..t * cached + first + t + 1).unwrap_or(&[]);
            let max = st.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = st.iter().map(|x| (x - max).exp()).sum();
            for (m, x) in mass.iter_mut().zip(st) {
                *m += (x - max).exp() / total;
            }
            if logits {
                lse.push(max + total.ln());
            }
        }
        let hl = logits.then(|| {
            let mut l = Vec::with_capacity(tokens * span);
            for t in 0..tokens {
                let row = s.get(t * cached..t * cached + span.min(cached)).unwrap_or(&[]);
                l.extend_from_slice(row);
                l.resize((t + 1) * span, f32::NEG_INFINITY);
            }
            HeadLogits { head: h, logits: l, lse }
        });
        (mass, hl)
    };
    let per: Vec<(Vec<f32>, Option<HeadLogits>)> =
        nmtype::budget::run(heads, |h| if only.is_none_or(|o| o.contains(&h)) { per_head(h) } else { (vec![0.0; span], None) })
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect();
    let mut mass = Vec::with_capacity(heads * span);
    let mut out = Vec::new();
    for (m, l) in per {
        mass.extend(m);
        out.extend(l);
    }
    (mass, out)
}

/// `r16(r16(silu(g)) * u)`.
#[must_use]
pub fn swigate(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.iter().zip(u).map(|(a, b)| r16(r16(a / (1.0 + (-a).exp())) * b)).collect()
}

/// `r16(a + b)`.
#[must_use]
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| r16(x + y)).collect()
}

/// Embedding rows for `ids` from the table view (binary16 values).
#[must_use]
pub fn embed(table: &View, ids: &[i32]) -> Vec<f32> {
    embed_from(table, &table.tensor_bytes(), ids)
}

/// `embed` over the table's tensor bytes `bytes` (a held copy, `Provisioner::bytes`).
#[must_use]
pub fn embed_from(table: &View, bytes: &[u8], ids: &[i32]) -> Vec<f32> {
    let width = table.cluster.dims().1;
    let mut y = Vec::with_capacity(ids.len() * width);
    for id in ids {
        let row = usize::try_from(*id).unwrap_or(0);
        y.extend(table.row_in(bytes, row, 2).as_chunks::<2>().0.iter().map(|s| w16(u16::from_le_bytes(*s))));
    }
    y
}

/// A norm weight vector as binary16 values.
#[must_use]
pub fn vector(view: &View) -> Vec<f32> {
    vector_from(view, &view.tensor_bytes())
}

/// `vector` over the view's tensor bytes `bytes`.
#[must_use]
pub fn vector_from(view: &View, bytes: &[u8]) -> Vec<f32> {
    view.row_in(bytes, 0, 2).as_chunks::<2>().0.iter().map(|s| w16(u16::from_le_bytes(*s))).collect()
}

/// Self-check of `cpu`: a 2x2 attention with one masked head, rmsnorm of a known row, the
/// rope of position 0 is the identity, swigate and add round to binary16.
pub(crate) fn check() -> Result<(), String> {
    let n = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 0.0);
    if (n[0] - r16(3.0 / 12.5_f32.sqrt())).abs() > 1e-6 {
        return Err(format!("nmengn.cpu: rmsnorm {n:?}"));
    }
    let mut x = vec![0.5, -0.25, 1.0, 2.0];
    rope(&mut x, 1, 4, &[0], 1_000_000.0);
    if x != [0.5, -0.25, 1.0, 2.0] {
        return Err("nmengn.cpu: rope at position 0".into());
    }
    let y = attention(&[1.0, 0.0, 1.0, 0.0], &[1.0, 0.0], &[2.0, 3.0], 1, 2, 1, 2, &[false, true]);
    if y != [2.0, 3.0, 0.0, 0.0] {
        return Err(format!("nmengn.cpu: attention {y:?}"));
    }
    // The streaming kernel over cached binary16 rows agrees with the plain kernel within one
    // binary16 step (3 cached positions, 2 new tokens, 4 heads over 2 key-value heads, width 4).
    let val = |i: usize| r16(((i * 37 % 23) as f32 - 11.0) / 7.0);
    let (tok, heads, kv, d, cached) = (2, 4, 2, 4, 3);
    let q: Vec<f32> = (0..tok * heads * d).map(|i| val(i + 5)).collect();
    let k: Vec<f32> = (0..(cached + tok) * kv * d).map(val).collect();
    let v: Vec<f32> = (0..(cached + tok) * kv * d).map(|i| val(i + 11)).collect();
    let split = cached * kv * d;
    let (ck, cv): (Vec<u16>, Vec<u16>) =
        (k[..split].iter().map(|x| crate::half::bits16(*x)).collect(), v[..split].iter().map(|x| crate::half::bits16(*x)).collect());
    let kr = Rows { cached: vec![&ck], new: &k[split..], width: kv * d, ..Rows::default() };
    let vr = Rows { cached: vec![&cv], new: &v[split..], width: kv * d, ..Rows::default() };
    let off = [false, false, true, false];
    let (fast, plain) = (attention_rows(&q, &kr, &vr, tok, heads, kv, d, &off), attention(&q, &k, &v, tok, heads, kv, d, &off));
    if fast.len() != plain.len() || fast.iter().zip(&plain).any(|(a, b)| (a - b).abs() > a.abs().max(b.abs()) / 512.0 + 1e-6) {
        return Err("nmengn.cpu: streaming attention differs from the plain kernel".into());
    }
    if add(&[1.0], &[1.0 / 4096.0]) != [1.0] || swigate(&[0.0], &[5.0]) != [0.0] {
        return Err("nmengn.cpu: elementwise rounding".into());
    }
    Ok(())
}
