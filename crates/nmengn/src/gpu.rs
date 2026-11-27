//! The GPU path: the copied cukern kernels launched through `nmffi::Gpu` (no `unsafe` here),
//! with the launch geometry of the cbridg host code (`element_launch`, `launch_linear`,
//! `attention_launch`, `prefill_launch`). Activations are binary16 device buffers; weights are
//! packed binary16 uploads of the bound clusters (one device copy per component binding,
//! recorded in NPCS residency).
use nmffi::{DevBuf, Geometry, Gpu, OpArgs, Param, TensorArg};
use std::sync::Arc;

pub(crate) const F16: i32 = 2;
const I32: i32 = 5;
const U8: i32 = 6;
pub(crate) const I8: i32 = 4;
/// Short batches (2 to 15 tokens) also take the tiled kernel: cbridg sends them to the
/// one-warp-per-row kernel, which measured 180 ms against 117 ms for a longer input on the
/// 0.6B population (runs/S6-place-1794094905865.txt); the tiled kernel handles any count, and
/// the parity gate is rerun after this change.
const TILED_MIN: usize = 2;

/// Half-precision math for dense binary16 linears of 64 tokens or more (the `wide_h2_kernel`):
/// off in the engine, which keeps the reference path's outputs byte for byte; `serve` and `bench`
/// switch it on for the served models (`--half-math off` keeps binary32 products). Inside a
/// `ServedScope` it also gives int8 linears half products (`wide_h2q8_kernel`).
pub static HALF_MATH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 8-bit activations and int8 dot products for weight-only int8 linears of 64 tokens or more
/// (`q8_blocks_kernel` then `wide_dp4a_kernel`): activations in blocks of 32 with one scale each.
/// Forced on in any process by `--dp4a`; without it, on for the forwards of a model whose role
/// passed the integer path's parity check (`int8_dot_part`, `INT8_DOT`) wherever `HALF_MATH` is on
/// (the served models), unless `NANO_DP4A=off` (`dp4a_served`). The half-product scopes keep
/// `wide_h2q8_kernel`; the reference path (half math off) keeps the widening `wide_kernel`.
pub static DP4A: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether 8-bit linears of 64 tokens or more take the integer path when half math is on: yes,
/// unless the environment sets `NANO_DP4A=off` (the widening kernel, for comparison).
pub fn dp4a_served() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_DP4A").map_or(true, |v| v != "off"))
}

/// Whether a population's 8-bit linears take the integer path by default: the roles whose outputs
/// stayed within the 8-bit closeness of the widening kernel against the 16-bit reference. The
/// embedder passed (least cosine 0.9974 over 16 texts against 0.9979 widening); the reader and the
/// writer did not (greedy tokens identical for 33 and 11 of 64, against 64 of 64 widening); the
/// judge keeps the half products of its scope (equal job measure and seconds). `NANO_INT8_DOT` names
/// the parts instead, comma separated (`embedder`, `base`, `reranker`, `writer`), for a measurement.
#[must_use]
pub fn int8_dot_part(pop_id: &str) -> bool {
    static PARTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let parts = PARTS.get_or_init(|| {
        std::env::var("NANO_INT8_DOT").map_or_else(
            |_| vec!["embedder".to_owned(), "base".to_owned()],
            |v| v.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect(),
        )
    });
    let part = if pop_id.ends_with("qwen3-embedding-0.6b") {
        "embedder"
    } else if pop_id.ends_with("qwen3-reranker-0.6b") {
        "reranker"
    } else if pop_id.ends_with("qwen3-1.7b") {
        "writer"
    } else {
        "base"
    };
    parts.iter().any(|p| p == part)
}
/// Attention of packed sequences over a cached prefix through `segment_kernel` (one key per lane,
/// two query rows per warp). Off in the engine (the prefill kernels are bit-identical to the
/// one-at-a-time path); `serve` and `bench` switch it on for the served models, where it passed
/// the gate for arithmetic changes with half math (`--segment-attention off` keeps the prefill
/// kernels). Inside a `ServedScope` it also sends single sequences to `flash_kernel`.
pub static SEGMENT_ATTENTION: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Set while a model bound at 16 by request runs a forward (`--precision`, 2026-11-13): its
    /// dense linears keep the binary32 math even when `HALF_MATH` is on.
    pub static NO_HALF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set while a model whose role takes the integer path (`int8_dot_part`) runs a forward.
    pub static INT8_DOT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set inside a `ServedScope`.
    static SERVED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set inside a `Q8HalfScope`.
    static Q8_HALF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The last input quantised to 8-bit blocks on this thread, with its blocks and scales.
    /// Weak, so the input is freed when its last reader has run (the address cannot be taken by
    /// another tensor while the weak count holds the allocation of its counter).
    static Q8_LAST: std::cell::RefCell<Option<(std::sync::Weak<DevBuf>, DevT, DevT)>> = const { std::cell::RefCell::new(None) };
    /// Head masks on the card by content and device (lane S): uploaded once each.
    static MASKS: std::cell::RefCell<Vec<(Gpu, Vec<u8>, DevT)>> = const { std::cell::RefCell::new(Vec::new()) };
    /// The last int32 ids uploaded on this thread (a run's positions, read by every rope node).
    static IDS_LAST: std::cell::RefCell<Option<(Gpu, Vec<i32>, DevT)>> = const { std::cell::RefCell::new(None) };
    /// The last block table of packed sequences (`attention_segments`), read by every layer.
    static TABLE_LAST: std::cell::RefCell<Option<(Gpu, Vec<i32>, DevT)>> = const { std::cell::RefCell::new(None) };
}

/// Drop what this thread holds for the run that ends: its 8-bit activation blocks and its ids.
pub fn clear_run_holds() {
    Q8_LAST.with(|c| c.borrow_mut().take());
    IDS_LAST.with(|c| c.borrow_mut().take());
    TABLE_LAST.with(|c| c.borrow_mut().take());
}

/// Whether packed sequences outside a served scope read their attention in one launch
/// (`segments_kernel`, lane S 2026-11-18): yes unless `NANO_SEGMENTS_ONE=off` (one copy and
/// launch of `segment_kernel` per sequence). Same outputs bit for bit (`check_segments`).
pub fn segments_one() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_SEGMENTS_ONE").map_or(true, |v| v != "off"))
}

/// Whether `attention_segments` can serve this thread now (the switches and the scope; the shapes
/// are checked by the call).
#[must_use]
pub fn segments_ready() -> bool {
    segments_one() && SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed) && (!SERVED.with(std::cell::Cell::get) || segments_served())
}

/// Whether packed sequences inside a served scope also read in one launch (`segments_kernel`,
/// lane S package 2) instead of one copy of the shared prefix and one `flash_kernel` launch per
/// sequence: yes unless `NANO_SEGMENTS_SERVED=off`. With `segments_kernel` (2 rows a warp) it was
/// no faster (T-073); with `segments_wide_kernel` (package 3) the attention of 32 short sequences
/// over a 160-token prefix took 163 ms at 8 rows a warp against 181 ms of flash launches plus
/// 170 ms of copies and allocations. Outputs within rounding of the flash path (largest logit
/// difference 0.023 against separate forwards, yes-no order 32 of 32; both kernels are within
/// rounding of prefill2_kernel); the served path's existing tolerance.
/// Whether served batches with a sequence over `SERVED_SEGMENT_ROWS` read their attention through
/// `flash_segments_kernel` in one launch (lane S package 4): yes unless `NANO_FLASH_SEGMENTS=off`
/// (one copy and one `flash_kernel` launch per sequence). Bit-identical (`check_flash_segments`).
pub fn flash_segments() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_FLASH_SEGMENTS").map_or(true, |v| v != "off"))
}

/// Whether `flash_segments_kernel` takes binary16 products (`flash_segments_h_kernel`, lane S
/// package 5): only with `NANO_FLASH_HALF=on`. Measured not worth its rounding: integration shape
/// 4 x 1,500 0.348 to 0.330 ms a token, 4 x 600 unchanged, and the packed group reads' raw scores
/// move 0.348 from the plain read at 1,500 tokens (0.052 with binary32); see the troubleshooting
/// log. Outputs within 1/64 of `flash_segments_kernel` (`check_flash_h`).
pub fn flash_half() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_FLASH_HALF").is_ok_and(|v| v == "on"))
}

/// Longest sequence a served scope sends to the one-launch packed attention (longer: flash).
const SERVED_SEGMENT_ROWS: usize = 256;

pub fn segments_served() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_SEGMENTS_SERVED").map_or(true, |v| v != "off"))
}

/// Drop this thread's held 8-bit activation blocks once their input is gone (called after a
/// run releases activations), so the blocks live no longer than the input they were made from.
pub fn release_q8_blocks_if_dead() {
    Q8_LAST.with(|c| {
        let dead = c.borrow().as_ref().is_some_and(|(src, _, _)| src.strong_count() == 0);
        if dead {
            c.borrow_mut().take();
        }
    });
}

/// Whether norms, SwiGLU gates, adds and rotary encodings take the lane S element kernels
/// (`norm_warp_kernel`, `binary8_kernel`, `rope2_kernel`; 2026-11-18): yes unless
/// `NANO_FAST_ELEMENT=off`. Same outputs bit for bit (`check_fastel`).
pub fn fast_element() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_FAST_ELEMENT").map_or(true, |v| v != "off"))
}

/// Whether the served half products over int8 weights take `h2q8_tile_kernel` (lane S,
/// 2026-11-18): only with `NANO_H2Q8_TILE=on`. Same outputs bit for bit (`check_h2q8_tile`), but no
/// faster (176.8 against 176.6 ms over 196 launches at 512 tokens: the shared reads were not the
/// limit; T-066), so off; `h2q8_fetch_kernel` replaces it.
pub fn h2q8_tile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_H2Q8_TILE").is_ok_and(|v| v == "on"))
}

/// Whether the served half products over int8 weights take `h2q8_fetch_kernel<4>` (lane S package
/// 2, 2026-11-18: 16-byte activation and 4-byte weight loads, the products of `wide_h2q8_kernel`
/// in the same order): yes unless `NANO_H2Q8_FETCH=off`. Bit-identical (`check_h2q8_tile`); 191.0 to
/// 162.1 ms a pass of the 196 linears at 512 tokens.
pub fn h2q8_fetch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_H2Q8_FETCH").map_or(true, |v| v != "off"))
}

/// Query rows per warp of the one-launch packed attention (lane S package 3): 2 is
/// `segments_kernel` (16 rows a block), 4 and 8 are `segments_wide_kernel` (32 and 64 rows a block,
/// one pass over the prefix for up to 64 rows of a sequence); `NANO_SEGMENTS_PER`, default 4 (32
/// short sequences over a 160-token prefix: attention 275, 112 and 157 ms at 2, 4 and 8). All
/// bit-identical (`check_segments`).
pub fn segments_per() -> usize {
    static ON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_SEGMENTS_PER").ok().and_then(|v| v.parse().ok()).filter(|p| [2, 4, 8].contains(p)).unwrap_or(4))
}

/// Whether plain-path int8 linears of 64 tokens or more take `wide_q8f_kernel` (lane S package 3:
/// 16-byte activation and 4-byte weight loads, the binary32 products of `wide_kernel<8, 4, true>` in
/// the same order): yes unless `NANO_WIDE_Q8F=off`. Bit-identical (`check_wide_q8f`).
pub fn wide_q8f() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_WIDE_Q8F").map_or(true, |v| v != "off"))
}

/// Whether int8 dot-product linears take `dp4a_tile_kernel` (lane S, 2026-11-18): yes unless
/// `NANO_DP4A_TILE=off` (the earlier `wide_dp4a_kernel`, for comparison). Same outputs bit for bit.
pub fn dp4a_tile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NANO_DP4A_TILE").map_or(true, |v| v != "off"))
}
/// Whether `int8_dot_part` roles also use `dp4a_tile_kernel` / `dp4a_tt_kernel<4,8,2>` inside
/// `ServedScope` / `Q8HalfScope` instead of falling back to `h2q8_fetch_kernel` (opt-in via
/// `NANO_SERVED_DP4A=on` for paired A/B verification on Turing `sm_75`).
/// Served-scope `dp4a` mode (`NANO_SERVED_DP4A`): `0` = off (baseline), `1` = `judge` (`Q8HalfScope`
/// uses `dp4a` while `ServedScope` keeps `h2q8_fetch`), `2` = `on` (both `ServedScope` and
/// `Q8HalfScope` use `dp4a`), `3` = `served` (`ServedScope` uses `dp4a` while `Q8HalfScope` keeps
/// `h2q8_fetch`). Roadmap Phase 0 item 6 (2026-11-24): on the full sets `judge` and `on` gave the
/// same answers on both boards, so dp4a inside `ServedScope` moved no answer and only saved time
/// (median 3.26 to 2.94 s); the answer changes came from what both share (dp4a in `Q8HalfScope`, or
/// the 16-token threshold). `served` keeps the served-scope speed without the `Q8HalfScope` change.
/// The quick board then placed every answer change on the 16-token threshold: `on` and `served` at
/// 64 tokens gave the base's answers exactly (first 7, top 3 10, pack 14, every 13, decoys 4/4),
/// `on` 10 percent faster (median 3.306 to 2.967 s). So `on` serves by default with the 64-token
/// threshold (`dp4a_min_tokens`); `NANO_SERVED_DP4A=off` gives the earlier kernels for diagnosis.
pub fn served_dp4a_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("NANO_SERVED_DP4A").as_deref() {
        Ok("off" | "0") => 0,
        Ok("judge") => 1,
        Ok("served") => 3,
        _ => 2,
    })
}
/// Minimum token count for `dp4a` linears (`NANO_DP4A_MIN`, default `64`; `16` sends short
/// `16..=64`-token passes to `dp4a_tt_kernel<4,8,2>` instead of FP32 `tiled_kernel<float>`, which
/// moved answers on the item 6 boards).
pub fn dp4a_min_tokens() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("NANO_DP4A_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            // Roadmap Phase 0 item 6 (2026-11-24): 64 in every mode; 16 moved answers (see
            // `served_dp4a_mode`) and stays reachable as `NANO_DP4A_MIN=16`.
            .map_or(64, |n| n.max(2))
    })
}

/// While alive, int8 linears of 64 tokens or more on this thread take the half products of
/// `wide_h2q8_kernel` when `HALF_MATH` is on, as inside a `ServedScope`; attention and every other
/// kernel choice are unchanged. Outputs equal the widening int8 kernels' within rounding
/// (`check_h2q8`).
#[derive(Debug)]
pub struct Q8HalfScope {
    before: bool,
}
impl Q8HalfScope {
    /// Open the scope on this thread.
    #[must_use]
    pub fn new() -> Self {
        Self { before: Q8_HALF.with(|c| c.replace(true)) }
    }
}
impl Default for Q8HalfScope {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for Q8HalfScope {
    fn drop(&mut self) {
        Q8_HALF.with(|c| c.set(self.before));
    }
}

/// While alive, this thread's runs use the served path's faster kernels where the served switches
/// allow it: one causal sequence's attention (not packed sequences read outside the scope) through
/// `flash_kernel` (64 query rows by 64 keys per tile, register tiles) when `SEGMENT_ATTENTION` is
/// on, and int8 linears of 64 tokens or more through `wide_h2q8_kernel` (the half products of
/// `wide_h2_kernel` over the int8 weights) when `HALF_MATH` is on; shapes that do not fit keep the
/// usual kernels, and with both switches off nothing changes. Outputs equal the usual kernels'
/// within rounding (`check_flash`, `check_h2q8`). The caller opens it around the runs it vouches
/// for (the joint read), so other readers keep the kernels their stored results were made with.
#[derive(Debug)]
pub struct ServedScope {
    before: bool,
}
impl ServedScope {
    /// Open the scope on this thread.
    #[must_use]
    pub fn new() -> Self {
        Self { before: SERVED.with(|c| c.replace(true)) }
    }
}
impl Default for ServedScope {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for ServedScope {
    fn drop(&mut self) {
        SERVED.with(|c| c.set(self.before));
    }
}

/// A device tensor: buffer, shape, element type.
#[derive(Clone, Debug)]
pub struct DevT {
    pub buf: Arc<DevBuf>,
    pub shape: Vec<usize>,
    pub dtype: i32,
}
impl DevT {
    #[must_use]
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    fn arg(&self) -> TensorArg {
        let mut shape = [0_i64; 4];
        for (d, s) in shape.iter_mut().zip(&self.shape) {
            *d = i64::try_from(*s).unwrap_or(0);
        }
        TensorArg {
            data: self.buf.ptr(),
            shape,
            rank: i32::try_from(self.shape.len()).unwrap_or(0),
            dtype: self.dtype,
            owner: 1,
            ..TensorArg::default()
        }
    }
    /// The same buffer seen with another shape of the same element count.
    #[must_use]
    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        Self { buf: Arc::clone(&self.buf), shape, dtype: self.dtype }
    }
}

fn blocks(n: usize) -> u32 {
    u32::try_from(n.div_ceil(128)).unwrap_or(u32::MAX)
}
fn u(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// GPU operations over one device.
#[derive(Clone, Debug)]
pub struct Ops {
    pub gpu: Gpu,
}
impl Ops {
    /// A new binary16 tensor of `shape`.
    /// # Errors
    /// Out of device memory.
    pub fn alloc(&self, shape: Vec<usize>, dtype: i32) -> Result<DevT, String> {
        let w = if dtype == I32 {
            4
        } else if dtype == U8 || dtype == I8 {
            1
        } else {
            2
        };
        Ok(DevT { buf: Arc::new(self.gpu.alloc(shape.iter().product::<usize>() * w)?), shape, dtype })
    }
    /// Upload binary16 bytes as a tensor.
    /// # Errors
    /// Device memory or copy failure.
    pub fn upload16(&self, bytes: &[u8], shape: Vec<usize>) -> Result<DevT, String> {
        let t = self.alloc(shape, F16)?;
        self.gpu.upload(&t.buf, bytes)?;
        Ok(t)
    }
    /// Upload f32 values (already binary16-exact) as a binary16 tensor.
    /// # Errors
    /// As `upload16`.
    pub fn upload_f32(&self, x: &[f32], shape: Vec<usize>) -> Result<DevT, String> {
        let b: Vec<u8> = x.iter().flat_map(|v| crate::half::bits16(*v).to_le_bytes()).collect();
        self.upload16(&b, shape)
    }
    /// Upload int32 values.
    /// # Errors
    /// As `upload16`.
    pub fn upload_i32(&self, x: &[i32]) -> Result<DevT, String> {
        let t = self.alloc(vec![x.len()], I32)?;
        self.gpu.upload(&t.buf, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(t)
    }
    /// Upload a head mask (1 on, 0 off).
    /// # Errors
    /// As `upload16`.
    pub fn upload_mask(&self, on: &[u8]) -> Result<DevT, String> {
        let t = self.alloc(vec![on.len()], U8)?;
        self.gpu.upload(&t.buf, on)?;
        Ok(t)
    }
    /// A head mask on the card, uploaded once per distinct mask and device on this thread (lane
    /// S, 2026-11-18): a copy from pageable host memory first waits for the device to finish its
    /// queue, which `upload_mask` did once per attention layer. Long-lived (at most 64 masks).
    /// # Errors
    /// As `upload16`.
    pub fn mask_cached(&self, on: &[u8]) -> Result<DevT, String> {
        let hit = MASKS.with(|c| c.borrow().iter().find(|(g, m, _)| g.same(&self.gpu) && m == on).map(|(_, _, t)| t.clone()));
        if let Some(t) = hit {
            return Ok(t);
        }
        let t = {
            let _k = nmffi::KeptScope::new();
            self.upload_mask(on)?
        };
        MASKS.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() >= 64 {
                c.remove(0);
            }
            c.push((self.gpu.clone(), on.to_vec(), t.clone()));
        });
        Ok(t)
    }
    /// Int32 ids on the card, the last upload on this thread reused while the ids are the same
    /// (lane S, 2026-11-18: a run's positions were copied up, with a wait for the device, by every
    /// rotary node). Held until other ids come or the run ends (`clear_run_holds`).
    /// # Errors
    /// As `upload_i32`.
    pub fn ids_cached(&self, v: &[i32]) -> Result<DevT, String> {
        let hit = IDS_LAST.with(|c| c.borrow().as_ref().filter(|(g, x, _)| g.same(&self.gpu) && x == v).map(|(_, _, t)| t.clone()));
        if let Some(t) = hit {
            return Ok(t);
        }
        let t = self.upload_i32(v)?;
        IDS_LAST.with(|c| *c.borrow_mut() = Some((self.gpu.clone(), v.to_vec(), t.clone())));
        Ok(t)
    }
    /// Download a binary16 tensor as f32 values.
    /// # Errors
    /// A failed kernel or copy.
    pub fn download(&self, t: &DevT) -> Result<Vec<f32>, String> {
        let mut b = vec![0_u8; t.elements() * 2];
        self.gpu.download(&t.buf, &mut b)?;
        Ok(b.as_chunks::<2>().0.iter().map(|c| crate::half::from16(u16::from_le_bytes(*c))).collect())
    }
    /// Download a binary16 tensor as its u16 bits, exactly as the device holds them (step P2:
    /// key and value capture lands in storage layout, no conversion through f32).
    /// # Errors
    /// A failed kernel or copy.
    pub fn download16(&self, t: &DevT) -> Result<Vec<u16>, String> {
        let mut b = vec![0_u16; t.elements()];
        self.gpu.download_u16(&t.buf, &mut b)?;
        Ok(b)
    }
    /// A binary16 tensor of `shape` filled from u16 segments placed one after another (step
    /// P2): one device buffer, one plain copy per segment at its offset, no host repacking.
    /// # Errors
    /// Segments that do not fill the shape exactly, device memory or copy failure.
    pub fn upload_parts16(&self, parts: &[&[u16]], shape: Vec<usize>) -> Result<DevT, String> {
        let n: usize = parts.iter().map(|p| p.len()).sum();
        if n != shape.iter().product::<usize>() {
            return Err(format!("upload of {n} values into a tensor of {shape:?}"));
        }
        let t = self.alloc(shape, F16)?;
        let mut at = 0;
        for p in parts.iter().filter(|p| !p.is_empty()) {
            self.gpu.upload_u16_at(&t.buf, at, p)?;
            at += p.len() * 2;
        }
        Ok(t)
    }
    /// Rotary encoding of keys only, `k [T, KV, D]` at `pos` (step P2): the same kernel and
    /// launch `rope` uses for its keys, so the result is the same bit for bit; the query half
    /// is not run (a window's keys need no queries).
    /// # Errors
    /// Device failure.
    pub fn rope_k(&self, k: &DevT, pos: &DevT, theta: f32) -> Result<DevT, String> {
        let ko = self.alloc(k.shape.clone(), F16)?;
        let mut a = Self::args(&[k, k, pos], &[], &[&ko, &ko]);
        a.attrs.theta = theta;
        self.gpu.launch(
            "_ZN4emma11rope_kernelE12emma_op_argsi",
            (blocks(k.elements() / 2), 1, 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::I32(1)],
        )?;
        Ok(ko)
    }
    pub(crate) fn args(inputs: &[&DevT], weights: &[&DevT], outputs: &[&DevT]) -> OpArgs {
        let mut a =
            OpArgs { input_count: u(inputs.len()), weight_count: u(weights.len()), output_count: u(outputs.len()), ..OpArgs::default() };
        for (d, t) in a.inputs.iter_mut().zip(inputs) {
            *d = t.arg();
        }
        for (d, t) in a.weights.iter_mut().zip(weights) {
            *d = t.arg();
        }
        for (d, t) in a.outputs.iter_mut().zip(outputs) {
            *d = t.arg();
        }
        a
    }
    /// Embedding rows: `ids` int32 `[T]`, table `[V, D]` -> `[T, D]`.
    /// # Errors
    /// Device failure.
    pub fn embed(&self, ids: &DevT, table: &DevT) -> Result<DevT, String> {
        let d = table.shape.get(1).copied().unwrap_or(0);
        let y = self.alloc(vec![ids.elements(), d], F16)?;
        let a = Self::args(&[ids], &[table], &[&y]);
        self.gpu.launch(
            "_ZN4emma12embed_kernelE12emma_op_args",
            (blocks(y.elements()), 1, 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a))],
        )?;
        Ok(y)
    }
    /// RMSNorm over rows of the weight's width (one block per row, as `element_launch`).
    /// # Errors
    /// Device failure.
    pub fn norm(&self, x: &DevT, w: &DevT, eps: f32) -> Result<DevT, String> {
        let y = self.alloc(x.shape.clone(), F16)?;
        let width = w.elements().max(1);
        let rows = x.elements() / width;
        let mut a = Self::args(&[x], &[w], &[&y]);
        a.attrs.eps = eps;
        let warp = match width {
            128 => Some("_ZN4emma16norm_warp_kernelILi4EEEv12emma_op_args"),
            1024 => Some("_ZN4emma16norm_warp_kernelILi32EEEv12emma_op_args"),
            _ => None,
        };
        if let Some(name) = warp.filter(|_| fast_element() && x.dtype == F16) {
            // Lane S: one warp per row, eight rows per block (bit-identical, `check_fastel`).
            self.gpu.launch(name, (u(rows.div_ceil(8)), 1, 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
        } else if width * 4 <= 32768 {
            self.gpu.launch(
                "_ZN4emma15norm_row_kernelE12emma_op_args",
                (u(rows), 1, 1),
                (256, 1, 1),
                u(width * 4),
                &[Param::Op(Box::new(a))],
            )?;
        } else {
            self.gpu.launch("_ZN4emma11norm_kernelE12emma_op_args", (blocks(rows), 1, 1), (128, 1, 1), 0, &[Param::Op(Box::new(a))])?;
        }
        Ok(y)
    }
    /// Dense binary16 linear (op 7): `x [T, K]`, `w [N, K]` -> `[T, N]`, kernel choice as
    /// `launch_linear` for float16 inputs.
    /// # Errors
    /// Device failure.
    pub fn linear(&self, x: &DevT, w: &DevT) -> Result<DevT, String> {
        let rows = w.shape.first().copied().unwrap_or(0);
        let k = w.shape.get(1).copied().unwrap_or(1);
        let tokens = x.elements() / k.max(1);
        let y = self.alloc(vec![tokens, rows], F16)?;
        let a = Self::args(&[&x.reshape(vec![tokens, k])], &[w], &[&y]);
        if tokens == 1 {
            self.gpu.launch(
                "_ZN4emma18vector_half_kernelE12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
        } else if tokens >= 64 && HALF_MATH.load(std::sync::atomic::Ordering::Relaxed) && !NO_HALF.with(std::cell::Cell::get) {
            // Lane S (T-079): the 16-byte fetch for binary16 weights, bit-identical (`check_h2_fetch`).
            let fetch = h2q8_fetch() && x.dtype == F16 && w.dtype == F16 && k.is_multiple_of(16) && x.buf.ptr().is_multiple_of(16) && w.buf.ptr().is_multiple_of(8);
            self.gpu.launch(
                if fetch { "_ZN4emma15h2_fetch_kernelE12emma_op_args" } else { "_ZN4emma14wide_h2_kernelILi8ELi4EEEv12emma_op_args" },
                (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
        } else if tokens >= 64 {
            self.gpu.launch(
                "_ZN4emma11wide_kernelILi8ELi4ELb0EEEvi12emma_op_args",
                (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                (256, 1, 1),
                0,
                &[Param::I32(7), Param::Op(Box::new(a))],
            )?;
        } else if tokens >= TILED_MIN {
            self.gpu.launch(
                "_ZN4emma12tiled_kernelIfEEvi12emma_op_args",
                (u(rows.div_ceil(64)), u(tokens.div_ceil(64)), 1),
                (256, 1, 1),
                0,
                &[Param::I32(7), Param::Op(Box::new(a))],
            )?;
        } else {
            self.gpu.launch(
                "_ZN4emma13linear_kernelIfEEvi12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::I32(7), Param::Op(Box::new(a))],
            )?;
        }
        Ok(y)
    }
    /// 4-bit linear (2026-11-14): `wp [N, K/2]` packed nibbles, `ws [N, K/32]` binary16 group
    /// scales; unpacked to a temporary binary16 `[N, K]` by `q4_unpack_kernel`, then the binary16
    /// linear. The weights rest on the card at 4 bits.
    /// # Errors
    /// Device failure.
    pub fn linear_q4(&self, x: &DevT, wp: &DevT, ws: &DevT, n: usize, k: usize) -> Result<DevT, String> {
        let w = self.alloc(vec![n, k], F16)?;
        let a = Self::args(&[], &[wp, ws], &[&w]);
        let bytes = n * k / 2;
        self.gpu.launch(
            "_ZN4emma16q4_unpack_kernelE12emma_op_args",
            (u(bytes.div_ceil(256)), 1, 1),
            (256, 1, 1),
            0,
            &[Param::Op(Box::new(a))],
        )?;
        self.linear(x, &w)
    }
    /// Weight-only int8 linear (op 6): `x [T, K]` binary16, `w [N, K]` int8, `s [N]` binary16
    /// row scales -> `[T, N]`, kernel choice as `launch_linear` for float16 inputs and scales.
    /// # Errors
    /// Device failure.
    pub fn linear_q8(&self, x: &DevT, w: &DevT, s: &DevT) -> Result<DevT, String> {
        let rows = w.shape.first().copied().unwrap_or(0);
        let k = w.shape.get(1).copied().unwrap_or(1);
        let tokens = x.elements() / k.max(1);
        let y = self.alloc(vec![tokens, rows], F16)?;
        let wt = DevT { buf: Arc::clone(&w.buf), shape: w.shape.clone(), dtype: I8 };
        let a = Self::args(&[&x.reshape(vec![tokens, k])], &[&wt, s], &[&y]);
        let half_on = HALF_MATH.load(std::sync::atomic::Ordering::Relaxed) && !NO_HALF.with(std::cell::Cell::get);
        let in_served = SERVED.with(std::cell::Cell::get);
        let in_q8_half = Q8_HALF.with(std::cell::Cell::get);
        let in_half_scope = in_served || in_q8_half;
        let dp4a_role = DP4A.load(std::sync::atomic::Ordering::Relaxed) || (INT8_DOT.with(std::cell::Cell::get) && half_on && dp4a_served());
        let mode = served_dp4a_mode();
        let scope_allows_dp4a = match mode {
            2 => true,
            1 => !in_served,
            3 => !in_q8_half,
            _ => !in_half_scope,
        };
        let use_dp4a = tokens >= dp4a_min_tokens() && k % 32 == 0 && dp4a_role && scope_allows_dp4a;
        if tokens == 1 {
            self.gpu.launch(
                "_ZN4emma23vector_int8_half_kernelE12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
        } else if use_dp4a {
            let (xq, xs) = self.q8_blocks(x, tokens, k)?;
            let ga = Self::args(&[&xq, &xs], &[&wt, s], &[&y]);
            let aligned = dp4a_tile() && k.is_multiple_of(64) && xq.buf.ptr().is_multiple_of(16) && wt.buf.ptr().is_multiple_of(16);
            if aligned {
                let (name, br, bt) = crate::autotune::dp4a_kernel(tokens <= 64);
                self.gpu.launch(
                    name,
                    (u(rows.div_ceil(br)), u(tokens.div_ceil(bt)), 1),
                    (256, 1, 1),
                    0,
                    &[Param::Op(Box::new(ga))],
                )?;
            } else {
                self.gpu.launch(
                    "_ZN4emma16wide_dp4a_kernelILi8ELi4EEEv12emma_op_args",
                    (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                    (256, 1, 1),
                    0,
                    &[Param::Op(Box::new(ga))],
                )?;
            }
        } else if tokens >= 64 && half_on && in_half_scope && s.dtype == F16 {
            // Lane S: vector reads of the shared tiles, bit-identical (`check_h2q8_tile`).
            self.gpu.launch(
                if h2q8_fetch() && k.is_multiple_of(16) && x.buf.ptr().is_multiple_of(16) && wt.buf.ptr().is_multiple_of(4) {
                    "_ZN4emma17h2q8_fetch_kernelILi4EEEv12emma_op_args"
                } else if h2q8_tile() {
                    "_ZN4emma16h2q8_tile_kernelE12emma_op_args"
                } else {
                    "_ZN4emma16wide_h2q8_kernelILi8ELi4EEEv12emma_op_args"
                },
                (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
        } else if tokens >= 64
            && wide_q8f()
            && s.dtype == F16
            && x.dtype == F16
            && k.is_multiple_of(16)
            && x.buf.ptr().is_multiple_of(16)
            && wt.buf.ptr().is_multiple_of(4)
        {
            // Lane S package 3: the 16-byte fetch, bit-identical (`check_wide_q8f`).
            self.gpu.launch(
                "_ZN4emma15wide_q8f_kernelE12emma_op_args",
                (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
        } else if tokens >= 64 {
            self.gpu.launch(
                "_ZN4emma11wide_kernelILi8ELi4ELb1EEEvi12emma_op_args",
                (u(rows.div_ceil(64)), u(tokens.div_ceil(128)), 1),
                (256, 1, 1),
                0,
                &[Param::I32(6), Param::Op(Box::new(a))],
            )?;
        } else if tokens >= TILED_MIN {
            self.gpu.launch(
                "_ZN4emma12tiled_kernelIfEEvi12emma_op_args",
                (u(rows.div_ceil(64)), u(tokens.div_ceil(64)), 1),
                (256, 1, 1),
                0,
                &[Param::I32(6), Param::Op(Box::new(a))],
            )?;
        } else {
            self.gpu.launch(
                "_ZN4emma13linear_kernelIfEEvi12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::I32(6), Param::Op(Box::new(a))],
            )?;
        }
        Ok(y)
    }
    /// Upload a `GpuBitmaskQ8` compressed matrix onto the GPU for `bmq8.cu` (`linear_bmq8`).
    /// # Errors
    /// Device allocation or transfer failure.
    pub fn upload_bmq8(&self, b: &crate::pmap::GpuBitmaskQ8) -> Result<DevBitmaskQ8, String> {
        let mask_bytes: Vec<u8> = b.mask_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mask_words = self.alloc(vec![b.rows, b.words_per_row.max(1)], I32)?;
        if !mask_bytes.is_empty() {
            self.gpu.upload(&mask_words.buf, &mask_bytes)?;
        }
        let grp_bytes: Vec<u8> = b.group_offs.iter().flat_map(|g| g.to_le_bytes()).collect();
        let group_offs = self.alloc(vec![b.rows, b.groups_per_row.max(1)], F16)?;
        if !grp_bytes.is_empty() {
            self.gpu.upload(&group_offs.buf, &grp_bytes)?;
        }
        let row_bytes: Vec<u8> = b.row_offs.iter().flat_map(|r| r.to_le_bytes()).collect();
        let row_offs = self.alloc(vec![b.rows.max(1)], I32)?;
        if !row_bytes.is_empty() {
            self.gpu.upload(&row_offs.buf, &row_bytes)?;
        }
        let packed_bytes: Vec<u8> = b.packed_i8.iter().map(|&v| v.to_ne_bytes()[0]).collect();
        let packed_i8 = self.alloc(vec![packed_bytes.len().max(1)], I8)?;
        if !packed_bytes.is_empty() {
            self.gpu.upload(&packed_i8.buf, &packed_bytes)?;
        }
        let scale_bytes: Vec<u8> = b.row_scales_f16.iter().flat_map(|s| s.to_le_bytes()).collect();
        let row_scales = self.alloc(vec![b.rows.max(1)], F16)?;
        if !scale_bytes.is_empty() {
            self.gpu.upload(&row_scales.buf, &scale_bytes)?;
        }
        Ok(DevBitmaskQ8 {
            rows: b.rows,
            cols: b.cols,
            groups_per_row: b.groups_per_row,
            mask_words,
            group_offs,
            row_offs,
            packed_i8,
            row_scales,
        })
    }

    /// Compressed `bitmask-q8` linear (`bmq8.cu`):
    /// - `tokens == 1`: runs `bmq8_vec_kernel` directly over compressed weights (zero scratch buffer,
    ///   reduced VRAM bandwidth via O(1) `__popc()` indexing).
    /// - `2 <= tokens <= 16` (below `TILED_MIN`): runs `bmq8_small_kernel` directly over compressed
    ///   weights (zero scratch allocation, zero dense unpack round-trip, bit-identical to
    ///   `linear_kernel<float>`).
    /// - `tokens > 16`: unpacks compressed weights via `bmq8_unpack_kernel` into a pooled temporary
    ///   `int8` `[rows, cols]` buffer and dispatches to `linear_q8` (`h2q8_fetch` / `dp4a_tile` /
    ///   `wide_q8f`) at peak SM throughput, releasing the scratch buffer immediately after.
    /// # Errors
    /// Device failure.
    pub fn linear_bmq8(&self, x: &DevT, w: &DevBitmaskQ8) -> Result<DevT, String> {
        let rows = w.rows;
        let k = w.cols;
        let tokens = x.elements() / k.max(1);
        if tokens == 1 {
            let y = self.alloc(vec![1, rows], F16)?;
            let a = Self::args(
                &[&x.reshape(vec![1, k]), &w.mask_words, &w.group_offs, &w.row_offs],
                &[&w.packed_i8, &w.row_scales],
                &[&y],
            );
            self.gpu.launch(
                "_ZN4emma15bmq8_vec_kernelE12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
            return Ok(y);
        }
        if tokens <= 16 {
            let y = self.alloc(vec![tokens, rows], F16)?;
            let a = Self::args(
                &[&x.reshape(vec![tokens, k]), &w.mask_words, &w.group_offs, &w.row_offs],
                &[&w.packed_i8, &w.row_scales],
                &[&y],
            );
            self.gpu.launch(
                "_ZN4emma17bmq8_small_kernelE12emma_op_args",
                (u(rows.div_ceil(4)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::Op(Box::new(a))],
            )?;
            return Ok(y);
        }
        let dense_i8 = self.alloc(vec![rows, k], I8)?;
        let ua = Self::args(&[&w.mask_words, &w.group_offs, &w.row_offs], &[&w.packed_i8], &[&dense_i8]);
        self.gpu.launch(
            "_ZN4emma18bmq8_unpack_kernelE12emma_op_args",
            (u(w.groups_per_row.div_ceil(8)), u(rows), 1),
            (256, 1, 1),
            0,
            &[Param::Op(Box::new(ua))],
        )?;
        self.linear_q8(x, &dense_i8, &w.row_scales)
    }

    /// 8-bit activation blocks of `x [tokens, k]` (`q8_blocks_kernel`), made once per input:
    /// the projections that read the same activation (query, key and value; gate and up) share
    /// one quantisation (lane S, 2026-11-18; the same values, so outputs are unchanged). The last
    /// blocks are held on this thread while their input lives (a weak reference to it), until
    /// another input comes, the input is released (`release_q8_blocks_if_dead`) or the run ends.
    fn q8_blocks(&self, x: &DevT, tokens: usize, k: usize) -> Result<(DevT, DevT), String> {
        let hit = Q8_LAST.with(|c| {
            c.borrow()
                .as_ref()
                .filter(|(src, xq, _)| src.strong_count() > 0 && std::ptr::eq(src.as_ptr(), Arc::as_ptr(&x.buf)) && xq.shape == [tokens, k])
                .map(|(_, xq, xs)| (xq.clone(), xs.clone()))
        });
        if let Some(h) = hit {
            return Ok(h);
        }
        let xq = self.alloc(vec![tokens, k], I8)?;
        let xs = self.alloc(vec![tokens, k / 32], F16)?;
        let qa = Self::args(&[&x.reshape(vec![tokens, k])], &[], &[&xq, &xs]);
        let blocks = tokens * (k / 32);
        // Lane S: 16-byte reads and writes when aligned (bit-identical, `check_dp4a_tile`).
        let name = if dp4a_tile() && x.buf.ptr().is_multiple_of(16) && k.is_multiple_of(32) {
            "_ZN4emma17q8_blocks2_kernelE12emma_op_args"
        } else {
            "_ZN4emma16q8_blocks_kernelE12emma_op_args"
        };
        self.gpu.launch(name, (u(blocks.div_ceil(256)), 1, 1), (256, 1, 1), 0, &[Param::Op(Box::new(qa))])?;
        Q8_LAST.with(|c| *c.borrow_mut() = Some((Arc::downgrade(&x.buf), xq.clone(), xs.clone())));
        Ok((xq, xs))
    }
    /// Rotary encoding of `q [T, H, D]` and `k [T, KV, D]` at `pos`.
    /// # Errors
    /// Device failure.
    pub fn rope(&self, q: &DevT, k: &DevT, pos: &DevT, theta: f32) -> Result<(DevT, DevT), String> {
        let (qo, ko) = (self.alloc(q.shape.clone(), F16)?, self.alloc(k.shape.clone(), F16)?);
        let mut a = Self::args(&[q, k, pos], &[], &[&qo, &ko]);
        a.attrs.theta = theta;
        let (qw, kw) = (q.shape.last().copied().unwrap_or(0), k.shape.last().copied().unwrap_or(0));
        let (qh, kh) = (q.shape.len().checked_sub(2).and_then(|i| q.shape.get(i)).copied().unwrap_or(0), k.shape.len().checked_sub(2).and_then(|i| k.shape.get(i)).copied().unwrap_or(0));
        if fast_element() && q.dtype == F16 && k.dtype == F16 && qw == kw && qw % 2 == 0 && qh > 0 && kh > 0 && q.elements() / (qh * qw) == k.elements() / (kh * kw) {
            // Lane S: one launch, the angle once per token and pair (bit-identical, `check_fastel`).
            let pairs = q.elements() / (qh * qw) * (qw / 2);
            self.gpu.launch("_ZN4emma12rope2_kernelE12emma_op_args", (blocks(pairs), 1, 1), (128, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            return Ok((qo, ko));
        }
        self.gpu.launch(
            "_ZN4emma11rope_kernelE12emma_op_argsi",
            (blocks(q.elements() / 2), 1, 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::I32(0)],
        )?;
        self.gpu.launch(
            "_ZN4emma11rope_kernelE12emma_op_argsi",
            (blocks(k.elements() / 2), 1, 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::I32(1)],
        )?;
        Ok((qo, ko))
    }
    /// Causal grouped-query attention with a head mask, as `attention_launch`.
    /// # Errors
    /// Device failure.
    /// Causal attention of one packed sequence (`q` rows the last of `k`'s) through
    /// `segment_kernel` when its head width is 128 and `SEGMENT_ATTENTION` is on; `attention`
    /// otherwise. Outputs equal `attention`'s within rounding.
    /// # Errors
    /// Out of device memory or a launch failure.
    pub fn attention_segment(&self, q: &DevT, k: &DevT, v: &DevT, mask: &DevT) -> Result<DevT, String> {
        let s = |t: &DevT, i: usize| t.shape.get(i).copied().unwrap_or(1) as u64;
        // Inside a `ServedScope` a sequence reads like any single sequence (`attention` then takes
        // the tiled kernel when the served kernels are on).
        if !SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed)
            || SERVED.with(std::cell::Cell::get)
            || s(q, 2) != 128
            || s(k, 1) == 0
            || s(q, 1) % s(k, 1) != 0
        {
            return self.attention(q, k, v, mask);
        }
        let y = self.alloc(q.shape.clone(), F16)?;
        let mut a = Self::args(&[q, k, v, mask], &[], &[&y]);
        a.attrs.causal = 1;
        let g = Geometry { tokens: s(q, 0), cached: s(k, 0), heads: s(q, 1), kv_heads: s(k, 1), width: s(q, 2), clones: 1 };
        self.gpu.launch(
            "_ZN4emma14segment_kernelE12emma_op_argsNS_8GeometryE",
            (u(usize::try_from(g.tokens.div_ceil(16)).unwrap_or(0)), u(usize::try_from(g.heads).unwrap_or(0)), 1),
            (256, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::Geo(g)],
        )?;
        Ok(y)
    }
    /// Block-diagonal attention of packed sequences in one launch (lane S, 2026-11-18): `q [T, H,
    /// 128]`, `k` and `v [T, KV, 128]` hold the sequences of `lens` one after another; each reads
    /// the shared `prefix` keys and values (`[P, KV, 128]`, when given) and then its own rows,
    /// causally. The outputs of `attention_segment` on each sequence's copied rows, bit for bit
    /// (`segments_kernel`, `check_segments`). `None` where that path would not take
    /// `segment_kernel` (inside a served scope, packed attention off, another head width or
    /// layout, or `NANO_SEGMENTS_ONE=off`): the caller reads sequence by sequence.
    /// # Errors
    /// Out of device memory or a launch failure.
    #[expect(clippy::too_many_arguments, reason = "the three tensors, the mask, the shared prefix and the lengths of one attention")]
    /// Package 5: `bodies[i] = Some(j)` makes sequence `i` a tail that reads sequence `j`'s rows
    /// (its body, which has none itself) after the prefix and before its own rows.
    pub fn attention_segments(
        &self,
        q: &DevT,
        k: &DevT,
        v: &DevT,
        mask: &DevT,
        prefix: Option<(&DevT, &DevT)>,
        lens: &[usize],
        bodies: Option<&[Option<usize>]>,
    ) -> Result<Option<DevT>, String> {
        let s = |t: &DevT, i: usize| t.shape.get(i).copied().unwrap_or(0);
        let total: usize = lens.iter().sum();
        let (heads, kv) = (s(q, 1), s(k, 1));
        // In a served scope short sequences go to `segments_wide_kernel`; from 600 rows a group the
        // flash tiling reads faster (integration shape, package 3: 0.354 against 0.403 ms a token
        // at 1,500 rows; 0.372 against 0.358 at 200), so batches with a longer sequence take
        // `flash_segments_kernel` (package 4: flash in one launch, no copies), or the per-sequence
        // flash path when `NANO_FLASH_SEGMENTS=off`.
        let long = SERVED.with(std::cell::Cell::get) && lens.iter().any(|n| *n > SERVED_SEGMENT_ROWS);
        if !segments_ready()
            || (long && !flash_segments())
            || [q, k, v].iter().any(|t| t.shape.len() != 3 || t.dtype != F16 || s(t, 2) != 128 || s(t, 0) != total)
            || kv == 0
            || !heads.is_multiple_of(kv)
            || lens.contains(&0)
            || prefix.is_some_and(|(pk, pv)| pk.shape != pv.shape || pk.shape.len() != 3 || s(pk, 1) != kv || s(pk, 2) != 128 || pk.dtype != F16)
        {
            return Ok(None);
        }
        self.launch_segments(q, k, v, mask, prefix, lens, if long { 0 } else { segments_per() }, bodies).map(Some)
    }
    /// The launch of `attention_segments`, shapes already checked.
    #[expect(clippy::too_many_arguments, reason = "the three tensors, the mask, the shared prefix and the lengths of one attention")]
    fn launch_segments(
        &self,
        q: &DevT,
        k: &DevT,
        v: &DevT,
        mask: &DevT,
        prefix: Option<(&DevT, &DevT)>,
        lens: &[usize],
        per: usize,
        bodies: Option<&[Option<usize>]>,
    ) -> Result<DevT, String> {
        let s = |t: &DevT, i: usize| t.shape.get(i).copied().unwrap_or(0);
        let total: usize = lens.iter().sum();
        let (heads, kv) = (s(q, 1), s(k, 1));
        let mut table: Vec<i32> = Vec::new();
        let mut at = 0_usize;
        // `per` 0 is `flash_segments_kernel` (64-row tiles, the tiles that read the most keys first).
        // `per` 1 is `flash_segments_h_kernel` whatever the switch (checks).
        let rows_b = if per <= 1 { 64 } else { 8 * per };
        let starts: Vec<usize> = lens.iter().scan(0, |a, n| {
            let s = *a;
            *a += n;
            Some(s)
        }).collect();
        let mut entries: Vec<[usize; 5]> = Vec::new();
        for (i, &n) in lens.iter().enumerate() {
            let body = bodies.and_then(|b| b.get(i).copied().flatten()).filter(|j| *j < i);
            let (bs, bl) = body.map_or((0, 0), |j| (starts.get(j).copied().unwrap_or(0), lens.get(j).copied().unwrap_or(0)));
            for b in 0..n.div_ceil(rows_b) {
                entries.push([at, n, b * rows_b, bs, bl]);
            }
            at += n;
        }
        if per <= 1 {
            entries.sort_by(|a, b| b[2].cmp(&a[2]));
        }
        for e in &entries {
            table.extend(e.iter().map(|x| i32::try_from(*x).unwrap_or(i32::MAX)));
        }
        let blocks_x = table.len() / 5;
        let hit = TABLE_LAST.with(|c| c.borrow().as_ref().filter(|(g, x, _)| g.same(&self.gpu) && *x == table).map(|(_, _, t)| t.clone()));
        let tab = match hit {
            Some(t) => t,
            None => {
                let t = self.upload_i32(&table)?;
                TABLE_LAST.with(|c| *c.borrow_mut() = Some((self.gpu.clone(), table, t.clone())));
                t
            }
        };
        let (pk, pv, p) = match prefix {
            Some((pk, pv)) => (pk, pv, s(pk, 0)),
            None => (k, v, 0),
        };
        let y = self.alloc(q.shape.clone(), F16)?;
        let mut a = Self::args(&[q, k, v, mask], &[pk, pv], &[&y]);
        a.attrs.causal = 1;
        let w = |x: usize| x as u64;
        let g = Geometry { tokens: w(total), cached: w(p), heads: w(heads), kv_heads: w(kv), width: 128, clones: 1 };
        let name = match per {
            0 if flash_half() => "_ZN4emma23flash_segments_h_kernelE12emma_op_argsNS_8GeometryEPKi",
            0 => "_ZN4emma21flash_segments_kernelE12emma_op_argsNS_8GeometryEPKi",
            1 => "_ZN4emma23flash_segments_h_kernelE12emma_op_argsNS_8GeometryEPKi",
            4 => "_ZN4emma20segments_wide_kernelILi4EEEv12emma_op_argsNS_8GeometryEPKi",
            8 => "_ZN4emma20segments_wide_kernelILi8EEEv12emma_op_argsNS_8GeometryEPKi",
            _ => "_ZN4emma15segments_kernelE12emma_op_argsNS_8GeometryEPKi",
        };
        self.gpu.launch(
            name,
            (u(blocks_x), u(heads), 1),
            (256, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(tab.buf.ptr())],
        )?;
        Ok(y)
    }
    /// Per-part log-sum-exp of question rows' raw scores on the card (`part_lse_kernel`, lane S
    /// package 5): `q [T, H, 128]` and `k [T, KV, 128]` packed (roped), `prefix` roped keys `[P, KV,
    /// 128]`; `items` 8 ints each (question start, rows, body start, body rows, local head, first
    /// part, parts, first output), `parts` 2 ints each (first position, positions in the span:
    /// prefix then body). Returns one float per (item, part) at the item's output offsets.
    /// # Errors
    /// Device failure.
    #[expect(clippy::too_many_arguments, reason = "the packed rows, the prefix and the two tables of one reduction")]
    pub fn part_lse(&self, q: &DevT, k: &DevT, prefix: Option<&DevT>, items: &[i32], parts: &[i32], outputs: usize) -> Result<Vec<f32>, String> {
        let s = |t: &DevT, i: usize| t.shape.get(i).copied().unwrap_or(0) as u64;
        let n_items = items.len() / 8;
        let max_parts = items.as_chunks::<8>().0.iter().map(|c| c[6]).max().unwrap_or(0);
        if n_items == 0 || max_parts <= 0 || outputs == 0 {
            return Ok(vec![0.0; outputs]);
        }
        let ti = self.upload_i32(items)?;
        let tp = self.upload_i32(parts)?;
        let out = self.alloc(vec![outputs], I32)?;
        let pk = prefix.unwrap_or(k);
        let a = Self::args(&[q, k], &[pk], &[&out]);
        let g = Geometry { tokens: s(q, 0), cached: prefix.map_or(0, |p| s(p, 0)), heads: s(q, 1), kv_heads: s(k, 1), width: s(q, 2), clones: 1 };
        self.gpu.launch(
            "_ZN4emma15part_lse_kernelE12emma_op_argsNS_8GeometryEPKiS3_",
            (u(n_items), u32::try_from(max_parts).unwrap_or(0), 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(ti.buf.ptr()), Param::Ptr(tp.buf.ptr())],
        )?;
        let mut b = vec![0_u8; outputs * 4];
        self.gpu.download(&out.buf, &mut b)?;
        Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
    pub fn attention(&self, q: &DevT, k: &DevT, v: &DevT, mask: &DevT) -> Result<DevT, String> {
        let y = self.alloc(q.shape.clone(), F16)?;
        let mut a = Self::args(&[q, k, v, mask], &[], &[&y]);
        a.attrs.causal = 1;
        let s = |t: &DevT, i: usize| t.shape.get(i).copied().unwrap_or(1) as u64;
        let g = Geometry { tokens: s(q, 0), cached: s(k, 0), heads: s(q, 1), kv_heads: s(k, 1), width: s(q, 2), clones: 1 };
        if SERVED.with(std::cell::Cell::get)
            && SEGMENT_ATTENTION.load(std::sync::atomic::Ordering::Relaxed)
            && g.tokens >= 2
            && g.width == 128
            && g.kv_heads > 0
            && g.heads.is_multiple_of(g.kv_heads)
            && g.cached >= g.tokens
            && [q, k, v].iter().all(|t| t.dtype == F16 && t.shape.len() == 3)
        {
            self.gpu.launch(
                "_ZN4emma12flash_kernelE12emma_op_argsNS_8GeometryE",
                (u(usize::try_from(g.tokens.div_ceil(64)).unwrap_or(0)), u(usize::try_from(g.heads).unwrap_or(0)), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(g)],
            )?;
            return Ok(y);
        }
        if g.tokens >= 32 && g.width <= 128 {
            // 2026-11-13: two query rows per warp (16 per block): half the key and value reads,
            // the same bytes (`run s6 fingerprint` identical).
            self.gpu.launch(
                "_ZN4emma15prefill2_kernelIfEEv12emma_op_argsNS_8GeometryE",
                (u(usize::try_from(g.tokens.div_ceil(16)).unwrap_or(0)), u(usize::try_from(g.heads).unwrap_or(0)), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(g)],
            )?;
            return Ok(y);
        }
        let rows = usize::try_from(g.tokens * g.heads).unwrap_or(0);
        let cached = usize::try_from(g.cached).unwrap_or(0);
        let scores = self.gpu.alloc(rows * cached * 4)?;
        if g.width <= 128 && g.kv_heads > 0 && g.heads.is_multiple_of(g.kv_heads) {
            // E1 (2026-11-10): the shared key tile kernel, same scores bit for bit, linear in
            // the window (the per-score kernel grew faster than the window).
            self.gpu.launch(
                "_ZN4emma17score_tile_kernelE12emma_op_argsNS_8GeometryEPf",
                (u(cached.div_ceil(32)), u(usize::try_from(g.kv_heads).unwrap_or(0)), u(usize::try_from(g.clones).unwrap_or(1))),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(scores.ptr())],
            )?;
        } else {
            self.gpu.launch(
                "_ZN4emma12score_kernelE12emma_op_argsNS_8GeometryEPf",
                (blocks(rows * cached), 1, 1),
                (128, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(scores.ptr())],
            )?;
        }
        self.gpu.launch(
            "_ZN4emma18softmax_row_kernelE12emma_op_argsNS_8GeometryEPf",
            (u(rows), 1, 1),
            (256, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(scores.ptr())],
        )?;
        self.gpu.launch(
            "_ZN4emma12value_kernelE12emma_op_argsNS_8GeometryEPKf",
            (blocks(y.elements()), 1, 1),
            (128, 1, 1),
            0,
            &[Param::Op(Box::new(a)), Param::Geo(g), Param::Ptr(scores.ptr())],
        )?;
        drop(scores);
        Ok(y)
    }
    /// Element-wise: op 4 swigate, op 5 add.
    /// # Errors
    /// Device failure.
    pub fn binary(&self, op: i32, a: &DevT, b: &DevT) -> Result<DevT, String> {
        let y = self.alloc(a.shape.clone(), F16)?;
        let args = Self::args(&[a, b], &[], &[&y]);
        if fast_element()
            && a.dtype == F16
            && b.dtype == F16
            && a.elements() == b.elements()
            && [a.buf.ptr(), b.buf.ptr(), y.buf.ptr()].iter().all(|p| p.is_multiple_of(16))
        {
            // Lane S: eight values per thread, 16-byte reads (bit-identical, `check_fastel`).
            self.gpu.launch(
                "_ZN4emma14binary8_kernelEi12emma_op_args",
                (blocks(a.elements().div_ceil(8)), 1, 1),
                (128, 1, 1),
                0,
                &[Param::I32(op), Param::Op(Box::new(args))],
            )?;
            return Ok(y);
        }
        self.gpu.launch(
            "_ZN4emma13binary_kernelEi12emma_op_args",
            (blocks(a.elements()), 1, 1),
            (128, 1, 1),
            0,
            &[Param::I32(op), Param::Op(Box::new(args))],
        )?;
        Ok(y)
    }
    /// A float32 vector kept on the device (long-lived: exact size, never pooled).
    /// # Errors
    /// Device memory or copy failure.
    pub fn upload_vec32(&self, x: &[f32]) -> Result<DevT, String> {
        let _k = nmffi::KeptScope::new();
        let t = DevT { buf: Arc::new(self.gpu.alloc(x.len() * 4)?), shape: vec![x.len()], dtype: 1 };
        self.gpu.upload(&t.buf, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        Ok(t)
    }
    /// A cartridge filter over `x [.., width]`: `y = round(fma(x, gain, bias))` per channel, with
    /// float32 `gain` and `bias` of the last dimension's width (`filter_kernel`).
    /// # Errors
    /// Device failure.
    pub fn filter(&self, x: &DevT, gain: &DevT, bias: &DevT) -> Result<DevT, String> {
        let y = self.alloc(x.shape.clone(), x.dtype)?;
        let args = Self::args(&[x], &[gain, bias], &[&y]);
        self.gpu.launch("_ZN4emma13filter_kernelE12emma_op_args", (blocks(x.elements()), 1, 1), (128, 1, 1), 0, &[Param::Op(Box::new(args))])?;
        Ok(y)
    }
}

/// Section check of the two-rows-per-warp attention kernel (2026-11-13): on random binary16
/// queries, keys and values (causal, grouped heads 16 over 8, width 128; 33, 100 and 257 tokens,
/// so blocks end inside and past the rows), `prefill2_kernel` gives bit-identical outputs to
/// `prefill_kernel`. Passes on a PC without a CUDA device.
pub(crate) fn check_prefill2() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x2545_f491_u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                crate::half::r16(x)
            })
            .collect()
    };
    for tokens in [33_usize, 100, 257] {
        let (h, g, w) = (16_usize, 8_usize, 128_usize);
        let q = o.upload_f32(&rnd(tokens * h * w), vec![tokens, h, w])?;
        let k = o.upload_f32(&rnd(tokens * g * w), vec![tokens, g, w])?;
        let v = o.upload_f32(&rnd(tokens * g * w), vec![tokens, g, w])?;
        let m = o.upload_mask(&vec![1_u8; h])?;
        let run = |name: &str, per: u64| -> Result<Vec<f32>, String> {
            let y = o.alloc(q.shape.clone(), F16)?;
            let mut a = Ops::args(&[&q, &k, &v, &m], &[], &[&y]);
            a.attrs.causal = 1;
            let geo =
                Geometry { tokens: tokens as u64, cached: tokens as u64, heads: h as u64, kv_heads: g as u64, width: w as u64, clones: 1 };
            o.gpu.launch(
                name,
                (u32::try_from((tokens as u64).div_ceil(per)).unwrap_or(0), u32::try_from(h).unwrap_or(0), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(geo)],
            )?;
            o.download(&y)
        };
        let old = run("_ZN4emma14prefill_kernelIfEEv12emma_op_argsNS_8GeometryE", 8)?;
        let new = run("_ZN4emma15prefill2_kernelIfEEv12emma_op_argsNS_8GeometryE", 16)?;
        if old.iter().zip(&new).any(|(a, b)| a.to_bits() != b.to_bits()) || old.len() != new.len() {
            return Err(format!("nmengn.gpu.prefill2: {tokens} tokens: outputs differ from prefill_kernel"));
        }
    }
    Ok(())
}

/// The spacing of binary16 values at `x` (one unit in the last place).
fn ulp16(x: f32) -> f32 {
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        return 5.960_464_5e-8;
    }
    #[expect(clippy::cast_possible_truncation, reason = "the binary exponent of a finite binary16 value")]
    let e = a.log2().floor() as i32;
    2.0_f32.powi(e - 10)
}

/// Parity of the tiled attention kernel (`flash_kernel`) with the reference prefill kernel
/// (`prefill2_kernel`), by the gate the prefill kernels were held to: against a float64 truth on
/// random binary16 queries, keys and values (causal, grouped heads 16 over 8, width 128, one head
/// masked off; 33, 100 and 257 rows alone, 40 rows over 340 keys, 130 over 1,230), every output's
/// error is at most the reference kernel's error on it or one binary16 step at the true value,
/// whichever is larger (plus 2^-20 for float summation order), and a masked head's rows are zero
/// in both. Passes on a PC without a CUDA device.
pub(crate) fn check_flash() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x9e37_79b9_u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                crate::half::r16(x)
            })
            .collect()
    };
    let (h, g, w) = (16_usize, 8_usize, 128_usize);
    let mut on = vec![1_u8; h];
    on[3] = 0;
    let m = o.upload_mask(&on)?;
    for (tokens, cached) in [(33_usize, 33_usize), (100, 100), (257, 257), (40, 340), (130, 1230)] {
        let (qh, kh, vh) = (rnd(tokens * h * w), rnd(cached * g * w), rnd(cached * g * w));
        let q = o.upload_f32(&qh, vec![tokens, h, w])?;
        let k = o.upload_f32(&kh, vec![cached, g, w])?;
        let v = o.upload_f32(&vh, vec![cached, g, w])?;
        let run = |name: &str, per: u64| -> Result<Vec<f32>, String> {
            let y = o.alloc(q.shape.clone(), F16)?;
            let mut a = Ops::args(&[&q, &k, &v, &m], &[], &[&y]);
            a.attrs.causal = 1;
            let geo =
                Geometry { tokens: tokens as u64, cached: cached as u64, heads: h as u64, kv_heads: g as u64, width: w as u64, clones: 1 };
            o.gpu.launch(
                name,
                (u32::try_from((tokens as u64).div_ceil(per)).unwrap_or(0), u32::try_from(h).unwrap_or(0), 1),
                (256, 1, 1),
                0,
                &[Param::Op(Box::new(a)), Param::Geo(geo)],
            )?;
            o.download(&y)
        };
        let reference = run("_ZN4emma15prefill2_kernelIfEEv12emma_op_argsNS_8GeometryE", 16)?;
        let tiled = run("_ZN4emma12flash_kernelE12emma_op_argsNS_8GeometryE", 64)?;
        if reference.len() != tiled.len() || reference.len() != tokens * h * w {
            return Err(format!("nmengn.gpu.flash: {tokens} over {cached}: output lengths differ"));
        }
        let root = (w as f64).sqrt();
        let mut scores = vec![0.0_f64; cached];
        for t in 0..tokens {
            let seen = cached - tokens + t + 1;
            for head in 0..h {
                let row = (t * h + head) * w;
                if head == 3 {
                    if reference[row..row + w].iter().chain(&tiled[row..row + w]).any(|x| *x != 0.0) {
                        return Err(format!("nmengn.gpu.flash: {tokens} over {cached}: a masked head's output is not zero"));
                    }
                    continue;
                }
                let kv = head / (h / g);
                let qr = &qh[row..row + w];
                let mut top = f64::NEG_INFINITY;
                for (s, x) in scores.iter_mut().enumerate().take(seen) {
                    let kr = &kh[(s * g + kv) * w..(s * g + kv) * w + w];
                    *x = qr.iter().zip(kr).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() / root;
                    top = top.max(*x);
                }
                let mut sum = 0.0;
                for x in scores.iter_mut().take(seen) {
                    *x = (*x - top).exp();
                    sum += *x;
                }
                for d in 0..w {
                    let truth = scores.iter().take(seen).enumerate().map(|(s, p)| p * f64::from(vh[(s * g + kv) * w + d])).sum::<f64>() / sum;
                    #[expect(clippy::cast_possible_truncation, reason = "a binary16-range value")]
                    let step = f64::from(ulp16(truth as f32));
                    let (a, b) = (f64::from(reference[row + d]), f64::from(tiled[row + d]));
                    let limit = (a - truth).abs().max(step) + 2.0_f64.powi(-20);
                    if !b.is_finite() || (b - truth).abs() > limit {
                        return Err(format!(
                            "nmengn.gpu.flash: {tokens} over {cached}: row {t} head {head} dimension {d}: {b} against the true {truth} (reference kernel {a})"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Half products over int8 weights (`wide_h2q8_kernel`) are exactly the served half math of
/// binary16 linears over the dequantized weights: on random binary16 activations, int8 weights and
/// binary16 row scales (64, 130 and 300 tokens; 200 rows, so tiles end inside and past them;
/// width 1,024), its outputs are bit-identical to `wide_h2_kernel` run on the weights each int8
/// value stands for (int8 times its row scale, rounded to binary16), which is how `wide_kernel`'s
/// int8 path dequantizes. Passes on a PC without a CUDA device.
pub(crate) fn check_h2q8() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x51ed_270b_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (n, k) = (200_usize, 1024_usize);
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "a test value in -127..=127")]
    let q: Vec<i8> = (0..n * k).map(|_| ((next() % 255) as i64 - 127) as i8).collect();
    #[expect(clippy::cast_precision_loss, reason = "a test value")]
    let scales: Vec<f32> = (0..n).map(|_| crate::half::r16(0.0005 + (next() % 1000) as f32 * 0.00002)).collect();
    let dequant: Vec<f32> = q.iter().enumerate().map(|(i, v)| crate::half::r16(f32::from(*v) * scales[i / k])).collect();
    let wq = o.alloc(vec![n, k], I8)?;
    o.gpu.upload(&wq.buf, &q.iter().map(|v| v.to_le_bytes()[0]).collect::<Vec<u8>>())?;
    let ws = o.upload_f32(&scales, vec![n])?;
    let wd = o.upload_f32(&dequant, vec![n, k])?;
    for tokens in [64_usize, 130, 300] {
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let xs: Vec<f32> = (0..tokens * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 250.0)).collect();
        let x = o.upload_f32(&xs, vec![tokens, k])?;
        let launch = |name: &str, weights: &[&DevT]| -> Result<Vec<f32>, String> {
            let y = o.alloc(vec![tokens, n], F16)?;
            let a = Ops::args(&[&x], weights, &[&y]);
            o.gpu.launch(name, (u(n.div_ceil(64)), u(tokens.div_ceil(128)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            o.download(&y)
        };
        let new = launch("_ZN4emma16wide_h2q8_kernelILi8ELi4EEEv12emma_op_args", &[&wq, &ws])?;
        let half = launch("_ZN4emma14wide_h2_kernelILi8ELi4EEEv12emma_op_args", &[&wd])?;
        if new.len() != half.len() || new.iter().zip(&half).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(format!("nmengn.gpu.h2q8: {tokens} tokens: outputs differ from wide_h2_kernel over the dequantized weights"));
        }
    }
    Ok(())
}

/// Lane S (2026-11-18): `dp4a_tile_kernel` gives the outputs of `wide_dp4a_kernel` bit for bit on
/// random int8 activations with binary16 block scales, int8 weights and binary16 row scales (64,
/// 130 and 300 tokens; 200 rows, so tiles end inside and past them; widths 1,024 and 3,072); and
/// `q8_blocks2_kernel` gives the int8 values and block scales of `q8_blocks_kernel` bit for bit on
/// random binary16 activations (300 tokens of 3,072, one block all zero). Passes on a PC without a
/// CUDA device.
pub(crate) fn check_dp4a_tile() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x7f4a_7c15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut bytes = |n: usize| -> Vec<u8> { (0..n).map(|_| (next() % 255) as u8).collect() };
    {
        let (tokens, k) = (300_usize, 3072_usize);
        let mut xv: Vec<f32> = bytes(tokens * k).iter().map(|b| crate::half::r16((f32::from(*b) - 127.0) / 9.0)).collect();
        xv[64..96].iter_mut().for_each(|v| *v = 0.0);
        let x = o.upload_f32(&xv, vec![tokens, k])?;
        let run = |name: &str| -> Result<(Vec<u8>, Vec<f32>), String> {
            let (xq, xs) = (o.alloc(vec![tokens, k], I8)?, o.alloc(vec![tokens, k / 32], F16)?);
            let a = Ops::args(&[&x], &[], &[&xq, &xs]);
            o.gpu.launch(name, (u((tokens * k / 32).div_ceil(256)), 1, 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            let mut q = vec![0_u8; tokens * k];
            o.gpu.download(&xq.buf, &mut q)?;
            Ok((q, o.download(&xs)?))
        };
        let (old, new) = (run("_ZN4emma16q8_blocks_kernelE12emma_op_args")?, run("_ZN4emma17q8_blocks2_kernelE12emma_op_args")?);
        if old.0 != new.0 || old.1.iter().zip(&new.1).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err("nmengn.gpu.dp4a_tile: q8_blocks2_kernel differs from q8_blocks_kernel".to_owned());
        }
    }
    for k in [1024_usize, 3072] {
        let n = 200_usize;
        let wq = o.alloc(vec![n, k], I8)?;
        o.gpu.upload(&wq.buf, &bytes(n * k))?;
        let scales: Vec<f32> = bytes(n).iter().map(|b| crate::half::r16(0.0005 + f32::from(*b) * 0.00002)).collect();
        let ws = o.upload_f32(&scales, vec![n])?;
        for tokens in [7_usize, 33, 64, 130, 300] {
            let xq = o.alloc(vec![tokens, k], I8)?;
            o.gpu.upload(&xq.buf, &bytes(tokens * k))?;
            let xsv: Vec<f32> = bytes(tokens * k / 32).iter().map(|b| crate::half::r16(0.001 + f32::from(*b) * 0.0001)).collect();
            let xs = o.upload_f32(&xsv, vec![tokens, k / 32])?;
            let launch = |name: &str| -> Result<Vec<f32>, String> {
                let y = o.alloc(vec![tokens, n], F16)?;
                let a = Ops::args(&[&xq, &xs], &[&wq, &ws], &[&y]);
                o.gpu.launch(name, (u(n.div_ceil(64)), u(tokens.div_ceil(128)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
                o.download(&y)
            };
            let old = launch("_ZN4emma16wide_dp4a_kernelILi8ELi4EEEv12emma_op_args")?;
            let new = launch("_ZN4emma16dp4a_tile_kernelE12emma_op_args")?;
            if old.len() != new.len() || old.iter().zip(&new).any(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(format!("nmengn.gpu.dp4a_tile: {tokens} tokens, width {k}: outputs differ from wide_dp4a_kernel"));
            }
            // 2026-11-25: every variant the auto-tuner may pick (`autotune::DP4A_VARIANTS`), each
            // on its own grid, gives the reference outputs bit for bit (only `dp4a_tile` was
            // checked; the tuner relies on all four being identical).
            for &(name, br, bt) in &crate::autotune::DP4A_VARIANTS {
                let y = o.alloc(vec![tokens, n], F16)?;
                let a = Ops::args(&[&xq, &xs], &[&wq, &ws], &[&y]);
                o.gpu.launch(name, (u(n.div_ceil(br)), u(tokens.div_ceil(bt)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
                let got = o.download(&y)?;
                if got.len() != old.len() || got.iter().zip(&old).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(format!("nmengn.gpu.dp4a_tile: {tokens} tokens, width {k}: tuner variant {name} differs from wide_dp4a_kernel"));
                }
            }
        }
    }
    Ok(())
}

/// Lane S (2026-11-18): `h2q8_tile_kernel` gives the outputs of `wide_h2q8_kernel` bit for bit on
/// random binary16 activations, int8 weights and binary16 row scales (64, 130 and 300 tokens; 200
/// rows; width 1,024). Passes on a PC without a CUDA device.
pub(crate) fn check_h2q8_tile() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x1b87_3593_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (n, k) = (200_usize, 1024_usize);
    let q: Vec<u8> = (0..n * k).map(|_| (next() % 255) as u8).collect();
    let wq = o.alloc(vec![n, k], I8)?;
    o.gpu.upload(&wq.buf, &q)?;
    #[expect(clippy::cast_precision_loss, reason = "a test value")]
    let scales: Vec<f32> = (0..n).map(|_| crate::half::r16(0.0005 + (next() % 1000) as f32 * 0.00002)).collect();
    let ws = o.upload_f32(&scales, vec![n])?;
    for tokens in [64_usize, 130, 300] {
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let xs: Vec<f32> = (0..tokens * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 250.0)).collect();
        let x = o.upload_f32(&xs, vec![tokens, k])?;
        let launch = |name: &str, per: usize| -> Result<Vec<f32>, String> {
            let y = o.alloc(vec![tokens, n], F16)?;
            let a = Ops::args(&[&x], &[&wq, &ws], &[&y]);
            o.gpu.launch(name, (u(n.div_ceil(per)), u(tokens.div_ceil(128)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            o.download(&y)
        };
        let old = launch("_ZN4emma16wide_h2q8_kernelILi8ELi4EEEv12emma_op_args", 64)?;
        for (name, per) in [
            ("_ZN4emma16h2q8_tile_kernelE12emma_op_args", 64),
            ("_ZN4emma17h2q8_fetch_kernelILi4EEEv12emma_op_args", 64),
            ("_ZN4emma17h2q8_fetch_kernelILi8EEEv12emma_op_args", 128),
        ] {
            let new = launch(name, per)?;
            if old.len() != new.len() || old.iter().zip(&new).any(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(format!("nmengn.gpu.h2q8_tile: {tokens} tokens: {name} differs from wide_h2q8_kernel"));
            }
        }
    }
    Ok(())
}

/// Lane S (2026-11-18): `segments_kernel` (all packed sequences in one launch) gives the outputs
/// of `segment_kernel` run on each sequence's copied rows bit for bit, on random binary16 queries,
/// keys and values (16 heads over 8 of width 128, one head masked off; sequences of 37, 16, 100
/// and 1 rows, alone and over a shared prefix of 50 rows; package 3: and 70 rows, and the
/// `segments_wide_kernel` forms with 4 and 8 rows a warp). Passes on a PC without a CUDA device.
pub(crate) fn check_segments() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0xa54f_f53a_u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                crate::half::r16(x)
            })
            .collect()
    };
    let (h, g, w) = (16_usize, 8_usize, 128_usize);
    let lens = [37_usize, 16, 100, 1, 70];
    let total: usize = lens.iter().sum();
    let mut on = vec![1_u8; h];
    on[5] = 0;
    let m = o.upload_mask(&on)?;
    let q = o.upload_f32(&rnd(total * h * w), vec![total, h, w])?;
    let k = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    let v = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    for p in [0_usize, 50] {
        let pk = o.upload_f32(&rnd(p.max(1) * g * w), vec![p.max(1), g, w])?;
        let pv = o.upload_f32(&rnd(p.max(1) * g * w), vec![p.max(1), g, w])?;
        let prefix = (p > 0).then_some((&pk, &pv));
        let ones: Vec<Vec<f32>> = [2, 4, 8].iter().map(|per| o.download(&o.launch_segments(&q, &k, &v, &m, prefix, &lens, *per, None)?)).collect::<Result<_, _>>()?;
        let mut at = 0;
        for &n in &lens {
            let take = |x: &DevT, heads: usize, pre: Option<&DevT>| -> Result<DevT, String> {
                let rows = n + pre.map_or(0, |_| p);
                let out = o.alloc(vec![rows, heads, w], F16)?;
                let mut off = 0;
                if let Some(pre) = pre {
                    o.gpu.copy(&pre.buf, 0, &out.buf, 0, p * heads * w * 2)?;
                    off = p * heads * w * 2;
                }
                o.gpu.copy(&x.buf, at * heads * w * 2, &out.buf, off, n * heads * w * 2)?;
                Ok(out)
            };
            let (qs, ks, vs) = (take(&q, h, None)?, take(&k, g, prefix.map(|x| x.0))?, take(&v, g, prefix.map(|x| x.1))?);
            let y = o.alloc(qs.shape.clone(), F16)?;
            let mut a = Ops::args(&[&qs, &ks, &vs, &m], &[], &[&y]);
            a.attrs.causal = 1;
            let geo = Geometry { tokens: n as u64, cached: (n + if p > 0 { p } else { 0 }) as u64, heads: h as u64, kv_heads: g as u64, width: w as u64, clones: 1 };
            o.gpu.launch("_ZN4emma14segment_kernelE12emma_op_argsNS_8GeometryE", (u(n.div_ceil(16)), u(h), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a)), Param::Geo(geo)])?;
            let old = o.download(&y)?;
            for (one, per) in ones.iter().zip([2, 4, 8]) {
                let new = one.get(at * h * w..(at + n) * h * w).ok_or("short output")?;
                if old.iter().zip(new).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    return Err(format!("nmengn.gpu.segments: prefix {p}, sequence of {n} at {at}, {per} rows a warp: outputs differ from segment_kernel"));
                }
            }
            at += n;
        }
    }
    Ok(())
}

/// Lane S package 3 (2026-11-18): `wide_q8f_kernel` gives the outputs of `wide_kernel<8, 4, true>`
/// (int8 weights, binary16 activations and row scales, binary32 products) bit for bit (64, 130 and
/// 300 tokens; 200 rows; width 1,024). Passes on a PC without a CUDA device.
pub(crate) fn check_wide_q8f() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x68e3_1da4_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (n, k) = (200_usize, 1024_usize);
    let wq = o.alloc(vec![n, k], I8)?;
    o.gpu.upload(&wq.buf, &(0..n * k).map(|_| (next() % 255) as u8).collect::<Vec<u8>>())?;
    #[expect(clippy::cast_precision_loss, reason = "a test value")]
    let scales: Vec<f32> = (0..n).map(|_| crate::half::r16(0.0005 + (next() % 1000) as f32 * 0.00002)).collect();
    let ws = o.upload_f32(&scales, vec![n])?;
    for tokens in [64_usize, 130, 300] {
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let xs: Vec<f32> = (0..tokens * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 250.0)).collect();
        let x = o.upload_f32(&xs, vec![tokens, k])?;
        let run = |name: &str, op: bool| -> Result<Vec<f32>, String> {
            let y = o.alloc(vec![tokens, n], F16)?;
            let a = Ops::args(&[&x], &[&wq, &ws], &[&y]);
            let p: Vec<Param> = if op { vec![Param::I32(6), Param::Op(Box::new(a))] } else { vec![Param::Op(Box::new(a))] };
            o.gpu.launch(name, (u(n.div_ceil(64)), u(tokens.div_ceil(128)), 1), (256, 1, 1), 0, &p)?;
            o.download(&y)
        };
        let old = run("_ZN4emma11wide_kernelILi8ELi4ELb1EEEvi12emma_op_args", true)?;
        let new = run("_ZN4emma15wide_q8f_kernelE12emma_op_args", false)?;
        if old.len() != new.len() || old.iter().zip(&new).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(format!("nmengn.gpu.wide_q8f: {tokens} tokens: outputs differ from wide_kernel"));
        }
    }
    Ok(())
}

/// Lane S package 4 (2026-11-18): `flash_segments_kernel` (every packed sequence in one launch, keys
/// and values read in place) gives the outputs of one `flash_kernel` launch per sequence on its
/// copied rows (prefix, then its own) bit for bit: random binary16 values, 16 heads over 8 of width
/// 128, one head masked off, sequences of 37, 300, 130 and 64 rows, alone and over a 50-row prefix.
/// Passes on a PC without a CUDA device.
pub(crate) fn check_flash_segments() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x2f69_3b11_u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                crate::half::r16(x)
            })
            .collect()
    };
    let (h, g, w) = (16_usize, 8_usize, 128_usize);
    let lens = [37_usize, 300, 130, 64];
    let total: usize = lens.iter().sum();
    let mut on = vec![1_u8; h];
    on[9] = 0;
    let m = o.upload_mask(&on)?;
    let q = o.upload_f32(&rnd(total * h * w), vec![total, h, w])?;
    let k = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    let v = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    for p in [0_usize, 50] {
        let pk = o.upload_f32(&rnd(p.max(1) * g * w), vec![p.max(1), g, w])?;
        let pv = o.upload_f32(&rnd(p.max(1) * g * w), vec![p.max(1), g, w])?;
        let prefix = (p > 0).then_some((&pk, &pv));
        let one = o.download(&o.launch_segments(&q, &k, &v, &m, prefix, &lens, 0, None)?)?;
        let mut at = 0;
        for &n in &lens {
            let take = |x: &DevT, heads: usize, pre: Option<&DevT>| -> Result<DevT, String> {
                let rows = n + pre.map_or(0, |_| p);
                let out = o.alloc(vec![rows, heads, w], F16)?;
                let mut off = 0;
                if let Some(pre) = pre {
                    o.gpu.copy(&pre.buf, 0, &out.buf, 0, p * heads * w * 2)?;
                    off = p * heads * w * 2;
                }
                o.gpu.copy(&x.buf, at * heads * w * 2, &out.buf, off, n * heads * w * 2)?;
                Ok(out)
            };
            let (qs, ks, vs) = (take(&q, h, None)?, take(&k, g, prefix.map(|x| x.0))?, take(&v, g, prefix.map(|x| x.1))?);
            let y = o.alloc(qs.shape.clone(), F16)?;
            let mut a = Ops::args(&[&qs, &ks, &vs, &m], &[], &[&y]);
            a.attrs.causal = 1;
            let geo = Geometry { tokens: n as u64, cached: (n + p) as u64, heads: h as u64, kv_heads: g as u64, width: w as u64, clones: 1 };
            o.gpu.launch("_ZN4emma12flash_kernelE12emma_op_argsNS_8GeometryE", (u(n.div_ceil(64)), u(h), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a)), Param::Geo(geo)])?;
            let old = o.download(&y)?;
            let new = one.get(at * h * w..(at + n) * h * w).ok_or("short output")?;
            if old.iter().zip(new).any(|(a, b)| a.to_bits() != b.to_bits()) {
                return Err(format!("nmengn.gpu.flash_segments: prefix {p}, sequence of {n} at {at}: outputs differ from flash_kernel"));
            }
            at += n;
        }
    }
    Ok(())
}

/// Lane S package 5 (2026-11-18): `flash_segments_h_kernel` (binary16 products) against
/// `flash_segments_kernel` (binary32) on random binary16 values (16 heads over 8 of width 128, one
/// head masked off; sequences of 37, 300, 130 and 64 rows over a 50-row prefix, and a tail of 20
/// rows over the 300-row sequence as its body): every output within 1/64 of the binary32 kernel's
/// (outputs are weighted means of values in -2..2) and the masked head zero. Passes on a PC without
/// a CUDA device.
pub(crate) fn check_flash_h() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x7a3c_51e9_u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                crate::half::r16(x)
            })
            .collect()
    };
    let (h, g, w) = (16_usize, 8_usize, 128_usize);
    let lens = [37_usize, 300, 130, 64, 20];
    let bodies = [None, None, None, None, Some(1)];
    let total: usize = lens.iter().sum();
    let mut on = vec![1_u8; h];
    on[9] = 0;
    let m = o.upload_mask(&on)?;
    let q = o.upload_f32(&rnd(total * h * w), vec![total, h, w])?;
    let k = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    let v = o.upload_f32(&rnd(total * g * w), vec![total, g, w])?;
    let pk = o.upload_f32(&rnd(50 * g * w), vec![50, g, w])?;
    let pv = o.upload_f32(&rnd(50 * g * w), vec![50, g, w])?;
    let f = o.download(&o.launch_segments(&q, &k, &v, &m, Some((&pk, &pv)), &lens, 0, Some(&bodies))?)?;
    let hh = o.download(&o.launch_segments(&q, &k, &v, &m, Some((&pk, &pv)), &lens, 1, Some(&bodies))?)?;
    let worst = f.iter().zip(&hh).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    if f.len() != hh.len() || !worst.is_finite() || worst > 1.0 / 64.0 {
        return Err(format!("nmengn.gpu.flash_h: largest difference {worst} from flash_segments_kernel"));
    }
    for t in 0..total {
        if hh[(t * h + 9) * w..(t * h + 10) * w].iter().any(|x| *x != 0.0) {
            return Err("nmengn.gpu.flash_h: a masked head's output is not zero".to_owned());
        }
    }
    Ok(())
}

/// Lane S (2026-11-18, T-079): `h2_fetch_kernel` gives the outputs of `wide_h2_kernel` (binary16
/// weights, half products) bit for bit (64, 130 and 300 tokens; 200 rows; width 1,024). Passes on a
/// PC without a CUDA device.
pub(crate) fn check_h2_fetch() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x4b1d_9e27_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let (n, k) = (200_usize, 1024_usize);
    #[expect(clippy::cast_precision_loss, reason = "a test value")]
    let wv: Vec<f32> = (0..n * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 20000.0)).collect();
    let w = o.upload_f32(&wv, vec![n, k])?;
    for tokens in [64_usize, 130, 300] {
        #[expect(clippy::cast_precision_loss, reason = "a test value")]
        let xs: Vec<f32> = (0..tokens * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 250.0)).collect();
        let x = o.upload_f32(&xs, vec![tokens, k])?;
        let run = |name: &str| -> Result<Vec<f32>, String> {
            let y = o.alloc(vec![tokens, n], F16)?;
            let a = Ops::args(&[&x], &[&w], &[&y]);
            o.gpu.launch(name, (u(n.div_ceil(64)), u(tokens.div_ceil(128)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
            o.download(&y)
        };
        let old = run("_ZN4emma14wide_h2_kernelILi8ELi4EEEv12emma_op_args")?;
        let new = run("_ZN4emma15h2_fetch_kernelE12emma_op_args")?;
        if old.len() != new.len() || old.iter().zip(&new).any(|(a, b)| a.to_bits() != b.to_bits()) {
            return Err(format!("nmengn.gpu.h2_fetch: {tokens} tokens: outputs differ from wide_h2_kernel"));
        }
    }
    Ok(())
}

/// Device-resident compressed `bitmask-q8` linear matrix for `kernels/cukern/bmq8.cu`.
#[derive(Clone)]
pub struct DevBitmaskQ8 {
    pub rows: usize,
    pub cols: usize,
    pub groups_per_row: usize,
    pub mask_words: DevT,
    pub group_offs: DevT,
    pub row_offs: DevT,
    pub packed_i8: DevT,
    pub row_scales: DevT,
}

/// Wave 2 (2026-11-21): `bmq8.cu` (`bmq8_vec_kernel` for `tokens == 1` and `bmq8_unpack_kernel` ->
/// `linear_q8` for `tokens >= 2`) matches dense masked `linear_q8` on `[200, 1024]` and `[200, 3072]`
/// matrices at 35% and 50% unstructured sparsity while reducing device VRAM bytes. Passes on a PC
/// without a CUDA device.
pub(crate) fn check_bmq8() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x51a7_9b2d_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (k, sparsity_pct) in [(1024_usize, 35_u64), (3072_usize, 50_u64)] {
        let rows = 200_usize;
        let mut mask = crate::pmap::BitMask::new(rows, k, true);
        let mut dense_i8 = vec![0_u8; rows * k];
        let mut packed_i8 = Vec::with_capacity(rows * k);
        let mut scales = Vec::with_capacity(rows);
        for r in 0..rows {
            let sc = crate::half::r16(0.0005 + f32::from((next() & 0xFF) as u8) * 0.00002);
            scales.push(sc);
            for c in 0..k {
                let keep = (next() % 100) >= sparsity_pct;
                mask.set(r, c, keep);
                if keep {
                    #[expect(clippy::cast_possible_truncation, reason = "bounded i8 test value")]
                    let q = ((next() % 251) as i32 - 125) as i8;
                    dense_i8[r * k + c] = q.to_ne_bytes()[0];
                    packed_i8.push(q);
                }
            }
        }
        let packed = crate::pmap::PackedBitmaskQ8 { mask, row_scales: scales.clone(), packed_i8 };
        let gpu_layout = packed.to_gpu_layout();
        if gpu_layout.device_bytes() >= gpu_layout.dense_q8_bytes() {
            return Err(format!(
                "nmengn.gpu.bmq8: expected VRAM compression at {sparsity_pct}% sparsity (bmq8={} >= dense={})",
                gpu_layout.device_bytes(),
                gpu_layout.dense_q8_bytes()
            ));
        }
        let dev_bm = o.upload_bmq8(&gpu_layout)?;
        let wq_dense = o.alloc(vec![rows, k], I8)?;
        o.gpu.upload(&wq_dense.buf, &dense_i8)?;
        let ws_dense = o.upload_f32(&scales, vec![rows])?;

        for tokens in [1_usize, 5, 12, 64, 130] {
            let xv: Vec<f32> = (0..tokens * k)
                .map(|_| crate::half::r16((f32::from((next() & 0xFF) as u8) - 127.0) / 16.0))
                .collect();
            let x = o.upload_f32(&xv, vec![tokens, k])?;
            let ref_y = o.download(&o.linear_q8(&x, &wq_dense, &ws_dense)?)?;
            let got_y = o.download(&o.linear_bmq8(&x, &dev_bm)?)?;
            if ref_y.len() != got_y.len() {
                return Err(format!("nmengn.gpu.bmq8: shape mismatch at k={k} tokens={tokens}"));
            }
            if (2..=16).contains(&tokens) {
                // `bmq8_small_kernel` matches `bmq8_vec_kernel` bit-for-bit on every token row
                // (exact same warp reduction order) and matches `tiled_kernel` within f16 rounding.
                for t in 0..tokens {
                    let row_x = o.upload_f32(&xv[t * k..(t + 1) * k], vec![1, k])?;
                    let single_y = o.download(&o.linear_bmq8(&row_x, &dev_bm)?)?;
                    let slice_y = &got_y[t * rows..(t + 1) * rows];
                    if single_y.iter().zip(slice_y).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        return Err(format!("nmengn.gpu.bmq8: bmq8_small_kernel tokens={tokens} row {t} differs from bmq8_vec_kernel"));
                    }
                }
                // Relative to `tiled_kernel` (different f32 summation order), allow at most 1 f16 ULP.
                for (&a, &b) in ref_y.iter().zip(&got_y) {
                    let ulp = (a.abs().max(b.abs()) / 1024.0).max(1e-3);
                    if (a - b).abs() > ulp * 2.0 {
                        return Err(format!(
                            "nmengn.gpu.bmq8: tokens={tokens} k={k} differs from dense linear_q8 (a={a}, b={b}, ulp={ulp:.6})"
                        ));
                    }
                }
            } else if ref_y.iter().zip(&got_y).any(|(a, b)| a.to_bits() != b.to_bits()) {
                let max_diff = ref_y.iter().zip(&got_y).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                return Err(format!("nmengn.gpu.bmq8: tokens={tokens} k={k} differs from dense linear_q8 (max_diff={max_diff:.6})"));
            }
        }
    }
    Ok(())
}

/// Lane S (2026-11-18): the element kernels of `fastel.cu` give the outputs of the kernels they
/// replace bit for bit on random binary16 values: `norm_warp_kernel` against `norm_row_kernel`
/// (widths 128 over 1,001 rows and 1,024 over 37), `binary8_kernel` against `binary_kernel` (SwiGLU
/// gate and add over 8,005 values, so the last thread takes a short tail), `rope2_kernel` against
/// `rope_kernel` on queries and keys (37 tokens, 16 and 8 heads of 128, positions up to 40,000,
/// theta 1,000,000). Passes on a PC without a CUDA device.
pub(crate) fn check_fastel() -> Result<(), String> {
    let gpu = match nmffi::Gpu::open(0) {
        Ok(g) => g,
        Err(e) if e.contains("nvcuda.dll not found") || e.contains(nmffi::cuda::NO_USABLE) => return Ok(()),
        Err(e) => return Err(e),
    };
    let o = Ops { gpu };
    let mut seed = 0x3c6e_f372_u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                #[expect(clippy::cast_precision_loss, reason = "a test value")]
                let x = ((seed % 20001) as f32 - 10000.0) / 10000.0 * scale;
                crate::half::r16(x)
            })
            .collect()
    };
    let same = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    for (width, rows) in [(128_usize, 1001_usize), (1024, 37)] {
        let x = o.upload_f32(&rnd(rows * width, 6.0), vec![rows, width])?;
        let w = o.upload_f32(&rnd(width, 2.0), vec![width])?;
        let y_old = o.alloc(vec![rows, width], F16)?;
        let mut a = Ops::args(&[&x], &[&w], &[&y_old]);
        a.attrs.eps = 1e-6;
        o.gpu.launch("_ZN4emma15norm_row_kernelE12emma_op_args", (u(rows), 1, 1), (256, 1, 1), u(width * 4), &[Param::Op(Box::new(a))])?;
        let new = o.norm(&x, &w, 1e-6)?;
        if !fast_element() || !same(&o.download(&y_old)?, &o.download(&new)?) {
            return Err(format!("nmengn.gpu.fastel: norm of width {width}: outputs differ from norm_row_kernel (or NANO_FAST_ELEMENT=off)"));
        }
    }
    let n = 8005_usize;
    let (l, r) = (o.upload_f32(&rnd(n, 12.0), vec![n])?, o.upload_f32(&rnd(n, 4.0), vec![n])?);
    for op in [4, 5] {
        let y_old = o.alloc(vec![n], F16)?;
        let a = Ops::args(&[&l, &r], &[], &[&y_old]);
        o.gpu.launch("_ZN4emma13binary_kernelEi12emma_op_args", (blocks(n), 1, 1), (128, 1, 1), 0, &[Param::I32(op), Param::Op(Box::new(a))])?;
        let new = o.binary(op, &l, &r)?;
        if !same(&o.download(&y_old)?, &o.download(&new)?) {
            return Err(format!("nmengn.gpu.fastel: binary op {op}: outputs differ from binary_kernel"));
        }
    }
    let (t, h, g, d) = (37_usize, 16_usize, 8_usize, 128_usize);
    let q = o.upload_f32(&rnd(t * h * d, 4.0), vec![t, h, d])?;
    let k = o.upload_f32(&rnd(t * g * d, 4.0), vec![t, g, d])?;
    #[expect(clippy::cast_possible_truncation, reason = "positions under 40,001")]
    let pos: Vec<i32> = rnd(t, 1.0).iter().map(|v| ((v + 1.0) * 20000.0) as i32).collect();
    let p = o.upload_i32(&pos)?;
    let (qo, ko) = (o.alloc(q.shape.clone(), F16)?, o.alloc(k.shape.clone(), F16)?);
    let mut a = Ops::args(&[&q, &k, &p], &[], &[&qo, &ko]);
    a.attrs.theta = 1_000_000.0;
    for (tensor, count) in [(0, q.elements()), (1, k.elements())] {
        o.gpu.launch("_ZN4emma11rope_kernelE12emma_op_argsi", (blocks(count / 2), 1, 1), (128, 1, 1), 0, &[Param::Op(Box::new(a)), Param::I32(tensor)])?;
    }
    let (qn, kn) = o.rope(&q, &k, &p, 1_000_000.0)?;
    if !same(&o.download(&qo)?, &o.download(&qn)?) || !same(&o.download(&ko)?, &o.download(&kn)?) {
        return Err("nmengn.gpu.fastel: rope: outputs differ from rope_kernel".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod speed {
    use super::{DevT, F16, I8, Ops, clear_run_holds, u};
    use nmffi::{Geometry, Param};

    /// The lane S kernels against the kernels they replace (section checks, run alone).
    #[test]
    fn lane_s_parity() -> Result<(), String> {
        super::check_dp4a_tile()?;
        super::check_h2q8_tile()?;
        super::check_segments()?;
        super::check_wide_q8f()?;
        super::check_flash_segments()?;
        super::check_flash_h()?;
        super::check_h2_fetch()?;
        super::check_fastel()
    }

    /// Lane S (2026-11-18): the batched forward with a block mask per sequence group
    /// (`RunOpts::segments`: one card pass, attention block-diagonal by sequence, positions
    /// restarting per sequence, optionally over one shared prefix) equals separate forwards of each
    /// sequence within tolerance. Reader (8-bit linears, served switches) with and without a served
    /// scope, and with a shared 64-token prefix (each separate forward reads prefix then sequence
    /// from position 0); embedder pooled vectors. Sequence lengths 37, 128, 90 and 200, so the
    /// separate forwards take other kernels than the batch (under 64 tokens: the small-batch
    /// linears). Pass: per sequence the largest logit difference is under 0.25 (logits span about
    /// 30) and the greedy token is the same, or differs only where the separate forward's two
    /// tokens are within twice that difference (a tie); the embedder's cosines at least 0.999.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture segments_equal_separate`.
    #[test]
    #[ignore = "needs the graphics card and the open weights"]
    fn segments_equal_separate() -> Result<(), String> {
        use crate::model::{Act, Cache, Model, Population, RunOpts};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let text = std::fs::read_to_string(home.join("crates/nmengn/src/gpu.rs")).map_err(|e| e.to_string())?;
        let lens = [37_usize, 128, 90, 200];
        let host = |r: &std::collections::BTreeMap<(String, String), Act>, t: &(String, String)| -> Result<Vec<f32>, String> {
            match r.get(t) {
                Some(Act::Cpu(v)) => Ok(v.clone()),
                _ => Err("no host output".to_owned()),
            }
        };
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |x| x.0);
        let mut fails = Vec::new();
        for (pop_id, asm_name) in [("qwen3-0.6b", "qwen3-reference"), ("qwen3-embedding-0.6b", "qwen3-embedding")] {
            let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), asm_name).map_err(|e| e.to_string())?;
            asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
            let pop = Population::open(&home, pop_id, true, true).map_err(|e| e.to_string())?;
            let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
            let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
            let mut at = 64;
            let seqs: Vec<Vec<i32>> = lens
                .iter()
                .map(|n| {
                    let s = all[at..at + n].to_vec();
                    at += n + 11;
                    s
                })
                .collect();
            let prefix: Vec<i32> = all[..64].to_vec();
            let packed: Vec<i32> = seqs.concat();
            let segs = Some(Arc::new(lens.to_vec()));
            let reader = pop_id == "qwen3-0.6b";
            let target = if reader { (m.role_node("head").map_err(|e| e.to_string())?, "logits".to_owned()) } else { (m.role_node("pool").map_err(|e| e.to_string())?, "y".to_owned()) };
            let width = if reader { m.pop.dims.vocab } else { m.pop.dims.hidden };
            let cases: Vec<(&str, bool, bool)> = if reader { vec![("plain", false, false), ("served scope", true, false), ("shared prefix", false, true)] } else { vec![("plain", false, false)] };
            for (case, served, with_prefix) in cases {
                let _s = served.then(super::ServedScope::new);
                let batched = if with_prefix {
                    let cache = Arc::new(Cache::default());
                    let warm = RunOpts { cache: Some(Arc::clone(&cache)), ..RunOpts::default() };
                    m.forward(Device::Gpu, &prefix, std::slice::from_ref(&target), &warm).map_err(|e| e.to_string())?;
                    let opts = RunOpts { cache: Some(cache), segments: segs.clone(), ..RunOpts::default() };
                    host(&m.forward_at(Device::Gpu, &packed, prefix.len(), std::slice::from_ref(&target), &opts).map_err(|e| e.to_string())?, &target)?
                } else {
                    let opts = RunOpts { segments: segs.clone(), ..RunOpts::default() };
                    host(&m.forward(Device::Gpu, &packed, std::slice::from_ref(&target), &opts).map_err(|e| e.to_string())?, &target)?
                };
                for (i, s) in seqs.iter().enumerate() {
                    let ids: Vec<i32> = if with_prefix { prefix.iter().chain(s).copied().collect() } else { s.clone() };
                    let one = host(&m.forward(Device::Gpu, &ids, std::slice::from_ref(&target), &RunOpts::default()).map_err(|e| e.to_string())?, &target)?;
                    let row = batched.get(i * width..(i + 1) * width).ok_or("short batched output")?;
                    if reader {
                        let worst = row.iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                        // A changed greedy token passes only as a tie: the separate forward's own
                        // margin between those two tokens is within twice the largest difference.
                        let tie = (one[argmax(&one)] - one[argmax(row)]).abs() <= 2.0 * worst;
                        let ok = (argmax(row) == argmax(&one) || tie) && worst < 0.25;
                        eprintln!(
                            "{pop_id} {case}: sequence {i} ({} tokens): greedy token {} against {} (separate margin {:.4}), largest logit difference {worst:.4}{}",
                            s.len(),
                            argmax(row),
                            argmax(&one),
                            one[argmax(&one)] - one[argmax(row)],
                            if ok { "" } else { "  FAIL" }
                        );
                        if !ok {
                            fails.push(format!("{pop_id} {case} {i}"));
                        }
                    } else {
                        let cos = row.iter().zip(&one).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>()
                            / (row.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt() * one.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt()).max(1e-30);
                        eprintln!("{pop_id} {case}: sequence {i} ({} tokens): cosine {cos:.6}{}", s.len(), if cos >= 0.999 { "" } else { "  FAIL" });
                        if cos < 0.999 {
                            fails.push(format!("{pop_id} {case} {i}"));
                        }
                    }
                }
            }
        }
        if fails.is_empty() { Ok(()) } else { Err(format!("segments differ from separate forwards: {fails:?}")) }
    }

    /// Lane S package 2 (2026-11-18): kernel time of each linear variant over one pass of the
    /// 0.6B MicroModel's projections (query 2,048, key and value 1,024, output 1,024 over 2,048,
    /// gate and up 3,072, down 1,024 over 3,072; 28 layers) at `NANO_LANE_S_TOKENS` tokens
    /// (default 512), launched back to back with one synchronisation (no per-launch wait), median
    /// of 3; and each variant's outputs on the query shape against its reference kernel, bit for bit.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture lane_s_kernels`.
    #[test]
    #[ignore = "needs the graphics card; timing only"]
    fn lane_s_kernels() -> Result<(), String> {
        let o = Ops { gpu: nmffi::Gpu::open(0)? };
        let tokens: usize = std::env::var("NANO_LANE_S_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        let shapes = [(2048_usize, 1024_usize), (1024, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (3072, 1024), (1024, 3072)];
        let mut seed = 0x5bd1_e995_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Per shape: x binary16, x as int8 blocks with scales, int8 weights, row scales.
        let mut data = Vec::new();
        for (n, k) in shapes {
            #[expect(clippy::cast_precision_loss, reason = "a test value")]
            let xv: Vec<f32> = (0..tokens * k).map(|_| crate::half::r16(((next() % 2000) as f32 - 1000.0) / 300.0)).collect();
            let x = o.upload_f32(&xv, vec![tokens, k])?;
            let (xq, xs) = o.q8_blocks(&x, tokens, k)?;
            clear_run_holds();
            let wq = o.alloc(vec![n, k], I8)?;
            o.gpu.upload(&wq.buf, &(0..n * k).map(|_| (next() % 255) as u8).collect::<Vec<u8>>())?;
            #[expect(clippy::cast_precision_loss, reason = "a test value")]
            let sc: Vec<f32> = (0..n).map(|_| crate::half::r16(0.0005 + (next() % 1000) as f32 * 0.00002)).collect();
            let ws = o.upload_f32(&sc, vec![n])?;
            data.push((n, k, x, xq, xs, wq, ws));
        }
        // (name, rows a block, tokens a block, int8 activations)
        let variants: [(&str, usize, usize, bool); 9] = [
            ("_ZN4emma16wide_h2q8_kernelILi8ELi4EEEv12emma_op_args", 64, 128, false),
            ("_ZN4emma16h2q8_tile_kernelE12emma_op_args", 64, 128, false),
            ("_ZN4emma17h2q8_fetch_kernelILi4EEEv12emma_op_args", 64, 128, false),
            ("_ZN4emma17h2q8_fetch_kernelILi8EEEv12emma_op_args", 128, 128, false),
            ("_ZN4emma16wide_dp4a_kernelILi8ELi4EEEv12emma_op_args", 64, 128, true),
            ("_ZN4emma16dp4a_tile_kernelE12emma_op_args", 64, 128, true),
            ("_ZN4emma14dp4a_tt_kernelILi8ELi4ELi2EEEv12emma_op_args", 64, 128, true),
            ("_ZN4emma14dp4a_tt_kernelILi8ELi8ELi1EEEv12emma_op_args", 128, 128, true),
            ("_ZN4emma14dp4a_tt_kernelILi4ELi8ELi2EEEv12emma_op_args", 128, 64, true),
        ];
        let mut reference: [Option<Vec<f32>>; 2] = [None, None];
        for (name, br, bt, int8) in variants {
            let outs: Vec<DevT> = data.iter().map(|d| o.alloc(vec![tokens, d.0], F16)).collect::<Result<_, _>>()?;
            let launch_all = || -> Result<(), String> {
                for (d, y) in data.iter().zip(&outs) {
                    let a = if int8 { Ops::args(&[&d.3, &d.4], &[&d.5, &d.6], &[y]) } else { Ops::args(&[&d.2], &[&d.5, &d.6], &[y]) };
                    o.gpu.launch(name, (u(d.0.div_ceil(br)), u(tokens.div_ceil(bt)), 1), (256, 1, 1), 0, &[Param::Op(Box::new(a))])?;
                }
                Ok(())
            };
            launch_all()?;
            o.gpu.sync()?;
            let mut times = Vec::new();
            for _ in 0..3 {
                let t = std::time::Instant::now();
                for _ in 0..28 {
                    launch_all()?;
                }
                o.gpu.sync()?;
                times.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(f64::total_cmp);
            let first = o.download(&outs[0])?;
            let slot = usize::from(int8);
            let same = match &reference[slot] {
                None => {
                    reference[slot] = Some(first);
                    "reference".to_owned()
                }
                Some(r) => {
                    let ok = r.len() == first.len() && r.iter().zip(&first).all(|(a, b)| a.to_bits() == b.to_bits());
                    if ok { "bit-identical".to_owned() } else { "DIFFERS".to_owned() }
                }
            };
            #[expect(clippy::cast_precision_loss, reason = "a rate for a report")]
            let tops = 2.0 * 15_728_640.0 * 28.0 * tokens as f64 / (times[1] / 1000.0) / 1e12;
            eprintln!("{tokens} tokens: {name}: {:.1} ms a pass of 196 linears ({tops:.2} T a second); {same}", times[1]);
        }
        Ok(())
    }

    /// Lane S package 2 (2026-11-18): the reader's short batched prompts with a shared prefix (the
    /// tentacles' poses, lane R's packed yes-or-no prompts): a 160-token prefix, then 8 or 32
    /// sequences of 20 to 60 tokens, the head reading two rows (as the label reads do), 8-bit
    /// linears and the served switches, outside and inside a served scope. Timed (median of 5 after
    /// a warm-up): separate forwards of prefix and sequence; the prefix prefilled once into a cache
    /// and the sequences packed in one run over it (`RunOpts::cache` with `segments`). The packed
    /// rows are compared with the separate forwards' (largest difference, yes-or-no order).
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture short_batches`.
    #[test]
    #[ignore = "needs the graphics card and the open weights; timing only"]
    fn short_batches() -> Result<(), String> {
        use crate::model::{Act, Cache, Model, Population, RunOpts};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let text = std::fs::read_to_string(home.join("crates/nmengn/src/model.rs")).map_err(|e| e.to_string())?;
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let first = |s: &str| m.pop.tokenizer.encode(s).ok().and_then(|v| v.first().and_then(|x| usize::try_from(*x).ok())).unwrap_or(0);
        let rows = Arc::new(vec![first("yes"), first("no")]);
        let target = (m.role_node("head").map_err(|e| e.to_string())?, "logits".to_owned());
        let prefix: Vec<i32> = all[..160].to_vec();
        let host = |r: &std::collections::BTreeMap<(String, String), Act>| -> Result<Vec<f32>, String> {
            match r.get(&target) {
                Some(Act::Cpu(v)) => Ok(v.clone()),
                _ => Err("no host output".to_owned()),
            }
        };
        let med = |f: &mut dyn FnMut() -> Result<Vec<f32>, String>| -> Result<(f64, Vec<f32>), String> {
            let out = f()?;
            let mut t = Vec::new();
            for _ in 0..5 {
                let s = std::time::Instant::now();
                f()?;
                t.push(s.elapsed().as_secs_f64() * 1000.0);
            }
            t.sort_by(f64::total_cmp);
            Ok((t[2], out))
        };
        for n in [8_usize, 32] {
            let mut at = 200;
            let seqs: Vec<Vec<i32>> = (0..n)
                .map(|i| {
                    let len = 20 + (i * 13) % 41;
                    let s = all[at..at + len].to_vec();
                    at += len + 7;
                    s
                })
                .collect();
            let tokens: usize = seqs.iter().map(Vec::len).sum();
            for served in [false, true] {
                let _s = served.then(super::ServedScope::new);
                let (t_sep, sep) = med(&mut || {
                    let mut v = Vec::new();
                    for s in &seqs {
                        let ids: Vec<i32> = prefix.iter().chain(s).copied().collect();
                        let o = RunOpts { head_rows: Arc::clone(&rows), ..RunOpts::default() };
                        v.extend(host(&m.forward(Device::Gpu, &ids, std::slice::from_ref(&target), &o).map_err(|e| e.to_string())?)?);
                    }
                    Ok(v)
                })?;
                let packed: Vec<i32> = seqs.concat();
                let lens = Arc::new(seqs.iter().map(Vec::len).collect::<Vec<_>>());
                let (t_pk, pk) = med(&mut || {
                    let cache = Arc::new(Cache::default());
                    let warm = RunOpts { cache: Some(Arc::clone(&cache)), head_rows: Arc::clone(&rows), ..RunOpts::default() };
                    m.forward(Device::Gpu, &prefix, std::slice::from_ref(&target), &warm).map_err(|e| e.to_string())?;
                    let o = RunOpts { cache: Some(cache), segments: Some(Arc::clone(&lens)), head_rows: Arc::clone(&rows), ..RunOpts::default() };
                    host(&m.forward_at(Device::Gpu, &packed, prefix.len(), std::slice::from_ref(&target), &o).map_err(|e| e.to_string())?)
                })?;
                if std::env::var("NANO_LANE_S_PROF").is_ok() && n == 32 {
                    let cache = Arc::new(Cache::default());
                    let warm = RunOpts { cache: Some(Arc::clone(&cache)), head_rows: Arc::clone(&rows), ..RunOpts::default() };
                    m.forward(Device::Gpu, &prefix, std::slice::from_ref(&target), &warm).map_err(|e| e.to_string())?;
                    let o = RunOpts { cache: Some(cache), segments: Some(Arc::clone(&lens)), head_rows: Arc::clone(&rows), ..RunOpts::default() };
                    nmffi::prof::reset();
                    nmffi::prof::set(true);
                    nmffi::prof::set_kernels(true);
                    m.forward_at(Device::Gpu, &packed, prefix.len(), std::slice::from_ref(&target), &o).map_err(|e| e.to_string())?;
                    nmffi::prof::set_kernels(false);
                    nmffi::prof::set(false);
                    let mut k: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
                    k.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let mut t: Vec<(String, f64, u64)> = nmffi::prof::table();
                    t.sort_by(|a, b| b.1.total_cmp(&a.1));
                    for (name, secs, c) in k.iter().take(8).chain(t.iter().take(8)) {
                        eprintln!("    {:7.2} ms {c:6}  {name}", secs * 1000.0);
                    }
                }
                let worst = sep.iter().zip(&pk).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                let order = sep.chunks(2).zip(pk.chunks(2)).filter(|(a, b)| (a[0] > a[1]) == (b[0] > b[1])).count();
                eprintln!(
                    "{n} sequences ({tokens} tokens) over a 160-token prefix, {}: separate {t_sep:.1} ms, prefix once and packed {t_pk:.1} ms ({:.1} times); largest logit difference {worst:.4}, yes-or-no order same on {order} of {n}",
                    if served { "served scope" } else { "plain" },
                    t_sep / t_pk.max(1e-9)
                );
            }
        }
        Ok(())
    }

    /// Lane S package 3 (2026-11-18): the tentacles' integration shape. 1, 2 or 4 groups of 200, 600
    /// or 1,500 tokens each, block-masked (`RunOpts::segments`: each group attends only to itself),
    /// through layer 17 of 28 (the residual after L17 stays on the card), reader at 8 bits with the
    /// served switches, outside and inside a served scope. Median of 3 after a warm-up; ms a token
    /// and the card tokens a 1 s budget buys at that rate.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture integration_shape`.
    #[test]
    #[ignore = "needs the graphics card and the open weights; timing only"]
    fn integration_shape() -> Result<(), String> {
        use crate::model::{Model, Population, RunOpts};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let mut text = String::new();
        for f in ["crates/nmengn/src/model.rs", "crates/nmengn/src/gpu.rs", "crates/nmengn/src/kvstore.rs"] {
            text.push_str(&std::fs::read_to_string(home.join(f)).map_err(|e| e.to_string())?);
        }
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let target = (crate::assembly::nid("L17.res2"), "y".to_owned());
        let o = m.pop.gpu.clone().ok_or("no card")?;
        for served in [false, true] {
            let _s = served.then(super::ServedScope::new);
            for len in [200_usize, 600, 1500] {
                for groups in [1_usize, 2, 4] {
                    let total = len * groups;
                    let ids: Vec<i32> = all.iter().copied().cycle().take(total).collect();
                    let opts = RunOpts { segments: Some(Arc::new(vec![len; groups])), device_targets: true, ..RunOpts::default() };
                    let pass = || -> Result<(), String> {
                        m.forward(Device::Gpu, &ids, std::slice::from_ref(&target), &opts).map_err(|e| e.to_string())?;
                        o.gpu.sync()
                    };
                    pass()?;
                    let mut t = Vec::new();
                    for _ in 0..3 {
                        let s = std::time::Instant::now();
                        pass()?;
                        t.push(s.elapsed().as_secs_f64() * 1000.0);
                    }
                    t.sort_by(f64::total_cmp);
                    #[expect(clippy::cast_precision_loss, reason = "a rate for a report")]
                    let per = t[1] / total as f64;
                    eprintln!(
                        "{}: {groups} groups of {len} ({total} tokens) through L17: {:.1} ms, {per:.3} ms a token, {:.0} tokens in a 1 s budget",
                        if served { "served scope" } else { "plain" },
                        t[1],
                        1000.0 / per
                    );
                }
            }
        }
        Ok(())
    }

    /// Lane S package 4 (2026-11-18): the judge MicroModel (Qwen3-Reranker-0.6B, linears at 8 bits,
    /// the served switches) scoring 10 candidates of about 128 tokens for one question over the
    /// shared question prefix (`roles::card_rerank_batch`, the packed path): median seconds of 5
    /// after a warm-up, one profiled pass (synchronised kernels, shares only), and the scores against
    /// one candidate at a time (`card_rerank_set_capped`): largest P(yes) difference and order.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture judge_batch`.
    #[test]
    #[ignore = "needs the graphics card and the open weights; timing only"]
    fn judge_batch() -> Result<(), String> {
        use crate::model::{Model, Population};
        use crate::placement::Device;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reranker").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-reranker-0.6b", true, true).map_err(|e| e.to_string())?;
        // As served: the head bound to its label rows. `NANO_LANE_S_JUDGE=dp4a` reads with 8-bit
        // activations and int8 dot products (with `NANO_INT8_DOT` naming `reranker`) instead of the
        // judge's half products; scores are saved per mode and compared with the half-product file.
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?.label_rows();
        let mode = std::env::var("NANO_LANE_S_JUDGE").unwrap_or_else(|_| "half".to_owned());
        if mode == "dp4a" {
            crate::roles::JUDGE_Q8_HALF.store(false, Ordering::Relaxed);
        }
        let text = std::fs::read_to_string(home.join("crates/nmengn/src/model.rs")).map_err(|e| e.to_string())?;
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let docs: Vec<String> = (0..10).map(|i| m.pop.tokenizer.decode(&all[200 + i * 300..200 + i * 300 + 128])).collect();
        let q = "Where does the engine release an activation after its last reader has run?";
        let ins = crate::roles::CODE_INSTRUCTION;
        let batch = || crate::roles::card_rerank_batch(&m, Device::Gpu, ins, q, &docs, None).map_err(|e| e.to_string());
        let first = batch()?;
        let mut t = Vec::new();
        for _ in 0..5 {
            let s = std::time::Instant::now();
            batch()?;
            t.push(s.elapsed().as_secs_f64());
        }
        t.sort_by(f64::total_cmp);
        eprintln!("judge, 10 candidates of 128 tokens, packed over the question prefix: median {:.3} s (runs {:.3} to {:.3})", t[2], t[0], t[4]);
        nmffi::prof::reset();
        nmffi::prof::set(true);
        nmffi::prof::set_kernels(true);
        batch()?;
        nmffi::prof::set_kernels(false);
        nmffi::prof::set(false);
        let mut k: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
        k.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut tb = nmffi::prof::table();
        tb.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (name, secs, c) in k.iter().take(8).chain(tb.iter().take(6)) {
            eprintln!("    {:7.2} ms {c:6}  {name}", secs * 1000.0);
        }
        let one = crate::roles::card_rerank_set_capped(&m, Device::Gpu, ins, q, &docs, None).map_err(|e| e.to_string())?;
        let worst = first.iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        let rank = |v: &[f32]| {
            let mut o: Vec<usize> = (0..v.len()).collect();
            o.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
            o
        };
        eprintln!("  {mode}: against one at a time: largest P(yes) difference {worst:.4}; order identical {}; first place same {}", rank(&first) == rank(&one), rank(&first).first() == rank(&one).first());
        let dir = home.join("scratch/lane-s");
        let _ = std::fs::write(dir.join(format!("judge-{mode}.f32")), first.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        if mode != "half"
            && let Ok(b) = std::fs::read(dir.join("judge-half.f32"))
        {
            let half: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let worst = half.iter().zip(&first).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            let z = |p: f32| {
                let p = p.clamp(1e-6, 1.0 - 1e-6);
                (p / (1.0 - p)).ln()
            };
            let dz = half.iter().zip(&first).map(|(a, b)| (z(*a) - z(*b)).abs()).fold(0.0_f32, f32::max);
            eprintln!("  {mode} against the half products: largest P(yes) difference {worst:.4}, largest log-odds change {dz:.3}; order identical {}; first place same {}", rank(&half) == rank(&first), rank(&half).first() == rank(&first).first());
        }
        Ok(())
    }

    /// Lane S (2026-11-18, T-074): `kvstore::read_groups` (every group's pieces and question in one
    /// packed card run, relevance heads recorded per group) against the plain per-group read (the
    /// pieces encoded behind the anchor through layer 17, `encode_through`, then the question over
    /// that window with a probe). Reader at 8 bits, served switches, an anchor of about 40 tokens,
    /// heads (15, 3), (16, 7), (17, 1), (17, 9); groups of 200 and of 1,500 tokens, 4 each. Reports
    /// per configuration the largest raw-score, log-normaliser and mass differences, whether each
    /// head's top position is the same, and the time of the plain reads against the packed one,
    /// outside and inside a served scope (with `NANO_LANE_S_SERVED=on`).
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture read_groups_equal`.
    #[test]
    #[ignore = "needs the graphics card and the open weights"]
    fn read_groups_equal() -> Result<(), String> {
        use crate::kvstore::{Probe, Window};
        use crate::model::{Model, Population, RunOpts};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let mut text = String::new();
        for f in ["crates/nmengn/src/model.rs", "crates/nmengn/src/kvstore.rs"] {
            text.push_str(&std::fs::read_to_string(home.join(f)).map_err(|e| e.to_string())?);
        }
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
        let _served = (std::env::var("NANO_LANE_S_SERVED").as_deref() == Ok("on")).then(super::ServedScope::new);
        let last = 17;
        let heads = [(15_usize, 3_usize), (16, 7), (17, 1), (17, 9)];
        let stop = crate::kvstore::score_targets(&m, last).map_err(|e| e.to_string())?;
        let anchor = tok("You read source code. The pieces below come from one repository.\n\n")?;
        let ak = Arc::new(crate::kvstore::encode_through(&m, Device::Gpu, None, &anchor, last).map_err(|e| e.to_string())?);
        let aw = Arc::new(Window::build(&m, &[Arc::clone(&ak)]).map_err(|e| e.to_string())?);
        let questions = ["Where is an activation released after its last reader?", "How is the key cache kept between decode steps?", "Which kernel reads packed sequences?", "Where are the window's keys roped?"];
        let mut fails = Vec::new();
        for len in [200_usize, 1500] {
            let groups: Vec<(Vec<i32>, Vec<i32>)> = (0..4)
                .map(|g| -> Result<(Vec<i32>, Vec<i32>), String> {
                    let at = 500 + g * 1700;
                    Ok((all[at..at + len].to_vec(), tok(&format!("\nQuestion: {}", questions[g]))?))
                })
                .collect::<Result<_, _>>()?;
            let t = std::time::Instant::now();
            let mut plain = Vec::new();
            for (pieces, q) in &groups {
                let kv = Arc::new(crate::kvstore::encode_through(&m, Device::Gpu, Some(&aw), pieces, last).map_err(|e| e.to_string())?);
                let w = Arc::new(Window::build(&m, &[Arc::clone(&ak), kv]).map_err(|e| e.to_string())?);
                let probe = Arc::new(Probe::of_heads(&heads).with_logits());
                m.forward_at(Device::Gpu, q, w.len, &stop, &RunOpts { window: Some(Arc::clone(&w)), probe: Some(Arc::clone(&probe)), ..RunOpts::default() })
                    .map_err(|e| e.to_string())?;
                plain.push(probe);
            }
            let t_plain = t.elapsed().as_secs_f64();
            let _ = crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &groups).map_err(|e| e.to_string())?;
            let t = std::time::Instant::now();
            let packed = crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &groups).map_err(|e| e.to_string())?;
            let t_packed = t.elapsed().as_secs_f64();
            // Posed and content-free questions of every group in one run (8 sequences).
            let both: Vec<(Vec<i32>, Vec<i32>)> =
                groups.iter().flat_map(|(p, q)| [(p.clone(), q.clone()), (p.clone(), tok("\nQuestion: N/A").unwrap_or_default())]).collect();
            let t = std::time::Instant::now();
            let _ = crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &both).map_err(|e| e.to_string())?;
            let t_both = t.elapsed().as_secs_f64();
            let (mut dl, mut dz, mut dm, mut same_top, mut n_top) = (0.0_f32, 0.0_f32, 0.0_f32, 0, 0);
            for (a, b) in plain.iter().zip(&packed) {
                for &(l, h) in &heads {
                    let (la, lb) = (a.logits(l, h).ok_or("plain logits")?, b.logits(l, h).ok_or("packed logits")?);
                    if la.logits.len() != lb.logits.len() || la.lse.len() != lb.lse.len() {
                        return Err(format!("layer {l} head {h}: shapes differ ({} and {})", la.logits.len(), lb.logits.len()));
                    }
                    dl = la.logits.iter().zip(&lb.logits).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (x - y).abs()).fold(dl, f32::max);
                    dz = la.lse.iter().zip(&lb.lse).map(|(x, y)| (x - y).abs()).fold(dz, f32::max);
                    let (ma, mb) = (a.get(l).ok_or("plain mass")?, b.get(l).ok_or("packed mass")?);
                    let span = la.logits.len() / la.lse.len().max(1);
                    let (ra, rb) = (&ma[h * span..(h + 1) * span], &mb[h * span..(h + 1) * span]);
                    dm = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(dm, f32::max);
                    let top = |v: &[f32]| v.iter().enumerate().skip(anchor.len()).max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |x| x.0);
                    n_top += 1;
                    if top(ra) == top(rb) {
                        same_top += 1;
                    }
                }
            }
            let ok = dl < 0.25 && dm < 0.05 && same_top * 10 >= n_top * 9;
            eprintln!(
                "4 groups of {len} tokens: plain reads {t_plain:.2} s, packed {t_packed:.2} s ({:.1} times), posed and content-free packed {t_both:.2} s; largest raw-score difference {dl:.4}, log-normaliser {dz:.4}, mass {dm:.4}; top piece position same on {same_top} of {n_top}{}",
                t_plain / t_packed.max(1e-9),
                if ok { "" } else { "  FAIL" }
            );
            if !ok {
                fails.push(len);
            }
        }
        if fails.is_empty() { Ok(()) } else { Err(format!("packed group reads differ from plain reads at {fails:?} tokens")) }
    }

    /// Lane S package 5 (2026-11-18): `read_groups_tails` (each group's pieces once, the posed and the
    /// content-free questions as two tails over them) against `read_groups` with the two questions as
    /// separate (pieces, question) groups: same setup and pass rule as `read_groups_equal` (raw score
    /// under 0.25, mass under 0.05, top position the same on 9 of 10), times of both, groups of 200
    /// and 1,500 tokens, 4 each; served scope with `NANO_LANE_S_SERVED=on`.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture read_groups_tails_equal`.
    #[test]
    #[ignore = "needs the graphics card and the open weights"]
    fn read_groups_tails_equal() -> Result<(), String> {
        use crate::kvstore::Window;
        use crate::model::{Model, Population};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let mut text = String::new();
        for f in ["crates/nmengn/src/model.rs", "crates/nmengn/src/kvstore.rs"] {
            text.push_str(&std::fs::read_to_string(home.join(f)).map_err(|e| e.to_string())?);
        }
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
        let _served = (std::env::var("NANO_LANE_S_SERVED").as_deref() == Ok("on")).then(super::ServedScope::new);
        let last = 17;
        let heads = [(15_usize, 3_usize), (16, 7), (17, 1), (17, 9)];
        let stop = crate::kvstore::score_targets(&m, last).map_err(|e| e.to_string())?;
        let anchor = tok("You read source code. The pieces below come from one repository.\n\n")?;
        let ak = Arc::new(crate::kvstore::encode_through(&m, Device::Gpu, None, &anchor, last).map_err(|e| e.to_string())?);
        let aw = Arc::new(Window::build(&m, &[ak]).map_err(|e| e.to_string())?);
        let questions = ["Where is an activation released after its last reader?", "How is the key cache kept between decode steps?", "Which kernel reads packed sequences?", "Where are the window's keys roped?"];
        let null = tok("\nQuestion: N/A")?;
        let mut fails = Vec::new();
        for len in [200_usize, 1500] {
            let groups: Vec<(Vec<i32>, Vec<Vec<i32>>)> = (0..4)
                .map(|g| -> Result<(Vec<i32>, Vec<Vec<i32>>), String> {
                    let at = 500 + g * 1700;
                    Ok((all[at..at + len].to_vec(), vec![tok(&format!("\nQuestion: {}", questions[g]))?, null.clone()]))
                })
                .collect::<Result<_, _>>()?;
            let pairs: Vec<(Vec<i32>, Vec<i32>)> = groups.iter().flat_map(|(p, qs)| qs.iter().map(|q| (p.clone(), q.clone()))).collect();
            let time = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
                f()?;
                let mut t = Vec::new();
                for _ in 0..3 {
                    let s = std::time::Instant::now();
                    f()?;
                    t.push(s.elapsed().as_secs_f64());
                }
                t.sort_by(f64::total_cmp);
                Ok(t[1])
            };
            let t_old = time(&|| crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &pairs).map(|_| ()).map_err(|e| e.to_string()))?;
            let t_new = time(&|| crate::kvstore::read_groups_tails(&m, Some(&aw), &stop, &heads, &groups).map(|_| ()).map_err(|e| e.to_string()))?;
            let old = crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &pairs).map_err(|e| e.to_string())?;
            let new: Vec<_> = crate::kvstore::read_groups_tails(&m, Some(&aw), &stop, &heads, &groups).map_err(|e| e.to_string())?.into_iter().flatten().collect();
            let (mut dl, mut dz, mut dm, mut same_top, mut n_top) = (0.0_f32, 0.0_f32, 0.0_f32, 0, 0);
            for (a, b) in old.iter().zip(&new) {
                for &(l, h) in &heads {
                    let (la, lb) = (a.logits(l, h).ok_or("old logits")?, b.logits(l, h).ok_or("new logits")?);
                    if la.logits.len() != lb.logits.len() {
                        return Err(format!("layer {l} head {h}: shapes differ ({} and {})", la.logits.len(), lb.logits.len()));
                    }
                    dl = la.logits.iter().zip(&lb.logits).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (x - y).abs()).fold(dl, f32::max);
                    dz = la.lse.iter().zip(&lb.lse).map(|(x, y)| (x - y).abs()).fold(dz, f32::max);
                    let (ma, mb) = (a.get(l).ok_or("old mass")?, b.get(l).ok_or("new mass")?);
                    let span = la.logits.len() / la.lse.len().max(1);
                    let (ra, rb) = (&ma[h * span..(h + 1) * span], &mb[h * span..(h + 1) * span]);
                    dm = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).fold(dm, f32::max);
                    let top = |v: &[f32]| v.iter().enumerate().skip(anchor.len()).max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |x| x.0);
                    n_top += 1;
                    if top(ra) == top(rb) {
                        same_top += 1;
                    }
                }
            }
            let ok = dl < 0.25 && dm < 0.05 && same_top * 10 >= n_top * 9;
            eprintln!(
                "4 groups of {len} tokens, posed and content-free: read_groups {t_old:.2} s, read_groups_tails {t_new:.2} s ({:.1} times); largest raw-score difference {dl:.4}, log-normaliser {dz:.4}, mass {dm:.4}; top piece position same on {same_top} of {n_top}{}",
                t_old / t_new.max(1e-9),
                if ok { "" } else { "  FAIL" }
            );
            if !ok {
                fails.push(len);
            }
        }
        if fails.is_empty() { Ok(()) } else { Err(format!("tails differ from read_groups at {fails:?} tokens")) }
    }

    /// Lane S package 5 (2026-11-18): lane T1's call shape (`tentacle_read::Integrator::read_packed`:
    /// the union of the tentacles' relevance heads, 16 here over layers 10 to 17; per tentacle the
    /// posed and the content-free question; parts = the anchor's window parts then 10 pieces of a
    /// tenth of the group) through the old form (`read_groups`, two groups a tentacle, logits to the
    /// processor, `part_lse` there) and the new (`read_group_parts`: the pieces once, both questions
    /// as tails, part scores reduced on the card). Groups of 200 and 1,500 tokens, 4 tentacles;
    /// served scope with `NANO_LANE_S_SERVED=on`. Pass: every part score within 0.05 of the old
    /// form's, and each tentacle's best piece (posed minus content-free summed over heads) the same.
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture group_parts_shape`.
    #[test]
    #[ignore = "needs the graphics card and the open weights"]
    fn group_parts_shape() -> Result<(), String> {
        use crate::kvstore::{PartGroup, Window};
        use crate::model::{Model, Population};
        use crate::placement::Device;
        use std::sync::Arc;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), "qwen3-reference").map_err(|e| e.to_string())?;
        asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
        let pop = Population::open(&home, "qwen3-0.6b", true, true).map_err(|e| e.to_string())?;
        let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
        let mut text = String::new();
        for f in ["crates/nmengn/src/model.rs", "crates/nmengn/src/kvstore.rs"] {
            text.push_str(&std::fs::read_to_string(home.join(f)).map_err(|e| e.to_string())?);
        }
        let all = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
        let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| e.to_string());
        let _served = (std::env::var("NANO_LANE_S_SERVED").as_deref() == Ok("on")).then(super::ServedScope::new);
        let last = 17;
        let heads: Vec<(usize, usize)> = (10..18).flat_map(|l| [(l, (l * 3) % 16), (l, (l * 7 + 5) % 16)]).collect();
        let stop = crate::kvstore::score_targets(&m, last).map_err(|e| e.to_string())?;
        let anchor = tok("You read source code. The pieces below come from one repository.\n\n")?;
        let ak = Arc::new(crate::kvstore::encode_through(&m, Device::Gpu, None, &anchor, last).map_err(|e| e.to_string())?);
        let aw = Arc::new(Window::build(&m, &[ak]).map_err(|e| e.to_string())?);
        let questions = ["Where is an activation released after its last reader?", "How is the key cache kept between decode steps?", "Which kernel reads packed sequences?", "Where are the window's keys roped?"];
        let null = tok("\nQuestion: N/A")?;
        let part_lse = |logits: &[f32], tokens: usize, span: usize, parts: &[(usize, usize)]| -> Vec<f32> {
            #[expect(clippy::cast_precision_loss, reason = "token counts are small")]
            let per = 1.0 / tokens.max(1) as f32;
            parts
                .iter()
                .map(|&(at, len)| {
                    let mut acc = 0.0_f32;
                    for t in 0..tokens {
                        let row = logits.get(t * span + at..t * span + at + len).unwrap_or(&[]);
                        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        if max.is_finite() {
                            acc += max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
                        }
                    }
                    acc * per
                })
                .collect()
        };
        let mut fails = Vec::new();
        for len in [200_usize, 1500] {
            let groups: Vec<PartGroup> = (0..4)
                .map(|g| -> Result<PartGroup, String> {
                    let at = 500 + g * 1700;
                    let mut parts = aw.parts.clone();
                    let piece = len / 10;
                    for k in 0..10 {
                        parts.push((aw.len + k * piece, if k == 9 { len - 9 * piece } else { piece }));
                    }
                    Ok(PartGroup { pieces: all[at..at + len].to_vec(), questions: vec![tok(&format!("\nQuestion: {}", questions[g]))?, null.clone()], heads: heads.clone(), parts })
                })
                .collect::<Result<_, _>>()?;
            // Old form: two (pieces, question) groups a tentacle, logits to the processor.
            let old_form = || -> Result<Vec<Vec<Vec<f32>>>, String> {
                let pairs: Vec<(Vec<i32>, Vec<i32>)> = groups.iter().flat_map(|g| g.questions.iter().map(|q| (g.pieces.clone(), q.clone()))).collect();
                let probes = crate::kvstore::read_groups(&m, Some(&aw), &stop, &heads, &pairs).map_err(|e| e.to_string())?;
                let mut out = Vec::new();
                for (gi, g) in groups.iter().enumerate() {
                    let span = aw.len + g.pieces.len();
                    let mut per_q = Vec::new();
                    for qi in 0..2 {
                        let p = &probes[2 * gi + qi];
                        let mut row = Vec::new();
                        for &(l, h) in &heads {
                            let lg = p.logits(l, h).ok_or("no logits")?;
                            row.extend(part_lse(&lg.logits, lg.lse.len(), span, &g.parts));
                        }
                        per_q.push(row);
                    }
                    out.push(per_q);
                }
                Ok(out)
            };
            let new_form = || crate::kvstore::read_group_parts(&m, Some(&aw), &stop, &groups).map_err(|e| e.to_string());
            old_form()?;
            let t = std::time::Instant::now();
            let old = old_form()?;
            let t_old = t.elapsed().as_secs_f64();
            new_form()?;
            let mut ts = Vec::new();
            let mut new = Vec::new();
            for _ in 0..3 {
                let t = std::time::Instant::now();
                new = new_form()?;
                ts.push(t.elapsed().as_secs_f64());
            }
            ts.sort_by(f64::total_cmp);
            let worst = old.iter().flatten().flatten().zip(new.iter().flatten().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            let best = |q: &[Vec<f32>], parts: usize, skip: usize| -> usize {
                (skip..parts)
                    .map(|p| (p, (0..heads.len()).map(|h| q[0][h * parts + p] - q[1][h * parts + p]).sum::<f32>()))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or(0, |x| x.0)
            };
            let same = groups.iter().enumerate().filter(|(gi, g)| best(&old[*gi], g.parts.len(), aw.parts.len()) == best(&new[*gi], g.parts.len(), aw.parts.len())).count();
            let ok = worst < 0.05 && same == groups.len();
            eprintln!(
                "4 tentacles of {len} tokens, {} heads, posed and content-free: old form (read_groups, processor part scores) {t_old:.2} s, read_group_parts {:.2} s ({:.1} times); largest part-score difference {worst:.4}; best piece same on {same} of 4{}",
                heads.len(),
                ts[1],
                t_old / ts[1].max(1e-9),
                if ok { "" } else { "  FAIL" }
            );
            if !ok {
                fails.push(len);
            }
        }
        if fails.is_empty() { Ok(()) } else { Err(format!("part scores differ at {fails:?} tokens")) }
    }

    /// Lane S prefill profile (2026-11-18, S1 and SPEED-1): the reader (Qwen3-0.6B, reference
    /// assembly) and the embedder (Qwen3-Embedding-0.6B) bound with 8-bit linears and the served
    /// switches (half math, packed attention), 512 tokens of this crate's source: the reader as one
    /// causal sequence outside and inside a served scope, the embedder as one sequence and as four
    /// packed sequences of 128. Median wall time of 5 passes after a warm-up, the card's working
    /// peak, then one pass with per-kernel timing (synchronised launches: shares only). Outputs are
    /// written to `scratch/lane-s/<label>.f32` and compared with a file of the same name in
    /// `NANO_LANE_S_AGAINST` when set (largest absolute difference, bit-identical count).
    /// `cargo test --release -p nmengn --lib -- --ignored --nocapture prefill_profile`.
    #[test]
    #[ignore = "needs the graphics card and the open weights; timing only"]
    fn prefill_profile() -> Result<(), String> {
        use crate::model::{Act, Model, Population, RunOpts};
        use crate::placement::Device;
        use std::sync::atomic::Ordering;
        let home = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let out_dir = home.join("scratch/lane-s");
        let _ = std::fs::create_dir_all(&out_dir);
        let against = std::env::var("NANO_LANE_S_AGAINST").ok().map(std::path::PathBuf::from);
        super::HALF_MATH.store(true, Ordering::Relaxed);
        super::SEGMENT_ATTENTION.store(true, Ordering::Relaxed);
        let text = std::fs::read_to_string(home.join("crates/nmengn/src/model.rs")).map_err(|e| e.to_string())?;
        for (pop_id, asm_name) in [("qwen3-0.6b", "qwen3-reference"), ("qwen3-embedding-0.6b", "qwen3-embedding")] {
            let mut asm = crate::Assembly::load(&home.join("designs/assemblies"), asm_name).map_err(|e| e.to_string())?;
            asm.nodes.iter_mut().filter(|n| n.op == "linear").for_each(|n| "q8".clone_into(&mut n.precision));
            let pop = Population::open(&home, pop_id, true, true).map_err(|e| e.to_string())?;
            let m = Model::bind(&pop, asm).map_err(|e| e.to_string())?;
            let o = m.pop.gpu.clone().ok_or("no card")?;
            let mut ids = m.pop.tokenizer.encode(&text).map_err(|e| e.to_string())?;
            ids.truncate(512);
            let reader = pop_id == "qwen3-0.6b";
            let target = if reader { (m.role_node("head").map_err(|e| e.to_string())?, "logits".to_owned()) } else { (m.role_node("pool").map_err(|e| e.to_string())?, "y".to_owned()) };
            let mut runs: Vec<(String, RunOpts, bool)> = vec![(format!("{pop_id} one sequence of 512"), RunOpts::default(), false)];
            if reader {
                runs.push((format!("{pop_id} one sequence of 512, served scope"), RunOpts::default(), true));
            } else {
                runs.push((format!("{pop_id} four packed sequences of 128"), RunOpts { segments: Some(std::sync::Arc::new(vec![128; 4])), ..RunOpts::default() }, false));
            }
            for (label, opts, served) in &runs {
                let pass = || -> Result<Vec<f32>, String> {
                    let _s = served.then(super::ServedScope::new);
                    let r = m.forward(Device::Gpu, &ids, std::slice::from_ref(&target), opts).map_err(|e| e.to_string())?;
                    match r.get(&target) {
                        Some(Act::Cpu(v)) => Ok(v.clone()),
                        _ => Err("no host output".to_owned()),
                    }
                };
                let first = pass()?;
                o.gpu.reset_peak();
                let mut times = Vec::new();
                for _ in 0..5 {
                    let t = std::time::Instant::now();
                    pass()?;
                    times.push(t.elapsed().as_secs_f64());
                }
                times.sort_by(f64::total_cmp);
                let med = times[2];
                let peak = o.gpu.pool_stats().peak_work;
                #[expect(clippy::cast_precision_loss, reason = "a rate for a report")]
                let tps = ids.len() as f64 / med;
                eprintln!("{label}: median {:.1} ms ({:.3} ms a token, {tps:.0} tokens a second); working peak {:.1} MiB", med * 1000.0, med * 1000.0 / 512.0, peak as f64 / 1_048_576.0);
                nmffi::prof::reset();
                nmffi::prof::set(true);
                nmffi::prof::set_kernels(true);
                pass()?;
                nmffi::prof::set_kernels(false);
                nmffi::prof::set(false);
                let mut k: Vec<(String, f64, u64)> = nmffi::prof::side().into_iter().filter(|r| r.0.starts_with("kernel ")).collect();
                k.sort_by(|a, b| b.1.total_cmp(&a.1));
                let tot: f64 = k.iter().map(|r| r.1).sum();
                let launches: u64 = k.iter().map(|r| r.2).sum();
                eprintln!("  kernels (synchronised): {:.1} ms over {launches} launches", tot * 1000.0);
                for (name, s, c) in &k {
                    eprintln!("    {:7.2} ms {c:5}  {name}", s * 1000.0);
                }
                let file = format!("{}.f32", label.replace([' ', ','], "_"));
                let bytes: Vec<u8> = first.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(out_dir.join(&file), &bytes);
                if let Some(dir) = &against
                    && let Ok(b) = std::fs::read(dir.join(&file))
                {
                    let old: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
                    let worst = old.iter().zip(&first).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                    let same = old.iter().zip(&first).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                    eprintln!("  against {}: {same} of {} values bit-identical, largest difference {worst}", dir.display(), first.len());
                }
            }
        }
        Ok(())
    }

    /// Time of each prefill attention kernel at the second look's sizes (one long causal
    /// sequence, and a short tail over a long cached prefix): `cargo test --release -p nmengn
    /// --lib -- --ignored --nocapture attention_kernels`.
    #[test]
    #[ignore = "needs the graphics card; timing only"]
    fn attention_kernels() -> Result<(), String> {
        let o = Ops { gpu: nmffi::Gpu::open(0)? };
        let mut seed = 0x2545_f491_u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    #[expect(clippy::cast_precision_loss, reason = "a test value")]
                    let x = ((seed % 2000) as f32 - 1000.0) / 500.0;
                    crate::half::r16(x)
                })
                .collect()
        };
        let (h, g, w) = (16_usize, 8_usize, 128_usize);
        for (tokens, cached) in [(1100_usize, 1100_usize), (40, 1180), (120, 1240)] {
            let q = o.upload_f32(&rnd(tokens * h * w), vec![tokens, h, w])?;
            let k = o.upload_f32(&rnd(cached * g * w), vec![cached, g, w])?;
            let v = o.upload_f32(&rnd(cached * g * w), vec![cached, g, w])?;
            let m = o.upload_mask(&vec![1_u8; h])?;
            let geo = Geometry { tokens: tokens as u64, cached: cached as u64, heads: h as u64, kv_heads: g as u64, width: w as u64, clones: 1 };
            let mut outs: Vec<(String, Vec<f32>)> = Vec::new();
            for (name, per) in [
                ("_ZN4emma15prefill2_kernelIfEEv12emma_op_argsNS_8GeometryE", 16_u64),
                ("_ZN4emma14segment_kernelE12emma_op_argsNS_8GeometryE", 16),
                ("_ZN4emma12flash_kernelE12emma_op_argsNS_8GeometryE", 64),
            ] {
                let y = o.alloc(q.shape.clone(), F16)?;
                let mut a = Ops::args(&[&q, &k, &v, &m], &[], &[&y]);
                a.attrs.causal = 1;
                let launch = |a: &nmffi::OpArgs| {
                    o.gpu.launch(
                        name,
                        (u32::try_from((tokens as u64).div_ceil(per)).unwrap_or(0), u32::try_from(h).unwrap_or(0), 1),
                        (256, 1, 1),
                        0,
                        &[Param::Op(Box::new(*a)), Param::Geo(geo)],
                    )
                };
                launch(&a)?;
                o.gpu.sync()?;
                let t = std::time::Instant::now();
                for _ in 0..28 {
                    launch(&a)?;
                }
                o.gpu.sync()?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                eprintln!("{tokens} rows over {cached}: {name}: {ms:.1} ms for 28 layers");
                outs.push((name.to_owned(), o.download(&y)?));
            }
            if let Some((_, base)) = outs.first() {
                for (name, o) in outs.iter().skip(1) {
                    let diff = base.iter().zip(o).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
                    eprintln!("  {name}: largest difference from prefill2 {diff}");
                }
            }
        }
        Ok(())
    }

    /// Time of the linear kernels at the joint read's sizes (1,100 and 120 tokens, the base
    /// model's projections): int8 weights (`wide_kernel` int8), binary16 weights with binary32
    /// products (`wide_kernel`) and with half products (`wide_h2_kernel`).
    #[test]
    #[ignore = "needs the graphics card; timing only"]
    fn linear_kernels() -> Result<(), String> {
        let o = Ops { gpu: nmffi::Gpu::open(0)? };
        let shapes = [(2048_usize, 1024_usize), (512, 1024), (512, 1024), (1024, 2048), (3072, 1024), (3072, 1024), (1024, 3072)];
        for tokens in [1100_usize, 120, 40] {
            let x = o.upload_f32(&vec![0.25; tokens * 3072], vec![tokens * 3072])?;
            let mut t8 = 0.0;
            let mut t16 = 0.0;
            let mut th = 0.0;
            let mut t8h = 0.0;
            for (n, k) in shapes {
                let xk = x.reshape(vec![tokens, k]);
                let xk = DevT { buf: std::sync::Arc::clone(&xk.buf), shape: vec![tokens, k], dtype: F16 };
                let w8 = o.alloc(vec![n, k], super::I8)?;
                let s8 = o.upload_f32(&vec![0.01; n], vec![n])?;
                let w16 = o.upload_f32(&vec![0.01; n * k], vec![n, k])?;
                let time = |f: &dyn Fn() -> Result<DevT, String>| -> Result<f64, String> {
                    f()?;
                    o.gpu.sync()?;
                    let t = std::time::Instant::now();
                    for _ in 0..28 {
                        f()?;
                    }
                    o.gpu.sync()?;
                    Ok(t.elapsed().as_secs_f64() * 1000.0)
                };
                t8 += time(&|| o.linear_q8(&xk, &w8, &s8))?;
                super::HALF_MATH.store(false, std::sync::atomic::Ordering::Relaxed);
                t16 += time(&|| o.linear(&xk, &w16))?;
                super::HALF_MATH.store(true, std::sync::atomic::Ordering::Relaxed);
                th += time(&|| o.linear(&xk, &w16))?;
                {
                    let _served = super::ServedScope::new();
                    t8h += time(&|| o.linear_q8(&xk, &w8, &s8))?;
                }
                super::HALF_MATH.store(false, std::sync::atomic::Ordering::Relaxed);
            }
            eprintln!(
                "{tokens} tokens, 28 layers of projections: int8 {t8:.1} ms, int8 half products {t8h:.1} ms, binary16 {t16:.1} ms, binary16 half products {th:.1} ms"
            );
        }
        Ok(())
    }
}
