// Gavriel Technologies. Lane S, card speed (2026-11-18): faster tilings of existing linears with
// the same arithmetic, so outputs are bit-identical to the kernels they replace (checked by
// nmengn.gpu.dp4a_tile). Separate image (fast.fatbin) so the existing images stay as they are.
//
// dp4a_tile_kernel: the int8 dot-product linear of wide_dp4a_kernel (linear.cu) over the same
// 128-token by 64-row block, rebuilt for throughput: 64-deep stages (two activation blocks of 32)
// loaded with 16-byte reads into a double-buffered shared tile (the next stage is read into
// registers while this one is summed), 16-byte shared reads, rows spaced 16 apart per thread (no
// bank conflicts, adjacent outputs from adjacent threads). Per output and per block of 32 the int32
// sum of products is exact in any order; it is added to the binary32 total times the activation
// block's scale in block order, and the row scale is applied once at the end, exactly as in
// wide_dp4a_kernel. Needs a width that is a multiple of 64 and 16-byte aligned operands (the host
// checks; otherwise wide_dp4a_kernel).
//
// h2q8_tile_kernel: wide_h2q8_kernel (q8half.cu), the half products over int8 weights of the
// served path, with its shared tiles read 16 bytes (eight activations) and 8 bytes (four weights)
// at a time in place of one value per read: the old kernel issued ten shared reads per sixteen
// half2 multiply-adds and was bound by them. Same tiles, same loads, same products in the same
// order, so outputs are bit-identical (nmengn.gpu.h2q8_tile).
//
// h2q8_fetch_kernel<TN> (lane S package 2, 2026-11-18): h2q8_tile_kernel with the global fetch
// rebuilt: each thread reads its activations as one 16-byte load (eight values of one token) and
// its weights as 4-byte loads (four int8 of one row), where the old fetch read one value a load
// (two-byte and one-byte loads, two tokens or rows per warp load). The values written to the
// shared tiles, the products and their order are those of wide_h2q8_kernel, so outputs are
// bit-identical (nmengn.gpu.h2q8_tile). TN = 4 is the 128 by 64 tile; TN = 8 a 128 by 128 tile.
// Needs a width that is a multiple of 16 and 16-byte aligned activations (the host checks).
//
// dp4a_tt_kernel<TM, TN, MINB> (lane S package 2): dp4a_tile_kernel with a 16*TM token by 16*TN
// row tile and MINB blocks an SM asked of the register allocator (fewer registers); the same
// arithmetic per output, so bit-identical to wide_dp4a_kernel (nmengn.gpu.dp4a_tile).
//
// wide_q8f_kernel (lane S package 3, 2026-11-18): wide_kernel<8, 4, true> (linear.cu: int8 weights,
// binary16 activations and row scales, binary32 products; the plain path of the reader at 8 bits)
// with the global fetch rebuilt as in h2q8_fetch_kernel: one 16-byte activation load (eight values
// of one token) and one 4-byte weight load (four int8 of one row) per thread per 16-deep slice. The
// float values put in the shared tiles, the products and their order are those of wide_kernel, so
// outputs are bit-identical (nmengn.gpu.wide_q8f). Needs a width that is a multiple of 16 and
// 16-byte aligned activations (the host checks).
//
// q8_blocks2_kernel: q8_blocks_kernel (linear.cu) with each thread's block of 32 activations read
// as four 16-byte loads and its 32 int8 values written as two (the old kernel made 32 two-byte
// reads and 32 one-byte writes per thread, adjacent threads 64 bytes apart). Same arithmetic.
#include "common.cuh"
namespace emma {
namespace {
constexpr int kThreads = 256;
constexpr int kTok = 128;     // tokens per block (16 thread rows x 8)
constexpr int kRow = 64;      // weight rows per block (16 thread columns x 4, spaced 16 apart)
constexpr int kStage = 64;    // int8 values per row per stage (two blocks of 32)
constexpr int kWords = 16;    // 32-bit words per row per stage
constexpr int kPad = kWords + 4;  // 80-byte row pitch: 16-byte reads of 8 rows hit 32 banks once
}  // namespace
__global__ __launch_bounds__(kThreads) void dp4a_tile_kernel(emma_op_args a) {
  __shared__ __align__(16) int xs[2][kTok][kPad];
  __shared__ __align__(16) int ws[2][kRow][kPad];
  __shared__ float xsc[2][2][kTok];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t blocks = width / 32;
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTok;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRow;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  const auto* xq = reinterpret_cast<const int8_t*>(a.inputs[0].data);
  const auto* xs_scale = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* wsh = reinterpret_cast<const __half*>(a.weights[1].data);
  // Loader roles: activations 128 rows x 4 chunks of 16 bytes (two per thread), weights 64 rows x
  // 4 chunks (one per thread), scales 2 blocks x 128 tokens (one per thread).
  int4 xr[2];
  int4 wr;
  float sr;
  auto fetch = [&](uint64_t k0) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int idx = tid + n * kThreads;
      const int r = idx / 4;
      const int c = idx % 4;
      const uint64_t t = token0 + r;
      xr[n] = t < tokens ? *reinterpret_cast<const int4*>(xq + t * width + k0 + c * 16) : make_int4(0, 0, 0, 0);
    }
    {
      const int r = tid / 4;
      const int c = tid % 4;
      const uint64_t w = row0 + r;
      wr = w < rows ? *reinterpret_cast<const int4*>(wq + w * width + k0 + c * 16) : make_int4(0, 0, 0, 0);
    }
    {
      const int bb = tid / kTok;
      const int r = tid % kTok;
      const uint64_t t = token0 + r;
      sr = t < tokens ? __half2float(xs_scale[t * blocks + k0 / 32 + bb]) : 0.0F;
    }
  };
  auto put = [&](int b) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int idx = tid + n * kThreads;
      *reinterpret_cast<int4*>(&xs[b][idx / 4][(idx % 4) * 4]) = xr[n];
    }
    *reinterpret_cast<int4*>(&ws[b][tid / 4][(tid % 4) * 4]) = wr;
    xsc[b][tid / kTok][tid % kTok] = sr;
  };
  float acc[8][4] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kStage) {
    const bool more = k0 + kStage < width;
    if (more) {
      fetch(k0 + kStage);
    }
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      int sum[8][4] = {};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int at = bb * 8 + q * 4;
        int4 wv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          wv[j] = *reinterpret_cast<const int4*>(&ws[b][tx + 16 * j][at]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int4 xv = *reinterpret_cast<const int4*>(&xs[b][ty * 8 + i][at]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            int s = sum[i][j];
            s = __dp4a(xv.x, wv[j].x, s);
            s = __dp4a(xv.y, wv[j].y, s);
            s = __dp4a(xv.z, wv[j].z, s);
            s = __dp4a(xv.w, wv[j].w, s);
            sum[i][j] = s;
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float sc = xsc[b][bb][ty * 8 + i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] += static_cast<float>(sum[i][j]) * sc;
        }
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint64_t t = token0 + ty * 8 + i;
      const uint64_t w = row0 + tx + 16 * j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j] * __half2float(wsh[w]));
      }
    }
  }
}
namespace {
constexpr int kDepth = 16;
}  // namespace
__global__ __launch_bounds__(kThreads) void h2q8_tile_kernel(emma_op_args a) {
  constexpr int TM = 8;
  constexpr int TN = 4;
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kXs = kDepth * kTokens / kThreads;
  constexpr int kWs = kDepth * kRows / kThreads;
  __shared__ __align__(16) __half xs[2][kDepth][kTokens + 8];
  __shared__ __align__(16) __half ws[2][kDepth][kRows + 8];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tx = static_cast<int>(threadIdx.x % 16);
  const int ty = static_cast<int>(threadIdx.x / 16);
  const int kk_load = static_cast<int>(threadIdx.x) % kDepth;
  const int r_load = static_cast<int>(threadIdx.x) / kDepth;
  constexpr int kRowsPerPass = kThreads / kDepth;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* sh = reinterpret_cast<const __half*>(a.weights[1].data);
  const __half zero = __float2half_rn(0.0F);
  float wscale[kWs];
#pragma unroll
  for (int n = 0; n < kWs; ++n) {
    const uint64_t w = row0 + r_load + n * kRowsPerPass;
    wscale[n] = w < rows ? __half2float(sh[w]) : 0.0F;
  }
  __half xr[kXs];
  __half wr[kWs];
  auto fetch = [&](uint64_t k0) {
    const uint64_t k = k0 + kk_load;
#pragma unroll
    for (int n = 0; n < kXs; ++n) {
      const uint64_t t = token0 + r_load + n * kRowsPerPass;
      xr[n] = (t < tokens && k < width) ? xh[t * width + k] : zero;
    }
#pragma unroll
    for (int n = 0; n < kWs; ++n) {
      const uint64_t w = row0 + r_load + n * kRowsPerPass;
      wr[n] = (w < rows && k < width) ? __float2half_rn(static_cast<float>(wq[w * width + k]) * wscale[n]) : zero;
    }
  };
  auto put = [&](int b) {
#pragma unroll
    for (int n = 0; n < kXs; ++n) {
      xs[b][kk_load][r_load + n * kRowsPerPass] = xr[n];
    }
#pragma unroll
    for (int n = 0; n < kWs; ++n) {
      ws[b][kk_load][r_load + n * kRowsPerPass] = wr[n];
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    const bool more = k0 + kDepth < width;
    if (more) {
      fetch(k0 + kDepth);
    }
    __half2 part[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        part[i][j] = __half2half2(zero);
      }
    }
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      const uint4 x8 = *reinterpret_cast<const uint4*>(&xs[b][kk][ty * TM]);
      const uint2 w4 = *reinterpret_cast<const uint2*>(&ws[b][kk][tx * TN]);
      const __half* xv8 = reinterpret_cast<const __half*>(&x8);
      const __half2* wv = reinterpret_cast<const __half2*>(&w4);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const __half2 xv = __half2half2(xv8[i]);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          part[i][j] = __hfma2(xv, wv[j], part[i][j]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        acc[i][2 * j] += __low2float(part[i][j]);
        acc[i][2 * j + 1] += __high2float(part[i][j]);
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j]);
      }
    }
  }
}
__global__ __launch_bounds__(256) void q8_blocks2_kernel(emma_op_args a) {
  const uint64_t width = static_cast<uint64_t>(a.inputs[0].shape[a.inputs[0].rank - 1]);
  const uint64_t blocks = width / 32;
  const uint64_t total = elements(a.inputs[0]) / 32;
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) {
    return;
  }
  const uint64_t t = i / blocks;
  const uint64_t blk = i % blocks;
  const auto* x = reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(a.inputs[0].data) + t * width + blk * 32);
  auto* q = reinterpret_cast<uint4*>(reinterpret_cast<int8_t*>(a.outputs[0].data) + t * width + blk * 32);
  auto* s = reinterpret_cast<__half*>(a.outputs[1].data);
  float v[32];
  float m = 0.0F;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 raw = x[c];
    const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      v[c * 8 + e] = __half2float(h[e]);
    }
  }
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    m = fmaxf(m, fabsf(v[k]));
  }
  const float scale = m / 127.0F;
  const float inv = m > 0.0F ? 127.0F / m : 0.0F;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    uint4 out;
    int8_t* o = reinterpret_cast<int8_t*>(&out);
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      o[e] = static_cast<int8_t>(__float2int_rn(v[c * 16 + e] * inv));
    }
    q[c] = out;
  }
  s[i] = __float2half_rn(scale);
}
template <int TN>
__global__ __launch_bounds__(kThreads) void h2q8_fetch_kernel(emma_op_args a) {
  constexpr int TM = 8;
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kWl = kRows * 4 / kThreads;  // 4-byte weight loads per thread per slice
  __shared__ __align__(16) __half xs[2][kDepth][kTokens + 8];
  __shared__ __align__(16) __half ws[2][kDepth][kRows + 8];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* sh = reinterpret_cast<const __half*>(a.weights[1].data);
  const __half zero = __float2half_rn(0.0F);
  // Activation role: token tid / 2, eight k from (tid % 2) * 8. Weight role n: row
  // (tid + n * 256) / 4, four k from ((tid + n * 256) % 4) * 4.
  const int xr_t = tid / 2;
  const int xr_k = (tid % 2) * 8;
  float wscale[kWl];
#pragma unroll
  for (int n = 0; n < kWl; ++n) {
    const uint64_t w = row0 + (tid + n * kThreads) / 4;
    wscale[n] = w < rows ? __half2float(sh[w]) : 0.0F;
  }
  uint4 xr;
  int wr[kWl];
  auto fetch = [&](uint64_t k0) {
    const uint64_t t = token0 + xr_t;
    xr = t < tokens ? *reinterpret_cast<const uint4*>(xh + t * width + k0 + xr_k) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < kWl; ++n) {
      const int idx = tid + n * kThreads;
      const uint64_t w = row0 + idx / 4;
      wr[n] = w < rows ? *reinterpret_cast<const int*>(wq + w * width + k0 + (idx % 4) * 4) : 0;
    }
  };
  auto put = [&](int b) {
    const __half* x8 = reinterpret_cast<const __half*>(&xr);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      xs[b][xr_k + e][xr_t] = x8[e];
    }
#pragma unroll
    for (int n = 0; n < kWl; ++n) {
      const int idx = tid + n * kThreads;
      const int8_t* w4 = reinterpret_cast<const int8_t*>(&wr[n]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        ws[b][(idx % 4) * 4 + e][idx / 4] = row0 + idx / 4 < rows ? __float2half_rn(static_cast<float>(w4[e]) * wscale[n]) : zero;
      }
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    const bool more = k0 + kDepth < width;
    if (more) {
      fetch(k0 + kDepth);
    }
    __half2 part[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        part[i][j] = __half2half2(zero);
      }
    }
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      const uint4 x8 = *reinterpret_cast<const uint4*>(&xs[b][kk][ty * TM]);
      __half2 wv[TN / 2];
      if constexpr (TN == 4) {
        const uint2 w4 = *reinterpret_cast<const uint2*>(&ws[b][kk][tx * TN]);
        const __half2* p = reinterpret_cast<const __half2*>(&w4);
        wv[0] = p[0];
        wv[1] = p[1];
      } else {
        const uint4 w8 = *reinterpret_cast<const uint4*>(&ws[b][kk][tx * TN]);
        const __half2* p = reinterpret_cast<const __half2*>(&w8);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          wv[j] = p[j];
        }
      }
      const __half* xv8 = reinterpret_cast<const __half*>(&x8);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const __half2 xv = __half2half2(xv8[i]);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          part[i][j] = __hfma2(xv, wv[j], part[i][j]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        acc[i][2 * j] += __low2float(part[i][j]);
        acc[i][2 * j + 1] += __high2float(part[i][j]);
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j]);
      }
    }
  }
}
template __global__ void h2q8_fetch_kernel<4>(emma_op_args a);
template __global__ void h2q8_fetch_kernel<8>(emma_op_args a);
template <int TM, int TN, int MINB>
__global__ __launch_bounds__(kThreads, MINB) void dp4a_tt_kernel(emma_op_args a) {
  constexpr int kT = 16 * TM;
  constexpr int kR = 16 * TN;
  constexpr int kXl = kT * 4 / kThreads;  // 16-byte activation loads per thread per stage
  constexpr int kWlo = kR * 4 / kThreads;  // 16-byte weight loads per thread per stage
  static_assert(kXl >= 1 && kWlo >= 1, "tile too small");
  __shared__ __align__(16) int xs[2][kT][kPad];
  __shared__ __align__(16) int ws[2][kR][kPad];
  __shared__ float xsc[2][2][kT];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t blocks = width / 32;
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kT;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kR;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  const auto* xq = reinterpret_cast<const int8_t*>(a.inputs[0].data);
  const auto* xs_scale = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* wsh = reinterpret_cast<const __half*>(a.weights[1].data);
  int4 xr[kXl];
  int4 wr[kWlo];
  float sr = 0.0F;
  auto fetch = [&](uint64_t k0) {
#pragma unroll
    for (int n = 0; n < kXl; ++n) {
      const int idx = tid + n * kThreads;
      const uint64_t t = token0 + idx / 4;
      xr[n] = t < tokens ? *reinterpret_cast<const int4*>(xq + t * width + k0 + (idx % 4) * 16) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int n = 0; n < kWlo; ++n) {
      const int idx = tid + n * kThreads;
      const uint64_t w = row0 + idx / 4;
      wr[n] = w < rows ? *reinterpret_cast<const int4*>(wq + w * width + k0 + (idx % 4) * 16) : make_int4(0, 0, 0, 0);
    }
    if (tid < 2 * kT) {
      const uint64_t t = token0 + tid % kT;
      sr = t < tokens ? __half2float(xs_scale[t * blocks + k0 / 32 + tid / kT]) : 0.0F;
    }
  };
  auto put = [&](int b) {
#pragma unroll
    for (int n = 0; n < kXl; ++n) {
      const int idx = tid + n * kThreads;
      *reinterpret_cast<int4*>(&xs[b][idx / 4][(idx % 4) * 4]) = xr[n];
    }
#pragma unroll
    for (int n = 0; n < kWlo; ++n) {
      const int idx = tid + n * kThreads;
      *reinterpret_cast<int4*>(&ws[b][idx / 4][(idx % 4) * 4]) = wr[n];
    }
    if (tid < 2 * kT) {
      xsc[b][tid / kT][tid % kT] = sr;
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kStage) {
    const bool more = k0 + kStage < width;
    if (more) {
      fetch(k0 + kStage);
    }
#pragma unroll
    for (int bb = 0; bb < 2; ++bb) {
      int sum[TM][TN] = {};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int at = bb * 8 + q * 4;
        int4 wv[TN];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          wv[j] = *reinterpret_cast<const int4*>(&ws[b][tx + 16 * j][at]);
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const int4 xv = *reinterpret_cast<const int4*>(&xs[b][ty * TM + i][at]);
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            int s = sum[i][j];
            s = __dp4a(xv.x, wv[j].x, s);
            s = __dp4a(xv.y, wv[j].y, s);
            s = __dp4a(xv.z, wv[j].z, s);
            s = __dp4a(xv.w, wv[j].w, s);
            sum[i][j] = s;
          }
        }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const float sc = xsc[b][bb][ty * TM + i];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += static_cast<float>(sum[i][j]) * sc;
        }
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx + 16 * j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j] * __half2float(wsh[w]));
      }
    }
  }
}
template __global__ void dp4a_tt_kernel<8, 4, 2>(emma_op_args a);
template __global__ void dp4a_tt_kernel<8, 8, 1>(emma_op_args a);
template __global__ void dp4a_tt_kernel<4, 8, 2>(emma_op_args a);
__global__ __launch_bounds__(kThreads) void wide_q8f_kernel(emma_op_args a) {
  constexpr int TM = 8;
  constexpr int TN = 4;
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  __shared__ __align__(16) float xs[2][kDepth][kTokens + 4];
  __shared__ __align__(16) float ws[2][kDepth][kRows + 4];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* sh = reinterpret_cast<const __half*>(a.weights[1].data);
  const int xr_t = tid / 2;
  const int xr_k = (tid % 2) * 8;
  const int wr_r = tid / 4;
  const int wr_k = (tid % 4) * 4;
  const bool w_in = row0 + wr_r < rows;
  const float wscale = w_in ? __half2float(sh[row0 + wr_r]) : 0.0F;
  uint4 xr;
  int wr;
  auto fetch = [&](uint64_t k0) {
    const uint64_t t = token0 + xr_t;
    xr = t < tokens ? *reinterpret_cast<const uint4*>(xh + t * width + k0 + xr_k) : make_uint4(0, 0, 0, 0);
    wr = w_in ? *reinterpret_cast<const int*>(wq + (row0 + wr_r) * width + k0 + wr_k) : 0;
  };
  auto put = [&](int b) {
    const __half* x8 = reinterpret_cast<const __half*>(&xr);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      xs[b][xr_k + e][xr_t] = __half2float(x8[e]);
    }
    const int8_t* w4 = reinterpret_cast<const int8_t*>(&wr);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      ws[b][wr_k + e][wr_r] = w_in ? __half2float(__float2half_rn(static_cast<float>(w4[e]) * wscale)) : 0.0F;
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    const bool more = k0 + kDepth < width;
    if (more) {
      fetch(k0 + kDepth);
    }
    float part[TM][TN] = {};
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      float xv[TM];
      float wv[TN];
#pragma unroll
      for (int i = 0; i < TM; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&xs[b][kk][ty * TM + i]);
        xv[i] = v.x;
        xv[i + 1] = v.y;
        xv[i + 2] = v.z;
        xv[i + 3] = v.w;
      }
#pragma unroll
      for (int j = 0; j < TN; j += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&ws[b][kk][tx * TN + j]);
        wv[j] = v.x;
        wv[j + 1] = v.y;
        wv[j + 2] = v.z;
        wv[j + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          part[i][j] += xv[i] * wv[j];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] += part[i][j];
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j]);
      }
    }
  }
}
// h2_fetch_kernel (lane S, 2026-11-18, T-079): wide_h2_kernel (binary16 weights, half products; the
// judge bound at 16 bits) with the 16-byte activation and 8-byte weight fetch of h2q8_fetch_kernel;
// the same values in the shared tiles and the same products, so bit-identical (nmengn.gpu.h2_fetch).
__global__ __launch_bounds__(kThreads) void h2_fetch_kernel(emma_op_args a) {
  constexpr int TN = 4;
  constexpr int TM = 8;
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kWl = kRows * 4 / kThreads;  // 4-byte weight loads per thread per slice
  __shared__ __align__(16) __half xs[2][kDepth][kTokens + 8];
  __shared__ __align__(16) __half ws[2][kDepth][kRows + 8];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 16;
  const int ty = tid / 16;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wh = reinterpret_cast<const __half*>(a.weights[0].data);
  const __half zero = __float2half_rn(0.0F);
  // Activation role: token tid / 2, eight k from (tid % 2) * 8. Weight role n: row
  // (tid + n * 256) / 4, four k from ((tid + n * 256) % 4) * 4.
  const int xr_t = tid / 2;
  const int xr_k = (tid % 2) * 8;
  uint4 xr;
  uint2 wr[kWl];
  auto fetch = [&](uint64_t k0) {
    const uint64_t t = token0 + xr_t;
    xr = t < tokens ? *reinterpret_cast<const uint4*>(xh + t * width + k0 + xr_k) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int n = 0; n < kWl; ++n) {
      const int idx = tid + n * kThreads;
      const uint64_t w = row0 + idx / 4;
      wr[n] = w < rows ? *reinterpret_cast<const uint2*>(wh + w * width + k0 + (idx % 4) * 4) : make_uint2(0, 0);
    }
  };
  auto put = [&](int b) {
    const __half* x8 = reinterpret_cast<const __half*>(&xr);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      xs[b][xr_k + e][xr_t] = x8[e];
    }
#pragma unroll
    for (int n = 0; n < kWl; ++n) {
      const int idx = tid + n * kThreads;
      const __half* w4 = reinterpret_cast<const __half*>(&wr[n]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        ws[b][(idx % 4) * 4 + e][idx / 4] = row0 + idx / 4 < rows ? w4[e] : zero;
      }
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    const bool more = k0 + kDepth < width;
    if (more) {
      fetch(k0 + kDepth);
    }
    __half2 part[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        part[i][j] = __half2half2(zero);
      }
    }
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      const uint4 x8 = *reinterpret_cast<const uint4*>(&xs[b][kk][ty * TM]);
      __half2 wv[TN / 2];
      if constexpr (TN == 4) {
        const uint2 w4 = *reinterpret_cast<const uint2*>(&ws[b][kk][tx * TN]);
        const __half2* p = reinterpret_cast<const __half2*>(&w4);
        wv[0] = p[0];
        wv[1] = p[1];
      } else {
        const uint4 w8 = *reinterpret_cast<const uint4*>(&ws[b][kk][tx * TN]);
        const __half2* p = reinterpret_cast<const __half2*>(&w8);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          wv[j] = p[j];
        }
      }
      const __half* xv8 = reinterpret_cast<const __half*>(&x8);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const __half2 xv = __half2half2(xv8[i]);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          part[i][j] = __hfma2(xv, wv[j], part[i][j]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        acc[i][2 * j] += __low2float(part[i][j]);
        acc[i][2 * j + 1] += __high2float(part[i][j]);
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j]);
      }
    }
  }
}
}  // namespace emma
