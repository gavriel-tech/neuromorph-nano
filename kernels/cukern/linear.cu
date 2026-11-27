// Gavriel Technologies. Weight-only int8, int4-group and dense matmul; ordered reductions.
// Float16 activations accumulate in float32; float32 uses float64 like the CPU reference.
// Reference: backnd Cpu, tblock-forward/full-forward per-file tolerance.
#include "common.cuh"
#include <type_traits>
namespace emma {
// Dequantized weight (row, k), rounded to float16 like the CPU reference.
__device__ inline float weight_at(int op, const emma_op_args& a, uint64_t row, uint64_t k,
                                  uint64_t width) {
  const uint64_t idx = row * width + k;
  if (op == 6) {
    const float scale = load(a.weights[1], row);
    float w =
        rounded(static_cast<float>(reinterpret_cast<const int8_t*>(a.weights[0].data)[idx]) * scale,
                EMMA_F16);
#ifdef EMMA_MUTANT_SCALE_TWICE
    w = rounded(w * scale, EMMA_F16);
#endif
    return w;
  }
  if (op == 10) {
    const uint8_t byte = reinterpret_cast<const uint8_t*>(a.weights[0].data)[idx / 2];
#ifdef EMMA_MUTANT_INT4_HIGH_FIRST
    const int nibble = (idx % 2 == 0) ? ((byte >> 4) & 15) : (byte & 15);
#else
    const int nibble = (byte >> ((idx % 2) * 4)) & 15;
#endif
    const int value = nibble >= 8 ? nibble - 16 : nibble;
    const uint64_t group = a.attrs.group_size;
    const float scale = load(a.weights[1], row * (width / group) + k / group);
    return rounded(static_cast<float>(value) * scale, EMMA_F16);
  }
  if (op == 12) {
    // MLX affine int4 (Flint): unsigned code, low nibble first, value = code * scale + bias
    // per group, in float32 like the CPU reference (no float16 rounding).
    const uint8_t byte = reinterpret_cast<const uint8_t*>(a.weights[0].data)[idx / 2];
    const int nibble = (byte >> ((idx % 2) * 4)) & 15;
    const uint64_t group = a.attrs.group_size;
    const uint64_t g = row * (width / group) + k / group;
    return static_cast<float>(nibble) * load(a.weights[1], g) + load(a.inputs[1], g);
  }
  return load(a.weights[0], idx);
}
// Tokens that share one read of each weight value.
constexpr int kChunk = 8;
constexpr unsigned kWarps = 4;
// Decode keeps the same lane order and shuffle tree, without eight token accumulators
// or runtime op branches. Batched matmuls continue through the existing kernels below.
template <int Op, typename Sum>
__global__ __launch_bounds__(128) void vector_kernel(emma_op_args a) {
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  if (row >= rows) {
    return;
  }
  Sum sum = 0;
  for (uint64_t k = lane; k < width; k += 32) {
    sum +=
        static_cast<Sum>(load(a.inputs[0], k)) * static_cast<Sum>(weight_at(Op, a, row, k, width));
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  if (lane == 0) {
    store(a.outputs[0], row, static_cast<float>(sum));
  }
}
// Int8 decode with float16 activations and scales (the common case): the same lane order,
// element values and shuffle tree as `vector_kernel<6, float>`, with the row's scale read once
// and the dtype branches resolved at compile time (2026-11-05).
__global__ __launch_bounds__(128) void vector_int8_half_kernel(emma_op_args a) {
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  if (row >= rows) {
    return;
  }
  const auto* x = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* w = reinterpret_cast<const int8_t*>(a.weights[0].data) + row * width;
  const float scale = __half2float(reinterpret_cast<const __half*>(a.weights[1].data)[row]);
  float sum = 0;
  uint64_t k = lane;
  for (; k < width; k += 32) {
    float weight = __half2float(__float2half_rn(static_cast<float>(w[k]) * scale));
#ifdef EMMA_MUTANT_SCALE_TWICE
    weight = __half2float(__float2half_rn(weight * scale));
#endif
    sum += __half2float(x[k]) * weight;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  if (lane == 0) {
    store(a.outputs[0], row, sum);
  }
}
// Float16 weights and activations at decode (the output head): the same lane order and
// shuffle tree as `vector_kernel<7, float>`, dtype branches resolved at compile time.
__global__ __launch_bounds__(128) void vector_half_kernel(emma_op_args a) {
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  if (row >= rows) {
    return;
  }
  const auto* x = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* w = reinterpret_cast<const __half*>(a.weights[0].data) + row * width;
  float sum = 0;
  for (uint64_t k = lane; k < width; k += 32) {
    sum += __half2float(x[k]) * __half2float(w[k]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffU, sum, offset);
  }
  if (lane == 0) {
    store(a.outputs[0], row, sum);
  }
}
// One warp per output row: lanes stride over k so reads coalesce, each lane keeps partial
// sums for up to kChunk tokens, and a fixed shuffle tree adds them (deterministic order).
template <typename Sum>
__global__ __launch_bounds__(128) void linear_kernel(int op, emma_op_args a) {
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  const unsigned lane = threadIdx.x % 32;
  if (row >= rows) {
    return;
  }
  const uint64_t tokens = elements(y) / rows;
  for (uint64_t first = 0; first < tokens; first += kChunk) {
    const uint64_t count = tokens - first < kChunk ? tokens - first : kChunk;
    Sum sum[kChunk] = {};
    for (uint64_t k = lane; k < width; k += 32) {
      const Sum w = static_cast<Sum>(weight_at(op, a, row, k, width));
#pragma unroll
      for (int t = 0; t < kChunk; ++t) {
        if (static_cast<uint64_t>(t) < count) {
          sum[t] += static_cast<Sum>(load(a.inputs[0], (first + t) * width + k)) * w;
        }
      }
    }
#pragma unroll
    for (int t = 0; t < kChunk; ++t) {
      Sum s = sum[t];
      for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffffU, s, offset);
      }
      if (lane == 0 && static_cast<uint64_t>(t) < count) {
        store(y, (first + t) * rows + row, static_cast<float>(s));
      }
    }
  }
}
// Batches: a block computes a 64-token x 64-row tile; 256 threads hold 4 x 4 outputs each.
// Each 16-deep k slice is summed on its own, then added to the running total.
constexpr int kTile = 64;
constexpr int kDepth = 16;
constexpr int kTiledThreads = 256;
constexpr uint64_t kTiledMinTokens = 16;
template <typename Sum>
__global__ __launch_bounds__(kTiledThreads) void tiled_kernel(int op, emma_op_args a) {
  __shared__ float xs[kDepth][kTile + 1];
  __shared__ float ws[kDepth][kTile + 1];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTile;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kTile;
  const int tx = static_cast<int>(threadIdx.x % 16);
  const int ty = static_cast<int>(threadIdx.x / 16);
  Sum acc[4][4] = {};
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    for (int i = static_cast<int>(threadIdx.x); i < kDepth * kTile; i += kTiledThreads) {
      const int kk = i % kDepth;
      const int r = i / kDepth;
      const uint64_t k = k0 + kk;
      const uint64_t t = token0 + r;
      const uint64_t w = row0 + r;
      xs[kk][r] = (t < tokens && k < width) ? load(a.inputs[0], t * width + k) : 0.0F;
      ws[kk][r] = (w < rows && k < width) ? weight_at(op, a, w, k, width) : 0.0F;
    }
    __syncthreads();
    Sum part[4][4] = {};
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          part[i][j] += static_cast<Sum>(xs[kk][ty * 4 + i]) * static_cast<Sum>(ws[kk][tx * 4 + j]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[i][j] += part[i][j];
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const uint64_t t = token0 + ty * 4 + i;
      const uint64_t w = row0 + tx * 4 + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, static_cast<float>(acc[i][j]));
      }
    }
  }
}
// Long prompts: 256 threads as 16 x 16, each holding TM tokens x TN rows spaced 16 apart (no
// shared bank conflicts). Every output adds the same 16-deep slices in the same order as
// `tiled_kernel`, so the bytes are identical (2026-11-05: prompt reading was bound by this
// matmul). `Fast` is int8 weights with float16 activations and scales, the dtype branches
// resolved at compile time with the same element values as `weight_at`.
template <int TM, int TN, bool Fast>
__global__ __launch_bounds__(kTiledThreads) void wide_kernel(int op, emma_op_args a) {
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kXs = kDepth * kTokens / kTiledThreads;
  constexpr int kWs = kDepth * kRows / kTiledThreads;
  // Two shared slices: the next slice is read into registers while this one is summed.
  __shared__ __align__(16) float xs[2][kDepth][kTokens + 4];
  __shared__ __align__(16) float ws[2][kDepth][kRows + 4];
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
  constexpr int kRowsPerPass = kTiledThreads / kDepth;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* sh = reinterpret_cast<const __half*>(a.weights[1].data);
  float wscale[kWs];
#pragma unroll
  for (int n = 0; n < kWs; ++n) {
    const uint64_t w = row0 + r_load + n * kRowsPerPass;
    wscale[n] = (Fast && w < rows) ? __half2float(sh[w]) : 0.0F;
  }
  float xr[kXs];
  float wr[kWs];
  auto fetch = [&](uint64_t k0) {
    const uint64_t k = k0 + kk_load;
#pragma unroll
    for (int n = 0; n < kXs; ++n) {
      const uint64_t t = token0 + r_load + n * kRowsPerPass;
      float v = 0.0F;
      if (t < tokens && k < width) {
        v = Fast ? __half2float(xh[t * width + k]) : load(a.inputs[0], t * width + k);
      }
      xr[n] = v;
    }
#pragma unroll
    for (int n = 0; n < kWs; ++n) {
      const uint64_t w = row0 + r_load + n * kRowsPerPass;
      float v = 0.0F;
      if (w < rows && k < width) {
        if constexpr (Fast) {
          v = __half2float(__float2half_rn(static_cast<float>(wq[w * width + k]) * wscale[n]));
#ifdef EMMA_MUTANT_SCALE_TWICE
          v = __half2float(__float2half_rn(v * wscale[n]));
#endif
        } else {
          v = weight_at(op, a, w, k, width);
        }
      }
      wr[n] = v;
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
#ifndef EMMA_WIDE_TM
#define EMMA_WIDE_TM 8
#endif
#ifndef EMMA_WIDE_TN
#define EMMA_WIDE_TN 4
#endif
constexpr uint64_t kWideMinTokens = 64;
template <bool Fast>
void launch_wide(int op, const emma_op_args& a, uint64_t rows, uint64_t tokens,
                 cudaStream_t stream) {
  constexpr int tm = EMMA_WIDE_TM;
  constexpr int tn = EMMA_WIDE_TN;
  const dim3 grid(static_cast<unsigned>((rows + 16 * tn - 1) / (16 * tn)),
                  static_cast<unsigned>((tokens + 16 * tm - 1) / (16 * tm)));
  wide_kernel<tm, tn, Fast><<<grid, kTiledThreads, 0, stream>>>(op, a);
}
// Half-precision math for dense binary16 linears (op 7), 2026-11-13 (speed item; kernel change
// allowed by Luis Jake Gabriel III at 18:56 UTC with the old files kept in
// retired/2026-11-13-kernels/). Same tiling as wide_kernel, but the tiles stay binary16 in shared
// memory and the products run as half2 fused multiply-adds (two rows at once; the GTX 1650 runs
// binary16 at twice the binary32 rate). Each 16-step slice is summed in binary16 and added to a
// binary32 total, so rounding grows with the slice, not with the whole width. Chosen from the
// host only when asked (`serve --half-math`); wide_kernel is unchanged.
template <int TM, int TN>
__global__ __launch_bounds__(kTiledThreads) void wide_h2_kernel(emma_op_args a) {
  static_assert(TN % 2 == 0, "rows go in pairs");
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kXs = kDepth * kTokens / kTiledThreads;
  constexpr int kWs = kDepth * kRows / kTiledThreads;
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
  constexpr int kRowsPerPass = kTiledThreads / kDepth;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wh = reinterpret_cast<const __half*>(a.weights[0].data);
  const __half zero = __float2half_rn(0.0F);
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
      wr[n] = (w < rows && k < width) ? wh[w * width + k] : zero;
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
      __half2 wv[TN / 2];
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        wv[j] = *reinterpret_cast<const __half2*>(&ws[b][kk][tx * TN + 2 * j]);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const __half2 xv = __half2half2(xs[b][kk][ty * TM + i]);
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
template __global__ void wide_h2_kernel<8, 4>(emma_op_args a);
// 8-bit activations and int8 dot products for weight-only int8 linears (op 6), 2026-11-13 (kernel
// change allowed by Luis Jake Gabriel III at 18:56 UTC, old files kept in retired/2026-11-13-kernels/).
// q8_blocks_kernel: binary16 activations [T, K] to int8 [T, K] with one binary16 scale per token and
// block of 32 (absolute maximum over 127), as llama.cpp's Q8 blocks. wide_dp4a_kernel: the wide
// tiling over 32-wide slices; each slice is one block, summed with __dp4a in int32 and added to a
// binary32 total times the block's scale; the row scale is applied once at the end. Chosen from the
// host only when asked (`serve --dp4a`); the existing kernels are unchanged.
__global__ __launch_bounds__(256) void q8_blocks_kernel(emma_op_args a) {
  const uint64_t width = static_cast<uint64_t>(a.inputs[0].shape[a.inputs[0].rank - 1]);
  const uint64_t blocks = width / 32;
  const uint64_t total = elements(a.inputs[0]) / 32;
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) {
    return;
  }
  const uint64_t t = i / blocks;
  const uint64_t blk = i % blocks;
  const auto* x = reinterpret_cast<const __half*>(a.inputs[0].data) + t * width + blk * 32;
  auto* q = reinterpret_cast<int8_t*>(a.outputs[0].data) + t * width + blk * 32;
  auto* s = reinterpret_cast<__half*>(a.outputs[1].data);
  float v[32];
  float m = 0.0F;
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    v[k] = __half2float(x[k]);
    m = fmaxf(m, fabsf(v[k]));
  }
  const float scale = m / 127.0F;
  const float inv = m > 0.0F ? 127.0F / m : 0.0F;
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    q[k] = static_cast<int8_t>(__float2int_rn(v[k] * inv));
  }
  s[i] = __float2half_rn(scale);
}
template <int TM, int TN>
__global__ __launch_bounds__(kTiledThreads) void wide_dp4a_kernel(emma_op_args a) {
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kWords = 8;  // 32 int8 per row of a slice, as 8 words
  __shared__ int xs[kTokens][kWords + 1];
  __shared__ int ws[kRows][kWords + 1];
  __shared__ float xsc[kTokens];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t blocks = width / 32;
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tx = static_cast<int>(threadIdx.x % 16);
  const int ty = static_cast<int>(threadIdx.x / 16);
  const auto* xq = reinterpret_cast<const int*>(a.inputs[0].data);
  const auto* xs_scale = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* wq = reinterpret_cast<const int*>(a.weights[0].data);
  const auto* wsh = reinterpret_cast<const __half*>(a.weights[1].data);
  const uint64_t words_per_row = width / 4;
  float acc[TM][TN] = {};
  for (uint64_t blk = 0; blk < blocks; ++blk) {
    for (int i = static_cast<int>(threadIdx.x); i < kTokens * kWords; i += kTiledThreads) {
      const int r = i / kWords;
      const int w = i % kWords;
      const uint64_t t = token0 + r;
      xs[r][w] = t < tokens ? xq[t * words_per_row + blk * kWords + w] : 0;
    }
    for (int i = static_cast<int>(threadIdx.x); i < kRows * kWords; i += kTiledThreads) {
      const int r = i / kWords;
      const int w = i % kWords;
      const uint64_t row = row0 + r;
      ws[r][w] = row < rows ? wq[row * words_per_row + blk * kWords + w] : 0;
    }
    for (int i = static_cast<int>(threadIdx.x); i < kTokens; i += kTiledThreads) {
      const uint64_t t = token0 + i;
      xsc[i] = t < tokens ? __half2float(xs_scale[t * blocks + blk]) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int r = ty * TM + i;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = tx * TN + j;
        int sum = 0;
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
          sum = __dp4a(xs[r][w], ws[c][w], sum);
        }
        acc[i][j] += static_cast<float>(sum) * xsc[r];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j] * __half2float(wsh[w]));
      }
    }
  }
}
template __global__ void wide_dp4a_kernel<8, 4>(emma_op_args a);
// 4-bit weights, unpacked per call (2026-11-14, precision down to 4 bits; kernel change allowed
// by Luis Jake Gabriel III, 2026-11-13 18:56 UTC, old files kept in retired/2026-11-13-kernels/):
// weights[0] packed nibbles [N, K/2] (low nibble first, value + 8), weights[1] binary16 scales
// [N, K/32]; outputs[0] binary16 [N, K] with w = round16(value * scale), as the processor's Q4.
__global__ __launch_bounds__(256) void q4_unpack_kernel(emma_op_args a) {
  const auto* p = reinterpret_cast<const uint8_t*>(a.weights[0].data);
  const auto* s = reinterpret_cast<const __half*>(a.weights[1].data);
  auto* w = reinterpret_cast<__half*>(a.outputs[0].data);
  const uint64_t bytes = elements(a.outputs[0]) / 2;
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= bytes) {
    return;
  }
  const uint64_t e = 2 * i;
  const float sc = __half2float(s[e / 32]);
  const uint8_t b = p[i];
  w[e] = __float2half_rn(static_cast<float>(static_cast<int>(b & 15) - 8) * sc);
  w[e + 1] = __float2half_rn(static_cast<float>(static_cast<int>(b >> 4) - 8) * sc);
}
template <typename Sum>
void launch_linear(int op, const emma_op_args& a, cudaStream_t stream) {
  const auto rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t tokens = elements(a.outputs[0]) / rows;
  if (tokens == 1 && (op == 6 || op == 7)) {
    const auto grid = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
    if (op == 6 && a.inputs[0].dtype == EMMA_F16 && a.weights[1].dtype == EMMA_F16) {
      vector_int8_half_kernel<<<grid, kWarps * 32, 0, stream>>>(a);
    } else if (op == 6) {
      vector_kernel<6, Sum><<<grid, kWarps * 32, 0, stream>>>(a);
    } else if (a.inputs[0].dtype == EMMA_F16 && a.weights[0].dtype == EMMA_F16) {
      vector_half_kernel<<<grid, kWarps * 32, 0, stream>>>(a);
    } else {
      vector_kernel<7, Sum><<<grid, kWarps * 32, 0, stream>>>(a);
    }
    return;
  }
  if ((op == 6 || op == 7) && std::is_same_v<Sum, float> && tokens >= kWideMinTokens) {
    if (op == 6 && a.inputs[0].dtype == EMMA_F16 && a.weights[1].dtype == EMMA_F16) {
      launch_wide<true>(op, a, rows, tokens, stream);
    } else {
      launch_wide<false>(op, a, rows, tokens, stream);
    }
    return;
  }
  if ((op == 6 || op == 7) && tokens >= kTiledMinTokens) {
    const dim3 grid(static_cast<unsigned>((rows + kTile - 1) / kTile),
                    static_cast<unsigned>((tokens + kTile - 1) / kTile));
    tiled_kernel<Sum><<<grid, kTiledThreads, 0, stream>>>(op, a);
    return;
  }
  const auto grid = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  linear_kernel<Sum><<<grid, kWarps * 32, 0, stream>>>(op, a);
}
void linear_launch(int op, const emma_op_args& a, cudaStream_t stream) {
  if (a.inputs[0].dtype == EMMA_F32) {
    launch_linear<double>(op, a, stream);
  } else {
    launch_linear<float>(op, a, stream);
  }
}
}  // namespace emma
