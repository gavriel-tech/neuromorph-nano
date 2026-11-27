// Gavriel Technologies. Compressed bitmask-q8 kernels (Wave 2 + WP-A1 X2, 2026-11-22):
// O(1) per-thread hardware __popc() decompression for unstructured parameter-mapped int8 linears.
//
// Format per [rows, cols] linear matrix:
// - packed_i8:  contiguous kept int8 weights in row-major order
// - row_scales: [rows] binary16 per-row scales (identical to linear_q8)
// - mask_words: [rows, words_per_row] uint32 bitmasks (words_per_row = ceil(cols / 32), bit c%32 = kept)
// - group_offs: [rows, groups_per_row] uint16 intra-row offsets at each 64-weight group start
// - row_offs:   [rows] uint32 global byte offset into packed_i8 where each row starts
//
// Three kernels:
// 1. bmq8_vec_kernel: tokens == 1 decode directly over compressed weights (zero scratch buffer,
//    deferred grp_offs fetch on empty groups, full-word 0xffffffff fast path, bit-identical to
//    vector_int8_half_kernel).
// 2. bmq8_small_kernel: 2 <= tokens <= 16 directly over compressed weights (zero scratch buffer,
//    chunked-8 register accumulators sharing each weight read across tokens, zero-group skip +
//    full-word fast path, bit-identical to linear_kernel<float>).
// 3. bmq8_unpack_kernel: tokens > 16 prefill unpacks [rows, cols] compressed weights into a pooled
//    temporary int8 matrix (block-shared row_base, zero-group skip, full-word fast path), then
//    feeds h2q8_fetch_kernel / dp4a_tile_kernel / wide_q8f_kernel at peak SM throughput.
#include "common.cuh"
namespace emma {

__global__ __launch_bounds__(128) void bmq8_vec_kernel(emma_op_args a) {
  const auto x_t = a.inputs[0];
  const auto mask_t = a.inputs[1];
  const auto grp_t = a.inputs[2];
  const auto row_t = a.inputs[3];
  const auto packed_t = a.weights[0];
  const auto scale_t = a.weights[1];
  const auto y_t = a.outputs[0];

  const uint32_t rows = static_cast<uint32_t>(y_t.shape[1]);
  const uint32_t cols = static_cast<uint32_t>(x_t.shape[1]);
  const uint32_t words_per_row = static_cast<uint32_t>(mask_t.shape[1]);
  const uint32_t groups_per_row = static_cast<uint32_t>(grp_t.shape[1]);

  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t r = blockIdx.x * 4u + warp;
  if (r >= rows) {
    return;
  }

  const auto* x = reinterpret_cast<const __half*>(x_t.data);
  const auto* mask_words = reinterpret_cast<const uint32_t*>(mask_t.data) + static_cast<uint64_t>(r) * words_per_row;
  const auto* grp_offs = reinterpret_cast<const uint16_t*>(grp_t.data) + static_cast<uint64_t>(r) * groups_per_row;
  const uint32_t row_base = __shfl_sync(0xffffffffu, lane == 0 ? reinterpret_cast<const uint32_t*>(row_t.data)[r] : 0u, 0);
  const auto* packed = reinterpret_cast<const int8_t*>(packed_t.data);
  const float scale = __shfl_sync(0xffffffffu, lane == 0 ? __half2float(reinterpret_cast<const __half*>(scale_t.data)[r]) : 0.0F, 0);
  const uint32_t below = (1u << lane) - 1u;

  float acc = 0.0F;
  // Warp-coalesced batch prefetch: load 32 mask words (16 groups = 1,024 weights) and 16 group
  // offsets in one coalesced transaction per batch, eliminating serial lane-0 DRAM stalls.
  for (uint32_t g0 = 0; g0 < groups_per_row; g0 += 16u) {
    const uint32_t w_prefetch = g0 * 2u + lane;
    const uint32_t my_w = (w_prefetch < words_per_row) ? mask_words[w_prefetch] : 0u;
    const uint32_t g_prefetch = g0 + lane;
    const uint32_t my_g = (lane < 16u && g_prefetch < groups_per_row) ? (row_base + static_cast<uint32_t>(grp_offs[g_prefetch])) : 0u;
    const uint32_t g_end = (g0 + 16u < groups_per_row) ? 16u : (groups_per_row - g0);
    for (uint32_t j = 0; j < g_end; ++j) {
      const uint32_t w0 = __shfl_sync(0xffffffffu, my_w, static_cast<int>(j * 2u));
      const uint32_t w1 = __shfl_sync(0xffffffffu, my_w, static_cast<int>(j * 2u + 1u));
      if ((w0 | w1) == 0u) {
        continue;
      }
      const uint32_t gbase = __shfl_sync(0xffffffffu, my_g, static_cast<int>(j));
      const uint32_t k0 = (g0 + j) * 64u + lane;
      if (w0 == 0xffffffffu && k0 < cols) {
        const float weight0 = __half2float(__float2half_rn(static_cast<float>(packed[gbase + lane]) * scale));
        acc += __half2float(x[k0]) * weight0;
      } else if (w0 != 0u && k0 < cols && ((w0 >> lane) & 1u)) {
        const uint32_t idx0 = gbase + static_cast<uint32_t>(__popc(w0 & below));
        const float weight0 = __half2float(__float2half_rn(static_cast<float>(packed[idx0]) * scale));
        acc += __half2float(x[k0]) * weight0;
      }
      const uint32_t k1 = k0 + 32u;
      if (w1 != 0u && k1 < cols) {
        const uint32_t base1 = gbase + static_cast<uint32_t>(__popc(w0));
        if (w1 == 0xffffffffu) {
          const float weight1 = __half2float(__float2half_rn(static_cast<float>(packed[base1 + lane]) * scale));
          acc += __half2float(x[k1]) * weight1;
        } else if ((w1 >> lane) & 1u) {
          const uint32_t idx1 = base1 + static_cast<uint32_t>(__popc(w1 & below));
          const float weight1 = __half2float(__float2half_rn(static_cast<float>(packed[idx1]) * scale));
          acc += __half2float(x[k1]) * weight1;
        }
      }
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) {
    reinterpret_cast<__half*>(y_t.data)[r] = __float2half_rn(acc);
  }
}

__global__ __launch_bounds__(128) void bmq8_small_kernel(emma_op_args a) {
  const auto x_t = a.inputs[0];
  const auto mask_t = a.inputs[1];
  const auto grp_t = a.inputs[2];
  const auto row_t = a.inputs[3];
  const auto packed_t = a.weights[0];
  const auto scale_t = a.weights[1];
  const auto y_t = a.outputs[0];

  const uint32_t tokens = static_cast<uint32_t>(x_t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(x_t.shape[1]);
  const uint32_t rows = static_cast<uint32_t>(y_t.shape[1]);
  const uint32_t words_per_row = static_cast<uint32_t>(mask_t.shape[1]);
  const uint32_t groups_per_row = static_cast<uint32_t>(grp_t.shape[1]);

  const uint32_t warp = threadIdx.x >> 5;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t r = blockIdx.x * 4u + warp;
  if (r >= rows) {
    return;
  }

  const auto* x = reinterpret_cast<const __half*>(x_t.data);
  const auto* mask_words = reinterpret_cast<const uint32_t*>(mask_t.data) + static_cast<uint64_t>(r) * words_per_row;
  const auto* grp_offs = reinterpret_cast<const uint16_t*>(grp_t.data) + static_cast<uint64_t>(r) * groups_per_row;
  const uint32_t row_base = __shfl_sync(0xffffffffu, lane == 0 ? reinterpret_cast<const uint32_t*>(row_t.data)[r] : 0u, 0);
  const auto* packed = reinterpret_cast<const int8_t*>(packed_t.data);
  const float scale = __shfl_sync(0xffffffffu, lane == 0 ? __half2float(reinterpret_cast<const __half*>(scale_t.data)[r]) : 0.0F, 0);
  const uint32_t below = (1u << lane) - 1u;
  auto* y = reinterpret_cast<__half*>(y_t.data);

  for (uint32_t t0 = 0; t0 < tokens; t0 += 8u) {
    const uint32_t n = (tokens - t0 < 8u) ? (tokens - t0) : 8u;
    float acc[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      const uint32_t w_idx = g * 2u;
      uint32_t w0 = 0u;
      uint32_t w1 = 0u;
      if (lane == 0) {
        w0 = mask_words[w_idx];
        w1 = (w_idx + 1u < words_per_row) ? mask_words[w_idx + 1u] : 0u;
      }
      w0 = __shfl_sync(0xffffffffu, w0, 0);
      w1 = __shfl_sync(0xffffffffu, w1, 0);
      if ((w0 | w1) == 0u) {
        continue;
      }
      const uint32_t gbase = __shfl_sync(0xffffffffu, lane == 0 ? (row_base + static_cast<uint32_t>(grp_offs[g])) : 0u, 0);
      const uint32_t k0 = g * 64u + lane;
      if (w0 != 0u && k0 < cols && ((w0 >> lane) & 1u)) {
        const uint32_t idx0 = (w0 == 0xffffffffu) ? (gbase + lane) : (gbase + static_cast<uint32_t>(__popc(w0 & below)));
        const float weight0 = __half2float(__float2half_rn(static_cast<float>(packed[idx0]) * scale));
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
          if (i < n) {
            acc[i] += __half2float(x[static_cast<uint64_t>(t0 + i) * cols + k0]) * weight0;
          }
        }
      }
      const uint32_t k1 = k0 + 32u;
      if (w1 != 0u && k1 < cols && ((w1 >> lane) & 1u)) {
        const uint32_t base1 = gbase + static_cast<uint32_t>(__popc(w0));
        const uint32_t idx1 = (w1 == 0xffffffffu) ? (base1 + lane) : (base1 + static_cast<uint32_t>(__popc(w1 & below)));
        const float weight1 = __half2float(__float2half_rn(static_cast<float>(packed[idx1]) * scale));
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
          if (i < n) {
            acc[i] += __half2float(x[static_cast<uint64_t>(t0 + i) * cols + k1]) * weight1;
          }
        }
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < 8u; ++i) {
      if (i < n) {
        float s = acc[i];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          s += __shfl_down_sync(0xffffffffu, s, offset);
        }
        if (lane == 0) {
          y[static_cast<uint64_t>(t0 + i) * rows + r] = __float2half_rn(s);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256) void bmq8_unpack_kernel(emma_op_args a) {
  const auto mask_t = a.inputs[0];
  const auto grp_t = a.inputs[1];
  const auto row_t = a.inputs[2];
  const auto packed_t = a.weights[0];
  const auto out_t = a.outputs[0];

  const uint32_t rows = static_cast<uint32_t>(out_t.shape[0]);
  const uint32_t cols = static_cast<uint32_t>(out_t.shape[1]);
  const uint32_t words_per_row = static_cast<uint32_t>(mask_t.shape[1]);
  const uint32_t groups_per_row = static_cast<uint32_t>(grp_t.shape[1]);

  // 2D grid: blockIdx.y = row, blockIdx.x * 8 + warp = 64-weight group (zero 64-bit divisions).
  const uint32_t r = blockIdx.y;
  if (r >= rows) {
    return;
  }
  __shared__ uint32_t s_row_base;
  if (threadIdx.x == 0) {
    s_row_base = reinterpret_cast<const uint32_t*>(row_t.data)[r];
  }
  __syncthreads();

  const uint32_t g = blockIdx.x * 8u + (threadIdx.x >> 5);
  const uint32_t lane = threadIdx.x & 31u;
  if (g >= groups_per_row) {
    return;
  }

  uint32_t w0 = 0u;
  uint32_t w1 = 0u;
  if (lane == 0) {
    const auto* mask_words = reinterpret_cast<const uint32_t*>(mask_t.data) + static_cast<uint64_t>(r) * words_per_row;
    const uint32_t w_idx = g * 2u;
    w0 = mask_words[w_idx];
    w1 = (w_idx + 1u < words_per_row) ? mask_words[w_idx + 1u] : 0u;
  }
  w0 = __shfl_sync(0xffffffffu, w0, 0);
  w1 = __shfl_sync(0xffffffffu, w1, 0);

  auto* dst = reinterpret_cast<int8_t*>(out_t.data) + static_cast<uint64_t>(r) * cols;
  const uint32_t k0 = g * 64u + lane;
  const uint32_t k1 = k0 + 32u;

  if ((w0 | w1) == 0u) {
    if (k0 < cols) {
      dst[k0] = static_cast<int8_t>(0);
    }
    if (k1 < cols) {
      dst[k1] = static_cast<int8_t>(0);
    }
    return;
  }

  uint32_t gbase = 0u;
  if (lane == 0) {
    const uint16_t grp_off = reinterpret_cast<const uint16_t*>(grp_t.data)[static_cast<uint64_t>(r) * groups_per_row + g];
    gbase = s_row_base + static_cast<uint32_t>(grp_off);
  }
  gbase = __shfl_sync(0xffffffffu, gbase, 0);

  const auto* packed = reinterpret_cast<const int8_t*>(packed_t.data);
  const uint32_t below = (1u << lane) - 1u;

  if (k0 < cols) {
    if (w0 == 0xffffffffu) {
      dst[k0] = packed[gbase + lane];
    } else if (w0 != 0u && ((w0 >> lane) & 1u)) {
      dst[k0] = packed[gbase + static_cast<uint32_t>(__popc(w0 & below))];
    } else {
      dst[k0] = static_cast<int8_t>(0);
    }
  }
  if (k1 < cols) {
    if (w1 == 0u) {
      dst[k1] = static_cast<int8_t>(0);
    } else {
      const uint32_t base1 = gbase + static_cast<uint32_t>(__popc(w0));
      if (w1 == 0xffffffffu) {
        dst[k1] = packed[base1 + lane];
      } else if ((w1 >> lane) & 1u) {
        dst[k1] = packed[base1 + static_cast<uint32_t>(__popc(w1 & below))];
      } else {
        dst[k1] = static_cast<int8_t>(0);
      }
    }
  }
}

}  // namespace emma
