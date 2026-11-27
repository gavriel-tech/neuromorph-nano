// Gavriel Technologies. sm_75 attention for prefill over a cached prefix (the batched reranker's
// packed sequences). Two query rows per warp, 16 per block, share each 32-key K/V tile; each lane
// scores one key of the tile with the whole dot product (no shuffle reduction per key), the online
// softmax state is kept per row, and each lane accumulates 4 output dimensions per row. Float
// accumulation, binary16 in and out, the causal and head-mask rules of prefill_kernel; a different
// summation order, so outputs match prefill2_kernel within rounding, not bit for bit.
#include "../cbridg/src/internal.hpp"
#include "attention.cuh"
namespace emma {
namespace {
constexpr int kTile = 32;
constexpr int kWarps = 8;
constexpr int kPer = 2;
constexpr int kWidth = 128;
__device__ inline float seg_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}
__device__ inline float seg_warp_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}
}  // namespace
__global__ __launch_bounds__(256) void segment_kernel(emma_op_args a, Geometry g) {
  __shared__ float key[kTile][kWidth + 1];
  __shared__ float value[kTile][kWidth];
  __shared__ float query[kWarps * kPer][kWidth];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint64_t kh = blockIdx.y / (g.heads / g.kv_heads);
  const float scale = 1.0f / sqrtf(static_cast<float>(g.width));
  uint64_t token[kPer], row[kPer], visible[kPer];
  bool active[kPer];
  float maximum[kPer], denominator[kPer], out[kPer][4];
  for (int r = 0; r < kPer; ++r) {
    token[r] = static_cast<uint64_t>(blockIdx.x) * kWarps * kPer + r * kWarps + warp;
    row[r] = token[r] * g.heads + blockIdx.y;
    active[r] = token[r] < g.tokens && enabled(a, g, row[r]);
    visible[r] = a.attrs.causal != 0 ? g.cached - g.tokens + token[r] + 1 : g.cached;
    maximum[r] = -CUDART_INF_F;
    denominator[r] = 0.0f;
    for (int j = 0; j < 4; ++j) {
      out[r][j] = 0.0f;
    }
    for (int d = lane; d < kWidth; d += 32) {
      query[r * kWarps + warp][d] = active[r] && d < static_cast<int>(g.width) ? load(a.inputs[0], row[r] * g.width + d) : 0.0f;
    }
  }
  const uint64_t last = min(g.tokens, static_cast<uint64_t>(blockIdx.x + 1) * kWarps * kPer);
  const uint64_t limit = a.attrs.causal != 0 ? min(g.cached, g.cached - g.tokens + last) : g.cached;
  for (uint64_t first = 0; first < limit; first += kTile) {
    __syncthreads();
    // Launched only for head width 128 (the host checks), so the tile index splits by shifts.
    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kTile * kWidth); i += blockDim.x) {
      const uint64_t s = i >> 7, d = i & 127;
      const uint64_t at = ((first + s) * g.kv_heads + kh) * g.width + d;
      const bool in = first + s < g.cached;
      key[s][d] = in ? load(a.inputs[1], at) : 0.0f;
      value[s][d] = in ? load(a.inputs[2], at) : 0.0f;
    }
    __syncthreads();
    // Lane `lane` scores key `first + lane` against both of this warp's query rows.
    float dot[kPer] = {0.0f, 0.0f};
#pragma unroll 8
    for (int d = 0; d < kWidth; ++d) {
      const float k = key[lane][d];
      for (int r = 0; r < kPer; ++r) {
        dot[r] += query[r * kWarps + warp][d] * k;
      }
    }
    float probability[kPer], previous[kPer];
    bool any[kPer];
    for (int r = 0; r < kPer; ++r) {
      const bool seen = active[r] && first + lane < visible[r];
      const float score = seen ? dot[r] * scale : -CUDART_INF_F;
      const float next = fmaxf(maximum[r], seg_warp_max(score));
      any[r] = isfinite(next);
      previous[r] = any[r] && isfinite(maximum[r]) ? expf(maximum[r] - next) : 0.0f;
      probability[r] = seen ? expf(score - next) : 0.0f;
      if (any[r]) {
        denominator[r] = denominator[r] * previous[r] + seg_warp_sum(probability[r]);
        maximum[r] = next;
        for (int j = 0; j < 4; ++j) {
          out[r][j] *= previous[r];
        }
      }
    }
    for (int s = 0; s < kTile; ++s) {
      const float p0 = __shfl_sync(0xffffffff, probability[0], s);
      const float p1 = __shfl_sync(0xffffffff, probability[1], s);
      for (int j = 0; j < 4; ++j) {
        const float v = value[s][lane + 32 * j];
        out[0][j] += p0 * v;
        out[1][j] += p1 * v;
      }
    }
  }
  for (int r = 0; r < kPer; ++r) {
    if (token[r] >= g.tokens) {
      continue;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = lane + 32 * j;
      if (d < g.width) {
        store(a.outputs[0], row[r] * g.width + d, active[r] && denominator[r] > 0.0f ? out[r][j] / denominator[r] : 0.0f);
      }
    }
  }
}
}  // namespace emma
