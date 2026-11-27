// Gavriel Technologies. sm_75 prefill: 8 query warps share 16-key K/V tiles.
// Online softmax keeps O(head_dim) state, no quadratic scratch. GQA and head masks
// use the same geometry as decode. Gate: CUDA error <= max(CPU error, dtype step)
// against float64, plus qwen3-1.7b-r2 and embedding golden parity.
#pragma once
#include "attention.cuh"
namespace emma {
template <typename T>
__device__ inline T warp_sum(T value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}
__device__ inline float warp_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}
template <typename Acc>
struct PrefillRow {
  float query[4], maximum;
  Acc output[4], denominator;
  uint64_t row, visible;
  bool active;
  uint64_t token = 0;
  // 2026-11-13 (prefill2): the query token given explicitly (two rows per warp).
  __device__ PrefillRow(emma_op_args a, Geometry g, uint64_t tok) {
    maximum = -CUDART_INF_F;
    denominator = 0;
    token = tok;
    row = (blockIdx.z * g.tokens + token) * g.heads + blockIdx.y;
    active = token < g.tokens && enabled(a, g, row);
    visible = a.attrs.causal != 0 ? g.cached - g.tokens + token + 1 : g.cached;
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = threadIdx.x % 32 + j * 32;
      query[j] = active && d < g.width ? load(a.inputs[0], row * g.width + d) : 0;
      output[j] = 0;
    }
  }
  __device__ void finish_token(emma_op_args a, Geometry g) const {
    if (token >= g.tokens) {
      return;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = threadIdx.x % 32 + j * 32;
      if (d < g.width) {
        store(a.outputs[0], row * g.width + d,
              active ? static_cast<float>(output[j] / denominator) : 0);
      }
    }
  }
  __device__ PrefillRow(emma_op_args a, Geometry g) {
    maximum = -CUDART_INF_F;
    denominator = 0;
    const uint64_t token = blockIdx.x * 8 + threadIdx.x / 32;
    row = (blockIdx.z * g.tokens + token) * g.heads + blockIdx.y;
    active = token < g.tokens && enabled(a, g, row);
    visible = a.attrs.causal != 0 ? g.cached - g.tokens + token + 1 : g.cached;
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = threadIdx.x % 32 + j * 32;
      query[j] = active && d < g.width ? load(a.inputs[0], row * g.width + d) : 0;
      output[j] = 0;
    }
  }
  __device__ void scores(Geometry g, uint64_t first, const float key[16][128], float* tile) {
    const int lane = threadIdx.x % 32;
    for (int s = 0; s < 16; ++s) {
      float dot = 0;
      for (int j = 0; j < 4; ++j) {
        const uint64_t d = lane + j * 32;
        if (d < g.width) {
          dot += query[j] * key[s][d];
        }
      }
      dot = warp_sum(dot);
      if (lane == 0) {
        tile[s] = active && first + s < visible ? dot / sqrtf(static_cast<float>(g.width))
                                                : -CUDART_INF_F;
      }
    }
    __syncwarp();
  }
  __device__ void update(Geometry g, const float value[16][128], float* tile) {
    const int lane = threadIdx.x % 32;
    const float score = lane < 16 ? tile[lane] : -CUDART_INF_F;
    const float next = fmaxf(maximum, warp_max(score));
    const float previous = isfinite(maximum) ? expf(maximum - next) : 0;
    const float probability = isfinite(score) ? expf(score - next) : 0;
    __syncwarp();  // Every lane has finished reading the score tile before reuse.
    if (lane < 16) {
      tile[lane] = probability;
    }
    denominator = denominator * previous + warp_sum(static_cast<Acc>(probability));
    __syncwarp();
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = lane + j * 32;
      output[j] *= previous;
      if (d < g.width) {
        for (int s = 0; s < 16; ++s) {
          output[j] += static_cast<Acc>(tile[s]) * value[s][d];
        }
      }
    }
    maximum = next;
  }
  __device__ void finish(emma_op_args a, Geometry g) const {
    if (blockIdx.x * 8 + threadIdx.x / 32 >= g.tokens) {
      return;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = threadIdx.x % 32 + j * 32;
      if (d < g.width) {
        store(a.outputs[0], row * g.width + d,
              active ? static_cast<float>(output[j] / denominator) : 0);
      }
    }
  }
};
}  // namespace emma
