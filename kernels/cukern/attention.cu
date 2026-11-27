// Gavriel Technologies. GQA decode and tiled online-softmax prefill, with explicit head masks.
// Reference: backnd Cpu, tblock-forward per-file tolerance; masked heads exact zero.
#include "../cbridg/src/internal.hpp"
#include "attention.cuh"
namespace emma {
__global__ __launch_bounds__(128) void score_kernel(emma_op_args a, Geometry g, float* scores) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.clones * g.tokens * g.heads * g.cached) {
    return;
  }
  const uint64_t row = i / g.cached;
  const uint64_t pos = i % g.cached;
  const uint64_t token = (row / g.heads) % g.tokens;
  const uint64_t clone = row / (g.heads * g.tokens);
  const uint64_t kh = (row % g.heads) / (g.heads / g.kv_heads);
  if (!enabled(a, g, row) || (a.attrs.causal != 0 && pos > g.cached - g.tokens + token)) {
    scores[i] = -CUDART_INF_F;
    return;
  }
  float sum = 0;
  for (uint64_t d = 0; d < g.width; ++d) {
    sum += load(a.inputs[0], row * g.width + d) *
           load(a.inputs[1], ((clone * g.cached + pos) * g.kv_heads + kh) * g.width + d);
  }
  scores[i] = sum / sqrtf(static_cast<float>(g.width));
}
// Scores with a shared key tile (E1, 2026-11-10): one block per (key tile, key-value head,
// clone). The tile is read from global memory once, coalesced, and reused by every token and
// head of the group; score_kernel read each key row once per (token, head) with uncoalesced
// loads, which made a short question over a long window grow faster than the window. Each
// score is the same sequential sum over the width as score_kernel, so results are
// bit-identical. Head widths up to 128.
constexpr unsigned kTile = 32;
constexpr unsigned kTileThreads = 256;
__global__ __launch_bounds__(kTileThreads) void score_tile_kernel(emma_op_args a, Geometry g,
                                                                  float* scores) {
  __shared__ float ks[kTile][129];
  const uint64_t kh = blockIdx.y;
  const uint64_t clone = blockIdx.z;
  const uint64_t p0 = static_cast<uint64_t>(blockIdx.x) * kTile;
  const uint64_t span = g.cached - p0 < kTile ? g.cached - p0 : kTile;
  for (uint64_t i = threadIdx.x; i < span * g.width; i += kTileThreads) {
    const uint64_t p = i / g.width;
    const uint64_t d = i % g.width;
    ks[p][d] = load(a.inputs[1], ((clone * g.cached + p0 + p) * g.kv_heads + kh) * g.width + d);
  }
  __syncthreads();
  const uint64_t group = g.heads / g.kv_heads;
  const uint64_t items = g.tokens * group * span;
  for (uint64_t it = threadIdx.x; it < items; it += kTileThreads) {
    const uint64_t p = it % span;
    const uint64_t r = it / span;
    const uint64_t token = r / group;
    const uint64_t row = (clone * g.tokens + token) * g.heads + kh * group + r % group;
    const uint64_t pos = p0 + p;
    float* out = scores + row * g.cached + pos;
    if (!enabled(a, g, row) || (a.attrs.causal != 0 && pos > g.cached - g.tokens + token)) {
      *out = -CUDART_INF_F;
      continue;
    }
    float sum = 0;
    for (uint64_t d = 0; d < g.width; ++d) {
      sum += load(a.inputs[0], row * g.width + d) * ks[p][d];
    }
    *out = sum / sqrtf(static_cast<float>(g.width));
  }
}
__global__ __launch_bounds__(128) void softmax_kernel(emma_op_args a, Geometry g, float* scores) {
  const uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= g.clones * g.tokens * g.heads || !enabled(a, g, row)) {
    return;
  }
  float max_value = -CUDART_INF_F;
  for (uint64_t s = 0; s < g.cached; ++s) {
    max_value = fmaxf(max_value, scores[row * g.cached + s]);
  }
  float sum = 0;
  for (uint64_t s = 0; s < g.cached; ++s) {
    const float v = expf(scores[row * g.cached + s] - max_value);
    scores[row * g.cached + s] = v;
    sum += v;
  }
  for (uint64_t s = 0; s < g.cached; ++s) {
    scores[row * g.cached + s] = rounded(scores[row * g.cached + s] / sum, a.inputs[0].dtype);
  }
}
// One block per row: the maximum and the exponentials are computed in parallel (both are
// exact whatever the order), one thread adds the exponentials in the same order as
// `softmax_kernel`, and the divisions run in parallel, so results equal it bit for bit
// (2026-11-05: one thread per row made decode slow down with every cached position).
constexpr unsigned kSoftThreads = 256;
__global__ __launch_bounds__(kSoftThreads) void softmax_row_kernel(emma_op_args a, Geometry g,
                                                                    float* scores) {
  __shared__ float partial[kSoftThreads];
  __shared__ float total;
  const uint64_t row = blockIdx.x;
  if (!enabled(a, g, row)) {
    return;
  }
  float* s = scores + row * g.cached;
  float local = -CUDART_INF_F;
  for (uint64_t i = threadIdx.x; i < g.cached; i += kSoftThreads) {
    local = fmaxf(local, s[i]);
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = kSoftThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_value = partial[0];
  for (uint64_t i = threadIdx.x; i < g.cached; i += kSoftThreads) {
    s[i] = expf(s[i] - max_value);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (uint64_t i = 0; i < g.cached; ++i) {
      sum += s[i];
    }
    total = sum;
  }
  __syncthreads();
  for (uint64_t i = threadIdx.x; i < g.cached; i += kSoftThreads) {
    s[i] = rounded(s[i] / total, a.inputs[0].dtype);
  }
}
__global__ __launch_bounds__(128) void value_kernel(emma_op_args a, Geometry g,
                                                    const float* scores) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements(a.outputs[0])) {
    return;
  }
  const uint64_t row = i / g.width;
  const uint64_t d = i % g.width;
  float sum = 0;
  if (enabled(a, g, row)) {
    const uint64_t clone = row / (g.heads * g.tokens);
    const uint64_t kh = (row % g.heads) / (g.heads / g.kv_heads);
    const uint64_t token = (row / g.heads) % g.tokens;
    const uint64_t visible = a.attrs.causal != 0 ? g.cached - g.tokens + token + 1 : g.cached;
    for (uint64_t s = 0; s < visible; ++s) {
      sum += scores[row * g.cached + s] *
             load(a.inputs[2], ((clone * g.cached + s) * g.kv_heads + kh) * g.width + d);
    }
  }
  store(a.outputs[0], i, sum);
}
void attention_launch(const emma_op_args& a, cudaStream_t stream) {
  Geometry g(a);
  // Decode and unsupported head widths retain the original rounded-softmax path.
  if (g.tokens >= 32 && g.width <= 128 && g.heads <= 65535 && g.clones <= 65535) {
    prefill_launch(a, g, stream);
    return;
  }
  float* scores = nullptr;
  const uint64_t rows = g.clones * g.tokens * g.heads;
  require(rows <= static_cast<uint64_t>(INT64_MAX) / g.cached / sizeof(float), EMMA_ERR_OVERFLOW,
          "attention scratch overflow");
  cuda_check(
      cudaMallocAsync(reinterpret_cast<void**>(&scores), rows * g.cached * sizeof(float), stream));
  score_kernel<<<blocks(rows * g.cached), 128, 0, stream>>>(a, g, scores);
  cuda_check(cudaGetLastError());
  if (rows <= 2147483647ULL) {
    softmax_row_kernel<<<static_cast<unsigned>(rows), kSoftThreads, 0, stream>>>(a, g, scores);
  } else {
    softmax_kernel<<<blocks(rows), 128, 0, stream>>>(a, g, scores);
  }
  cuda_check(cudaGetLastError());
  value_kernel<<<blocks(elements(a.outputs[0])), 128, 0, stream>>>(a, g, scores);
  cuda_check(cudaGetLastError());
  cuda_check(cudaFreeAsync(scores, stream));
}
}  // namespace emma
