// Gavriel Technologies. Tiled online-softmax prefill, float64-bounded parity.
#include "../cbridg/src/internal.hpp"
#include "prefill.cuh"
namespace emma {
template <typename Acc>
__global__ __launch_bounds__(256) void prefill_kernel(emma_op_args a, Geometry g) {
  __shared__ float key[16][128], value[16][128], scores[8][16];
  PrefillRow<Acc> row(a, g);
  const uint64_t kh = blockIdx.y / (g.heads / g.kv_heads);
  const uint64_t limit =
      a.attrs.causal != 0 ? min(g.cached, g.cached - g.tokens + (blockIdx.x + 1) * 8) : g.cached;
  for (uint64_t first = 0; first < limit; first += 16) {
    for (uint64_t i = threadIdx.x; i < 16 * g.width; i += blockDim.x) {
      const uint64_t s = i / g.width, d = i % g.width;
      const uint64_t at = ((blockIdx.z * g.cached + first + s) * g.kv_heads + kh) * g.width + d;
      key[s][d] = first + s < g.cached ? load(a.inputs[1], at) : 0;
      value[s][d] = first + s < g.cached ? load(a.inputs[2], at) : 0;
    }
    __syncthreads();
    row.scores(g, first, key, scores[threadIdx.x / 32]);
    row.update(g, value, scores[threadIdx.x / 32]);
    __syncthreads();
  }
  row.finish(a, g);
}
// 2026-11-13: two query rows per warp (16 per block) share each 16-key K/V tile, halving
// the key and value reads; each row's arithmetic is the same sequence as prefill_kernel,
// so its bytes are identical.
template <typename Acc>
__global__ __launch_bounds__(256) void prefill2_kernel(emma_op_args a, Geometry g) {
  __shared__ float key[16][128], value[16][128], scores[2][8][16];
  const uint64_t warp = threadIdx.x / 32;
  PrefillRow<Acc> r0(a, g, blockIdx.x * 16 + warp);
  PrefillRow<Acc> r1(a, g, blockIdx.x * 16 + 8 + warp);
  const uint64_t limit =
      a.attrs.causal != 0 ? min(g.cached, g.cached - g.tokens + (blockIdx.x + 1) * 16) : g.cached;
  const uint64_t kh = blockIdx.y / (g.heads / g.kv_heads);
  for (uint64_t first = 0; first < limit; first += 16) {
    for (uint64_t i = threadIdx.x; i < 16 * g.width; i += blockDim.x) {
      const uint64_t s = i / g.width, d = i % g.width;
      const uint64_t at = ((blockIdx.z * g.cached + first + s) * g.kv_heads + kh) * g.width + d;
      key[s][d] = first + s < g.cached ? load(a.inputs[1], at) : 0;
      value[s][d] = first + s < g.cached ? load(a.inputs[2], at) : 0;
    }
    __syncthreads();
    r0.scores(g, first, key, scores[0][warp]);
    r0.update(g, value, scores[0][warp]);
    r1.scores(g, first, key, scores[1][warp]);
    r1.update(g, value, scores[1][warp]);
    __syncthreads();
  }
  r0.finish_token(a, g);
  r1.finish_token(a, g);
}
template __global__ void prefill2_kernel<float>(emma_op_args, Geometry);
void prefill_launch(const emma_op_args& a, const Geometry& g, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((g.tokens + 7) / 8), static_cast<unsigned>(g.heads),
                  static_cast<unsigned>(g.clones));
  // F32's tight truth gate needs wider softmax/output reductions. Half inputs
  // retain F32 accumulation, with final rounding only at the output boundary.
  if (a.inputs[0].dtype == EMMA_F32) {
    prefill_kernel<double><<<grid, 256, 0, stream>>>(a, g);
  } else {
    prefill_kernel<float><<<grid, 256, 0, stream>>>(a, g);
  }
  cuda_check(cudaGetLastError());
}
}  // namespace emma
