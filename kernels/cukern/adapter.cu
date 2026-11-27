// Gavriel Technologies. Additive low-rank adapter, never merged into the base weight.
// y = round(base + round(round(hidden . b) * scale)), hidden = round(x . a), float32 sums in
// order, dtype rounding at the same boundaries as the backnd Cpu reference.
#include "../cbridg/src/internal.hpp"
#include "common.cuh"
namespace emma {
// hidden[t, j] = round(sum_k x[t, k] * a[j, k]), kept as float32 values.
__global__ __launch_bounds__(128) void adapter_down(emma_op_args a, float* hidden, uint64_t count) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const uint64_t rank = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t j = i % rank;
  const uint64_t token = i / rank;
  float sum = 0;
  for (uint64_t k = 0; k < width; ++k) {
    sum += load(a.inputs[0], token * width + k) * load(a.weights[0], j * width + k);
  }
  hidden[i] = rounded(sum, a.inputs[0].dtype);
}
__global__ __launch_bounds__(128) void adapter_up(emma_op_args a, const float* hidden) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = a.outputs[0];
  if (i >= elements(y)) {
    return;
  }
  const int dtype = a.inputs[0].dtype;
  const uint64_t rows = static_cast<uint64_t>(a.weights[1].shape[0]);
  const uint64_t rank = static_cast<uint64_t>(a.weights[1].shape[1]);
  const uint64_t n = i % rows;
  const uint64_t token = i / rows;
  float sum = 0;
  for (uint64_t j = 0; j < rank; ++j) {
    sum += hidden[token * rank + j] * load(a.weights[1], n * rank + j);
  }
  const float delta = rounded(sum, dtype);
  store(y, i, load(a.inputs[1], i) + rounded(delta * a.attrs.scale, dtype));
}
void adapter_launch(const emma_op_args& a, cudaStream_t stream) {
  const uint64_t rank = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t tokens = elements(a.inputs[0]) / static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t count = tokens * rank;
  float* hidden = nullptr;
  cuda_check(cudaMallocAsync(reinterpret_cast<void**>(&hidden), count * sizeof(float), stream));
  adapter_down<<<blocks(count), 128, 0, stream>>>(a, hidden, count);
  adapter_up<<<blocks(elements(a.outputs[0])), 128, 0, stream>>>(a, hidden);
  cuda_check(cudaFreeAsync(hidden, stream));
}
}  // namespace emma
