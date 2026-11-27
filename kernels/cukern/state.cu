// Gavriel Technologies. Session KV append/read and lowest-id greedy selection.
// Reference: full-forward; cache bytes and integer outputs exact.
#include "../cbridg/src/internal.hpp"
#include "common.cuh"
namespace emma {
__global__ __launch_bounds__(128) void cache_kernel(emma_op_args a, int pair) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto out = a.outputs[pair];
  const auto cache = a.inputs[pair];
  const auto added = a.inputs[pair + 2];
  if (i >= elements(out)) {
    return;
  }
  const int axis = cache.rank - 3;
  const uint64_t stride = static_cast<uint64_t>(cache.shape[axis + 1] * cache.shape[axis + 2]);
  const uint64_t used = static_cast<uint64_t>(out.shape[axis]) * stride;
  const uint64_t capacity = static_cast<uint64_t>(cache.shape[axis]) * stride;
  const uint64_t append = static_cast<uint64_t>(added.shape[axis]) * stride;
  const uint64_t clone = i / used;
  const uint64_t offset = i % used;
  const uint64_t boundary = a.attrs.pos * stride;
  if (offset < boundary) {
    copy_value(cache, clone * capacity + offset, out, i);
  } else {
    copy_value(added, clone * append + offset - boundary, cache, clone * capacity + offset);
    copy_value(added, clone * append + offset - boundary, out, i);
  }
}
__global__ __launch_bounds__(128) void greedy_kernel(emma_op_args a) {
  const uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= elements(a.outputs[0])) {
    return;
  }
  const auto x = a.inputs[0];
  const uint64_t width = static_cast<uint64_t>(x.shape[x.rank - 1]);
  int32_t best = 0;
  float value = load(x, row * width);
  for (uint64_t i = 1; i < width; ++i) {
    const float candidate = load(x, row * width + i);
    if (candidate > value) {
      value = candidate;
      best = static_cast<int32_t>(i);
    }
  }
  reinterpret_cast<int32_t*>(a.outputs[0].data)[row] = best;
}
void state_launch(int op, const emma_op_args& a, cudaStream_t stream) {
  if (op == 8) {
    cache_kernel<<<blocks(elements(a.outputs[0])), 128, 0, stream>>>(a, 0);
    cuda_check(cudaGetLastError());
    cache_kernel<<<blocks(elements(a.outputs[1])), 128, 0, stream>>>(a, 1);
  } else {
    greedy_kernel<<<blocks(elements(a.outputs[0])), 128, 0, stream>>>(a);
  }
}
void launch(int op, const emma_op_args& a, cudaStream_t stream) {
  if (op <= 2 || op == 4 || op == 5 || op == 13) {
    element_launch(op, a, stream);
  } else if (op == 3) {
    attention_launch(a, stream);
  } else if (op == 6 || op == 7 || op == 10 || op == 12) {
    linear_launch(op, a, stream);
  } else if (op == 11) {
    adapter_launch(a, stream);
  } else {
    state_launch(op, a, stream);
  }
  cuda_check(cudaGetLastError());
#ifndef NDEBUG
  cuda_check(cudaStreamSynchronize(stream));
#endif
}
}  // namespace emma
