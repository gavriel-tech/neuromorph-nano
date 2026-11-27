// Gavriel Technologies. Shared attention geometry and existing mutant mask guard.
#pragma once
#include "common.cuh"
namespace emma {
struct Geometry {
  uint64_t tokens, cached, heads, kv_heads, width, clones;
  explicit Geometry(const emma_op_args& a) {
    const auto& q = a.inputs[0];
    const auto& k = a.inputs[1];
    const int axis = q.rank - 3;
    tokens = static_cast<uint64_t>(q.shape[axis]);
    cached = static_cast<uint64_t>(k.shape[axis]);
    heads = static_cast<uint64_t>(q.shape[axis + 1]);
    kv_heads = static_cast<uint64_t>(k.shape[axis + 1]);
    width = static_cast<uint64_t>(q.shape[axis + 2]);
    clones = axis == 1 ? static_cast<uint64_t>(q.shape[0]) : 1;
  }
};
__device__ inline bool enabled(emma_op_args a, Geometry g, uint64_t row) {
#ifdef EMMA_MUTANT_IGNORE_MASK
  return true;
#else
  const uint64_t index = elements(a.inputs[3]) == g.heads
                             ? row % g.heads
                             : (row / (g.tokens * g.heads)) * g.heads + row % g.heads;
  return reinterpret_cast<const uint8_t*>(a.inputs[3].data)[index] != 0;
#endif
}
void prefill_launch(const emma_op_args& a, const Geometry& g, cudaStream_t stream);
}  // namespace emma
