// Gavriel Technologies. Cartridge filter at a socket: a per-channel gain and bias over one
// activation of a MicroModel, y[t, j] = round(fma(x[t, j], gain[j], bias[j])). One fused
// multiply-add in float32 and one rounding to the output's type, the same arithmetic as the
// processor path (`nmengn::filter::apply_cpu`), so both give the same bits. Gain and bias are
// float32 vectors of the activation's width; the activation and output may be any element type
// `load` and `store` read.
#include "common.cuh"
namespace emma {
__global__ __launch_bounds__(128) void filter_kernel(emma_op_args a) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = a.inputs[0];
  if (i >= elements(x)) {
    return;
  }
  const uint64_t width = elements(a.weights[0]);
  const uint64_t j = i % width;
  store(a.outputs[0], i, fmaf(load(x, i), load(a.weights[0], j), load(a.weights[1], j)));
}
}  // namespace emma
