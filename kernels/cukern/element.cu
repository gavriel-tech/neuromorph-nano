// Gavriel Technologies. embed/rmsnorm/rope/swiglu/residual-add.
// Reference: backnd Cpu; tblock-forward/full-forward per-file dtype tolerance.
#include "common.cuh"
namespace emma {
__global__ __launch_bounds__(128) void embed_kernel(emma_op_args a) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = a.outputs[0];
  if (i >= elements(y)) {
    return;
  }
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const int32_t id = reinterpret_cast<const int32_t*>(a.inputs[0].data)[i / width];
  copy_value(a.weights[0], static_cast<uint64_t>(id) * width + i % width, y, i);
}
__global__ __launch_bounds__(128) void norm_kernel(emma_op_args a) {
  const uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = a.inputs[0];
  const auto y = a.outputs[0];
  const uint64_t width = elements(a.weights[0]);
  if (row >= elements(x) / width) {
    return;
  }
  float sum = 0;
  for (uint64_t j = 0; j < width; ++j) {
    const float v = load(x, row * width + j);
    sum += v * v;
  }
  const float factor = 1.0F / sqrtf(sum / static_cast<float>(width) + a.attrs.eps);
  for (uint64_t j = 0; j < width; ++j) {
    store(y, row * width + j,
          rounded(load(x, row * width + j) * factor, x.dtype) * load(a.weights[0], j));
  }
}
// One block per row: the squares are loaded in parallel into shared memory, one thread adds
// them in the same left-to-right order as `norm_kernel`, and every thread then writes its
// share of the row. Results equal `norm_kernel` bit for bit (2026-11-05: one thread per row
// made each decode norm take about 0.35 ms).
constexpr unsigned kNormThreads = 256;
__global__ __launch_bounds__(kNormThreads) void norm_row_kernel(emma_op_args a) {
  extern __shared__ float squares[];
  __shared__ float factor;
  const uint64_t row = blockIdx.x;
  const auto x = a.inputs[0];
  const auto y = a.outputs[0];
  const uint64_t width = elements(a.weights[0]);
  for (uint64_t j = threadIdx.x; j < width; j += kNormThreads) {
    const float v = load(x, row * width + j);
    squares[j] = v * v;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0;
    for (uint64_t j = 0; j < width; ++j) {
      sum += squares[j];
    }
    factor = 1.0F / sqrtf(sum / static_cast<float>(width) + a.attrs.eps);
  }
  __syncthreads();
  for (uint64_t j = threadIdx.x; j < width; j += kNormThreads) {
    store(y, row * width + j,
          rounded(load(x, row * width + j) * factor, x.dtype) * load(a.weights[0], j));
  }
}
__global__ __launch_bounds__(128) void binary_kernel(int op, emma_op_args a) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = a.inputs[0];
  if (i >= elements(x)) {
    return;
  }
  float left = load(x, i);
  const float right = load(a.inputs[1], i);
  if (op == 4) {
    left = rounded(left / (1.0F + expf(-left)), x.dtype) * right;
  } else {
    left += right;
  }
  store(a.outputs[0], i, left);
}
// sparsify (op 13): keep values with |x| >= threshold (attrs.scale), zero the rest; the
// training-free activation sparsity option (TEAL style), off by default.
__global__ __launch_bounds__(128) void sparsify_kernel(emma_op_args a) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = a.inputs[0];
  if (i >= elements(x)) {
    return;
  }
  const float v = load(x, i);
  store(a.outputs[0], i, fabsf(v) >= a.attrs.scale ? v : 0.0F);
}
__global__ __launch_bounds__(128) void rope_kernel(emma_op_args a, int tensor) {
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = a.inputs[tensor];
  const uint64_t width = static_cast<uint64_t>(x.shape[x.rank - 1]);
  if (i >= elements(x) / 2) {
    return;
  }
  const uint64_t row = i / (width / 2);
  const uint64_t j = i % (width / 2);
  const uint64_t heads = static_cast<uint64_t>(x.shape[x.rank - 2]);
  const int32_t pos = reinterpret_cast<const int32_t*>(a.inputs[2].data)[row / heads];
  const float frequency =
      1.0F / powf(a.attrs.theta, static_cast<float>(2 * j) / static_cast<float>(width));
  const float angle = static_cast<float>(pos) * frequency;
  const float cs = rounded(cosf(angle), x.dtype);
  const float sn = rounded(sinf(angle), x.dtype);
  const float left = load(x, row * width + j);
  const float right = load(x, row * width + j + width / 2);
  store(a.outputs[tensor], row * width + j,
        rounded(left * cs, x.dtype) + rounded(-right * sn, x.dtype));
  store(a.outputs[tensor], row * width + j + width / 2,
        rounded(right * cs, x.dtype) + rounded(left * sn, x.dtype));
}
void element_launch(int op, const emma_op_args& a, cudaStream_t stream) {
  if (op == 0) {
    embed_kernel<<<blocks(elements(a.outputs[0])), 128, 0, stream>>>(a);
  } else if (op == 1) {
    const uint64_t width = elements(a.weights[0]);
    const uint64_t rows = elements(a.inputs[0]) / width;
    if (width * sizeof(float) <= 32768 && rows <= 2147483647ULL) {
      norm_row_kernel<<<static_cast<unsigned>(rows), kNormThreads, width * sizeof(float), stream>>>(
          a);
    } else {
      norm_kernel<<<blocks(rows), 128, 0, stream>>>(a);
    }
  } else if (op == 2) {
    rope_kernel<<<blocks(elements(a.inputs[0]) / 2), 128, 0, stream>>>(a, 0);
    rope_kernel<<<blocks(elements(a.inputs[1]) / 2), 128, 0, stream>>>(a, 1);
  } else if (op == 13) {
    sparsify_kernel<<<blocks(elements(a.inputs[0])), 128, 0, stream>>>(a);
  } else {
    binary_kernel<<<blocks(elements(a.inputs[0])), 128, 0, stream>>>(op, a);
  }
}
}  // namespace emma
