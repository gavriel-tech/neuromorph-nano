// Gavriel Technologies. Half products for weight-only int8 linears (op 6) of 64 tokens or more:
// the tiling and arithmetic of wide_h2_kernel (binary16 tiles in shared memory, half2 fused
// multiply-adds two rows at a time, each 16-step slice summed in binary16 and added to a binary32
// total), with each weight dequantized at fetch exactly as wide_kernel's int8 path does (int8
// times the row's binary16 scale, rounded to binary16). Chosen by the host only inside a served
// kernel scope with half math on; wide_kernel is unchanged.
#include "common.cuh"
namespace emma {
namespace {
constexpr int kDepth = 16;
constexpr int kThreads = 256;
}  // namespace
template <int TM, int TN>
__global__ __launch_bounds__(kThreads) void wide_h2q8_kernel(emma_op_args a) {
  static_assert(TN % 2 == 0, "rows go in pairs");
  constexpr int kTokens = 16 * TM;
  constexpr int kRows = 16 * TN;
  constexpr int kXs = kDepth * kTokens / kThreads;
  constexpr int kWs = kDepth * kRows / kThreads;
  __shared__ __align__(16) __half xs[2][kDepth][kTokens + 8];
  __shared__ __align__(16) __half ws[2][kDepth][kRows + 8];
  const auto y = a.outputs[0];
  const uint64_t rows = static_cast<uint64_t>(a.weights[0].shape[0]);
  const uint64_t width = static_cast<uint64_t>(a.weights[0].shape[1]);
  const uint64_t tokens = elements(y) / rows;
  const uint64_t token0 = static_cast<uint64_t>(blockIdx.y) * kTokens;
  const uint64_t row0 = static_cast<uint64_t>(blockIdx.x) * kRows;
  const int tx = static_cast<int>(threadIdx.x % 16);
  const int ty = static_cast<int>(threadIdx.x / 16);
  const int kk_load = static_cast<int>(threadIdx.x) % kDepth;
  const int r_load = static_cast<int>(threadIdx.x) / kDepth;
  constexpr int kRowsPerPass = kThreads / kDepth;
  const auto* xh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* wq = reinterpret_cast<const int8_t*>(a.weights[0].data);
  const auto* sh = reinterpret_cast<const __half*>(a.weights[1].data);
  const __half zero = __float2half_rn(0.0F);
  float wscale[kWs];
#pragma unroll
  for (int n = 0; n < kWs; ++n) {
    const uint64_t w = row0 + r_load + n * kRowsPerPass;
    wscale[n] = w < rows ? __half2float(sh[w]) : 0.0F;
  }
  __half xr[kXs];
  __half wr[kWs];
  auto fetch = [&](uint64_t k0) {
    const uint64_t k = k0 + kk_load;
#pragma unroll
    for (int n = 0; n < kXs; ++n) {
      const uint64_t t = token0 + r_load + n * kRowsPerPass;
      xr[n] = (t < tokens && k < width) ? xh[t * width + k] : zero;
    }
#pragma unroll
    for (int n = 0; n < kWs; ++n) {
      const uint64_t w = row0 + r_load + n * kRowsPerPass;
      wr[n] = (w < rows && k < width) ? __float2half_rn(static_cast<float>(wq[w * width + k]) * wscale[n]) : zero;
    }
  };
  auto put = [&](int b) {
#pragma unroll
    for (int n = 0; n < kXs; ++n) {
      xs[b][kk_load][r_load + n * kRowsPerPass] = xr[n];
    }
#pragma unroll
    for (int n = 0; n < kWs; ++n) {
      ws[b][kk_load][r_load + n * kRowsPerPass] = wr[n];
    }
  };
  float acc[TM][TN] = {};
  fetch(0);
  put(0);
  __syncthreads();
  int b = 0;
  for (uint64_t k0 = 0; k0 < width; k0 += kDepth) {
    const bool more = k0 + kDepth < width;
    if (more) {
      fetch(k0 + kDepth);
    }
    __half2 part[TM][TN / 2];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        part[i][j] = __half2half2(zero);
      }
    }
#pragma unroll
    for (int kk = 0; kk < kDepth; ++kk) {
      __half2 wv[TN / 2];
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        wv[j] = *reinterpret_cast<const __half2*>(&ws[b][kk][tx * TN + 2 * j]);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const __half2 xv = __half2half2(xs[b][kk][ty * TM + i]);
#pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
          part[i][j] = __hfma2(xv, wv[j], part[i][j]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN / 2; ++j) {
        acc[i][2 * j] += __low2float(part[i][j]);
        acc[i][2 * j + 1] += __high2float(part[i][j]);
      }
    }
    if (more) {
      put(b ^ 1);
    }
    __syncthreads();
    b ^= 1;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint64_t t = token0 + ty * TM + i;
      const uint64_t w = row0 + tx * TN + j;
      if (t < tokens && w < rows) {
        store(y, t * rows + w, acc[i][j]);
      }
    }
  }
}
template __global__ void wide_h2q8_kernel<8, 4>(emma_op_args a);
}  // namespace emma
