// Gavriel Technologies. Shared dtype boundaries, never fast math.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "emma_cbridg.h"
namespace emma {
__host__ __device__ inline uint64_t elements(const emma_tensor& t) {
  uint64_t count = 1;
  for (int i = 0; i < t.rank; ++i) {
    count *= static_cast<uint64_t>(t.shape[i]);
  }
  return count;
}
__device__ inline float load(const emma_tensor& t, uint64_t i) {
  const auto* p = reinterpret_cast<const void*>(t.data);
  if (t.dtype == EMMA_F32) {
    return static_cast<const float*>(p)[i];
  }
  if (t.dtype == EMMA_F16) {
    return __half2float(static_cast<const __half*>(p)[i]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}
__device__ inline void copy_value(const emma_tensor& src, uint64_t si, const emma_tensor& dst,
                                  uint64_t di) {
  if (src.dtype == EMMA_F32) {
    reinterpret_cast<uint32_t*>(dst.data)[di] = reinterpret_cast<const uint32_t*>(src.data)[si];
  } else {
    reinterpret_cast<uint16_t*>(dst.data)[di] = reinterpret_cast<const uint16_t*>(src.data)[si];
  }
}
__device__ inline float rounded(float x, int dtype) {
  if (dtype == EMMA_F16) {
    return __half2float(__float2half_rn(x));
  }
  if (dtype == EMMA_BF16) {
    return __bfloat162float(__float2bfloat16_rn(x));
  }
  return x;
}
__device__ inline void store(const emma_tensor& t, uint64_t i, float x) {
  auto* p = reinterpret_cast<void*>(t.data);
  if (t.dtype == EMMA_F32) {
    static_cast<float*>(p)[i] = x;
  } else if (t.dtype == EMMA_F16) {
    static_cast<__half*>(p)[i] = __float2half_rn(x);
  } else {
    static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(x);
  }
}
inline unsigned blocks(uint64_t count) { return static_cast<unsigned>((count + 127) / 128); }
void element_launch(int op, const emma_op_args& args, cudaStream_t stream);
void linear_launch(int op, const emma_op_args& args, cudaStream_t stream);
void adapter_launch(const emma_op_args& args, cudaStream_t stream);
void attention_launch(const emma_op_args& args, cudaStream_t stream);
void state_launch(int op, const emma_op_args& args, cudaStream_t stream);
}  // namespace emma
