#include <cstdio>
#include <cuda_runtime.h>
#include "scale.cuh"

// Clamp before scaling so one wild sample cannot overflow the half range.
constexpr float CLAMP_LIMIT = 65504.0f;

template <typename T>
__device__ __forceinline__ T clamp_abs(T x, T limit) {
    return x > limit ? limit : (x < -limit ? -limit : x);
}

__global__ void scale_kernel(float *x, int n, float gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = clamp_abs(x[i], CLAMP_LIMIT) * gain;
    }
}

extern "C" int scale_on_device(float *host, int n, float gain) {
    float *dev = nullptr;
    size_t bytes = (size_t)n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    CUDA_CHECK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    int blocks = (n + SCALE_BLOCK - 1) / SCALE_BLOCK;
    scale_kernel<<<blocks, SCALE_BLOCK>>>(dev, n, gain);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dev);
    return 0;
}
