#include <cuda_runtime.h>
#include "scale.cuh"

#define WARP 32

/* Sum within one warp by shuffles; lane 0 ends with the warp's total. */
__device__ float warp_sum(float v) {
    for (int offset = WARP / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void block_sum(const float *x, float *partial, int n) {
    __shared__ float lanes[SCALE_BLOCK / WARP];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = warp_sum(i < n ? x[i] : 0.0f);
    if ((threadIdx.x % WARP) == 0) {
        lanes[threadIdx.x / WARP] = v;
    }
    __syncthreads();
    if (threadIdx.x < WARP) {
        v = threadIdx.x < SCALE_BLOCK / WARP ? lanes[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) {
            partial[blockIdx.x] = v;
        }
    }
}

extern "C" float sum_on_device(const float *host, int n) {
    int blocks = (n + SCALE_BLOCK - 1) / SCALE_BLOCK;
    float *dev = nullptr, *partial = nullptr;
    cudaMalloc(&dev, n * sizeof(float));
    cudaMalloc(&partial, blocks * sizeof(float));
    cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    block_sum<<<blocks, SCALE_BLOCK>>>(dev, partial, n);
    float total = 0.0f;
    float *back = new float[blocks];
    cudaMemcpy(back, partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    for (int b = 0; b < blocks; b++) {
        total += back[b];
    }
    delete[] back;
    cudaFree(dev);
    cudaFree(partial);
    return total;
}
