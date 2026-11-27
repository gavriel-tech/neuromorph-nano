#pragma once

#define SCALE_BLOCK 256
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            fprintf(stderr, "cuda: %s\n", cudaGetErrorString(err_));  \
            return -1;                                                \
        }                                                             \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

int scale_on_device(float *host, int n, float gain);
float sum_on_device(const float *host, int n);

#ifdef __cplusplus
}
#endif
