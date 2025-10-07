#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
void launch_scale_f32(float* p, float alpha, size_t n, cudaStream_t s);
#ifdef __cplusplus
}
#endif
