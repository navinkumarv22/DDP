#include <cuda_runtime.h>
#include <cstddef>
#include <device_launch_parameters.h>

__global__ void scale_f32_kernel(float* __restrict__ p, float alpha, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= alpha;
}

extern "C" void launch_scale_f32(float* p, float alpha, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  const int grid  = static_cast<int>((n + block - 1) / block);
  scale_f32_kernel<<<grid, block, 0, s>>>(p, alpha, n);
}
