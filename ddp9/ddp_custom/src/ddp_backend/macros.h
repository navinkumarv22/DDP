#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDA_CHECK(cmd) do {                                       \
  cudaError_t e_ = (cmd);                                          \
  if (e_ != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
            cudaGetErrorString(e_));                               \
    std::exit(EXIT_FAILURE);                                       \
  }                                                                \
} while(0)

#define NCCL_CHECK(cmd) do {                                       \
  ncclResult_t r_ = (cmd);                                         \
  if (r_ != ncclSuccess) {                                         \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,  \
            ncclGetErrorString(r_));                               \
    std::exit(EXIT_FAILURE);                                       \
  }                                                                \
} while(0)