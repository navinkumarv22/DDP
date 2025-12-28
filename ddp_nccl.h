#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <iostream>

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                              \
  if (r != ncclSuccess) {                            \
    printf("NCCL error %s:%d '%s'\n",                \
           __FILE__, __LINE__, ncclGetErrorString(r));\
    exit(EXIT_FAILURE);                              \
  }                                                   \
} while(0)

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                               \
  if (e != cudaSuccess) {                            \
    printf("CUDA error %s:%d '%s'\n",                \
           __FILE__, __LINE__, cudaGetErrorString(e));\
    exit(EXIT_FAILURE);                              \
  }                                                   \
} while(0)
