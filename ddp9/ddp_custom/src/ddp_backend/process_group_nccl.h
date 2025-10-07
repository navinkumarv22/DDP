#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <nccl.h>

struct PGOptions {
  int device = 0;
  int rank = 0;
  int world = 1;
  int stream_priority = -1; // -1 = highest
};

class ProcessGroupNCCL {
public:
  ProcessGroupNCCL(const ncclUniqueId& uid, const PGOptions& opt);
  ~ProcessGroupNCCL();

  int rank() const { return rank_; }
  int world_size() const { return world_; }
  cudaStream_t stream() const { return stream_; }

  void allreduce_float32(float* dev_ptr, size_t count);
  // void allreduce_bfloat16(void* dev_ptr, size_t count);
  void broadcast_float32(float* dev_ptr, size_t count, int root);
  void barrier();

private:
  int device_ = 0;
  int rank_ = 0;
  int world_ = 1;
  cudaEvent_t default_stream_event_;
  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

