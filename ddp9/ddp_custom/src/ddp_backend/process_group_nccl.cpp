#include "process_group_nccl.h"
#include "macros.h"
#include <cstdio>
#include "kernels.h"


ProcessGroupNCCL::ProcessGroupNCCL(const ncclUniqueId& uid, const PGOptions& opt)
: device_(opt.device), rank_(opt.rank), world_(opt.world) {
  // Select device & eagerly initialize CUDA context
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaFree(0));  // context warmup

  // Non-blocking dedicated stream for NCCL
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  // Initialize NCCL communicator for this rank
  NCCL_CHECK(ncclCommInitRank(&comm_, world_, uid, rank_));

  // Surface any async init faults early
  CUDA_CHECK(cudaStreamSynchronize(stream_)); 
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  if (comm_)  ncclCommDestroy(comm_); 
  if (stream_) cudaStreamDestroy(stream_);
}

void ProcessGroupNCCL::allreduce_float32(float* dev_ptr, size_t count) {
  CUDA_CHECK(cudaSetDevice(device_));

  // Validate pointer/device mapping to avoid illegal access (700)
  cudaPointerAttributes attr{};
  auto perr = cudaPointerGetAttributes(&attr, dev_ptr);
  if (perr != cudaSuccess) {
    fprintf(stderr, "[DDP] cudaPointerGetAttributes failed on rank %d dev %d: %s\n",
            rank_, device_, cudaGetErrorString(perr));
    std::exit(EXIT_FAILURE);
  }
#if CUDART_VERSION >= 10000
  if (attr.type != cudaMemoryTypeDevice || attr.device != device_) {
    fprintf(stderr, "[DDP] BAD PTR on rank %d: ptr=%p type=%d device=%d (expected %d)\n",
            rank_, (void*)dev_ptr, (int)attr.type, attr.device, device_);
    std::exit(EXIT_FAILURE);
  }
#endif

  NCCL_CHECK(ncclAllReduce((const void*)dev_ptr, (void*)dev_ptr,
                           count, ncclFloat, ncclSum, comm_, stream_));

  const float alpha = 1.0f / static_cast<float>(world_);
  launch_scale_f32(dev_ptr, alpha, count, stream_);
}

void ProcessGroupNCCL::broadcast_float32(float* dev_ptr, size_t count, int root) {
  CUDA_CHECK(cudaSetDevice(device_));

  // Validate pointer/device mapping
  cudaPointerAttributes attr{};
  auto perr = cudaPointerGetAttributes(&attr, dev_ptr);
  if (perr != cudaSuccess) {
    fprintf(stderr, "[DDP] cudaPointerGetAttributes failed on rank %d dev %d: %s\n",
            rank_, device_, cudaGetErrorString(perr));
    std::exit(EXIT_FAILURE);
  }
#if CUDART_VERSION >= 10000
  if (attr.type != cudaMemoryTypeDevice || attr.device != device_) {
    fprintf(stderr, "[DDP] BAD PTR on rank %d: ptr=%p type=%d device=%d (expected %d)\n",
            rank_, (void*)dev_ptr, (int)attr.type, attr.device, device_);
    std::exit(EXIT_FAILURE);
  }
#endif

  NCCL_CHECK(ncclBroadcast((const void*)dev_ptr, (void*)dev_ptr,
                           count, ncclFloat, root, comm_, stream_));
}

void ProcessGroupNCCL::barrier() {
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}
