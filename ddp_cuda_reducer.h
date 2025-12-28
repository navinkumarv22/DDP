#pragma once
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <iostream>

//////////////////////////////////////////////////////////////
// CUDA helpers
//////////////////////////////////////////////////////////////

#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

//////////////////////////////////////////////////////////////
// Gradient View
//////////////////////////////////////////////////////////////

struct GradView {
    float* data;     // device pointer
    size_t numel;
};

//////////////////////////////////////////////////////////////
// Bucket
//////////////////////////////////////////////////////////////

struct DDPBucket {
    std::vector<GradView> grads;
    size_t total_numel = 0;

    float* buffer = nullptr;        // device flat buffer
    MPI_Request request = MPI_REQUEST_NULL;
    bool reduced = false;
};

//////////////////////////////////////////////////////////////
// CUDA DDP Reducer (PyTorch-like)
//////////////////////////////////////////////////////////////

class DDPCUDAReducer {
public:
    DDPCUDAReducer(size_t bucket_bytes,
                   cudaStream_t stream,
                   MPI_Comm comm = MPI_COMM_WORLD);

    ~DDPCUDAReducer();

    // Register parameter gradients (call once)
    void register_param(float* grad_ptr, size_t numel);

    // Build buckets (call once)
    void finalize();

    // Called when a gradient becomes ready
    void mark_ready(float* grad_ptr);

    // Call at end of backward
    void finalize_backward();

private:
    void launch_allreduce(DDPBucket& bucket);
    void pack_bucket(DDPBucket& bucket);
    void unpack_bucket(DDPBucket& bucket);

private:
    size_t bucket_bytes_;
    cudaStream_t stream_;
    MPI_Comm comm_;
    int rank_, world_;

    std::vector<GradView> params_;
    std::vector<DDPBucket> buckets_;

    std::unordered_map<float*, DDPBucket*> grad_to_bucket_;
    std::unordered_map<DDPBucket*, size_t> ready_count_;
};
