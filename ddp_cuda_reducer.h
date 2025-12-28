  #pragma once
#include <mpi.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_map>
#include <cassert>
#include <cstring>
#include <iostream>

//////////////////////////////////////////////////////////////
// CUDA CHECK
//////////////////////////////////////////////////////////////

#define CUDA_CHECK(x) do {                            \
    cudaError_t err = x;                              \
    if (err != cudaSuccess) {                         \
        printf("CUDA error %s:%d: %s\n",              \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                      \
    }                                                 \
} while(0)

//////////////////////////////////////////////////////////////
// Gradient View
//////////////////////////////////////////////////////////////

struct GradView {
    float* data;        // CUDA pointer
    size_t numel;
};

//////////////////////////////////////////////////////////////
// Bucket
//////////////////////////////////////////////////////////////

struct Bucket {
    std::vector<GradView> grads;
    size_t total_numel = 0;

    float* buffer = nullptr;   // CUDA flat buffer
    MPI_Request request;
    bool launched = false;
};

//////////////////////////////////////////////////////////////
// CUDA DDP Reducer
//////////////////////////////////////////////////////////////

class DDPReducerCUDA {
public:
    DDPReducerCUDA(size_t bucket_cap_bytes,
                   MPI_Comm comm = MPI_COMM_WORLD)
        : bucket_cap_bytes_(bucket_cap_bytes),
          comm_(comm) {

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_);

        CUDA_CHECK(cudaStreamCreate(&comm_stream_));
    }

    ~DDPReducerCUDA() {
        for (auto& b : buckets_) {
            if (b.buffer)
                CUDA_CHECK(cudaFree(b.buffer));
        }
        CUDA_CHECK(cudaStreamDestroy(comm_stream_));
    }

    //////////////////////////////////////////////////////////
    // Register parameter gradients (call once)
    //////////////////////////////////////////////////////////
    void register_param(float* grad_ptr, size_t numel) {
        params_.push_back({grad_ptr, numel});
    }

    //////////////////////////////////////////////////////////
    // Build buckets (call once)
    //////////////////////////////////////////////////////////
    void finalize() {
        buckets_.clear();

        Bucket cur;
        size_t bytes = 0;

        for (auto& p : params_) {
            size_t pbytes = p.numel * sizeof(float);

            if (bytes + pbytes > bucket_cap_bytes_ &&
                !cur.grads.empty()) {
                allocate_bucket(cur);
                buckets_.push_back(cur);
                cur = Bucket{};
                bytes = 0;
            }

            cur.grads.push_back(p);
            cur.total_numel += p.numel;
            bytes += pbytes;
        }

        if (!cur.grads.empty()) {
            allocate_bucket(cur);
            buckets_.push_back(cur);
        }

        if (rank_ == 0)
            std::cout << "[DDP] CUDA Buckets: "
                      << buckets_.size() << std::endl;
    }

    //////////////////////////////////////////////////////////
    // Mark gradient ready (called during backward)
    //////////////////////////////////////////////////////////
    void mark_ready(float* grad_ptr) {
        for (auto& b : buckets_) {
            for (auto& g : b.grads) {
                if (g.data == grad_ptr) {
                    ready_count_[&b]++;
                    if (ready_count_[&b] == b.grads.size()) {
                        launch_bucket(b);
                    }
                    return;
                }
            }
        }
        assert(false && "Unregistered gradient");
    }

    //////////////////////////////////////////////////////////
    // End of backward: wait + unpack
    //////////////////////////////////////////////////////////
    void finalize_backward() {
        for (auto& b : buckets_) {
            if (!b.launched)
                launch_bucket(b);

            MPI_Wait(&b.request, MPI_STATUS_IGNORE);
            unpack_bucket(b);
            b.launched = false;
        }
        ready_count_.clear();
    }

private:
    //////////////////////////////////////////////////////////
    // Allocate CUDA buffer
    //////////////////////////////////////////////////////////
    void allocate_bucket(Bucket& b) {
        CUDA_CHECK(cudaMalloc(&b.buffer,
                   b.total_numel * sizeof(float)));
    }

    //////////////////////////////////////////////////////////
    // Pack + async allreduce
    //////////////////////////////////////////////////////////
    void launch_bucket(Bucket& b) {
        if (b.launched) return;

        // pack
        size_t offset = 0;
        for (auto& g : b.grads) {
            CUDA_CHECK(cudaMemcpyAsync(
                b.buffer + offset,
                g.data,
                g.numel * sizeof(float),
                cudaMemcpyDeviceToDevice,
                comm_stream_));
            offset += g.numel;
        }

        CUDA_CHECK(cudaStreamSynchronize(comm_stream_));

        MPI_Iallreduce(
            MPI_IN_PLACE,
            b.buffer,
            b.total_numel,
            MPI_FLOAT,
            MPI_SUM,
            comm_,
            &b.request);

        b.launched = true;
    }

    //////////////////////////////////////////////////////////
    // Unpack + average
    //////////////////////////////////////////////////////////
    void unpack_bucket(Bucket& b) {
        float inv = 1.0f / world_;
        size_t offset = 0;

        for (auto& g : b.grads) {
            scale_and_copy(
                g.data,
                b.buffer + offset,
                g.numel,
                inv);
            offset += g.numel;
        }
    }

    //////////////////////////////////////////////////////////
    // CUDA kernel: scale + copy
    //////////////////////////////////////////////////////////
    static __global__ void scale_copy_kernel(
        float* dst, const float* src,
        size_t n, float scale) {

        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = src[i] * scale;
    }

    void scale_and_copy(
        float* dst, const float* src,
        size_t n, float scale) {

        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scale_copy_kernel<<<blocks, threads, 0, comm_stream_>>>(
            dst, src, n, scale);
    }

private:
    size_t bucket_cap_bytes_;
    MPI_Comm comm_;
    int rank_, world_;

    cudaStream_t comm_stream_;

    std::vector<GradView> params_;
    std::vector<Bucket> buckets_;
    std::unordered_map<Bucket*, size_t> ready_count_;
};
