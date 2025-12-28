#include "ddp_nccl.h"

//////////////////////////////////////////////////////////////
// Gradient View
//////////////////////////////////////////////////////////////

struct GradView {
    float* ptr;
    size_t numel;
};

//////////////////////////////////////////////////////////////
// Bucket
//////////////////////////////////////////////////////////////

struct Bucket {
    std::vector<GradView> grads;
    size_t total_numel = 0;

    float* buffer = nullptr;      // CUDA flat buffer
    cudaStream_t stream;
    bool reduced = false;
};

//////////////////////////////////////////////////////////////
// NCCL DDP Reducer
//////////////////////////////////////////////////////////////

class NCCLReducer {
public:
    NCCLReducer(size_t bucket_bytes,
                int rank,
                int world,
                ncclComm_t comm,
                cudaStream_t stream)
        : bucket_bytes_(bucket_bytes),
          rank_(rank),
          world_(world),
          comm_(comm),
          stream_(stream) {}

    //////////////////////////////////////////////////////////
    // Register parameter gradient
    //////////////////////////////////////////////////////////
    void register_param(float* grad_ptr, size_t numel) {
        params_.push_back({grad_ptr, numel});
    }

    //////////////////////////////////////////////////////////
    // Build buckets (call once)
    //////////////////////////////////////////////////////////
    void finalize() {
        size_t current_bytes = 0;
        Bucket current;

        for (auto& p : params_) {
            size_t bytes = p.numel * sizeof(float);

            if (current_bytes + bytes > bucket_bytes_ &&
                !current.grads.empty()) {
                allocate_bucket(current);
                buckets_.push_back(current);
                current = Bucket{};
                current_bytes = 0;
            }

            current.grads.push_back(p);
            current.total_numel += p.numel;
            current_bytes += bytes;
        }

        if (!current.grads.empty()) {
            allocate_bucket(current);
            buckets_.push_back(current);
        }

        if (rank_ == 0)
            std::cout << "[DDP/NCCL] Buckets built: "
                      << buckets_.size() << std::endl;
    }

    //////////////////////////////////////////////////////////
    // Mark gradient ready (called from backward)
    //////////////////////////////////////////////////////////
    void mark_ready(float* grad_ptr) {
        for (auto& bucket : buckets_) {
            if (bucket.reduced) continue;

            for (auto& g : bucket.grads) {
                if (g.ptr == grad_ptr) {
                    ready_count_[&bucket]++;
                    if (ready_count_[&bucket] == bucket.grads.size()) {
                        reduce_bucket(bucket);
                    }
                    return;
                }
            }
        }
    }

    //////////////////////////////////////////////////////////
    // Finalize backward (flush leftovers)
    //////////////////////////////////////////////////////////
    void finalize_backward() {
        for (auto& bucket : buckets_) {
            if (!bucket.reduced)
                reduce_bucket(bucket);
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
        b.stream = stream_;
    }

    //////////////////////////////////////////////////////////
    // Reduce bucket (pack → allreduce → unpack)
    //////////////////////////////////////////////////////////
    void reduce_bucket(Bucket& b) {
        size_t offset = 0;

        // pack
        for (auto& g : b.grads) {
            CUDA_CHECK(cudaMemcpyAsync(
                b.buffer + offset,
                g.ptr,
                g.numel * sizeof(float),
                cudaMemcpyDeviceToDevice,
                b.stream));
            offset += g.numel;
        }

        // allreduce
        NCCL_CHECK(ncclAllReduce(
            b.buffer,
            b.buffer,
            b.total_numel,
            ncclFloat,
            ncclSum,
            comm_,
            b.stream));

        // scale
        float scale = 1.0f / world_;
        scale_kernel<<<(b.total_numel+255)/256,256,0,b.stream>>>(
            b.buffer, b.total_numel, scale);

        // unpack
        offset = 0;
        for (auto& g : b.grads) {
            CUDA_CHECK(cudaMemcpyAsync(
                g.ptr,
                b.buffer + offset,
                g.numel * sizeof(float),
                cudaMemcpyDeviceToDevice,
                b.stream));
            offset += g.numel;
        }

        b.reduced = true;
    }

private:
    size_t bucket_bytes_;
    int rank_, world_;
    ncclComm_t comm_;
    cudaStream_t stream_;

    std::vector<GradView> params_;
    std::vector<Bucket> buckets_;
    std::unordered_map<Bucket*, size_t> ready_count_;

    //////////////////////////////////////////////////////////
    // CUDA scale kernel
    //////////////////////////////////////////////////////////
    static __global__ void scale_kernel(float* x, size_t n, float s) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] *= s;
    }
};
