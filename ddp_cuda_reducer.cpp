#include "ddp_cuda_reducer.h"

//////////////////////////////////////////////////////////////
// Constructor / Destructor
//////////////////////////////////////////////////////////////

DDPCUDAReducer::DDPCUDAReducer(size_t bucket_bytes,
                               cudaStream_t stream,
                               MPI_Comm comm)
    : bucket_bytes_(bucket_bytes),
      stream_(stream),
      comm_(comm) {

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_);
}

DDPCUDAReducer::~DDPCUDAReducer() {
    for (auto& b : buckets_) {
        if (b.buffer)
            cudaFree(b.buffer);
    }
}

//////////////////////////////////////////////////////////////
// Registration
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::register_param(float* grad_ptr, size_t numel) {
    params_.push_back({grad_ptr, numel});
}

//////////////////////////////////////////////////////////////
// Build buckets
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::finalize() {
    buckets_.clear();

    DDPBucket current;
    size_t current_bytes = 0;

    for (auto& p : params_) {
        size_t bytes = p.numel * sizeof(float);

        if (current_bytes + bytes > bucket_bytes_ &&
            !current.grads.empty()) {

            buckets_.push_back(current);
            current = DDPBucket{};
            current_bytes = 0;
        }

        current.grads.push_back(p);
        current.total_numel += p.numel;
        current_bytes += bytes;
    }

    if (!current.grads.empty())
        buckets_.push_back(current);

    // Allocate buffers + map grads
    for (auto& b : buckets_) {
        CUDA_CHECK(cudaMalloc(&b.buffer, b.total_numel * sizeof(float)));
        for (auto& g : b.grads)
            grad_to_bucket_[g.data] = &b;
    }

    if (rank_ == 0) {
        std::cout << "[DDP CUDA] Built " << buckets_.size()
                  << " buckets\n";
    }
}

//////////////////////////////////////////////////////////////
// Gradient ready hook
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::mark_ready(float* grad_ptr) {
    auto it = grad_to_bucket_.find(grad_ptr);
    assert(it != grad_to_bucket_.end());

    DDPBucket* bucket = it->second;
    ready_count_[bucket]++;

    if (ready_count_[bucket] == bucket->grads.size()) {
        launch_allreduce(*bucket);
    }
}

//////////////////////////////////////////////////////////////
// Pack gradients → flat buffer
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::pack_bucket(DDPBucket& bucket) {
    size_t offset = 0;
    for (auto& g : bucket.grads) {
        CUDA_CHECK(cudaMemcpyAsync(
            bucket.buffer + offset,
            g.data,
            g.numel * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_));
        offset += g.numel;
    }
}

//////////////////////////////////////////////////////////////
// Unpack buffer → gradients
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::unpack_bucket(DDPBucket& bucket) {
    size_t offset = 0;
    for (auto& g : bucket.grads) {
        CUDA_CHECK(cudaMemcpyAsync(
            g.data,
            bucket.buffer + offset,
            g.numel * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_));
        offset += g.numel;
    }
}

//////////////////////////////////////////////////////////////
// Launch async allreduce
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::launch_allreduce(DDPBucket& bucket) {
    if (bucket.reduced) return;

    pack_bucket(bucket);

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    MPI_Iallreduce(MPI_IN_PLACE,
                   bucket.buffer,
                   bucket.total_numel,
                   MPI_FLOAT,
                   MPI_SUM,
                   comm_,
                   &bucket.request);

    bucket.reduced = true;
}

//////////////////////////////////////////////////////////////
// Finalize backward
//////////////////////////////////////////////////////////////

void DDPCUDAReducer::finalize_backward() {
    for (auto& b : buckets_) {
        if (!b.reduced)
            launch_allreduce(b);
    }

    // Wait all
    for (auto& b : buckets_) {
        MPI_Wait(&b.request, MPI_STATUS_IGNORE);

        // average
        float scale = 1.0f / world_;
        int threads = 256;
        int blocks = (b.total_numel + threads - 1) / threads;

        // simple CUDA kernel inline
        auto kernel = [] __global__ (float* x, size_t n, float s) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) x[i] *= s;
        };

        kernel<<<blocks, threads, 0, stream_>>>(b.buffer, b.total_numel, scale);

        unpack_bucket(b);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    ready_count_.clear();
}
