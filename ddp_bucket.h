#pragma once
#include <mpi.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>

//////////////////////////////////////////////////////////////
// DDP Bucket
//////////////////////////////////////////////////////////////

struct GradView {
    float* data;
    size_t numel;
};

struct Bucket {
    std::vector<GradView> grads;
    size_t total_numel = 0;
    bool ready = false;
};

//////////////////////////////////////////////////////////////
// DDP Reducer (PyTorch-like)
//////////////////////////////////////////////////////////////

class DDPReducer {
public:
    DDPReducer(size_t bucket_cap_bytes,
               MPI_Comm comm = MPI_COMM_WORLD)
        : bucket_cap_bytes_(bucket_cap_bytes),
          comm_(comm) {

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &world_);
    }

    //////////////////////////////////////////////////////////
    // Register a parameter gradient
    // Call ONCE at model construction time
    //////////////////////////////////////////////////////////
    void register_param(float* grad_ptr, size_t numel) {
        params_.push_back({grad_ptr, numel});
    }

    //////////////////////////////////////////////////////////
    // Build buckets (call once after registration)
    //////////////////////////////////////////////////////////
    void finalize() {
        buckets_.clear();

        Bucket current;
        size_t current_bytes = 0;

        for (auto& p : params_) {
            size_t bytes = p.numel * sizeof(float);

            if (current_bytes + bytes > bucket_cap_bytes_
                && !current.grads.empty()) {

                buckets_.push_back(current);
                current = Bucket{};
                current_bytes = 0;
            }

            current.grads.push_back(p);
            current.total_numel += p.numel;
            current_bytes += bytes;
        }

        if (!current.grads.empty())
            buckets_.push_back(current);

        if (rank_ == 0) {
            std::cout << "[DDP] Built " << buckets_.size()
                      << " gradient buckets\n";
        }
    }

    //////////////////////////////////////////////////////////
    // Mark gradient as ready (called during backward)
    //////////////////////////////////////////////////////////
    void mark_ready(float* grad_ptr) {
        for (auto& bucket : buckets_) {
            for (auto& g : bucket.grads) {
                if (g.data == grad_ptr) {
                    ready_counts_[&bucket]++;
                    if (ready_counts_[&bucket] == bucket.grads.size()) {
                        bucket.ready = true;
                        allreduce_bucket(bucket);
                    }
                    return;
                }
            }
        }
        assert(false && "Gradient not registered");
    }

    //////////////////////////////////////////////////////////
    // Call at end of backward to flush leftovers
    //////////////////////////////////////////////////////////
    void finalize_backward() {
        for (auto& bucket : buckets_) {
            if (!bucket.ready)
                allreduce_bucket(bucket);
        }
        ready_counts_.clear();
    }

private:
    //////////////////////////////////////////////////////////
    // Allreduce a bucket (flat buffer)
    //////////////////////////////////////////////////////////
    void allreduce_bucket(Bucket& bucket) {
        if (bucket.total_numel == 0) return;

        std::vector<float> buffer(bucket.total_numel);
        size_t offset = 0;

        // pack
        for (auto& g : bucket.grads) {
            std::memcpy(buffer.data() + offset,
                        g.data,
                        g.numel * sizeof(float));
            offset += g.numel;
        }

        // allreduce
        MPI_Allreduce(MPI_IN_PLACE,
                      buffer.data(),
                      buffer.size(),
                      MPI_FLOAT,
                      MPI_SUM,
                      comm_);

        float inv_world = 1.0f / world_;
        for (auto& v : buffer)
            v *= inv_world;

        // unpack
        offset = 0;
        for (auto& g : bucket.grads) {
            std::memcpy(g.data,
                        buffer.data() + offset,
                        g.numel * sizeof(float));
            offset += g.numel;
        }

        bucket.ready = true;
    }

private:
    size_t bucket_cap_bytes_;
    MPI_Comm comm_;
    int rank_, world_;

    std::vector<GradView> params_;
    std::vector<Bucket> buckets_;
    std::unordered_map<Bucket*, size_t> ready_counts_;
};
