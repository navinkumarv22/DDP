#pragma once
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

inline ncclComm_t init_nccl(int rank, int world) {
    ncclUniqueId id;
    if (rank == 0)
        ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    ncclCommInitRank(&comm, world, id, rank);
    return comm;
}
