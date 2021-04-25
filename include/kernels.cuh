#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandStatate type
#include <stdio.h>

#include "types.hpp"

#define CUDA_ASSERT(ans)                                                       \
    { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void k_setup_rng(uint rows, uint cols, curandState *globalRandState,
                            unsigned long seed);

__global__ void k_init_grid(bool *grid, uint rows, uint cols,
                            curandState *globalRandState,
                            float spawnProbability);

__global__ void k_compute_grid_count_rule(bool *grid, bool *nextGrid, uint rows,
                                          uint cols,
                                          curandState *globalRandState,
                                          float virtualSpawnProbability,
                                          bool countAliveCells,
                                          uint *activeCellCount);

__global__ void k_update_grid_buffers(bool *grid, fvec2s *gridVertices,
                                      uint cols, uint numVerticesX,
                                      fvec2 cellDensity, ulim2 gridLimX,
                                      ulim2 gridLimY);

__global__ void k_reset_grid_buffers(fvec2s *gridVertices, uint numVerticesX,
                                     uint numVerticesY);

#endif