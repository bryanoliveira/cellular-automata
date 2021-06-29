#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandState type
#include <stdio.h>

#include "grid.hpp"  // GridType
#include "types.hpp" // uint, ulim, etc

#define CUDA_ASSERT(ans)                                                       \
    { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *const file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void k_setup_rng(const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const uint seed);

__global__ void k_init_grid(GridType *const grid, const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability);

__global__ void k_reset_grid_buffers(fvec2s *const __restrict__ gridVertices,
                                     const uvec2 numVertices);

__global__ void
k_update_grid_buffers(const GridType *const grid, const uvec2 dims,
                      fvec2s *const __restrict__ gridVertices,
                      const uint numVerticesX, const uvec2 cellDensity,
                      const ulim2 gridLimX, const ulim2 gridLimY);

__global__ void
k_evolve_count_rule(const GridType *const grid, GridType *const nextGrid,
                    const uvec2 dims,
                    curandState *const __restrict__ globalRandState,
                    const float virtualSpawnProbability,
                    const bool countAliveCells, uint *const activeCellCount);

__global__ void k_evolve_shmem(const GridType *const grid,
                               GridType *const nextGrid, const uvec2 dims,
                               curandState *const __restrict__ globalRandState,
                               const float virtualSpawnProbability,
                               const bool countAliveCells,
                               uint *const activeCellCount);

#endif