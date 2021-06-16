#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandState type
#include <stdio.h>

#include "types.hpp"

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

__global__ void k_setup_rng(const uint rows, const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const unsigned long seed);

__global__ void k_init_grid(bool *const grid, const uint rows, const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability);

__global__ void k_update_grid_buffers(const bool *const grid,
                                      fvec2s *const __restrict__ gridVertices,
                                      const uint cols, const uint numVerticesX,
                                      const uvec2 cellDensity,
                                      const ulim2 gridLimX,
                                      const ulim2 gridLimY);

__global__ void k_reset_grid_buffers(fvec2s *const __restrict__ gridVertices,
                                     const uint numVerticesX,
                                     const uint numVerticesY);

__global__ void
k_evolve_count_rule(const bool *const grid, bool *const nextGrid,
                    const uint rows, const uint cols,
                    curandState *const __restrict__ globalRandState,
                    const float virtualSpawnProbability,
                    const bool countAliveCells, uint *const activeCellCount);

#endif