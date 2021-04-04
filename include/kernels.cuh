#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandStatate type

#include "display.hpp" // for the vec2s type

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

__global__ void k_setup_rng(unsigned int rows, unsigned int cols,
                            curandState *globalRandState, unsigned long seed);

__global__ void k_init_grid(bool *grid, unsigned int rows, unsigned int cols,
                            curandState *globalRandState,
                            float spawnProbability);

__global__ void k_compute_grid_count_rule(bool *grid, bool *nextGrid,
                                          unsigned int rows, unsigned int cols,
                                          curandState *globalRandState,
                                          float virtualSpawnProbability,
                                          unsigned int *activeCellCount);

__global__ void k_update_grid_buffers(bool *grid, vec2s *gridVertices,
                                      unsigned int rows, unsigned int cols);

#endif