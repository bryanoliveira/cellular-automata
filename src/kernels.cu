// isolate cuda specific imports
#include <curand_kernel.h>

#include "kernels.cuh"

////// DEVICE FUNCTIONS

__device__ unsigned short count_neighbours(bool *grid, unsigned int rows,
                                           unsigned int cols, unsigned int x,
                                           unsigned int y) {
    unsigned short livingNeighbours = 0;
    // iterate over neighbourhood row by row
    for (short a = y - 1; a <= y + 1; a++) {
        // skip out-of-grid neighbours
        if (a < 0 || a >= rows)
            continue;
        // iterate over neighbourhood col by col
        for (short b = x - 1; b <= x + 1; b++) {
            // skip out-of-grid neighbours and self position
            if (b < 0 || b >= cols || (a == y && b == x))
                continue; // cols
            // if cell is alive, increase neighbours
            if (grid[a * cols + b])
                livingNeighbours++;
        }
    }
    return livingNeighbours;
}

__device__ bool game_of_life(bool isAlive, unsigned short livingNeighbours) {
    // 1. Any live cell with two or three live neighbours survives.
    if (isAlive)
        return livingNeighbours == 2 || livingNeighbours == 3;
    // 2. Any dead cell with three live neighbours becomes a live cell.
    else if (livingNeighbours == 3)
        return true;
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    else
        return false;
}

////// CUDA KERNELS

__global__ void k_setup_rng(unsigned int rows, unsigned int cols,
                            curandState *globalRandState, unsigned long seed) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            curand_init(seed, idx, 0, &globalRandState[idx]);
        }
    }
}

__global__ void k_init_grid(bool *grid, unsigned int rows, unsigned int cols,
                            curandState *globalRandState,
                            float spawnProbability) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            grid[idx] =
                curand_uniform(&globalRandState[idx]) < spawnProbability;
        }
    }
}

__global__ void k_update_grid_buffers(bool *grid, vec2s *gridVertices,
                                      unsigned int rows, unsigned int cols) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            gridVertices[idx].state = (float)grid[idx];
        }
    }
}

__global__ void k_compute_grid(bool *grid, bool *nextGrid, unsigned int rows,
                               unsigned int cols, curandState *globalRandState,
                               float virtualSpawnProbability,
                               unsigned int *activeCellCount) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (unsigned int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (unsigned int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            unsigned int idx = y * cols + x;
            bool newState = false;

            // 0. Virtual particle spawn probability
            if (virtualSpawnProbability > 0 &&
                curand_uniform(&globalRandState[idx]) < virtualSpawnProbability)
                newState = true;
            else
                newState = game_of_life(
                    grid[idx], count_neighbours(grid, rows, cols, x, y));

            if (newState)
                atomicAdd(activeCellCount, 1);
            nextGrid[idx] = newState;
        }
    }
}
