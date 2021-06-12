#include "kernels.cuh"

////// DEVICE FUNCTIONS

__device__ inline unsigned short count_moore_neighbours(const bool *const grid,
                                                        const uint rows,
                                                        const uint cols,
                                                        const uint idx) {
    unsigned short livingNeighbours = 0;
    // we can calculate the neighbours directly since we're using safety borders
    const uint neighbours[] = {
        idx - cols - 1, // top left
        idx - cols,     // top center
        idx - cols + 1, // top right
        idx - 1,        // middle left
        idx + 1,        // middle right
        idx + cols - 1, // bottom left
        idx + cols,     // bottom center
        idx + cols + 1, // bottom right
    };

#pragma unroll
    for (unsigned short nidx = 0; nidx < 8; ++nidx)
        // grid is currently bool, so summing true values is ok
        livingNeighbours += static_cast<unsigned short>(grid[neighbours[nidx]]);

    return livingNeighbours;
}

__device__ inline bool game_of_life(const bool isAlive,
                                    const unsigned short livingNeighbours) {
    // 1. Any live cell with two or three live neighbours survives.
    // 2. Any dead cell with three live neighbours becomes a live cell.
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    return (isAlive && livingNeighbours == 2) || livingNeighbours == 3;
}

////// CUDA KERNELS

__global__ void k_setup_rng(const uint rows, const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const unsigned long seed) {
    const dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    // note: we're using safety borders
    for (uint y = blockDim.y * blockIdx.y + threadIdx.y + 1; y < rows - 1;
         y += stride.y) {
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x + 1; x < cols - 1;
             x += stride.x) {
            uint idx = y * cols + x;
            curand_init(seed, idx, 0, &globalRandState[idx]);
        }
    }
}

__global__ void k_init_grid(bool *const grid, const uint rows, const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability) {
    const dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    // note: we're using safety borders
    for (uint y = blockDim.y * blockIdx.y + threadIdx.y + 1; y < rows - 1;
         y += stride.y) {
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x + 1; x < cols - 1;
             x += stride.x) {
            uint idx = y * cols + x;
            grid[idx] =
                curand_uniform(&globalRandState[idx]) < spawnProbability;
        }
    }
}

__global__ void k_update_grid_buffers(const bool *const grid,
                                      fvec2s *const __restrict__ gridVertices,
                                      const uint cols, const uint numVerticesX,
                                      const uvec2 cellDensity,
                                      const ulim2 gridLimX,
                                      const ulim2 gridLimY) {
    const dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    // no need to use safety borders here
    for (uint y = blockDim.y * blockIdx.y + threadIdx.y; y < gridLimY.end;
         y += stride.y) {
        // check if out of bounds
        if (y < gridLimY.start)
            continue;
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < gridLimX.end;
             x += stride.x) {
            // check if out of bounds
            if (x < gridLimX.start)
                continue;
            uint idx = y * cols + x;
            // try avoiding further operations when not needed
            // atomicMax is pretty expensive
            if (grid[idx]) {
                // calculate mapping between grid and vertice
                uint vx = (x - gridLimX.start) / cellDensity.x;
                uint vy = (y - gridLimY.start) / cellDensity.y;
                uint vidx = vy * numVerticesX + vx;
                // no need to be atomic on a read
                // we check before to avoid atomic writing bottleneck
                if (gridVertices[vidx].state == 0)
                    atomicMax(&gridVertices[vidx].state,
                              static_cast<int>(grid[idx]));
            }
        }
    }
}

__global__ void k_reset_grid_buffers(fvec2s *const __restrict__ gridVertices,
                                     const uint numVerticesX,
                                     const uint numVerticesY) {
    const dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (uint y = blockDim.y * blockIdx.y + threadIdx.y; y < numVerticesY;
         y += stride.y) {
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < numVerticesX;
             x += stride.x) {
            gridVertices[y * numVerticesX + x].state = 0;
        }
    }
}

__global__ void k_compute_grid_count_rule(
    const bool *const grid, bool *const nextGrid, const uint rows,
    const uint cols, curandState *const __restrict__ globalRandState,
    const float virtualSpawnProbability, const bool countAliveCells,
    uint *const activeCellCount) {
    const dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    // note: we're using safety borders
    for (uint y = blockDim.y * blockIdx.y + threadIdx.y + 1; y < rows - 1;
         y += stride.y) {
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x + 1; x < cols - 1;
             x += stride.x) {
            uint idx = y * cols + x;
            bool newState = false;

            // 0. Virtual particle spawn probability
            if (virtualSpawnProbability > 0 &&
                curand_uniform(&globalRandState[idx]) < virtualSpawnProbability)
                newState = true;
            else
                newState = game_of_life(
                    grid[idx], count_moore_neighbours(grid, rows, cols, idx));

            // avoid atomicAdd when not necessary
            if (countAliveCells && newState)
                atomicAdd(activeCellCount, 1);

            nextGrid[idx] = newState;
        }
    }
}
