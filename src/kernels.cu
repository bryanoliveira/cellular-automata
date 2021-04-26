#include "kernels.cuh"

////// DEVICE FUNCTIONS

__device__ inline unsigned short count_moore_neighbours(bool *grid, uint rows,
                                                        uint cols, uint idx) {
    unsigned short livingNeighbours = 0;
    // we can calculate the neighbours directly since we're using safety borders
    uint neighbours[] = {
        idx - cols - 1, // top left
        idx - cols,     // top center
        idx - cols + 1, // top right
        idx - 1,        // middle left
        idx + 1,        // middle right
        idx + cols - 1, // bottom left
        idx + cols,     // bottom center
        idx + cols + 1, // bottom right
    };
    for (unsigned short nidx = 0; nidx < 8; nidx++)
        if (grid[neighbours[nidx]])
            livingNeighbours++;

    return livingNeighbours;
}

__device__ inline bool game_of_life(bool isAlive,
                                    unsigned short livingNeighbours) {
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

__global__ void k_setup_rng(uint rows, uint cols, curandState *globalRandState,
                            unsigned long seed) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

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

__global__ void k_init_grid(bool *grid, uint rows, uint cols,
                            curandState *globalRandState,
                            float spawnProbability) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

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

__global__ void k_update_grid_buffers(bool *grid, fvec2s *gridVertices,
                                      uint cols, uint numVerticesX,
                                      uvec2 cellDensity, ulim2 gridLimX,
                                      ulim2 gridLimY) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

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
            if (grid[idx]) {
                // calculate mapping between grid and vertice
                uint vx = (x - gridLimX.start) / cellDensity.x;
                uint vy = (y - gridLimY.start) / cellDensity.y;
                uint vidx = vy * numVerticesX + vx;
                // no need to be atomic on a read
                // we check before to avoid atomic writing bottleneck
                if (gridVertices[vidx].state == 0)
                    atomicMax(&gridVertices[vidx].state, (int)grid[idx]);
            }
        }
    }
}

__global__ void k_reset_grid_buffers(fvec2s *gridVertices, uint numVerticesX,
                                     uint numVerticesY) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (uint y = blockDim.y * blockIdx.y + threadIdx.y; y < numVerticesY;
         y += stride.y) {
        for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < numVerticesX;
             x += stride.x) {
            gridVertices[y * numVerticesX + x].state = 0;
        }
    }
}

__global__ void k_compute_grid_count_rule(bool *grid, bool *nextGrid, uint rows,
                                          uint cols,
                                          curandState *globalRandState,
                                          float virtualSpawnProbability,
                                          bool countAliveCells,
                                          uint *activeCellCount) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

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
