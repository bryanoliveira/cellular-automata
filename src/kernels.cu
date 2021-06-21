#ifndef NH_RADIUS
#define NH_RADIUS 1
#endif

#include "kernels.cuh"

////// DEVICE FUNCTIONS

__device__ inline ushort count_nh(const GridType *const grid, const uint cols,
                                  const uint idx) {
    /** Counts immediate active Moore neighbours **/

    return static_cast<ushort>(
        // it's faster to calculate them directly
        grid[idx - cols - 1] + // top left
        grid[idx - cols] +     // top center
        grid[idx - cols + 1] + // top right
        grid[idx - 1] +        // middle left
        grid[idx + 1] +        // middle right
        grid[idx + cols - 1] + // bottom left
        grid[idx + cols] +     // bottom center
        grid[idx + cols + 1]   // bottom right
    );
}

__device__ inline ushort count_radius_neighbours(const GridType *const grid,
                                                 const uint rows,
                                                 const uint cols, const uint x,
                                                 const uint y,
                                                 const int radius = 1) {
    /** Counts active neighbours given radius **/
    ushort livingNeighbours = 0;

#pragma unroll
    for (int ny = -radius; ny <= radius; ++ny) {
#pragma unroll
        for (int nx = -radius; nx <= radius; ++nx) {
            // calculate neighbour position
            const int px = static_cast<int>(x) + nx;
            const int py = static_cast<int>(y) + ny;

            livingNeighbours +=
                static_cast<ushort>((
                                        // if pos is not current cell
                                        (ny != 0 || nx != 0) &&
                                        // and is valid in x
                                        0 < px && px < cols - 1 &&
                                        // and is valid in y
                                        0 < py && py < rows - 1) &&
                                    // sum its activity
                                    grid[py * cols + px]);
        }
    }

    return livingNeighbours;
}

__device__ inline GridType game_of_life(const GridType state,
                                        const ushort livingNeighbours) {
    // 1. Any live cell with two or three live neighbours survives.
    // 2. Any dead cell with three live neighbours becomes a live cell.
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    return (state && livingNeighbours == 2) || livingNeighbours == 3;
}

__device__ inline GridType msgol(const GridType state,
                                 const ushort livingNeighbours,
                                 const ubyte nStates) {
    if (livingNeighbours == 3 && state < nStates - 1)
        // cell would be born
        return state + 1;
    else if (livingNeighbours != 2 && state != 0)
        // cell would die
        return state - 1;
    // cell would continue living
    return state;
}

__device__ inline GridType msgol4(const GridType state,
                                  const ushort livingNeighbours,
                                  const ubyte nStates) {
    if (livingNeighbours == 3) {
        // cell would be born
        if (state < nStates - 1)
            return state + 1;
        // increasing energy of maxed cell kills it (except GoL)
        else if (state != 1)
            return 0;
    } else if (livingNeighbours < 2) {
        if (state > 0)
            return state - 1;
    } else if (livingNeighbours > 3)
        return 0;

    return state;
}

////// CUDA KERNELS

__global__ void k_setup_rng(const uint rows, const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const ulong seed) {
    const uint stride = gridDim.x * blockDim.x, idxMax = (rows - 1) * cols - 1,
               xMax = cols - NH_RADIUS;

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride) {
        const uint x = idx % cols;
        if (idx > cols + 1 && NH_RADIUS < x && x < xMax)
            curand_init(seed, idx, 0, &globalRandState[idx]);
    }
}

__global__ void k_init_grid(GridType *const grid, const uint rows,
                            const uint cols,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability) {
    const uint stride = gridDim.x * blockDim.x, idxMax = (rows - 1) * cols - 1,
               xMax = cols - NH_RADIUS;

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride) {
        const uint x = idx % cols;
        grid[idx] = (idx > cols + 1 && NH_RADIUS < x && x < xMax) &&
                    curand_uniform(&globalRandState[idx]) < spawnProbability;
    }
}

__global__ void k_update_grid_buffers(const GridType *const grid,
                                      fvec2s *const __restrict__ gridVertices,
                                      const uint cols, const uint numVerticesX,
                                      const uvec2 cellDensity,
                                      const ulim2 gridLimX,
                                      const ulim2 gridLimY) {
    const uint stride = gridDim.x * blockDim.x,
               idxMin = gridLimY.start * cols + gridLimX.start,
               idxMax = (gridLimY.end - 1) * cols + gridLimX.end,
               xMin = max(NH_RADIUS, gridLimX.start),
               xMax = min(cols - NH_RADIUS, gridLimX.end);

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride) {
        // to check if out of bounds and to map vertices
        const uint x = idx % cols, y = idx / cols;

        // try avoiding further operations when not needed
        // atomicMax is pretty expensive
        if (idxMin <= idx && xMin < x && x < xMax && grid[idx]) {
            // calculate mapping between grid and vertice
            const uint vx = (x - gridLimX.start) / cellDensity.x,
                       vy = (y - gridLimY.start) / cellDensity.y,
                       vidx = vy * numVerticesX + vx;

            // no need to be atomic on a read
            // we check before to avoid atomic writing bottleneck
            // gridVertices[vidx].state = 1;
            // gridVertices[vidx].state =
            //     max(gridVertices[vidx].state, static_cast<int>(grid[idx]));
            if (gridVertices[vidx].state == 0)
                atomicMax(&gridVertices[vidx].state,
                          static_cast<int>(grid[idx]));
        }
    }
}

__global__ void k_reset_grid_buffers(fvec2s *const __restrict__ gridVertices,
                                     const uint numVerticesX,
                                     const uint numVerticesY) {
    const uint stride = gridDim.x * blockDim.x,
               idxMax = numVerticesX * numVerticesY;

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride) {
        gridVertices[idx].state = 0;
    }
}

__global__ void
k_evolve_count_rule(const GridType *const grid, GridType *const nextGrid,
                    const uint rows, const uint cols,
                    curandState *const __restrict__ globalRandState,
                    const float virtualSpawnProbability,
                    const bool countAliveCells, uint *const activeCellCount) {
    const uint stride = gridDim.x * blockDim.x, idxMax = (rows - 1) * cols - 1,
               xMax = cols - NH_RADIUS;

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride) {
        const uint x = idx % cols;
        GridType newState = false;

        newState = idx > cols + 1 && NH_RADIUS < x && x < xMax &&
                   // add a "virtual particle" spawn probability
                   ((virtualSpawnProbability > 0 &&
                     curand_uniform(&globalRandState[idx]) <
                         virtualSpawnProbability) ||
                    // compute cell rule
                    game_of_life(grid[idx], count_nh(grid, cols, idx)));

        // newState = game_of_life(
        //     grid[idx],
        //     count_radius_neighbours(grid, rows, cols, x, y, 3));

        nextGrid[idx] = newState;

        // avoid atomicAdd when not necessary
        if (countAliveCells)
            atomicAdd(activeCellCount, newState);
    }
}
