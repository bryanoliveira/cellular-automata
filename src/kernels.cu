#include "commons_gpu.cuh"
#include "definitions.hpp"
#include "kernels.cuh"

__global__ void k_setup_rng(const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const uint seed) {
    // idxMin = thread ID + skip safety rows + skip safety cols
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        (dims.x * NH_RADIUS) + NH_RADIUS,
               idxMax = (dims.y - NH_RADIUS) * dims.x - NH_RADIUS,
               xMax = dims.x - NH_RADIUS;

    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x;
        if (NH_RADIUS < x && x < xMax)
            curand_init(seed, idx, 0, &globalRandState[idx]);
    }
}

__global__ void k_init_grid(GridType *const grid, const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability) {
    // idxMin = thread ID + skip safety rows + skip safety cols
    // idxMax = y - 1 full rows + last row cols OR max id given radius
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        (dims.x * NH_RADIUS) + NH_RADIUS,
               idxMax = (dims.y - NH_RADIUS) * dims.x - NH_RADIUS,
               xMax = dims.x - NH_RADIUS;

    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x;
        grid[idx] = (NH_RADIUS < x) * (x < xMax) *
                    (curand_uniform(&globalRandState[idx]) < spawnProbability);
    }
}

__global__ void k_reset_grid_buffers(fvec2s *const __restrict__ gridVertices,
                                     const uvec2 numVertices) {
    const uint stride = gridDim.x * blockDim.x,
               idxMax = numVertices.x * numVertices.y;

    for (uint idx = blockDim.x * blockIdx.x + threadIdx.x; idx < idxMax;
         idx += stride)
        gridVertices[idx].state = 0;
}

__global__ void
k_update_grid_buffers(const GridType *const grid, const uvec2 dims,
                      fvec2s *const __restrict__ gridVertices,
                      const uint numVerticesX, const uvec2 cellDensity,
                      const ulim2 gridLimX, const ulim2 gridLimY) {
    // idxMin = thread ID + render margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        max(gridLimY.start * dims.x + gridLimX.start,
                            (dims.x * NH_RADIUS) + NH_RADIUS),
               idxMax = min((gridLimY.end - 1) * dims.x + gridLimX.end,
                            (dims.y - NH_RADIUS) * dims.x - NH_RADIUS),
               xMin = max(NH_RADIUS, gridLimX.start),
               xMax = min(dims.x - NH_RADIUS, gridLimX.end);

    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        // to check if out of bounds and to map vertices
        const uint x = idx % dims.x, y = idx / dims.x;

        // try avoiding further operations when not needed
        // atomicMax is pretty expensive
        if (xMin <= x && x < xMax && grid[idx]) {
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

__global__ void
k_evolve_count_rule(const GridType *const grid, GridType *const nextGrid,
                    const uvec2 dims,
                    curandState *const __restrict__ globalRandState,
                    const float virtualSpawnProbability,
                    const bool countAliveCells, uint *const activeCellCount) {
    // idxMin = thread ID + skip safety rows + skip safety cols
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        (dims.x * NH_RADIUS) + NH_RADIUS,
               idxMax = (dims.y - NH_RADIUS) * dims.x - NH_RADIUS;

    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x;
        // if col is 0 or dims.x-1, given MxN grid and NH_RADIUS=1
        if (x < NH_RADIUS || dims.x - NH_RADIUS <= x)
            continue;

        // check safety borders & cell state
        nextGrid[idx] = game_of_life(grid[idx], count_nh(grid, dims.x, idx));

        // add a "virtual particle" spawn probability
        // note: this branching does not cause significant perf. hit
        if (virtualSpawnProbability > 0 && nextGrid[idx] == 0)
            nextGrid[idx] =
                curand_uniform(&globalRandState[idx]) < virtualSpawnProbability;

        // avoid atomicAdd when not necessary
        if (countAliveCells)
            atomicAdd(activeCellCount, static_cast<uint>(nextGrid[idx] > 0));
    }
}

__global__ void k_evolve_shmem(const GridType *const grid,
                               GridType *const nextGrid, const uvec2 dims,
                               curandState *const __restrict__ globalRandState,
                               const float virtualSpawnProbability,
                               const bool countAliveCells,
                               uint *const activeCellCount) {

    extern __shared__ GridType shgrid[];

    // idxMin = thread ID + skip safety rows + skip safety cols
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        (dims.x * NH_RADIUS) + NH_RADIUS,
               idxMax = (dims.y - NH_RADIUS) * dims.x - NH_RADIUS;

    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        // shared memory idx -
        const int shidx = (int(idx) - int(idxMin)) % stride;
        shgrid[shidx] = grid[idx];

        const uint x = idx % dims.x;
        // if col is 0 or dims.x-1, given MxN grid and NH_RADIUS=1
        if (!(x < NH_RADIUS || dims.x - NH_RADIUS <= x)) {

            const int inhs[] = {
                shidx - int(dims.x) - 1, // [* - -] top left
                shidx - int(dims.x),     // [- * -] top center
                shidx - int(dims.x) + 1, // [- - *] top right
                shidx - 1,               // [* - -] middle left
                shidx + 1,               // [- - *] middle right
                shidx + int(dims.x) - 1, // [* - -] bottom left
                shidx + int(dims.x),     // [- * -] bottom center
                shidx + int(dims.x) + 1  // [- - *] bottom right
            };
            ushort nhs = 0;
            for (auto inh : inhs) {
                if (inh >= 0 && inh < 32768)
                    nhs += shgrid[inh];
            }

            // check safety borders & cell state
            nextGrid[idx] = game_of_life(shgrid[shidx], nhs);

            // add a "virtual particle" spawn probability
            // note: this branching does not cause significant perf. hit
            if (virtualSpawnProbability > 0 && nextGrid[idx] == 0)
                nextGrid[idx] = curand_uniform(&globalRandState[idx]) <
                                virtualSpawnProbability;

            // avoid atomicAdd when not necessary
            if (countAliveCells)
                atomicAdd(activeCellCount,
                          static_cast<uint>(nextGrid[idx] > 0));
        }

        __syncthreads();
    }
}