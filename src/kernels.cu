#include "commons_gpu.cuh"
#include "definitions.hpp"
#include "kernels.cuh"

__global__ void k_setup_rng(const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const uint seed) {
    // idMin = thread ID + safety border margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idMin =
                   blockDim.x * blockIdx.x + threadIdx.x + dims.x + NH_RADIUS,
               idxMax = min((dims.y - 1) * dims.x - NH_RADIUS,
                            (dims.y - NH_RADIUS) * dims.x - NH_RADIUS),
               xMax = dims.x - NH_RADIUS;

    for (uint idx = idMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x;
        if (NH_RADIUS < x && x < xMax)
            curand_init(seed, idx, 0, &globalRandState[idx]);
    }
}

__global__ void k_init_grid(GridType *const grid, const uvec2 dims,
                            curandState *const __restrict__ globalRandState,
                            const float spawnProbability) {
    // idMin = thread ID + safety border margin
    // idxMax = y - 1 full rows + last row cols OR max id given radius
    const uint stride = gridDim.x * blockDim.x,
               idMin =
                   blockDim.x * blockIdx.x + threadIdx.x + dims.x + NH_RADIUS,
               idxMax = min((dims.y - 1) * dims.x - NH_RADIUS,
                            (dims.y - NH_RADIUS) * dims.x - NH_RADIUS),
               xMax = dims.x - NH_RADIUS;

    for (uint idx = idMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x;
        grid[idx] = (NH_RADIUS < x) * (x < xMax) *
                        curand_uniform(&globalRandState[idx]) <
                    spawnProbability;
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
    // idMin = thread ID + render margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        gridLimY.start * dims.x + gridLimX.start,
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
    // idMin = thread ID + safety border margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idMin =
                   blockDim.x * blockIdx.x + threadIdx.x + dims.x + NH_RADIUS,
               idxMax = min((dims.y - 1) * dims.x - NH_RADIUS,
                            (dims.y - NH_RADIUS) * dims.x - NH_RADIUS);

    for (uint idx = idMin; idx < idxMax; idx += stride) {
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

__global__ void
k_update_bit_grid_buffers(const ubyte *const grid, const uvec2 dims,
                          fvec2s *const __restrict__ gridVertices,
                          const uvec2 numVertices, const uvec2 cellDensity,
                          const ulim2 gridLimX, const ulim2 gridLimY,
                          const uint bytesPerThread) {
    // idMin = thread ID + render margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x,
               idxMin = blockDim.x * blockIdx.x + threadIdx.x +
                        gridLimY.start * dims.x + gridLimX.start,
               idxMax = min((gridLimY.end - 1) * dims.x + gridLimX.end,
                            (dims.y - NH_RADIUS) * dims.x - NH_RADIUS),
               xMin = max(NH_RADIUS, gridLimX.start),
               xMax = min(dims.x - NH_RADIUS, gridLimX.end);

    /**
    Rendering should only consider cells (all bits), not the number of array
    items. Render them considering that dims.x is already divided by 8, so each
    bit is mapped to a vertice.
    */
    // printf("\n\n\n%02u, %02u, %02u\n", stride, idxMin, idxMax);
    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x, y = idx / dims.x;
        // printf("%02u, x %02u, y %02u | ", idx, idx % dims.x, idx / dims.x);
        // try avoiding further operations when not needed
        // atomicMax is pretty expensive
        if (xMin <= x && x < xMax && grid[idx]) {
            for (int bit = 7; bit >= 0; --bit) {
                const uint vx = (x - gridLimX.start) / cellDensity.x + 7 - bit,
                           vy = (y - gridLimY.start) / cellDensity.y,
                           vidx = vy * numVertices.x + vx;
                // try avoiding further operations when not needed
                // atomicMax is pretty expensive
                // printf(" %u: %u,", (1 << bit), (grid[idx] & (1 << bit)));
                if (vx < numVertices.x && vy < numVertices.y) {
                    if (!!(grid[idx] & (1 << bit)) &&
                        gridVertices[vidx].state == 0)
                        atomicMax(&gridVertices[vidx].state, 1);
                }
            }
        }
        // printf("\n");
    }
}

/**
 * Code from
 * http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA/3-Advanced-bit-per-cell-implementation
 */
__global__ void k_bit_life(const ubyte *const grid, ubyte *const nextGrid,
                           const uvec2 dims, const uint bytesPerThread) {
    // idMin = thread ID + safety border margin
    // idxMax = y - 1 full rows + last row cols
    const uint stride = gridDim.x * blockDim.x * bytesPerThread,
               idMin = (blockDim.x * blockIdx.x + threadIdx.x) * bytesPerThread,
               idxMax = dims.y * dims.x;

    for (uint idx = idMin; idx < idxMax; idx += stride) {
        uint x = (idx + dims.x - 1) % dims.x; // Start at block x - 1.
        uint yAbs = (idx / dims.x) * dims.x;
        uint yAbsUp = (yAbs + idxMax - dims.x) % idxMax;
        uint yAbsDown = (yAbs + dims.x) % idxMax;

        // Initialize data with previous byte and current byte.
        uint row0 = (uint)grid[x + yAbsUp] << 16;
        uint row1 = (uint)grid[x + yAbs] << 16;
        uint row2 = (uint)grid[x + yAbsDown] << 16;

        x = (x + 1) % dims.x;
        row0 |= (uint)grid[x + yAbsUp] << 8;
        row1 |= (uint)grid[x + yAbs] << 8;
        row2 |= (uint)grid[x + yAbsDown] << 8;

        for (uint i = 0; i < bytesPerThread; ++i) {
            uint centerX = x;
            x = (x + 1) % dims.x;
            row0 |= (uint)grid[x + yAbsUp];
            row1 |= (uint)grid[x + yAbs];
            row2 |= (uint)grid[x + yAbsDown];

            uint result = 0;
            for (uint j = 0; j < 8; ++j) {
                // 0x14000 = 0b10100000000000000 (16 bits)
                uint aliveCells =
                    (row0 & 0x14000) + (row1 & 0x14000) + (row2 & 0x14000);
                aliveCells >>= 14;
                aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) +
                             ((row0 >> 15) & 0x1u) + ((row2 >> 15) & 0x1u);
                result =
                    result << 1 |
                    (aliveCells == 3 || (aliveCells == 2 && (row1 & 0x8000u))
                         ? 1u
                         : 0u);
                // next bit
                row0 <<= 1;
                row1 <<= 1;
                row2 <<= 1;
            }

            nextGrid[centerX + yAbs] = result;
        }
    }
}
