// isolate cuda specific imports
#include <chrono>
#include <cuda_gl_interop.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <iostream>

#include "automata_bit_gpu.cuh"
#include "config.hpp"
#include "definitions.hpp"
#include "grid.hpp"
#include "projection.hpp"

__global__ void k_init_bit_grid(GridType *const grid, const uvec2 dims,
                                curandState *const __restrict__ globalRandState,
                                const float spawnProbability);

__global__ void
k_update_bit_grid_buffers(const ubyte *const grid, const uvec2 dims,
                          fvec2s *const __restrict__ gridVertices,
                          const uvec2 numVertices, const uvec2 cellDensity,
                          const ulim2 gridLimX, const ulim2 gridLimY,
                          const uint bytesPerThread);

__global__ void k_bit_life(const ubyte *const grid, ubyte *const nextGrid,
                           const uvec2 dims, const uint bytesPerThread);

//// AUTOMATA OVERRIDES

namespace gpu {

void AutomataBit::run_init_kernel() {
    // initialize grid with fillProb
    if (config::fillProb > 0)
        k_init_bit_grid<<<config::gpuBlocks, config::gpuThreads>>>(
            grid, {config::cols, config::rows}, mGlobalRandState,
            config::fillProb);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBit::run_evolution_kernel(const bool countAliveCells) {
    k_bit_life<<<config::gpuBlocks, config::gpuThreads, 0, mEvolveStream>>>(
        grid, nextGrid, {config::cols, config::rows}, 1);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBit::run_render_kernel(fvec2s *gridVertices) {
#ifndef HEADLESS_ONLY
    k_update_bit_grid_buffers<<<config::gpuBlocks, config::gpuThreads, 0,
                                mBufferUpdateStream>>>(
        grid, {config::cols, config::rows}, gridVertices,
        proj::info.numVertices, proj::cellDensity, proj::gridLimX,
        proj::gridLimY, 1);
    CUDA_ASSERT(cudaGetLastError());
#endif
}

} // namespace gpu

//// KERNELS

__global__ void k_init_bit_grid(GridType *const grid, const uvec2 dims,
                                curandState *const __restrict__ globalRandState,
                                const float spawnProbability) {
    // idMin = thread ID + safety border margin
    // idxMax = y - 1 full rows + last row cols OR max id given radius
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

        for (uint bit = 0; bit < 8; ++bit)
            if (curand_uniform(&globalRandState[idx]) < spawnProbability)
                // enable each bit given probability
                grid[idx] |= 1 << bit;
    }
}

// ALERT: this currently does not work
// TODO: fix this!
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
    items. Render them considering that dims.x is already divided by 8, so
    each bit is mapped to a vertice.
    */
    // printf("\n\n\n%02u, %02u, %02u\n", stride, idxMin, idxMax);
    for (uint idx = idxMin; idx < idxMax; idx += stride) {
        const uint x = idx % dims.x, y = idx / dims.x;
        // printf("%02u, x %02u, y %02u | ", idx, idx % dims.x, idx /
        // dims.x); try avoiding further operations when not needed
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