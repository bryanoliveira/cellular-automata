#ifndef COMMONS_GPU_CUH_
#define COMMONS_GPU_CUH_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandState type

#include "definitions.hpp"
#include "grid.hpp"  // GridType
#include "types.hpp" // uint, ulim, etc

// NOTE: this code is repeated here and in commons_gpu.cu to avoid dependencies
// also, we unroll loops manually to avoid branching overhead
__device__ inline ushort count_nh(const GridType *const grid, const uint size1,
                                  const uint idx) {
    /** Counts immediate active Moore neighbours **/
    // size1 = cols
    ushort nhs = static_cast<ushort>(
        // it's faster to calculate them directly
        grid[idx - size1 - 1] + // [* - -] top left
        grid[idx - size1] +     // [- * -] top center
        grid[idx - size1 + 1] + // [- - *] top right
        grid[idx - 1] +         // [* - -] middle left
        grid[idx + 1] +         // [- - *] middle right
        grid[idx + size1 - 1] + // [* - -] bottom left
        grid[idx + size1] +     // [- * -] bottom center
        grid[idx + size1 + 1]   // [- - *] bottom right
    );

#if NH_RADIUS > 1 // 2 and above
    const uint size2 = size1 * 2;
    nhs += static_cast<ushort>(
        // top row
        grid[idx - size2 - 2] + //
        grid[idx - size2 - 1] + //
        grid[idx - size2] +     //
        grid[idx - size2 + 1] + //
        grid[idx - size2 + 2] + //
        // 2nd row
        grid[idx - size1 - 2] + //
        grid[idx - size1 + 2] + //
        // middle row
        grid[idx - 2] + //
        grid[idx + 2] + //
        // 4th row
        grid[idx + size1 - 2] + //
        grid[idx + size1 + 2] + //
        // bottom row
        grid[idx + size2 - 2] + //
        grid[idx + size2 - 1] + //
        grid[idx + size2] +     //
        grid[idx + size2 + 1] + //
        grid[idx + size2 + 2]   //
    );

#if NH_RADIUS > 2 // 3 and above
    const uint size3 = size1 * 3;
    nhs += static_cast<ushort>(
        // top row
        grid[idx - size3 - 3] + //
        grid[idx - size3 - 2] + //
        grid[idx - size3 - 1] + //
        grid[idx - size3] +     //
        grid[idx - size3 + 1] + //
        grid[idx - size3 + 2] + //
        grid[idx - size3 + 3] + //
        // 2nd row
        grid[idx - size2 - 3] + //
        grid[idx - size2 + 3] + //
        // 3rd row
        grid[idx - size - 3] + //
        grid[idx - size + 3] + //
        // middle row
        grid[idx - 3] + //
        grid[idx + 3] + //
        // 5th row
        grid[idx + size - 3] + //
        grid[idx + size + 3] + //
        // 6th row
        grid[idx + size2 - 3] + //
        grid[idx + size2 + 3] + //
        // bottom row
        grid[idx + size3 - 3] + //
        grid[idx + size3 - 2] + //
        grid[idx + size3 - 1] + //
        grid[idx + size3] +     //
        grid[idx + size3 + 1] + //
        grid[idx + size3 + 2] + //
        grid[idx + size3 + 3]   //
    );

#if RADIUS > 3
    const uint size4 = size1 * 4;
    nhs += static_cast<ushort>(
        // top row
        grid[idx - size4 - 4] + //
        grid[idx - size4 - 3] + //
        grid[idx - size4 - 2] + //
        grid[idx - size4 - 1] + //
        grid[idx - size4] +     //
        grid[idx - size4 + 1] + //
        grid[idx - size4 + 2] + //
        grid[idx - size4 + 3] + //
        grid[idx - size4 + 4] + //
                                // 2nd row
        grid[idx - size3 - 4] + //
        grid[idx - size3 + 4] + //
                                // 3rd row
        grid[idx - size2 - 4] + //
        grid[idx - size2 + 4] + //
                                // 4th row
        grid[idx - size - 4] +  //
        grid[idx - size + 4] +  //
                                // middle row
        grid[idx - 4] +         //
        grid[idx + 4] +         //
                                // 6th row
        grid[idx + size - 4] +  //
        grid[idx + size + 4] +  //
                                // 7th row
        grid[idx + size2 - 4] + //
        grid[idx + size2 + 4] + //
                                // 8th row
        grid[idx + size3 - 4] + //
        grid[idx + size3 + 4] + //
                                // bottom row
        grid[idx + size4 - 4] + //
        grid[idx + size4 - 3] + //
        grid[idx + size4 - 2] + //
        grid[idx + size4 - 1] + //
        grid[idx + size4] +     //
        grid[idx + size4 + 1] + //
        grid[idx + size4 + 2] + //
        grid[idx + size4 + 3] + //
        grid[idx + size4 + 4]   //
    );

#if RADIUS > 4
    const uint size5 = size1 * 5;
    nhs += static_cast<ushort>(
        // top row
        grid[idx - size5 - 5] + //
        grid[idx - size5 - 4] + //
        grid[idx - size5 - 3] + //
        grid[idx - size5 - 2] + //
        grid[idx - size5 - 1] + //
        grid[idx - size5] +     //
        grid[idx - size5 + 1] + //
        grid[idx - size5 + 2] + //
        grid[idx - size5 + 3] + //
        grid[idx - size5 + 4] + //
        grid[idx - size5 + 5] + //
                                // 2nd row
        grid[idx - size4 - 5] + //
        grid[idx - size4 + 5] + //
                                // 3rd row
        grid[idx - size3 - 5] + //
        grid[idx - size3 + 5] + //
                                // 4th row
        grid[idx - size2 - 5] + //
        grid[idx - size2 + 5] + //
                                // 5th row
        grid[idx - size - 5] +  //
        grid[idx - size + 5] +  //
                                // middle row
        grid[idx - 5] +         //
        grid[idx + 5] +         //
                                // 7th row
        grid[idx + size - 5] +  //
        grid[idx + size + 5] +  //
                                // 8th row
        grid[idx + size2 - 5] + //
        grid[idx + size2 + 5] + //
                                // 9th row
        grid[idx + size3 - 5] + //
        grid[idx + size3 + 5] + //
                                // 10th row
        grid[idx + size4 - 5] + //
        grid[idx + size4 + 5] + //
                                // bottom row
        grid[idx + size5 - 5] + //
        grid[idx + size5 - 4] + //
        grid[idx + size5 - 3] + //
        grid[idx + size5 - 2] + //
        grid[idx + size5 - 1] + //
        grid[idx + size5] +     //
        grid[idx + size5 + 1] + //
        grid[idx + size5 + 2] + //
        grid[idx + size5 + 3] + //
        grid[idx + size5 + 4] + //
        grid[idx + size5 + 5]   //
    );

#endif // RADIUS > 4
#endif // RADIUS > 3
#endif // RADIUS > 2
#endif // RADIUS > 1

    return nhs;
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

#endif