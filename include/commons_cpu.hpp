#ifndef COMMONS_CPU_HPP_
#define COMMONS_CPU_HPP_

#include "grid.hpp"
#include "types.hpp"

namespace cpu {

// These definitions must be in the .hpp since they may be inlined

/** Counts immediate active Moore neighbours **/
inline ushort count_nh(const uint idx) {
    return static_cast<ushort>(
        // it's faster to calculate them directly
        grid[idx - config::cols - 1] + // top left
        grid[idx - config::cols] +     // top center
        grid[idx - config::cols + 1] + // top right
        grid[idx - 1] +                // middle left
        grid[idx + 1] +                // middle right
        grid[idx + config::cols - 1] + // bottom left
        grid[idx + config::cols] +     // bottom center
        grid[idx + config::cols + 1]   // bottom right
    );
}

/**
 * Computes the game of life:
 * 1. Any live cell with two or three live neighbours survives.
 * 2. Any dead cell with three live neighbours becomes a live cell.
 * 3. All other live cells die in the next generation. Similarly, all other
 * dead cells stay dead.
 */
inline GridType game_of_life(const GridType state,
                             const ushort livingNeighbours) {
    return (state && livingNeighbours == 2) || livingNeighbours == 3;
}

} // namespace cpu

#endif // COMMONS_CPU_HPP_