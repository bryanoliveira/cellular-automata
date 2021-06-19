#ifndef NH_RADIUS
#define NH_RADIUS 1
#endif

#include <chrono>
#include <omp.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <iostream>

#include "automata_base_cpu.hpp"
#include "stats.hpp"

namespace cpu {

AutomataBase::AutomataBase(const unsigned long pRandSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           void (*pUpdateBuffersFunc)()) {
    spdlog::info("Initializing automata CPU engine...");
    srand(pRandSeed);

    grid = static_cast<GridType *>(
        calloc(config::rows * config::cols, sizeof(GridType)));
    nextGrid = static_cast<GridType *>(
        calloc(config::rows * config::cols, sizeof(GridType)));

    if (config::fillProb > 0)
        // note: we're using safety borders
        for (uint y = 1; y < config::rows - 1; ++y) {
            for (uint x = 1; x < config::cols - 1; ++x)
                grid[y * config::cols + x] =
                    (static_cast<float>(rand()) / RAND_MAX) < config::fillProb;
        }

    mLiveLogBuffer = pLiveLogBuffer;
    mUpdateBuffersFunc = pUpdateBuffersFunc;

    spdlog::info("Automata CPU engine is ready.");
}

AutomataBase::~AutomataBase() {
    free(grid);
    free(nextGrid);
}

void AutomataBase::evolve(const bool logEnabled) {
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    mActiveCellCount = 0;

    const uint idxMax = (config::rows - 1) * config::cols - 1,
               xMax = config::cols - NH_RADIUS;

#pragma omp parallel
    {
        uint myseed = omp_get_thread_num();
#pragma omp for private(myseed) reduction(+ : mActiveCellCount)
        // note: we're using safety borders
        for (uint idx = config::cols + 1; idx < idxMax; ++idx) {
            // check x index to skip borders
            const uint x = idx % config::cols;
            // check safety borders & cell state
            nextGrid[idx] = NH_RADIUS < x && x < xMax &&
                            // add a "virtual particle" spawn probability
                            ((config::virtualFillProb &&
                              (static_cast<float>(rand_r(&myseed)) / RAND_MAX) <
                                  config::virtualFillProb) ||
                             // compute cell rule
                             compute_cell(idx));

            if (logEnabled && nextGrid[idx])
                // #pragma omp critical
                ++mActiveCellCount;
        }
    }

    GridType *tmpGrid = grid;
    grid = nextGrid;
    nextGrid = tmpGrid;

    // calculate timing
    const unsigned long duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - timeStart)
            .count();
    stats::totalEvolveTime += duration;
    // update live buffer
    if (logEnabled)
        *mLiveLogBuffer << " | Evolve Function: " << duration
                        << " ns | Active cells: " << mActiveCellCount;
}

inline bool AutomataBase::compute_cell(const uint idx) {
    unsigned short livingNeighbours = 0;
    // we can calculate the neighbours directly since we're using safety
    // borders
    const uint neighbours[] = {
        idx - config::cols - 1, // top left
        idx - config::cols,     // top center
        idx - config::cols + 1, // top right
        idx - 1,                // middle left
        idx + 1,                // middle right
        idx + config::cols - 1, // bottom left
        idx + config::cols,     // bottom center
        idx + config::cols + 1, // bottom right
    };

#pragma GCC unroll 8
    for (unsigned short nidx = 0; nidx < 8; ++nidx)
        livingNeighbours += static_cast<unsigned short>(grid[neighbours[nidx]]);

    // 1. Any living cell with two or three live neighbours survives.
    // 2. Any dead cell with three live neighbours becomes a live cell.
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    return (grid[idx] && livingNeighbours == 2) || livingNeighbours == 3;
}

} // namespace cpu