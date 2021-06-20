#ifndef NH_RADIUS
#define NH_RADIUS 1
#endif

#include <chrono>
#include <omp.h>
#include <sstream>
#include <spdlog/spdlog.h>

#include "automata_base_cpu.hpp"
#include "commons_cpu.hpp"
#include "stats.hpp"

namespace cpu {

AutomataBase::AutomataBase(const ulong pRandSeed,
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

void AutomataBase::prepare(){
    // currently does nothing
};

void AutomataBase::evolve(const bool logEnabled) {
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    mActiveCellCount = 0;

    const uint idxMax = (config::rows - 1) * config::cols - 1,
               xMax = config::cols - NH_RADIUS;

#pragma omp parallel
    {
        uint myseed = omp_get_thread_num();

#pragma omp for private(myseed) reduction(+ : mActiveCellCount) schedule(runtime)
        for (uint idx = config::cols + 1; idx < idxMax; ++idx) {
            // check x index to skip borders
            const uint x = idx % config::cols;
            if (NH_RADIUS < x && x < xMax) {
                // check safety borders & cell state
                nextGrid[idx] =
                    // add a "virtual particle" spawn probability
                    ((config::virtualFillProb &&
                      (static_cast<float>(rand_r(&myseed)) / RAND_MAX) <
                          config::virtualFillProb)
                     // compute cell rule
                     || game_of_life(grid[idx], count_nh(idx)));
            }

            if (logEnabled)
                mActiveCellCount += nextGrid[idx];
        }
    }

    std::swap(grid, nextGrid);

    // calculate timing
    const ulong duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - timeStart)
                               .count();
    stats::totalEvolveTime += duration;
    // update live buffer
    if (logEnabled)
        *mLiveLogBuffer << " | Evolve Function: " << duration
                        << " ns | Active cells: " << mActiveCellCount;
}

} // namespace cpu
