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

AutomataBase::AutomataBase(const uint pRandSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           void (*pUpdateBuffersFunc)()) {
    spdlog::info("Initializing automata CPU engine...");

    // init random
    srand(pRandSeed);
    randState = new uint[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        randState[i] = pRandSeed + i;
    }

    // allocate grids
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

#pragma omp parallel for reduction(+ : mActiveCellCount)
    //   // schedule(runtime)
    for (uint idx = config::cols + 1; idx < idxMax; ++idx) {
        // check x index to skip borders
        const uint x = idx % config::cols;

        // check safety borders & cell state
        nextGrid[idx] = (NH_RADIUS < x) * (x < xMax) *
                        game_of_life(grid[idx], count_nh(idx));

        // add a "virtual particle" spawn probability
        // note: this branching does not cause significant perf. hit
        if (config::virtualFillProb > 0 && NH_RADIUS < x && x < xMax &&
            nextGrid[idx] == 0)
            nextGrid[idx] =
                static_cast<float>(rand_r(&randState[omp_get_thread_num()])) /
                    RAND_MAX <
                config::virtualFillProb;

        if (logEnabled)
            mActiveCellCount += static_cast<uint>(nextGrid[idx] > 0);
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
