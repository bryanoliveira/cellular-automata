#include <chrono>
#include <omp.h>
#include <sstream>
#include <spdlog/spdlog.h>

#include "automata_base_cpu.hpp"
#include "stats.hpp"

namespace cpu {

AutomataBase::AutomataBase(const unsigned long pRandSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           void (*pUpdateBuffersFunc)()) {
    spdlog::info("Initializing automata CPU engine...");
    srand(pRandSeed);

    grid =
        static_cast<bool *>(calloc(config::rows * config::cols, sizeof(bool)));
    nextGrid =
        static_cast<bool *>(calloc(config::rows * config::cols, sizeof(bool)));

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

void AutomataBase::compute_grid(const bool logEnabled) {
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    mActiveCellCount = 0;

    // note: we're using safety borders
#pragma omp parallel
    {
        uint myseed = omp_get_thread_num();
#pragma omp for // private(y) shared(grid, nextGrid, y, config::cols
        for (uint y = 1; y < config::rows - 1; ++y) {
            // reduction(+ : mActiveCellCount)
            // #pragma omp for // private(x) shared(grid, nextGrid, y,
            // config::cols)
            for (uint x = 1; x < config::cols - 1; ++x) {
                // add a "virtual particle" spawn probability
                nextGrid[y * config::cols + x] =
                    (config::virtualFillProb &&
                     (static_cast<float>(rand_r(&myseed)) / RAND_MAX) <
                         config::virtualFillProb) ||
                    compute_cell(y, x);

                // if (logEnabled && nextGrid[y * config::cols + x])
                //     // #pragma omp critical
                //     ++mActiveCellCount;
            }
        }
    }

    bool *tmpGrid = grid;
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

inline bool AutomataBase::compute_cell(const uint y, const uint x) {
    unsigned short livingNeighbours = 0;
    const uint idx = y * config::cols + x;
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

    // 1. Any live cell with two or three live neighbours survives.
    // 2. Any dead cell with three live neighbours becomes a live cell.
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    return (grid[idx] && livingNeighbours == 2) || livingNeighbours == 3;
}

} // namespace cpu