#include <chrono>
#include <sstream>

#include "automata_base_cpu.hpp"

namespace cpu {

AutomataBase::AutomataBase(unsigned long pRandSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           void (*pUpdateBuffersFunc)()) {
    srand(pRandSeed);

    grid = (bool *)calloc(config::rows * config::cols, sizeof(bool));
    nextGrid = (bool *)calloc(config::rows * config::cols, sizeof(bool));

    if (config::fillProb > 0)
        // note: we're using safety borders
        for (unsigned int y = 1; y < config::rows - 1; y++) {
            for (unsigned int x = 1; x < config::cols - 1; x++)
                grid[y * config::cols + x] =
                    (float(rand()) / RAND_MAX) < config::fillProb;
        }

    mLiveLogBuffer = pLiveLogBuffer;
    mUpdateBuffersFunc = pUpdateBuffersFunc;
}

AutomataBase::~AutomataBase() {
    free(grid);
    free(nextGrid);
}

void AutomataBase::compute_grid(bool logEnabled) {
    std::chrono::steady_clock::time_point timeStart;
    if (logEnabled)
        timeStart = std::chrono::steady_clock::now();

    mActiveCellCount = 0;

    // note: we're using safety borders
    for (unsigned int y = 1; y < config::rows - 1; y++) {
        for (unsigned int x = 1; x < config::cols - 1; x++) {
            // add a "virtual particle" spawn probability
            nextGrid[y * config::cols + x] =
                (float(rand()) / RAND_MAX) < config::virtualFillProb ||
                compute_cell(y, x);

            if (logEnabled && nextGrid[y * config::cols + x])
                mActiveCellCount++;
        }
    }
    bool *tmpGrid = grid;
    grid = nextGrid;
    nextGrid = tmpGrid;

    if (logEnabled)
        // calculate timings and update live buffer
        *mLiveLogBuffer << "| Evolve Function: "
                        << std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - timeStart)
                               .count()
                        << " ns | Active cells: " << mActiveCellCount << " |";
}

bool AutomataBase::compute_cell(unsigned int y, unsigned int x) {
    unsigned short livingNeighbours = 0;

    unsigned int idx = y * config::cols + x;
    // we can calculate the neighbours directly since we're using safety borders
    unsigned int neighbours[] = {
        idx - config::cols - 1, // top left
        idx - config::cols,     // top center
        idx - config::cols + 1, // top right
        idx - 1,                // middle left
        idx + 1,                // middle right
        idx + config::cols - 1, // bottom left
        idx + config::cols,     // bottom center
        idx + config::cols + 1, // bottom right
    };
    for (unsigned short nidx = 0; nidx < 8; nidx++)
        if (grid[neighbours[nidx]])
            livingNeighbours++;

    // 1. Any live cell with two or three live neighbours survives.
    if (grid[idx])
        return livingNeighbours == 2 || livingNeighbours == 3;
    // 2. Any dead cell with three live neighbours becomes a live cell.
    else if (livingNeighbours == 3)
        return true;
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    else
        return false;
}

} // namespace cpu