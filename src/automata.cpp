#include "automata.hpp"

namespace cpu {

void setup(unsigned long randSeed, float fill_prob) {
    srand(randSeed);

    grid = (bool *)calloc(config::rows * config::cols, sizeof(bool));
    nextGrid = (bool *)calloc(config::rows * config::cols, sizeof(bool));

    if (fill_prob > 0)
        for (unsigned int i = 0; i < config::rows; i++) {
            for (unsigned int j = 0; j < config::cols; j++)
                grid[i * config::cols + j] =
                    (float(rand()) / RAND_MAX) < fill_prob;
        }
}

void cleanUp() {
    free(grid);
    free(nextGrid);
}

bool computeCell(unsigned int i, unsigned int j) {
    int livingNeighbours = 0;

    // iterate over neighbourhood row by row
    for (unsigned int a = i - 1; a <= i + 1; a++) {
        // skip out-of-grid neighbours
        if (a < 0 || a >= config::rows)
            continue;
        // iterate over neighbourhood col by col
        for (unsigned int b = j - 1; b <= j + 1; b++) {
            // skip out-of-grid neighbours and self position
            if (b < 0 || b >= config::cols || (a == i && b == j))
                continue;
            // if cell is alive, increase neighbours
            if (grid[a * config::cols + b])
                livingNeighbours++;
        }
    }

    // 1. Any live cell with two or three live neighbours survives.
    if (grid[i * config::cols + j])
        return livingNeighbours == 2 || livingNeighbours == 3;
    // 2. Any dead cell with three live neighbours becomes a live cell.
    else if (livingNeighbours == 3)
        return true;
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    else
        return false;
}

void computeGrid() {
    for (unsigned int i = 0; i < config::rows; i++) {
        for (unsigned int j = 0; j < config::cols; j++) {
            // add a "virtual particle" spawn probability
            nextGrid[i * config::cols + j] =
                (float(rand()) / RAND_MAX) < config::virtual_fill_prob ||
                computeCell(i, j);
        }
    }
    bool *tmpGrid = grid;
    grid = nextGrid;
    nextGrid = tmpGrid;
}

} // namespace cpu