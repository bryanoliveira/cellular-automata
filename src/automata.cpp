#include "automata.hpp"

void computeGrid() {
    bool** prevGrid = grid;
    bool** nextGrid = initGrid();

    for (int i = 0; i < config::rows; i++) {
        for (int j = 0; j < config::cols; j++) {
            int livingNeighbours = 0;
            // iterate over neighbourhood row by row
            for (int a = i - 1; a <= i + 1; a++) {
                // skip out-of-grid neighbours
                if (a < 0 || a >= config::rows) continue;
                // iterate over neighbourhood col by col
                for (int b = j - 1; b <= j + 1; b++) {
                    // skip out-of-grid neighbours and self position
                    if (b < 0 || b >= config::cols || (a == i && b == j))
                        continue;
                    // if cell is alive, increase neighbours
                    if (prevGrid[a][b]) livingNeighbours++;
                }
            }

            // 1. Any live cell with two or three live neighbours survives.
            if (prevGrid[i][j])
                nextGrid[i][j] = livingNeighbours == 2 || livingNeighbours == 3;
            // 2. Any dead cell with three live neighbours becomes a live cell.
            else if (livingNeighbours == 3)
                nextGrid[i][j] = true;
            // 3. All other live cells die in the next generation. Similarly,
            // all other dead cells stay dead.
            else
                nextGrid[i][j] = false;
        }
    }

    freeGrid(prevGrid);
    grid = nextGrid;
}

void insertGlider(int row, int col) {
    grid[row - 1][row + 0] = true;
    grid[row + 0][row + 1] = true;
    grid[row + 1][row - 1] = true;
    grid[row + 1][row + 0] = true;
    grid[row + 1][row + 1] = true;
}

void insertBlinker(int row, int col) {
    grid[row - 1][col + 0] = true;
    grid[row + 0][col + 0] = true;
    grid[row + 1][col + 0] = true;
}