#include "grid.hpp"

// global grid
bool** grid;

bool** initGrid(bool random, float fill_prob) {
    if (fill_prob == -1) fill_prob = config::fill_prob;

    bool** tempGrid = (bool**)calloc(config::rows, sizeof(bool*));

    for (int i = 0; i < config::rows; i++) {
        tempGrid[i] = (bool*)calloc(config::cols, sizeof(bool));

        if (random)
            for (int j = 0; j < config::cols; j++)
                tempGrid[i][j] = (float(rand()) / RAND_MAX) < config::fill_prob;
    }

    return tempGrid;
}
void freeGrid(bool** tempGrid) {
    for (int i = 0; i < config::rows; i++) free(tempGrid[i]);
    free(tempGrid);
}