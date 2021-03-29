#include "grid.hpp"

// global grid
bool *grid;
bool *nextGrid;

bool *initGrid(bool random, float fill_prob) {
    if (fill_prob == -1)
        fill_prob = config::fill_prob;

    bool *tempGrid = (bool *)calloc(config::rows, sizeof(bool));

    if (random)
        for (unsigned int i = 0; i < config::rows; i++) {
            for (unsigned int j = 0; j < config::cols; j++)
                tempGrid[i * config::cols + j] =
                    (float(rand()) / RAND_MAX) < config::fill_prob;
        }

    return tempGrid;
}

void freeGrid(bool *tempGrid) { free(tempGrid); }