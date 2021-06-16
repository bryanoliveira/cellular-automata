#include "grid.hpp"

// global grid
GridType *grid;
GridType *nextGrid;

void insert_glider(const int row, const int col) {
    grid[(row - 1) * config::cols + col + 0] = 1;
    grid[(row + 0) * config::cols + col + 1] = 1;
    grid[(row + 1) * config::cols + col - 1] = 1;
    grid[(row + 1) * config::cols + col + 0] = 1;
    grid[(row + 1) * config::cols + col + 1] = 1;
}

void insert_blinker(const int row, const int col) {
    grid[(row - 1) * config::cols + col] = 1;
    grid[(row + 0) * config::cols + col] = 1;
    grid[(row + 1) * config::cols + col] = 1;
}