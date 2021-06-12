#include "grid.hpp"

// global grid
bool *grid;
bool *nextGrid;

void insert_glider(const int row, const int col) {
    grid[(row - 1) * config::cols + col + 0] = true;
    grid[(row + 0) * config::cols + col + 1] = true;
    grid[(row + 1) * config::cols + col - 1] = true;
    grid[(row + 1) * config::cols + col + 0] = true;
    grid[(row + 1) * config::cols + col + 1] = true;
}

void insert_blinker(const int row, const int col) {
    grid[(row - 1) * config::cols + col] = true;
    grid[(row + 0) * config::cols + col] = true;
    grid[(row + 1) * config::cols + col] = true;
}