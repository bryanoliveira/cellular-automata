#ifndef GRID_HPP_
#define GRID_HPP_

#include "config.hpp"

extern bool *grid;
extern bool *nextGrid;

void insert_glider(int row, int col);
void insert_blinker(int row, int col);

#endif