#ifndef GRID_HPP_
#define GRID_HPP_

#include "config.hpp"

extern bool *grid;
extern bool *nextGrid;

void insert_glider(const int row, const int col);
void insert_blinker(const int row, const int col);

#endif