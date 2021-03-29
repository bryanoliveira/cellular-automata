#ifndef GRID_HPP_
#define GRID_HPP_

#include "config.hpp"

extern bool *grid;
extern bool *nextGrid;

void insertGlider(int row, int col);
void insertBlinker(int row, int col);

#endif