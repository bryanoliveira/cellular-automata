#ifndef GRID_HPP_
#define GRID_HPP_

#include "config.hpp"
#include "types.hpp"

typedef ubyte GridType;

extern GridType *grid;
extern GridType *nextGrid;

void insert_glider(const int row, const int col);
void insert_blinker(const int row, const int col);

#endif