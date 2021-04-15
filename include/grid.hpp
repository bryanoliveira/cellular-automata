#ifndef GRID_HPP_
#define GRID_HPP_

#include <vector>
#include "config.hpp"

// to avoid having to change everywhere when we update the grid type
typedef bool GridType;
typedef std::vector<GridType> Grid;

extern Grid grid;
extern Grid nextGrid;

void insert_glider(int row, int col);
void insert_blinker(int row, int col);

#endif