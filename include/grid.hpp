#ifndef GRID_HPP_
#define GRID_HPP_

#include <iostream>

#include "config.hpp"

extern bool** grid;

bool** initGrid(bool random = false);
void freeGrid(bool** tempGrid);

#endif