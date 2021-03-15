#ifndef GRID_HPP_
#define GRID_HPP_

#include <iostream>

#include "config.hpp"

extern bool** grid;

bool** initGrid(bool random = false, float fill_prob = -1);
void freeGrid(bool** tempGrid);

#endif