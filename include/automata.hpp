#ifndef AUTOMATA_HPP_
#define AUTOMATA_HPP_

#include <iostream>

#include "config.hpp"
#include "grid.hpp"

namespace cpu {

void setup(unsigned long pRandSeed, float pFillProb = 0);
void clean_up();
void compute_grid();

} // namespace cpu

#endif