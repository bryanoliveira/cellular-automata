#ifndef AUTOMATA_HPP_
#define AUTOMATA_HPP_

#include <iostream>

#include "config.hpp"
#include "grid.hpp"

namespace cpu {

void setup(unsigned long randSeed, float fill_prob = 0);
void cleanUp();
void computeGrid();

} // namespace cpu

#endif