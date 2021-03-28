#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include "config.hpp"
#include "grid.hpp"


namespace gpu {

void setup(unsigned long seed);

void computeGrid();

void cleanUp();

}  // namespace gpu

#endif