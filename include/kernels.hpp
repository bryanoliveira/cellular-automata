#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include "config.hpp"
#include "display.hpp" // for the vec3s type
#include "grid.hpp"

namespace gpu {

void setup(unsigned long seed, unsigned long gridVBO);
void computeGrid();
void updateGridBuffers();
void cleanUp();

} // namespace gpu

#endif