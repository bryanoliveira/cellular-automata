#ifndef KERNELS_HPP_
#define KERNELS_HPP_

#include "config.hpp"
#include "display.hpp" // for the vec2s type
#include "grid.hpp"

namespace gpu {

void setup(unsigned long seed, std::ostringstream *pLiveLogBuffer,
           const unsigned int *gridVBO = NULL);
void compute_grid();
void update_grid_buffers();
void clean_up();

} // namespace gpu

#endif