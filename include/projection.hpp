#ifndef PROJECTION_HPP_
#define PROJECTION_HPP_

#include "config.hpp"
#include "controls.hpp"
#include "types.hpp"

namespace proj {

extern GridRenderInfo info;

// The mapping from grid cells to rendering vertices
extern uvec2 cellDensity;
// The X,Y limits where grid mapping occur
extern ulim2 gridLimX, gridLimY;

// Calculate number of vertices and projection limits
void init();
// Update projection limits given control updates
void update();
// Transform a X,Y Grid position into a 1D vertice index
uint getVerticeIdx(const uvec2 gridPos);

}; // namespace proj

#endif // PROJECTION_HPP_