#include "config.hpp"
#include "controls.hpp"
#include "types.hpp"

namespace proj {

extern GridRenderInfo renderInfo;

// The mapping from grid cells to rendering vertices
extern uvec2 cellDensity;
// The X,Y position where vertice updates will begin
extern uvec2 vStart;
// The X,Y position where grid mapping will begin
extern uvec2 gridStart;
// The X,Y position where grid mapping will end
extern uvec2 gridEnd;

// Calculate number of vertices and projection limits
void init();
// Update projection limits given control updates
void update();
// Transform a X,Y Grid position into a 1D vertice index
uint getVerticeIdx(uvec2 gridPos);

}; // namespace proj