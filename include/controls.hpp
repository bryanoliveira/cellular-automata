#ifndef CONTROLS_HPP_
#define CONTROLS_HPP_

#include "types.hpp"

namespace controls {

extern bool paused;
extern bool singleStep;
extern fvec2 rotation;
extern fvec2 position;
extern float scale;
extern float minScale;
extern float maxScale;
extern float translateFactor;
extern float scaleFactor;

void mouse(int button, int state, int x, int y);
void keyboard(unsigned char key, int x, int y);
void motion(int x, int y);

} // namespace controls

#endif