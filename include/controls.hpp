#ifndef CONTROLS_HPP_
#define CONTROLS_HPP_

#include "types.hpp"

namespace controls {

extern bool paused;
extern bool singleStep;
extern fvec2 rotation;
extern fvec2 position;
extern fvec2 secondaryPosition;
extern float scale;
extern float secondaryScale;
extern float minScale;
extern float maxScale;
extern float translateFactor;
extern float secondaryTranslateFactor;
extern float scaleFactor;
extern float secondaryScaleFactor;

void mouse(int button, int state, int x, int y);
void keyboard(unsigned char key, int x, int y);
void motion(int x, int y);

} // namespace controls

#endif