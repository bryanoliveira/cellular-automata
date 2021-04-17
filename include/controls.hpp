#ifndef CONTROLS_HPP_
#define CONTROLS_HPP_

namespace controls {

extern bool paused;
extern bool singleStep;
extern float center[];
extern float rotate_x;
extern float rotate_y;
extern float scale;
extern float minScale;
extern float maxScale;

void mouse(int button, int state, int x, int y);
void keyboard(unsigned char key, int x, int y);
void motion(int x, int y);

} // namespace controls

#endif