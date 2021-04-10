#ifndef CONTROLS_HPP_
#define CONTROLS_HPP_

namespace controls {

extern float center[];
extern float scale;
extern float rotate_x;
extern float rotate_y;

void mouse(int button, int state, int x, int y);
void motion(int x, int y);

} // namespace controls

#endif