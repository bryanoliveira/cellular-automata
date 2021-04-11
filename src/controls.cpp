#include <GL/freeglut.h>
#include <iostream>
#include "controls.hpp"

namespace controls {

// public
bool paused = true;
bool singleStep = false;
float center[] = {0.0, 0.0};
float scale = 1.0;
float rotate_x = 0.0;
float rotate_y = 0.0;

// private
int mouseOldX;
int mouseOldY;
int mouseButtons = 0;
const float translateFactor = 0.002f;
const float scaleFactor = 0.99f;

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        mouseButtons |= 1 << button;

        // middle click (for some reason it doesn't work like the others)
        if (button == GLUT_MIDDLE_BUTTON) {
            // reset
            center[0] = center[1] = rotate_x = rotate_y = 0;
            scale = 1;
        }
    } else if (state == GLUT_UP) {
        mouseButtons = 0;
    }

    // these buttons behave differently (they don't go up and down)
    // scroll up
    if (button == 3)
        scale = (1e-2 + scale) / scaleFactor;
    // scroll down
    if (button == 4)
        scale = (1e-2 + scale) * scaleFactor;

    mouseOldX = x;
    mouseOldY = y;
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    // std::cout << "key " << (int)key << std::endl;
    switch (key) {
    case 32:
        paused = !paused;
        break;

    case 13:
        singleStep = true;
        break;
    }
}

void motion(int x, int y) {
    float dx, dy;
    dx = (float)(x - mouseOldX);
    dy = (float)(y - mouseOldY);

    // left click
    if (mouseButtons & 1 << GLUT_LEFT_BUTTON) {
        // motion speed should be proportional to scale
        center[0] += dx * (translateFactor / scale);
        center[1] += dy * (translateFactor / scale);
    }
    // right click
    if (mouseButtons & 1 << GLUT_RIGHT_BUTTON) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }

    mouseOldX = x;
    mouseOldY = y;
}

} // namespace controls