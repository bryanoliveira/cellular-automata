#include <GL/freeglut.h>
#include <iostream>
#include "controls.hpp"

namespace controls {

// public
bool paused = true;
bool singleStep = false;
fvec2 rotation(0, 0);
fvec2 position(0, 0);          // grid translation
fvec2 secondaryPosition(0, 0); // projection translation
float scale = 1.0;             // grid scale
float secondaryScale = 1.0;    // projection scale
float minScale = 1.0;
float maxScale = 10.0;
float translateFactor = 10.0f;
float secondaryTranslateFactor = 0.002f;
float scaleFactor = 1.0f;
float secondaryScaleFactor = 0.99f;

// private
vec2 mouseOld;
int mouseButtons = 0;
int secondaryMouseButtons = 0;

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        // secondary mouse events when pressing CTRL
        if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
            secondaryMouseButtons |= 1 << button;
        else
            mouseButtons |= 1 << button;

        // middle click (for some reason it doesn't work like the others)
        if (button == GLUT_MIDDLE_BUTTON) {
            // reset
            position = secondaryPosition = rotation = {0, 0};
            scale = secondaryScale = 1;
        }
    } else if (state == GLUT_UP) {
        mouseButtons = 0;
        secondaryMouseButtons = 0;
    }

    // these buttons behave differently (they don't go up and down)
    // scroll up
    if (button == 3) {
        // secondary scaling when pressing CTRL
        if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
            secondaryScale = (1e-2 + secondaryScale) / secondaryScaleFactor;
        else
            scale += scaleFactor;
    }
    // scroll down
    if (button == 4) {
        // secondary scaling when pressing CTRL
        if (glutGetModifiers() & GLUT_ACTIVE_CTRL)
            secondaryScale = (1e-2 + secondaryScale) * secondaryScaleFactor;
        else
            scale -= scaleFactor;
    }

    if (scale < minScale)
        scale = minScale;
    else if (scale > maxScale)
        scale = maxScale;

    mouseOld = {x, y};
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
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
    dx = (float)(x - mouseOld.x);
    dy = (float)(y - mouseOld.y);

    // left click
    if (mouseButtons & 1 << GLUT_LEFT_BUTTON) {
        // motion speed should be proportional to scale
        position.x -= dx * (translateFactor / scale);
        position.y -= dy * (translateFactor / scale);
    }
    if (secondaryMouseButtons & 1 << GLUT_LEFT_BUTTON) {
        // motion speed should be proportional to scale
        secondaryPosition.x -= dx * (secondaryTranslateFactor / secondaryScale);
        secondaryPosition.y -= dy * (secondaryTranslateFactor / secondaryScale);
    }
    // right click
    if (mouseButtons & 1 << GLUT_RIGHT_BUTTON) {
        rotation.x += dx * 0.2f;
        rotation.y += dy * 0.2f;
    }

    mouseOld = {x, y};
}

} // namespace controls