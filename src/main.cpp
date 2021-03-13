/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 * Grid drawing inspired by
 * https://github.com/asegizek/418-Final-Project/blob/master/display.cpp
 */

#include <GL/glu.h>
#include <GL/glut.h>

#include "automata.hpp"
#include "config.hpp"
#include "display.hpp"
#include "grid.hpp"

void loop();

int main(int argc, char** argv) {
    srand(time(NULL));

    grid = initGrid(true);
    // insertGlider(ROWS/2, COLS/2);
    // insertBlinker(ROWS/2, COLS/);

    glutInit(&argc, argv);
    glutInitWindowSize(config::width, config::height);
    glutCreateWindow(config::program_name.c_str());

    glClear(GL_COLOR_BUFFER_BIT);
    glutDisplayFunc(loop);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}

void loop() {
    if (display())
        // compute grid afterwards to display initial state
        computeGrid();
}