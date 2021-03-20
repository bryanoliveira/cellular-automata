/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 * Grid drawing inspired by
 * https://github.com/asegizek/418-Final-Project/blob/master/display.cpp
 */

#include <GL/glu.h>
#include <GL/glut.h>
#include <chrono>
#include <thread>

#include "automata.hpp"
#include "automata_gpu.h"
#include "config.hpp"
#include "display.hpp"
#include "grid.hpp"

void loop();

int main(int argc, char** argv) {
    srand(time(NULL));

    grid = initGrid(true);
    // insertGlider(config::rows / 2 - 12, config::cols / 2 - 12);
    // insertBlinker(config::rows / 2, config::cols / 2);

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
    // limit framerate
    std::this_thread::sleep_for(std::chrono::milliseconds(config::render_delay_ms));
    // display initial grid
    display();
    // compute next grid
    computeGrid();
}