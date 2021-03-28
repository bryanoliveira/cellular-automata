// isolate opengl specific imports
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <ctime>
#include <sstream>

#include "display.hpp"


namespace display {

void setup(int * argc, char ** argv, void (*loopFunc)()) {
    // init glut
    glutInit(argc, argv);
    glutInitWindowSize(config::width, config::height);
    glutCreateWindow(config::program_name.c_str());
    glutDisplayFunc(loopFunc);
    glutReshapeFunc(reshape);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // init glew
    // this is required to setup GL functions (like glGenBuffers)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* glewInit failed, something is seriously wrong */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(1);
    }
}

void start() {
    glutMainLoop();
}

void stop() {
    if(glutGetWindow()) glutDestroyWindow(glutGetWindow());
}

void reshape(int w, int h) {
    config::width = w;
    config::height = h;

    glViewport(0, 0, config::width, config::height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(config::left, config::right, config::bottom, config::top);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutPostRedisplay();
}

void _displayFrameRate() {
    static clock_t delta_ticks;
    static clock_t current_ticks = 0;
    static clock_t fps = 0;

    // init clock if needed
    if (current_ticks == 0) current_ticks = clock();

    // the time, in ms, that took to render the scene
    delta_ticks = clock() - current_ticks;
    if (delta_ticks > 0) fps = CLOCKS_PER_SEC / delta_ticks;

    std::ostringstream title;
    title << config::program_name << " (" << fps << " fps)";
    glutSetWindowTitle(title.str().c_str());

    // get clock for the next call
    current_ticks = clock();
}

/// Displays the global 'grid' if it is time. Returns true if grid was
/// rendered.
void draw() {
    // glClear(GL_COLOR_BUFFER_BIT);

    float xSize = 1.0f / ((float)config::cols);
    float ySize = 1.0f / ((float)config::rows);
    bool printedAnything = false;

    for (unsigned int y = 0; y < config::rows; y++) {
        for (unsigned int x = 0; x < config::cols; x++) {
            unsigned int gridIdx = y * config::cols + x;

            printedAnything = printedAnything || grid[gridIdx];
            float c = grid[gridIdx] ? 1 : 0;
            glColor3f(c, c, c);
            float cx = (x + 0.5f) * xSize;
            float cy = (y + 0.5f) * ySize;

            glBegin(GL_POLYGON);
            glVertex2f(cx - xSize / 2, cy - ySize / 2);  // top left
            glVertex2f(cx + xSize / 2, cy - ySize / 2);  // top right
            glVertex2f(cx + xSize / 2, cy + ySize / 2);  // bottom right
            glVertex2f(cx - xSize / 2, cy + ySize / 2);  // bottom left
            glEnd();
        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    _displayFrameRate();

    if (!printedAnything && glutGetWindow()) glutDestroyWindow(glutGetWindow());
}

}  // namespace display