#include "display.hpp"

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
// TODO: make the program sleep instead of busy waiting, perhaps?
bool display() {
    // execute display every X seconds
    static clock_t last_exec_clock = 0;
    if (last_exec_clock > 0 &&
        float(clock() - last_exec_clock) / CLOCKS_PER_SEC <
            config::render_delay) {
        glutPostRedisplay();
        return false;
    }
    last_exec_clock = clock();

    // glClear(GL_COLOR_BUFFER_BIT);

    float xSize = 1.0f / ((float)config::cols);
    float ySize = 1.0f / ((float)config::rows);
    bool printedAnything = false;

    for (int i = 0; i < config::rows; i++) {
        for (int j = 0; j < config::cols; j++) {
            printedAnything = printedAnything || grid[i][j];
            float c = grid[i][j] ? 1 : 0;
            glColor3f(c, c, c);
            float cx = (j + 0.5f) * xSize;
            float cy = (i + 0.5f) * ySize;

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
    return true;
}
