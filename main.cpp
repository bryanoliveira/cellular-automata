/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 * Grid drawing inspired by https://github.com/asegizek/418-Final-Project/blob/master/display.cpp
 */

#include <iostream>
#include <ctime>
#include <sstream>
#include <time.h>
#include <GL/glu.h>
#include <GL/glut.h>

#define WIDTH 640
#define HEIGHT 480
#define ROWS 100
#define COLS 100
#define RENDER_DELAY 0.1
#define FILL_PROB 0.2

GLint width = WIDTH;
GLint height = HEIGHT;
GLint rows = ROWS;
GLint cols = COLS;
GLfloat left = 0.0;
GLfloat right = 1.0;
GLfloat bottom = 1.0;
GLfloat top = 0.0;

//// GRID HELPERS

bool** grid;

bool** initGrid(bool random = false) {
    bool** tempGrid = (bool**) calloc(ROWS, sizeof(bool*));

    for (int i = 0; i < ROWS; i++) {
        tempGrid[i] = (bool*) calloc(COLS, sizeof(bool));

        if(random) {
            for (int j = 0; j < COLS; j++) {
                tempGrid[i][j] = (float(rand()) / RAND_MAX) < FILL_PROB;
            }
        }
    }

    return tempGrid;
}
void freeGrid(bool ** tempGrid) {
    for (int i = 0; i < ROWS; i++)
        free(tempGrid[i]);
    free(tempGrid);
}

//// GRID COMPUTATION

void computeGrid() {
    bool** prevGrid = grid;
    bool** nextGrid = initGrid();

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int livingNeighbours = 0;
            // iterate over neighbourhood row by row
            for (int a = i - 1; a <= i + 1; a++) {
                // skip out-of-grid neighbours
                if (a < 0 || a >= ROWS) continue;
                // iterate over neighbourhood col by col
                for (int b = j - 1; b <= j + 1; b++) {
                    // skip out-of-grid neighbours and self position
                    if (b < 0 || b >= COLS || (a == i && b == j)) continue;
                    // if cell is alive, increase neighbours
                    if (prevGrid[a][b])
                        livingNeighbours++;
                }
            }

            // 1. Any live cell with two or three live neighbours survives.
            if (prevGrid[i][j])
                nextGrid[i][j] = livingNeighbours == 2 || livingNeighbours == 3;
            // 2. Any dead cell with three live neighbours becomes a live cell.
            else if (livingNeighbours == 3)
                nextGrid[i][j] = true;
            // 3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.
            else nextGrid[i][j] = false;
        }
    }

    freeGrid(prevGrid);
    grid = nextGrid;
}

void insertGlider(int row, int col) {
    grid[row - 1][row + 0] = true;
    grid[row + 0][row + 1] = true;
    grid[row + 1][row - 1] = true;
    grid[row + 1][row + 0] = true;
    grid[row + 1][row + 1] = true;
}

void insertBlinker(int row, int col) {
    grid[row - 1][col + 0] = true;
    grid[row + 0][col + 0] = true;
    grid[row + 1][col + 0] = true;
}

//// DISPLAY FUNCTIONS

void reshape(int w, int h) {
	width = w;
	height = h;

	glViewport(0, 0, width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(left, right, bottom, top);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}

void displayFrameRate() {
    static clock_t delta_ticks;
    static clock_t current_ticks = 0;
    static clock_t fps = 0;

    // init clock if needed
    if (current_ticks == 0) 
        current_ticks = clock();

    delta_ticks = clock() - current_ticks; // the time, in ms, that took to render the scene
    if(delta_ticks > 0)
        fps = CLOCKS_PER_SEC / delta_ticks;

    std::ostringstream title;
    title << "Grid (" << fps << " fps)";
    glutSetWindowTitle(title.str().c_str());

    // get clock for the next call
    current_ticks = clock();
}

void display() {
    // execute display every X seconds
    static clock_t last_exec_clock = 0;
    if (last_exec_clock > 0 && float(clock() - last_exec_clock) / CLOCKS_PER_SEC < RENDER_DELAY) {
        glutPostRedisplay();
        return;
    }
    last_exec_clock = clock();

    // glClear(GL_COLOR_BUFFER_BIT);

    float xSize = 1.0f/ ((float)cols);
    float ySize = 1.0f / ((float)rows);
    float size = 0.05f;
    bool printedAnything = false;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printedAnything = printedAnything || grid[i][j];
            float c = grid[i][j] ? 1 : 0;
            glColor3f(c, c, c);
            float cx = (j+0.5f)*xSize ;
            float cy = (i+0.5f)*ySize;
            
            glBegin(GL_POLYGON);
            glVertex2f(cx-xSize/2, cy-ySize/2); // top left
            glVertex2f(cx+xSize/2, cy-ySize/2); // top right
            glVertex2f(cx+xSize/2, cy+ySize/2); // bottom right
            glVertex2f(cx-xSize/2, cy+ySize/2); // bottom left
            glEnd();
            
        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    displayFrameRate();

    // compute grid afterwards to display initial state
    computeGrid();

    if (!printedAnything && glutGetWindow())
        glutDestroyWindow(glutGetWindow());
}

int main(int argc, char **argv) {
    srand (time(NULL));

    grid = initGrid(true);
    // insertGlider(ROWS/2, COLS/2);
    // insertBlinker(ROWS/2, COLS/);

    glutInit(&argc, argv);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Grid");
    
    glClear(GL_COLOR_BUFFER_BIT);
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
    return 0;
}
