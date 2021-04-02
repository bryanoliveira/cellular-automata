// isolate opengl specific imports
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <ctime>
#include <sstream>

#include "display.hpp"

/**
 * Sets up GLUT, GLEW, OpenGL methods and buffers.
 */
Display::Display(int *argc, char **argv, void (*loopFunc)(), bool _gpuOnly) {
    Display::gpuOnly = _gpuOnly;

    // init glut
    glutInit(argc, argv);
    glutInitWindowSize(config::width, config::height);
    glutCreateWindow(config::program_name.c_str());
    glutDisplayFunc(loopFunc);
    glutReshapeFunc(reshape);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    GLenum gl_error = glGetError();
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "GLUT Error: %d\n", gl_error);
        exit(EXIT_FAILURE);
    }

    // init glew
    // this is required to setup GL functions (like glGenBuffers)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* glewInit failed, something is seriously wrong */
        fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // unchanging gl param
    glPointSize(5.0f);

    // setup shaders
    Display::setupShaderProgram();
    // setup grid & OpenGL buffers
    Display::setupGridBuffers();
}

Display::~Display() {
    // free buffer objects
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    // free RAM vertices
    if (!Display::gpuOnly)
        free(gridVertices);
}

void Display::start() { glutMainLoop(); }

void Display::stop() {
    if (glutGetWindow())
        glutDestroyWindow(glutGetWindow());
}

/**
 * Draws on screen by iterating on the grid, the naive way.
 */
void Display::drawNaive() {
    // draw grid without proper OpenGL Buffers, the naive way
    // glClear(GL_COLOR_BUFFER_BIT);

    float xSize = 1.0f / ((float)config::cols);
    float ySize = 1.0f / ((float)config::rows);

    for (unsigned int y = 0; y < config::rows; y++) {
        for (unsigned int x = 0; x < config::cols; x++) {
            unsigned int gridIdx = y * config::cols + x;

            float c = grid[gridIdx] ? 1 : 0;
            glColor3f(c, c, c);
            float cx = (x + 0.5f) * xSize;
            float cy = (y + 0.5f) * ySize;

            glBegin(GL_POLYGON);
            glVertex2f(cx - xSize / 2, cy - ySize / 2); // top left
            glVertex2f(cx + xSize / 2, cy - ySize / 2); // top right
            glVertex2f(cx + xSize / 2, cy + ySize / 2); // bottom right
            glVertex2f(cx - xSize / 2, cy + ySize / 2); // bottom left
            glEnd();
        }
    }
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    Display::calcFrameRate();
}

/**
 * Draws on screen using OpenGL buffers, which is much faster.
 * Requires that the buffers are updated before this function is called.
 */
void Display::draw() {
    // use configured shaders
    glUseProgram(shaderProgram);
    // bind VAO, which implicitly binds our VBO
    glBindVertexArray(gridVAO);
    // params:
    //      the OpenGL primitive we will draw
    //      the starting index of the vertex array
    //      how many vertices to draw (a square has 4 vertices)
    glDrawArrays(GL_POINTS, 0, Display::nGridVertices);
    // unbind VAO
    glBindVertexArray(0);

    // actually in the screen
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    Display::calcFrameRate();
}

// this could be on the automata file
void Display::updateGridBuffersCPU() {
    if (Display::gpuOnly) {
        fprintf(stderr, "Display Error: cannot call updateGridBuffersCPU on "
                        "GPU Only mode!\n");
        exit(EXIT_FAILURE);
    }

    // update vertice states
    for (unsigned int idx = 0; idx < Display::nGridVertices; idx++) {
        // remember we have 4 contiguous vertices for each cell, so each vertice
        // index/4 corresponds to the grid cell
        gridVertices[idx].state = (float)grid[idx]; // int(idx / 4)];
    }

    // bind VBO to be updated
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    // update the VBO data
    glBufferData(GL_ARRAY_BUFFER, gridVerticesSize, gridVertices,
                 GL_STATIC_DRAW);
    // unbind VBO
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
}

void Display::reshape(int w, int h) {
    config::width = w;
    config::height = h;

    glViewport(0, 0, config::width, config::height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // left, right, bottom, top
    gluOrtho2D(0, 1, 1, 0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutPostRedisplay();
}

void Display::calcFrameRate() {
    static clock_t delta_ticks;
    static clock_t current_ticks = 0;
    static clock_t fps = 0;

    // init clock if needed
    if (current_ticks == 0)
        current_ticks = clock();

    // the time, in ms, that took to render the scene
    delta_ticks = clock() - current_ticks;
    if (delta_ticks > 0)
        fps = CLOCKS_PER_SEC / delta_ticks;

    std::ostringstream title;
    title << config::program_name << " (" << fps << " fps)";
    glutSetWindowTitle(title.str().c_str());

    // get clock for the next call
    current_ticks = clock();
}

void Display::setupShaderProgram() {
    // variables to store shader compiling errors
    int success;
    char infoLog[512];

    // create our vertex shader
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // compile the shader code
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check compiler errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n",
                infoLog);
        exit(1);
    }

    // create our fragment shader
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // compile the shader code
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check compiler errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n",
                infoLog);
        exit(1);
    }

    // create a shader program to link our shaders
    shaderProgram = glCreateProgram();
    // link the shaders
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check linker errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n",
                infoLog);
        exit(1);
    }
    // clear already linked shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Display::setupGridBuffers() {
    // we should probably free the vertices arrays at Display::stop, but we're
    // destroying the program after they are no longer needed
    gridVerticesSize = Display::nGridVertices * sizeof(vec2s);
    gridVertices = (vec2s *)malloc(gridVerticesSize);

    // configure vertices
    Display::setupGridVertices(gridVertices);

    // configure the Vertex Array Object so we configure our objects only once
    glGenVertexArrays(1, &gridVAO);
    // bind it
    glBindVertexArray(gridVAO);

    // generate Vertex Buffer Object and store it's ID
    glGenBuffers(1, &gridVBO);
    // only 1 buffer of a type can be bound simultaneously, that's how OpenGL
    // knows what object we're talking about on each command that refers to the
    // type (GL_ARRAY_BUFFER in this case) bind buffer to context
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    // copy vertext data to buffer
    //  GL_STREAM_DRAW: the data is set only once and used by the GPU at most a
    //  few times. GL_STATIC_DRAW: the data is set only once and used many
    //  times. GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    glBufferData(GL_ARRAY_BUFFER, gridVerticesSize, gridVertices,
                 GL_STATIC_DRAW);

    // tell OpenGL how to interpret the vertex buffer data
    // params:
    //      *location* of the position vertex (as in the vertex shader)
    //      size of the vertex attribute, which is a vec2s (size 2)
    //      type of each attribute (vec2s is made of floats)
    //      use_normalization?
    //      stride of each position vertex in the array. It could be 0 as data
    //      is tightly packed. offset in bytes where the data start in the
    //      buffer
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2s), (void *)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(vec2s),
                          (void *)(2 * sizeof(float)));
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // unbind buffers
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // free RAM since vertices will be updated in GPU
    if (Display::gpuOnly)
        free(gridVertices);
}

void Display::setupGridVertices(vec2s *vertices) {
    // setup vertices
    // iterate over the number of cells
    for (unsigned int y = 0, idx = 0; y < config::rows; y++) {
        for (unsigned int x = 0; x < config::cols; ++x) {
            // vertices live in an (-1, 1) tridimensional space
            // we need to calculate the position of each vertice inside a 2d
            // grid top left
            vertices[idx] = vec2s(-1.0f + x * (2.0f / config::cols),
                                  -1.0f + y * (2.0f / config::rows), false);
            idx++;
            // // top right
            // vertices[idx] = vec2s(-1.0f + (x + 1) * (2.0f / config::cols),
            //                       -1.0f + y * (2.0f / config::rows), false);
            // idx++;
            // // bottom right
            // vertices[idx] =
            //     vec2s(-1.0f + (x + 1) * (2.0f / config::cols),
            //           -1.0f + (y + 1) * (2.0f / config::rows), false);
            // idx++;
            // // bottom left
            // vertices[idx] =
            //     vec2s(-1.0f + x * (2.0f / config::cols),
            //           -1.0f + (y + 1) * (2.0f / config::rows), false);
            // idx++;
        }
    }
}