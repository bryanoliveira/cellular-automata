// isolate opengl specific imports
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <ctime>
#include <sstream>

#include "display.hpp"

/**
 * Sets up GLUT, GLEW, OpenGL methods and buffers.
 */
Display::Display(int *argc, char **argv, void (*loopFunc)()) {
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

    // setup shaders
    Display::setupShaderProgram();
    // setup grid & OpenGL buffers
    Display::setupGridBuffers();
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
    // bind VAO, which implicitly binds our EBO
    glBindVertexArray(gridVAO);
    // params:
    //      mode / OpenGL primitive
    //      count of elements to draw (6 vertices)
    //      type of the indices
    //      offset in the EBO or an index array
    glDrawElements(GL_QUADS, Display::nGridVertices, GL_UNSIGNED_INT, 0);
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
    // update vertice states
    for (unsigned int idx = 0; idx < Display::nGridVertices; idx++) {
        // remember we have 4 contiguous vertices for each cell, so each vertice
        // index/4 corresponds to the grid cell
        gridVertices[idx].state = (float)grid[int(idx / 4)];
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
    gluOrtho2D(config::left, config::right, config::bottom, config::top);

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
    gridVerticesSize = Display::nGridVertices * sizeof(vec3s);
    gridVertices = (vec3s *)malloc(gridVerticesSize);
    size_t indicesSize = Display::nGridVertices * sizeof(unsigned int);
    unsigned int *indices = (unsigned int *)malloc(indicesSize);

    // configure vertices and indices
    Display::setupGridVertices(gridVertices, indices);

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

    // generate an Element Buffer Object to iterate on the vertices of the VBO
    unsigned int gridEBO;
    glGenBuffers(1, &gridEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gridEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize, indices, GL_STATIC_DRAW);
    // free indices memory since it is already copied to the buffer
    // free(indices);

    // tell OpenGL how to interpret the vertex buffer data
    // params:
    //      *location* of the position vertex (as in the vertex shader)
    //      size of the vertex attribute, which is a vec3s (size 3)
    //      type of each attribute (vec3s is made of floats)
    //      use_normalization?
    //      stride of each position vertex in the array. It could be 0 as data
    //      is tightly packed. offset in bytes where the data start in the
    //      buffer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3s), (void *)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(vec3s),
                          (void *)(3 * sizeof(float)));
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // unbind buffers
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Display::setupGridVertices(vec3s *vertices, unsigned int *indices) {
    // setup vertices
    // iterate over the number of cells
    for (unsigned int y = 0, idx = 0; y < config::rows; y++) {
        for (unsigned int x = 0; x < config::cols; ++x) {
            // vertices live in an (-1, 1) tridimensional space
            // we need to calculate the position of each vertice inside a 2d
            // grid top left
            vertices[idx] =
                vec3s(-1.0f + x * (2.0f / config::cols),
                      -1.0f + y * (2.0f / config::rows), 0.0f, 0.0f);
            indices[idx] = idx;
            idx++;
            // top right
            vertices[idx] =
                vec3s(-1.0f + (x + 1) * (2.0f / config::cols),
                      -1.0f + y * (2.0f / config::rows), 0.0f, 0.0f);
            indices[idx] = idx;
            idx++;
            // bottom right
            vertices[idx] =
                vec3s(-1.0f + (x + 1) * (2.0f / config::cols),
                      -1.0f + (y + 1) * (2.0f / config::rows), 0.0f, 0.0f);
            indices[idx] = idx;
            idx++;
            // bottom left
            vertices[idx] =
                vec3s(-1.0f + x * (2.0f / config::cols),
                      -1.0f + (y + 1) * (2.0f / config::rows), 0.0f, 0.0f);
            indices[idx] = idx;
            idx++;
        }
    }
}