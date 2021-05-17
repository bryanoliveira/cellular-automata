// isolate opengl specific imports
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "display.hpp"

/**
 * Sets up GLUT, GLEW, OpenGL methods and buffers.
 */
Display::Display(int *pArgc, char **pArgv, void (*pLoopFunc)(), bool pCpuOnly) {
    std::cout << "Initializing display..." << std::endl;

    if (config::width % 2 == 1 || config::height % 2 == 1) {
        fprintf(stderr, "Width and Height must be even integers!\n");
        exit(EXIT_FAILURE);
    }

    proj::init();

    // will we be using only gpu?
    Display::mGpuOnly = !pCpuOnly;

    // init glut
    glutInit(pArgc, pArgv);
    glutInitWindowSize(config::width, config::height);
    glutCreateWindow(config::programName.c_str());
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    // callbacks
    glutDisplayFunc(pLoopFunc);
    glutMouseFunc(controls::mouse);
    glutKeyboardFunc(controls::keyboard);
    glutMotionFunc(controls::motion);
    glutReshapeFunc(reshape);
    // default initialization
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(5.0f);
    reshape(config::width, config::height); // set viewport, etc

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
    Display::setup_shader_program();
    // setup grid & OpenGL buffers
    Display::setup_grid_buffers();

    std::cout << "Display is ready." << std::endl;
}

Display::~Display() {
    // free buffer objects
    glDeleteVertexArrays(1, &mGridVAO);
    glDeleteBuffers(1, &mGridVBO);
    // free RAM vertices
    if (!Display::mGpuOnly)
        free(mGridVertices);
}

void Display::start() { glutMainLoop(); }

void Display::stop() {
    if (glutGetWindow())
        glutDestroyWindow(glutGetWindow());
}

/**
 * Draws on screen using OpenGL buffers, which is much faster.
 * Requires that the buffers are updated before this function is called.
 */
void Display::draw(bool logEnabled, unsigned long itsPerSecond) {
    glClear(GL_COLOR_BUFFER_BIT);

    // create default transform matrix
    glm::mat4 trans = glm::mat4(1.0f);
    // rotate
    trans =
        glm::rotate(trans, controls::rotation.x / 50, glm::vec3(1.0, 0.0, 0.0));
    trans =
        glm::rotate(trans, controls::rotation.y / 50, glm::vec3(0.0, 1.0, 0.0));
    // scale
    trans = glm::scale(trans, glm::vec3(controls::secondaryScale,
                                        controls::secondaryScale, 1));
    // translate
    trans =
        glm::translate(trans, glm::vec3(-controls::secondaryPosition.x,
                                        controls::secondaryPosition.y, 0.0f));

    // apply transforms to the shaders
    unsigned int transformLoc =
        glGetUniformLocation(mShaderProgram, "transform");
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));

    // use configured shaders
    glUseProgram(mShaderProgram);
    // bind VAO, which implicitly binds our VBO
    glBindVertexArray(mGridVAO);
    // params:
    //      the OpenGL primitive we will draw
    //      the starting index of the vertex array
    //      how many vertices to draw (a square has 4 vertices)
    glDrawArrays(GL_POINTS, 0, proj::info.totalVertices);
    // unbind VAO
    glBindVertexArray(0);

    // actually in the screen
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    if (logEnabled)
        update_title(itsPerSecond);
}

// this method is implemented here so it can access the vertice array directly
void Display::update_grid_buffers_cpu() {
    if (mGpuOnly) {
        fprintf(stderr, "Display Error: cannot call updateGridBuffersCPU on "
                        "GPU Only mode!\n");
        exit(EXIT_FAILURE);
    }

    // update projection limits
    proj::update();

    // reset vertices states
    for (uint idx = 0; idx < proj::info.totalVertices; idx++) {
        mGridVertices[idx].state = 0;
    }

    // update vertice states
    for (uint y = proj::gridLimY.start; y < proj::gridLimY.end; y++) {
        for (uint x = proj::gridLimX.start; x < proj::gridLimX.end; x++) {
            // the grid index
            uint idx = y * config::cols + x;
            // if grid state is on
            if (grid[idx]) {
                // map
                uint vIdx = proj::getVerticeIdx({x, y});
                // update vertice state
                mGridVertices[vIdx].state =
                    std::max(mGridVertices[vIdx].state, int(grid[idx]));
            }
        }
    }

    // bind VBO to be updated
    glBindBuffer(GL_ARRAY_BUFFER, mGridVBO);
    // update the VBO data
    glBufferData(GL_ARRAY_BUFFER, mGridVerticesSize, mGridVertices,
                 GL_STATIC_DRAW);
    // unbind VBO
    glBindBuffer(GL_ARRAY_BUFFER, mGridVBO);
}

void Display::update_title(unsigned long itsPerSecond) {
    std::ostringstream title;
    title << config::programName << " | " << config::patternFileName << " | "
          << config::rows << "x" << config::cols << " | gpos " << std::fixed
          << std::setprecision(2) << controls::position.x << "x"
          << controls::position.y << " | ppos " << controls::secondaryPosition.x
          << "x" << controls::secondaryPosition.y << " | gs " << controls::scale
          << "x | ps " << controls::secondaryScale << "x | " << itsPerSecond
          << " it/s";
    glutSetWindowTitle(title.str().c_str());
}

void Display::reshape(int pWidth, int pHeight) {
    glViewport(0, 0, pWidth, pHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // left, right, bottom, top
    gluOrtho2D(-1, 1, 1, -1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutPostRedisplay();
}

void Display::setup_shader_program() {
    // define shaders
    const char *vertexShaderSource =
        "#version 460 core\n"
        "layout (location = 0) in vec2 posA;\n"
        // implicit int->float conversion
        "layout (location = 1) in int state;\n"
        "out float v_state;\n"
        "uniform mat4 transform;\n"
        "void main() {\n"
        "   gl_Position = transform * vec4(posA.x, posA.y, 0, 1.0);\n"
        "   v_state = state;\n"
        "}\0";

    const char *fragmentShaderSource =
        "#version 460 core\n"
        "in float v_state;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(v_state, v_state, v_state, v_state);\n"
        "}\0";

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
    mShaderProgram = glCreateProgram();
    // link the shaders
    glAttachShader(mShaderProgram, vertexShader);
    glAttachShader(mShaderProgram, fragmentShader);
    glLinkProgram(mShaderProgram);
    // check linker errors
    glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(mShaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n",
                infoLog);
        exit(1);
    }
    // clear already linked shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Display::setup_grid_buffers() {
    // we should probably free the vertices arrays at Display::stop, but we're
    // destroying the program after they are no longer needed
    mGridVerticesSize = proj::info.totalVertices * sizeof(fvec2s);
    mGridVertices = (fvec2s *)malloc(mGridVerticesSize);

    // configure vertices
    setup_grid_vertices(mGridVertices);

    // configure the Vertex Array Object so we configure our objects only once
    glGenVertexArrays(1, &mGridVAO);
    // bind it
    glBindVertexArray(mGridVAO);

    // generate Vertex Buffer Object and store it's ID
    glGenBuffers(1, &mGridVBO);
    // only 1 buffer of a type can be bound simultaneously, that's how OpenGL
    // knows what object we're talking about on each command that refers to the
    // type (GL_ARRAY_BUFFER in this case) bind buffer to context
    glBindBuffer(GL_ARRAY_BUFFER, mGridVBO);
    // copy vertext data to buffer
    //  GL_STREAM_DRAW: the data is set only once and used by the GPU at most a
    //  few times. GL_STATIC_DRAW: the data is set only once and used many
    //  times. GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    glBufferData(GL_ARRAY_BUFFER, mGridVerticesSize, mGridVertices,
                 GL_STATIC_DRAW);

    // tell OpenGL how to interpret the vertex buffer data
    // params:
    //      - *location* of the position vertex (as in the vertex shader)
    //      - size of the vertex attribute, which is a vec2s (size 2)
    //      - type of each attribute (vec2s is made of floats)
    //      - use_normalization?
    //      - stride of each position vertex in the array. It could be 0 as data
    //      is tightly packed. offset in bytes where the data start in the
    //      buffer
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(fvec2s), (void *)0);
    glVertexAttribPointer(1, 1, GL_INT, GL_FALSE, sizeof(fvec2s),
                          (void *)(2 * sizeof(float)));
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // unbind buffers
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // free RAM since vertices will be updated in GPU
    if (mGpuOnly)
        free(mGridVertices);
}

void Display::setup_grid_vertices(fvec2s *vertices) {
    // setup vertices
    // iterate over the number of cells
    for (unsigned int y = 0, idx = 0; y < proj::info.numVertices.y; y++) {
        for (unsigned int x = 0; x < proj::info.numVertices.x; ++x) {
            // vertices live in an (-1, 1) tridimensional space
            // we need to calculate the position of each vertice inside a 2d
            // grid top left
            vertices[idx] =
                fvec2s({-1.0f + x * (2.0f / proj::info.numVertices.x),
                        1.0f - y * (2.0f / proj::info.numVertices.y)},
                       false);
            idx++;
        }
    }
}