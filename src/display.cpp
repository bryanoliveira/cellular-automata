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
    if (config::width % 2 == 1 || config::height % 2 == 1) {
        fprintf(stderr, "Width and Height must be even integers!\n");
        exit(EXIT_FAILURE);
    }

    // define sizes & proportions
    mRenderInfo.numVerticesX =
        config::cols > config::width ? config::width : config::cols;
    mRenderInfo.numVerticesY =
        config::rows > config::height ? config::height : config::rows;
    mRenderInfo.numVertices =
        mRenderInfo.numVerticesX * mRenderInfo.numVerticesY;
    mRenderInfo.cellsPerVerticeX = config::cols / mRenderInfo.numVerticesX;
    mRenderInfo.cellsPerVerticeY = config::rows / mRenderInfo.numVerticesY;

    // config controls mins and maxes
    controls::minScale = 1;
    controls::maxScale =
        std::max(mRenderInfo.cellsPerVerticeX, mRenderInfo.cellsPerVerticeY);

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
 * Draws on screen by iterating on the grid, the naive way.
 */
void Display::draw_naive(bool logEnabled, unsigned long itsPerSecond) {
    // draw grid without proper OpenGL Buffers, the naive way
    // glClear(GL_COLOR_BUFFER_BIT);

    // draw objects
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

    // glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    if (logEnabled)
        update_title(itsPerSecond);
}

/**
 * Draws on screen using OpenGL buffers, which is much faster.
 * Requires that the buffers are updated before this function is called.
 */
void Display::draw(bool logEnabled, unsigned long itsPerSecond) {
    glClear(GL_COLOR_BUFFER_BIT);

    // use configured shaders
    glUseProgram(mShaderProgram);
    // bind VAO, which implicitly binds our VBO
    glBindVertexArray(mGridVAO);
    // params:
    //      the OpenGL primitive we will draw
    //      the starting index of the vertex array
    //      how many vertices to draw (a square has 4 vertices)
    glDrawArrays(GL_POINTS, 0, mRenderInfo.numVertices);
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

    // reset vertices
    for (unsigned int idx = 0; idx < mRenderInfo.numVertices; idx++) {
        mGridVertices[idx].state = 0;
    }

    unsigned int sectionSizeX =
        std::floor(config::cols / float(controls::scale));
    unsigned int sectionSizeY =
        std::floor(config::rows / float(controls::scale));
    unsigned int densityX =
        std::ceil(sectionSizeX / float(mRenderInfo.numVerticesX));
    unsigned int densityY =
        std::ceil(sectionSizeY / float(mRenderInfo.numVerticesY));

    int startX = (config::cols / 2 + controls::position[0]) - sectionSizeX / 2;
    int endX = (config::cols / 2 + controls::position[0]) + sectionSizeX / 2;
    int startY = (config::rows / 2 + controls::position[1]) - sectionSizeY / 2;
    int endY =
        (config::rows / 2 + controls::position[1]) / 2 + sectionSizeY / 2;

    unsigned int gridStartX, gridEndX, gridStartY, gridEndY, vStartX, vEndX,
        vStartY, vEndY;

    // fix out of border positions
    // startX
    if (startX < 0)
        gridStartX = vStartX = 0;
    else
        gridStartX = vStartX = startX;
    // endX
    if (endX > (int)config::cols)
        gridEndX = vEndX = config::cols;
    else
        gridEndX = vEndX = endX;
    // startY
    if (startY < 0)
        gridStartY = vStartY = 0;
    else
        gridStartY = vStartY = startY;
    // endY
    if (endY > (int)config::rows)
        gridEndY = vEndY = config::rows;
    else
        gridEndY = vEndY = endY;

    std::cout << controls::position[0] << " x " << controls::position[1]
              << " | " << sectionSizeX << " - " << sectionSizeY << " sec xy / "
              << startX << " - " << endX << " x / " << startY << " - " << endY
              << " y / " << densityX << " - " << densityY << " dens xy / "
              << sectionSizeX / densityX << " - " << sectionSizeY / densityY
              << " map xy / " << controls::scale << " scale / "
              << (endX - startX - 1) / densityX << " maxX / cmaxV "
              << ((endY - startY - 1) / densityY) * mRenderInfo.numVerticesX +
                     (endX - 1) / densityX
              << " - maxV " << mRenderInfo.numVertices << std::endl;

    // update vertice states
    for (unsigned int y = gridStartY; y < gridEndY; y++) {
        for (unsigned int x = gridStartX; x < gridEndX; x++) {
            // the grid index
            unsigned int idx = y * config::cols + x;
            // if grid state is on
            if (grid[idx]) {
                unsigned int vx = (x - vStartX) / densityX;
                unsigned int vy = (y - vStartY) / densityY;
                // std::cout << vx << " " << vy << " = "
                //           << vy * mRenderInfo.numVerticesX + vx << " | ";
                // if (!mGridVertices[vy * mRenderInfo.numVerticesX + vx].state)
                // if buffer state is off, store the max
                mGridVertices[vy * mRenderInfo.numVerticesX + vx].state =
                    std::max(
                        mGridVertices[vy * mRenderInfo.numVerticesX + vx].state,
                        int(grid[idx]));
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
          << config::rows << "x" << config::cols << " | pos "
          << (int)controls::position[0] << "x" << (int)controls::position[1]
          << " | " << std::fixed << std::setprecision(2) << controls::scale
          << "x | " << itsPerSecond << " it/s";
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
        "void main() {\n"
        "   gl_Position = vec4(posA.x, posA.y, 0, 1.0);\n"
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
    mGridVerticesSize = mRenderInfo.numVertices * sizeof(vec2s);
    mGridVertices = (vec2s *)malloc(mGridVerticesSize);

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
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2s), (void *)0);
    glVertexAttribPointer(1, 1, GL_INT, GL_FALSE, sizeof(vec2s),
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

void Display::setup_grid_vertices(vec2s *vertices) {
    // setup vertices
    // iterate over the number of cells
    for (unsigned int y = 0, idx = 0; y < mRenderInfo.numVerticesY; y++) {
        for (unsigned int x = 0; x < mRenderInfo.numVerticesX; ++x) {
            // vertices live in an (-1, 1) tridimensional space
            // we need to calculate the position of each vertice inside a 2d
            // grid top left
            vertices[idx] =
                vec2s(-1.0f + x * (2.0f / mRenderInfo.numVerticesX),
                      1.0f - y * (2.0f / mRenderInfo.numVerticesY), false);
            idx++;
        }
    }
}