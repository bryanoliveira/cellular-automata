#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "config.hpp"
#include "grid.hpp"

typedef struct _vec3s {
    float x, y, z;
    float state;
    _vec3s(float _x, float _y, float _z, float _s)
        : x(_x), y(_y), z(_z), state(_s){};
} vec3s;

class Display {
  private:
    const char *vertexShaderSource =
        "#version 460 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in float state;\n"
        "out float v_state;\n"
        "void main() {\n"
        "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
        "   v_state = state;\n"
        "}\0";

    const char *fragmentShaderSource =
        "#version 460 core\n"
        "in float v_state;\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(v_state, v_state, v_state, v_state);\n"
        "}\0";
    unsigned int shaderProgram;
    // we define 4 vertices for each cell so we can color them
    // neighbours independently
    unsigned int nGridVertices = 4 * config::rows * config::cols;
    unsigned int nGridCells = config::rows * config::cols;
    unsigned int gridVAO;
    vec3s *gridVertices;
    size_t gridVerticesSize;

    static void reshape(int w, int h);
    void calcFrameRate();
    void setupShaderProgram();
    void setupGridBuffers();
    void setupGridVertices(vec3s *vertices, unsigned int *indices);

  public:
    // cuda code may probably have to update this
    unsigned int gridVBO;

    Display(int *argc, char **argv, void (*loopFunc)());
    void start();
    void stop();
    void draw();
    void updateGridBuffersCPU();
    void drawNaive();

}; // namespace display

#endif