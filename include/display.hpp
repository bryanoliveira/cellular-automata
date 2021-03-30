#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "config.hpp"
#include "grid.hpp"

typedef struct _vec2s {
    float x, y;
    float state;
    _vec2s(float _x, float _y, float _s) : x(_x), y(_y), state(_s){};
} vec2s;

class Display {
  private:
    const char *vertexShaderSource =
        "#version 460 core\n"
        "layout (location = 0) in vec2 posA;\n"
        "layout (location = 1) in float state;\n"
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
    unsigned int shaderProgram;
    // we define 4 vertices for each cell so we can color them
    // neighbours independently
    unsigned int nGridVertices = config::rows * config::cols; // * 4
    unsigned int nGridCells = config::rows * config::cols;
    unsigned int gridVAO;
    vec2s *gridVertices;
    size_t gridVerticesSize;
    bool gpuOnly;

    static void reshape(int w, int h);
    void calcFrameRate();
    void setupShaderProgram();
    void setupGridBuffers();
    void setupGridVertices(vec2s *vertices);

  public:
    // cuda code may probably have to update this
    unsigned int gridVBO;

    Display(int *argc, char **argv, void (*loopFunc)(), bool _gpuOnly);
    ~Display();
    void start();
    void stop();
    void draw();
    void updateGridBuffersCPU();
    void drawNaive();

}; // namespace display

#endif