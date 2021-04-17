#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "config.hpp"
#include "controls.hpp"
#include "grid.hpp"

typedef struct svec2s {
    float x, y;
    float state;
    svec2s(float _x, float _y, float _s) : x(_x), y(_y), state(_s){};
} vec2s;

class Display {
  public:
    Display(int *pArgc, char **pArgv, void (*pLoopFunc)(), bool pCpuOnly);
    ~Display();
    void start();
    void stop();
    void draw(bool logEnabled = false);
    void update_grid_buffers_cpu();
    void draw_naive(bool logEnabled = false);
    // cuda code may need to register this buffer
    unsigned int const &grid_vbo() const { return mGridVBO; }

  private:
    unsigned int mShaderProgram;
    unsigned int mGridVAO;
    unsigned int mGridVBO;
    vec2s *mGridVertices;
    size_t mGridVerticesSize;
    unsigned int mGridVerticesX;
    unsigned int mGridVerticesY;
    unsigned int mNumGridVertices;
    unsigned int mCellsPerVerticeX;
    unsigned int mCellsPerVerticeY;
    bool mGpuOnly;

    static void reshape(int pWidth, int pHeight);
    void calc_frameRate();
    void setup_shader_program();
    void setup_grid_buffers();
    void setup_grid_vertices(vec2s *vertices);

}; // namespace display

#endif