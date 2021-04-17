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

typedef struct sGridRenderInfo {
    unsigned int numVertices;
    unsigned int numVerticesX;
    unsigned int numVerticesY;
    unsigned int cellsPerVerticeX;
    unsigned int cellsPerVerticeY;
} GridRenderInfo;

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
    GridRenderInfo const &get_grid_render_info() const { return mRenderInfo; }

  private:
    unsigned int mShaderProgram;
    unsigned int mGridVAO;
    unsigned int mGridVBO;
    vec2s *mGridVertices;
    size_t mGridVerticesSize;
    GridRenderInfo mRenderInfo;
    bool mGpuOnly;

    static void reshape(int pWidth, int pHeight);
    void calc_frameRate();
    void setup_shader_program();
    void setup_grid_buffers();
    void setup_grid_vertices(vec2s *vertices);

}; // namespace display

#endif