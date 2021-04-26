#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "config.hpp"
#include "controls.hpp"
#include "grid.hpp"
#include "projection.hpp"
#include "types.hpp"

class Display {
  public:
    Display(int *pArgc, char **pArgv, void (*pLoopFunc)(), bool pCpuOnly);
    ~Display();
    void start();
    void stop();
    void draw(bool logEnabled = false, unsigned long itsPerSecond = 0);
    void update_grid_buffers_cpu();
    // cuda code may need to register this buffer
    unsigned int const &grid_vbo() const { return mGridVBO; }

  private:
    unsigned int mShaderProgram;
    unsigned int mGridVAO;
    unsigned int mGridVBO;
    fvec2s *mGridVertices;
    size_t mGridVerticesSize;
    bool mGpuOnly;

    static void reshape(int pWidth, int pHeight);
    void update_title(unsigned long itsPerSecond);
    void setup_shader_program();
    void setup_grid_buffers();
    void setup_grid_vertices(fvec2s *vertices);

}; // namespace display

#endif