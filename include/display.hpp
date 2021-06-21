#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "types.hpp"

class Display {
  public:
    Display(int *const argc, char **const argv, void (&loopFunc)(),
            const bool pCpuOnly);
    ~Display();
    void start();
    void stop();
    void draw(const bool logEnabled = false, const ulong itsPerSecond = 0);
    void update_grid_buffers_cpu();
    // cuda code may need to register this buffer
    uint const &grid_vbo() const { return mGridVBO; }

  private:
    uint mShaderProgram;
    uint mGridVAO;
    uint mGridVBO;
    fvec2s *mGridVertices;
    size_t mGridVerticesSize;
    bool mGpuOnly;

    static void reshape(const int width, int height);
    void update_title(const ulong itsPerSecond);
    void setup_shader_program();
    void setup_grid_buffers();
    void setup_grid_vertices();

}; // namespace display

#endif