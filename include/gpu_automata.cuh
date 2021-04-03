#ifndef GPU_AUTOMATA_HPP_
#define GPU_AUTOMATA_HPP_

// this is needed to declare CUDA-specific private variables
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "config.hpp"
#include "display.hpp" // for the vec2s type
#include "grid.hpp"
#include "kernels.cuh"

namespace gpu {

class BaseAutomata {
  public:
    BaseAutomata(unsigned long seed, std::ostringstream *pLiveLogBuffer,
                 const unsigned int *gridVBO = NULL);
    ~BaseAutomata();
    void compute_grid();
    void update_grid_buffers();

  private:
    dim3 mGpuBlocks;
    dim3 mGpuThreadsPerBlock;
    curandState *mGlobalRandState;
    struct cudaGraphicsResource *mGridVBOResource = NULL;
    cudaStream_t mEvolveStream, mBufferUpdateStream;
    unsigned int *mActiveCellCount;
    std::ostringstream *mLiveLogBuffer;
};

} // namespace gpu

#endif