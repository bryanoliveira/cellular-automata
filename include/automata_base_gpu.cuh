#ifndef AUTOMATA_BASE_GPU_HPP_
#define AUTOMATA_BASE_GPU_HPP_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandStatate type

#include "automata_interface.hpp"
#include "config.hpp"
#include "display.hpp" // for the vec2s type
#include "grid.hpp"
#include "kernels.cuh"

namespace gpu {

class AutomataBase : public AutomataInterface {
  public:
    AutomataBase(unsigned long seed, std::ostringstream *const pLiveLogBuffer,
                 const unsigned int *gridVBO = NULL,
                 const GridRenderInfo *pRenderInfo = NULL);
    virtual ~AutomataBase();
    virtual void compute_grid(bool logEnabled = false);
    virtual void update_grid_buffers();

  protected:
    dim3 mGpuBlocks;
    dim3 mGpuThreadsPerBlock;
    curandState *mGlobalRandState;
    cudaStream_t mEvolveStream, mBufferUpdateStream;
    unsigned int *mActiveCellCount;
    std::ostringstream *mLiveLogBuffer;
    // rendering stuff
    struct cudaGraphicsResource *mGridVBOResource = NULL;
    GridRenderInfo mRenderInfo;

    virtual void run_evolution_kernel(bool countAliveCells = false);
};

} // namespace gpu

#endif