#ifndef AUTOMATA_BASE_GPU_HPP_
#define AUTOMATA_BASE_GPU_HPP_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandStatate type

#include "automata_interface.hpp"
#include "config.hpp"
#include "grid.hpp"
#include "kernels.cuh"
#include "projection.hpp"
#include "types.hpp"

namespace gpu {

class AutomataBase : public AutomataInterface {
  public:
    AutomataBase(const uint seed, std::ostringstream *const pLiveLogBuffer,
                 const uint *const gridVBO = nullptr);
    virtual ~AutomataBase();
    virtual void prepare() override;
    virtual void evolve(const bool logEnabled = false) override;
    virtual void update_grid_buffers() override;

  protected:
    int mGpuDeviceId;
    const size_t mGridSize = config::rows * config::cols;
    const size_t mGridBytes = mGridSize * sizeof(GridType);
    curandState *mGlobalRandState;
    cudaStream_t mEvolveStream, mBufferUpdateStream;
    unsigned int *mActiveCellCount;
    std::ostringstream *mLiveLogBuffer;
    // rendering stuff
    struct cudaGraphicsResource *mGridVBOResource = nullptr;

    virtual void run_init_kernel();
    virtual void run_evolution_kernel(const bool countAliveCells = false);
    virtual void run_render_kernel(fvec2s *gridVertices);
};

} // namespace gpu

#endif