#ifndef AUTOMATA_BIT_GPU_HPP_
#define AUTOMATA_BIT_GPU_HPP_

// this is needed to declare CUDA-specific variable types
#include <cuda_runtime.h>
#include <curand_kernel.h> // for the curandStatate type

#include "automata_base_gpu.cuh"
#include "types.hpp"

namespace gpu {

class AutomataBit : public AutomataBase {
  public:
    // call superclass constructor as is
    AutomataBit(const uint seed, std::ostringstream *const pLiveLogBuffer,
                const uint *const gridVBO = nullptr)
        : AutomataBase(seed, pLiveLogBuffer, gridVBO) {}

  protected:
    // override kernel calls
    void run_init_kernel() override;
    void run_evolution_kernel(const bool countAliveCells = false) override;
    void run_render_kernel(fvec2s *gridVertices) override;
};

} // namespace gpu

#endif