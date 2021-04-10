#ifndef AUTOMATA_COUNT_GPU_HPP_
#define AUTOMATA_COUNT_GPU_HPP_

#include "automata_base_gpu.cuh"
#include "config.hpp"
#include "grid.hpp"
#include "kernels.cuh"

namespace gpu {

struct count_rule {
    int *aliveRange;
    int *deadRange;
};

class CountAutomata : public AutomataBase {
  public:
    using AutomataBase::AutomataBase;
    ~CountAutomata();

  protected:
    void run_evolution_kernel(bool countAliveCells) override;
};

} // namespace gpu

#endif