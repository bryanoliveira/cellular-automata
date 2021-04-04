#include "automata_count_gpu.cuh"

namespace gpu {

CountAutomata::~CountAutomata() {}

void CountAutomata::run_evolution_kernel() {
    k_compute_grid_count_rule<<<mGpuBlocks, mGpuThreadsPerBlock, 0,
                                mEvolveStream>>>(
        grid, nextGrid, config::rows, config::cols, mGlobalRandState,
        config::virtualFillProb, mActiveCellCount);
    CUDA_ASSERT(cudaGetLastError());
}

} // namespace gpu