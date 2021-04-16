#include "grid_gpu.hpp"
#include "kernels.cuh" // CUDA_ASSERT, etc

namespace gpu {

Grid::Grid(size_t pSize, bool init, GridType initValue) {
    mSize = pSize * sizeof(GridType);

    CUDA_ASSERT(cudaMallocManaged(&mData, mSize));
    if (init)
        CUDA_ASSERT(cudaMemset(mData, initValue, mSize));

    // prefetch grid to GPU
    CUDA_ASSERT(cudaMemPrefetchAsync(grid, gridBytes, gpuDeviceId));
    CUDA_ASSERT(cudaMemPrefetchAsync(nextGrid, gridBytes, gpuDeviceId));
}

Grid::~Grid() { CUDA_ASSERT(cudaFree(mData)); }

void Grid::swap(Grid &pGrid) {
    GridType *tempData = mData;
    mData = pGrid.mData;
    pGrid.mData = tempData;
}

__device__ GridType Grid::get(unsigned int position) { return mData[position]; }

} // namespace gpu