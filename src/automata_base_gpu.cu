// isolate cuda specific imports
#include <chrono>
#include <cuda_gl_interop.h>
#include <sstream>

#include "automata_base_gpu.cuh"

namespace gpu {

AutomataBase::AutomataBase(unsigned long randSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           const unsigned int *gridVBO) {
    int gpuDeviceId;
    cudaDeviceProp gpuProps;
    size_t gridSize = config::rows * config::cols;

    // define common kernel configs
    cudaGetDevice(&gpuDeviceId);
    cudaGetDeviceProperties(&gpuProps, gpuDeviceId);
    // blocks should be a multiple of #SMs on the grid (assume #SMs is even) !
    // actually the number of blocks should
    mGpuBlocks = dim3(gpuProps.multiProcessorCount / 2,
                      gpuProps.multiProcessorCount /
                          2); // 1156 on a 3080 - 17 blocks per SM
    // threads should be a multiple of warpSize on the block (assume warpSize is
    // even - it usually is 32)
    mGpuThreadsPerBlock = dim3(
        gpuProps.warpSize / 2,
        gpuProps.warpSize / 2); // 256 on a 3080 - 8 threads per SP (I think)

    // allocate memory
    CUDA_ASSERT(
        cudaMalloc(&mGlobalRandState,
                   gridSize * sizeof(curandState))); // gpu only, not managed
    CUDA_ASSERT(cudaMallocManaged(&mActiveCellCount, sizeof(unsigned int)));
    // instantiate grids
    grid = Grid(gridSize, true, false);
    nextGrid = Grid(gridSize, false);

    // initialize RNG
    k_setup_rng<<<mGpuBlocks, mGpuThreadsPerBlock>>>(
        config::rows, config::cols, mGlobalRandState, randSeed);

    // initialize grid
    k_init_grid<<<mGpuBlocks, mGpuThreadsPerBlock>>>(
        &grid, config::rows, config::cols, mGlobalRandState, config::fillProb);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    // create grid evolving CUDA stream
    CUDA_ASSERT(cudaStreamCreate(&mEvolveStream));

    // if rendering is enabled
    if (gridVBO) {
        // create buffer updating CUDA stream
        CUDA_ASSERT(cudaStreamCreate(&mBufferUpdateStream));
        // register OpenGL VBO to use with CUDA
        CUDA_ASSERT(cudaGraphicsGLRegisterBuffer(
            &mGridVBOResource, *gridVBO, cudaGraphicsMapFlagsWriteDiscard));
    }

    // define the live log buffer
    mLiveLogBuffer = pLiveLogBuffer;
}

AutomataBase::~AutomataBase() {
    // wait for pending operations to complete
    CUDA_ASSERT(cudaDeviceSynchronize());
    // destroy secondary streams
    cudaStreamDestroy(mEvolveStream);
    cudaStreamDestroy(mBufferUpdateStream);
    // unregister mapped resource (needs the GL context to still be set)
    // CUDA_ASSERT(cudaGraphicsUnregisterResource(mGridVBOResource));
    // free the grid
    delete grid;
    delete nextGrid;
    CUDA_ASSERT(cudaFree(mGlobalRandState));
}

void AutomataBase::compute_grid(bool logEnabled) {
    std::chrono::steady_clock::time_point timeStart;
    if (logEnabled)
        timeStart = std::chrono::steady_clock::now();

    *mActiveCellCount = 0; // this will be moved CPU<->GPU automatically
    run_evolution_kernel(logEnabled); // count alive cells if log is enabled
    // should I call cudaDeviceSynchronize?
    CUDA_ASSERT(cudaDeviceSynchronize());

    // simply swap buffers to avoid reallocation
    grid->swap(nextGrid);

    // calculate timings and update live buffer
    if (logEnabled)
        *mLiveLogBuffer << "| Evolve Kernel: "
                        << std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - timeStart)
                               .count()
                        << " ns | Active cells: " << *mActiveCellCount << " |";
}

void AutomataBase::run_evolution_kernel(bool countAliveCells) {
    k_compute_grid_count_rule<<<mGpuBlocks, mGpuThreadsPerBlock, 0,
                                mEvolveStream>>>(
        &grid, &nextGrid, config::rows, config::cols, mGlobalRandState,
        config::virtualFillProb, countAliveCells, mActiveCellCount);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBase::update_grid_buffers() {
    if (!mGridVBOResource) {
        fprintf(stderr, "ERROR: Cannot call update_grid_buffers with rendering "
                        "disabled.\n");
        exit(EXIT_FAILURE);
    }

    // map OpenGL buffer object for writing from CUDA
    vec2s *gridVertices;
    CUDA_ASSERT(cudaGraphicsMapResources(1, &mGridVBOResource, 0));
    size_t numBytes;
    CUDA_ASSERT(cudaGraphicsResourceGetMappedPointer(
        (void **)&gridVertices, &numBytes, mGridVBOResource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", numBytes);

    // launch kernel
    k_update_grid_buffers<<<mGpuBlocks, mGpuThreadsPerBlock, 0,
                            mBufferUpdateStream>>>(&grid, gridVertices,
                                                   config::rows, config::cols);

    CUDA_ASSERT(cudaGetLastError());
    // should I call cudaDeviceSynchronize?
    // CUDA_ASSERT(cudaDeviceSynchronize());

    // unmap buffer object
    CUDA_ASSERT(cudaGraphicsUnmapResources(1, &mGridVBOResource, 0));
}

} // namespace gpu