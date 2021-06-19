// isolate cuda specific imports
#include <chrono>
#include <cuda_gl_interop.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <iostream>

#include "automata_base_gpu.cuh"
#include "stats.hpp"

namespace gpu {

AutomataBase::AutomataBase(const unsigned long randSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           const uint *const gridVBO) {
    spdlog::info("Initializing automata GPU engine...");

    cudaDeviceProp gpuProps;
    cudaGetDevice(&mGpuDeviceId);
    cudaGetDeviceProperties(&gpuProps, mGpuDeviceId);
    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1))

    // allocate memory
    // gpu only, not managed
    CUDA_ASSERT(cudaMalloc(&mGlobalRandState, mGridSize * sizeof(curandState)));
    // must be managed so pattern can be loaded from file
    CUDA_ASSERT(cudaMallocManaged(&grid, mGridBytes));
    CUDA_ASSERT(cudaMallocManaged(&nextGrid, mGridBytes));
    CUDA_ASSERT(cudaMallocManaged(&mActiveCellCount, sizeof(uint)));
    // initialization
    *mActiveCellCount = 0;
    CUDA_ASSERT(cudaMemset(grid, 0, mGridBytes)); // init grid with zeros (gpu)

    // define common kernel configs
    // set_optimal_kernel_config(mGpuDeviceId);
    int tBlockSize;   // The launch configurator returned block size
    int tMinGridSize; // The minimum grid size needed to achieve the
                      // maximum occupancy for a full device launch
    int tGridSize;    // The actual grid size needed, based on input size

    /* int* minGridSize,
     * int* blockSize,
     * T func,
     * size_t dynamicSMemSize = 0,
     * int blockSizeLimit = 0 */
    cudaOccupancyMaxPotentialBlockSize(&tMinGridSize, &tBlockSize,
                                       k_evolve_count_rule, 0, 0);
    // Round up according to array size
    tGridSize = (mGridSize + tBlockSize - 1) / tBlockSize;

    k_evolve_count_rule<<<tGridSize, tBlockSize>>>(
        grid, nextGrid, config::rows, config::cols, NULL, 0, false, NULL);

    cudaDeviceSynchronize();

    // calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, k_evolve_count_rule, tBlockSize, 0);

    float occupancy =
        (maxActiveBlocks * tBlockSize / gpuProps.warpSize) /
        (float)(gpuProps.maxThreadsPerMultiProcessor / gpuProps.warpSize);

    spdlog::info(
        "Kernel launch config set to {} blocks of size {} with occupancy of {}",
        tGridSize, tBlockSize, occupancy);

    // blocks should be a multiple of #SMs on the grid (assume #SMs is even) !
    // actually the number of blocks should
    mGpuBlocks = tGridSize; // 1156 on a 3080 - 17 blocks per SM
    // threads should be a multiple of warpSize on the block
    mGpuThreadsPerBlock = tBlockSize;

    ///////////////////////////////////////////

    // initialize RNG
    k_setup_rng<<<mGpuBlocks, mGpuThreadsPerBlock>>>(
        config::rows, config::cols, mGlobalRandState, randSeed);

    // prefetch grid to CPU so it can fill it properly
    CUDA_ASSERT(cudaMemPrefetchAsync(grid, mGridBytes, cudaCpuDeviceId));
    CUDA_ASSERT(cudaMemPrefetchAsync(nextGrid, mGridBytes, cudaCpuDeviceId));

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

    spdlog::info("Automata GPU engine is ready.");
}

AutomataBase::~AutomataBase() {
    // I believe member streams are destroyed automatically (since I get
    // segfault if do it here).
    // Also, graphics resources are automatically unbound.

    // free the grid
    CUDA_ASSERT(cudaFree(grid));
    CUDA_ASSERT(cudaFree(nextGrid));
    CUDA_ASSERT(cudaFree(mActiveCellCount));
    CUDA_ASSERT(cudaFree(mGlobalRandState));
    cudaDeviceReset();
}

void AutomataBase::prepare() {
    // prefetch grid data to GPU
    CUDA_ASSERT(cudaMemPrefetchAsync(grid, mGridBytes, mGpuDeviceId));
    CUDA_ASSERT(cudaMemPrefetchAsync(nextGrid, mGridBytes, mGpuDeviceId));
    CUDA_ASSERT(
        cudaMemPrefetchAsync(mActiveCellCount, sizeof(uint), mGpuDeviceId));

    // initialize grid with fillProb
    if (config::fillProb > 0)
        k_init_grid<<<mGpuBlocks, mGpuThreadsPerBlock>>>(
            grid, config::rows, config::cols, mGlobalRandState,
            config::fillProb);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void AutomataBase::evolve(const bool logEnabled) {
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    run_evolution_kernel(logEnabled); // count alive cells if log is enabled
    // should I call cudaDeviceSynchronize?
    CUDA_ASSERT(cudaDeviceSynchronize());

    // simply swap buffers to avoid reallocation
    GridType *tmpGrid = grid;
    grid = nextGrid;
    nextGrid = tmpGrid;

    // calculate timing
    const unsigned long duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - timeStart)
            .count();
    stats::totalEvolveTime += duration;
    // update live buffer
    if (logEnabled) {
        // bring back variable to CPU
        CUDA_ASSERT(cudaMemPrefetchAsync(mActiveCellCount, sizeof(uint),
                                         cudaCpuDeviceId));

        *mLiveLogBuffer << " | Evolve Kernel: " << duration
                        << " ns | Active cells: " << *mActiveCellCount;

        // reset it send it back to GPU
        *mActiveCellCount = 0;
        CUDA_ASSERT(
            cudaMemPrefetchAsync(mActiveCellCount, sizeof(uint), mGpuDeviceId));
    }
}

void AutomataBase::run_evolution_kernel(const bool countAliveCells) {
    k_evolve_count_rule<<<mGpuBlocks, mGpuThreadsPerBlock, 0, mEvolveStream>>>(
        grid, nextGrid, config::rows, config::cols, mGlobalRandState,
        config::virtualFillProb, countAliveCells, mActiveCellCount);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBase::update_grid_buffers() {
#ifndef HEADLESS_ONLY

    if (!mGridVBOResource) {
        fprintf(stderr, "ERROR: Cannot call update_grid_buffers with rendering "
                        "disabled.\n");
        exit(EXIT_FAILURE);
    }

    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    // update projection limits
    proj::update();

    // map OpenGL buffer object for writing from CUDA
    fvec2s *gridVertices;
    CUDA_ASSERT(cudaGraphicsMapResources(1, &mGridVBOResource, 0));
    size_t numBytes;
    CUDA_ASSERT(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void **>(&gridVertices), &numBytes, mGridVBOResource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", numBytes);

    // launch kernels
    // reset buffers
    k_reset_grid_buffers<<<mGpuBlocks, mGpuThreadsPerBlock, 0,
                           mBufferUpdateStream>>>(
        gridVertices, proj::info.numVertices.x, proj::info.numVertices.y);
    CUDA_ASSERT(cudaGetLastError());
    // update buffers
    k_update_grid_buffers<<<mGpuBlocks, mGpuThreadsPerBlock, 0,
                            mBufferUpdateStream>>>(
        grid, gridVertices, config::cols, proj::info.numVertices.x,
        proj::cellDensity, proj::gridLimX, proj::gridLimY);
    CUDA_ASSERT(cudaGetLastError());

    // unmap buffer object
    CUDA_ASSERT(cudaGraphicsUnmapResources(1, &mGridVBOResource, 0));

    // calculate timing
    stats::totalBufferTime +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - timeStart)
            .count();

#endif // HEADLESS_ONLY
}

} // namespace gpu