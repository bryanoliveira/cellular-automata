// isolate cuda specific imports
#include <chrono>
#include <cuda_gl_interop.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <iostream>

#include "automata_base_gpu.cuh"
#include "commons_gpu.cuh"
#include "stats.hpp"

namespace gpu {

AutomataBase::AutomataBase(const uint randSeed,
                           std::ostringstream *const pLiveLogBuffer,
                           const uint *const gridVBO) {
    spdlog::info("Initializing Automata Base GPU engine...");

    int numGpus = 0;
    CUDA_ASSERT(cudaGetDeviceCount(&numGpus));
    if (numGpus == 0) {
        spdlog::critical(
            "ERROR! No GPUs available: cudaGetDeviceCount returned 0.");
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp gpuProps;
    CUDA_ASSERT(cudaGetDevice(&mGpuDeviceId));
    // define common kernel configs if needed
    CUDA_ASSERT(cudaGetDeviceProperties(&gpuProps, mGpuDeviceId));
    // threads should be a multiple of warpSize on the block
    // 32 on a 3080
    if (config::gpuThreads == 0)
        config::gpuThreads = gpuProps.warpSize * 8;
    // blocks is optimal when it's a multiple of #SMs on the grid
    // 68 on a 3080
    if (config::gpuBlocks == 0) {
        const int cores = getSPcores(gpuProps);
        if (cores <= 0) {
            spdlog::critical("ERROR! Unable to infer number of GPU cores: "
                             "unknown GPU device type. You may try seting "
                             "threads and blocks manually.");
            exit(EXIT_FAILURE);
        }
        config::gpuBlocks =
            std::min((mGridSize + config::gpuThreads - 1) / config::gpuThreads,
                     static_cast<ulong>(cores));
    }
    spdlog::info("GPU: Using {} blocks of {} threads.", config::gpuBlocks,
                 config::gpuThreads);

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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

    // initialize RNG
    k_setup_rng<<<config::gpuBlocks, config::gpuThreads>>>(
        {config::cols, config::rows}, mGlobalRandState, randSeed);

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

    spdlog::info("Automata Base GPU engine is ready.");
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

    run_init_kernel();

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
    std::swap(grid, nextGrid);

    // calculate timing
    const ulong duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
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
    k_reset_grid_buffers<<<config::gpuBlocks, config::gpuThreads, 0,
                           mBufferUpdateStream>>>(gridVertices,
                                                  proj::info.numVertices);
    CUDA_ASSERT(cudaGetLastError());
    // should I call cudaDeviceSynchronize?
    CUDA_ASSERT(cudaDeviceSynchronize());

    // update buffers
    run_render_kernel(gridVertices);

    // unmap buffer object
    CUDA_ASSERT(cudaGraphicsUnmapResources(1, &mGridVBOResource, 0));

    // calculate timing
    stats::totalBufferTime +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - timeStart)
            .count();

#endif // HEADLESS_ONLY
}

void AutomataBase::run_init_kernel() {
    // initialize grid with fillProb
    if (config::fillProb > 0)
        k_init_grid<<<config::gpuBlocks, config::gpuThreads>>>(
            grid, {config::cols, config::rows}, mGlobalRandState,
            config::fillProb);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBase::run_evolution_kernel(const bool countAliveCells) {
    k_evolve_count_rule<<<config::gpuBlocks, config::gpuThreads, 0,
                          mEvolveStream>>>(
        grid, nextGrid, {config::cols, config::rows}, mGlobalRandState,
        config::virtualFillProb, countAliveCells, mActiveCellCount);
    CUDA_ASSERT(cudaGetLastError());
}

void AutomataBase::run_render_kernel(fvec2s *gridVertices) {
#ifndef HEADLESS_ONLY
    k_update_grid_buffers<<<config::gpuBlocks, config::gpuThreads, 0,
                            mBufferUpdateStream>>>(
        grid, {config::cols, config::rows}, gridVertices,
        proj::info.numVertices.x, proj::cellDensity, proj::gridLimX,
        proj::gridLimY);
    CUDA_ASSERT(cudaGetLastError());
#endif
}

} // namespace gpu