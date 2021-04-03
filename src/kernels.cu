// isolate cuda specific imports
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "kernels.hpp"

namespace gpu {

// "private" global variables
int gpuDeviceId;
dim3 gpuBlocks, gpuThreadsPerBlock;
curandState *globalRandState;
size_t gridSize = config::rows * config::cols;
size_t gridBytes = gridSize * sizeof(bool);
struct cudaGraphicsResource *gridVBOResource = NULL;
cudaStream_t evolveStream, bufferUpdateStream;

#define CUDA_ASSERT(ans)                                                       \
    { cuda_assert((ans), __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t code, const char *file, int line,
                        bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void k_setup_rng(unsigned int rows, unsigned int cols,
                            curandState *globalRandState, unsigned long seed) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            curand_init(seed, idx, 0, &globalRandState[idx]);
        }
    }
}

__global__ void k_init_grid(bool *grid, unsigned int rows, unsigned int cols,
                            curandState *globalRandState,
                            float spawnProbability) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            grid[idx] =
                curand_uniform(&globalRandState[idx]) < spawnProbability;
        }
    }
}

__global__ void k_compute_grid(bool *grid, bool *nextGrid, unsigned int rows,
                               unsigned int cols, curandState *globalRandState,
                               float virtualSpawnProbability) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int livingNeighbours = 0;

            // iterate over neighbourhood row by row
            for (int a = y - 1; a <= y + 1; a++) {
                // skip out-of-grid neighbours
                if (a < 0 || a >= rows)
                    continue;
                // iterate over neighbourhood col by col
                for (int b = x - 1; b <= x + 1; b++) {
                    // skip out-of-grid neighbours and self position
                    if (b < 0 || b >= cols || (a == y && b == x))
                        continue; // cols
                    // if cell is alive, increase neighbours
                    if (grid[a * cols + b])
                        livingNeighbours++;
                }
            }

            int idx = y * cols + x;
            bool newState = false;

            // 0. Virtual particle spawn probability
            if (virtualSpawnProbability > 0 &&
                curand_uniform(&globalRandState[idx]) < virtualSpawnProbability)
                newState = true;
            // 1. Any live cell with two or three live neighbours survives.
            else if (grid[idx])
                newState = livingNeighbours == 2 || livingNeighbours == 3;
            // 2. Any dead cell with three live neighbours becomes a live cell.
            else if (livingNeighbours == 3)
                newState = true;
            // 3. All other live cells die in the next generation. Similarly,
            // all other dead cells stay dead.
            else
                newState = false;

            nextGrid[idx] = newState;
        }
    }
}

__global__ void k_update_grid_buffers(bool *grid, vec2s *gridVertices,
                                      unsigned int rows, unsigned int cols) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            gridVertices[idx].state = (float)grid[idx];
        }
    }
}

void setup(unsigned long randSeed, const unsigned int *gridVBO) {
    cudaDeviceProp gpuProps;

    // define common kernel configs
    cudaGetDevice(&gpuDeviceId);
    cudaGetDeviceProperties(&gpuProps, gpuDeviceId);
    // blocks should be a multiple of #SMs on the grid (assume #SMs is even) !
    // actually the number of blocks should
    gpuBlocks = dim3(gpuProps.multiProcessorCount / 2,
                     gpuProps.multiProcessorCount /
                         2); // 1156 on a 3080 - 17 blocks per SM
    // threads should be a multiple of warpSize on the block (assume warpSize is
    // even - it usually is 32)
    gpuThreadsPerBlock = dim3(
        gpuProps.warpSize / 2,
        gpuProps.warpSize / 2); // 256 on a 3080 - 8 threads per SP (I think)

    // allocate memory
    CUDA_ASSERT(
        cudaMalloc(&globalRandState,
                   gridSize * sizeof(curandState))); // gpu only, not managed
    CUDA_ASSERT(cudaMallocManaged(&grid, gridBytes));
    CUDA_ASSERT(cudaMallocManaged(&nextGrid, gridBytes));
    // prefetch grid to GPU
    CUDA_ASSERT(cudaMemPrefetchAsync(grid, gridBytes, gpuDeviceId));
    CUDA_ASSERT(cudaMemPrefetchAsync(nextGrid, gridBytes, gpuDeviceId));

    // initialize RNG
    k_setup_rng<<<gpuBlocks, gpuThreadsPerBlock>>>(config::rows, config::cols,
                                                   globalRandState, randSeed);

    // initialize grid
    k_init_grid<<<gpuBlocks, gpuThreadsPerBlock>>>(
        grid, config::rows, config::cols, globalRandState, config::fillProb);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    // create grid evolving CUDA stream
    CUDA_ASSERT(cudaStreamCreate(&evolveStream));

    // if rendering is enabled
    if (gridVBO) {
        // create buffer updating CUDA stream
        CUDA_ASSERT(cudaStreamCreate(&bufferUpdateStream));
        // register OpenGL VBO to use with CUDA
        CUDA_ASSERT(cudaGraphicsGLRegisterBuffer(
            &gridVBOResource, *gridVBO, cudaGraphicsMapFlagsWriteDiscard));
    }
}

void compute_grid() {
    k_compute_grid<<<gpuBlocks, gpuThreadsPerBlock, 0, evolveStream>>>(
        grid, nextGrid, config::rows, config::cols, globalRandState,
        config::virtualFillProb);
    CUDA_ASSERT(cudaGetLastError());
    // should I call cudaDeviceSynchronize?
    // CUDA_ASSERT(cudaDeviceSynchronize());

    // simply swap buffers to avoid reallocation
    bool *tmpGrid = grid;
    grid = nextGrid;
    nextGrid = tmpGrid;
}

void update_grid_buffers() {
    if (!gridVBOResource) {
        fprintf(stderr, "ERROR: Cannot call update_grid_buffers with rendering "
                        "disabled.\n");
        exit(EXIT_FAILURE);
    }

    // map OpenGL buffer object for writing from CUDA
    vec2s *gridVertices;
    CUDA_ASSERT(cudaGraphicsMapResources(1, &gridVBOResource, 0));
    size_t numBytes;
    CUDA_ASSERT(cudaGraphicsResourceGetMappedPointer(
        (void **)&gridVertices, &numBytes, gridVBOResource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", numBytes);

    // launch kernel
    k_update_grid_buffers<<<gpuBlocks, gpuThreadsPerBlock>>>(
        grid, gridVertices, config::rows, config::cols);
    CUDA_ASSERT(cudaGetLastError());
    // should I call cudaDeviceSynchronize?

    // unmap buffer object
    CUDA_ASSERT(cudaGraphicsUnmapResources(1, &gridVBOResource, 0));
}

void clean_up() {
    // wait for pending operations to complete
    CUDA_ASSERT(cudaDeviceSynchronize());
    // destroy secondary streams
    cudaStreamDestroy(evolveStream);
    cudaStreamDestroy(bufferUpdateStream);
    // unregister mapped resource (needs the GL context to still be set)
    // CUDA_ASSERT(cudaGraphicsUnregisterResource(gridVBOResource));
    // free the grid
    CUDA_ASSERT(cudaFree(grid));
    CUDA_ASSERT(cudaFree(nextGrid));
    CUDA_ASSERT(cudaFree(globalRandState));
}

} // namespace gpu