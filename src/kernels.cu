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
struct cudaGraphicsResource *gridVBOResource;

#define gpuAssert(ans)                                                         \
    { _gpuAssert((ans), __FILE__, __LINE__); }
inline void _gpuAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

__global__ void setupRNG(unsigned int rows, unsigned int cols,
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

__global__ void initGrid(bool *grid, unsigned int rows, unsigned int cols,
                         curandState *globalRandState, float spawnProbability) {
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

__global__ void evolveGrid(bool *grid, unsigned int rows, unsigned int cols) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
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

            // 1. Any live cell with two or three live neighbours survives.
            if (grid[idx])
                grid[idx] = livingNeighbours == 2 || livingNeighbours == 3;
            // 2. Any dead cell with three live neighbours becomes a live cell.
            else if (livingNeighbours == 3)
                grid[idx] = true;
            // 3. All other live cells die in the next generation. Similarly,
            // all other dead cells stay dead.
            else
                grid[idx] = false;
        }
    }
}

__global__ void updateGridBuffers(bool *grid, unsigned int rows,
                                  unsigned int cols, vec3s *gridVertices) {
    dim3 stride(gridDim.x * blockDim.x, gridDim.y * blockDim.x);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < rows;
         y += stride.y) {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < cols;
             x += stride.x) {
            int idx = y * cols + x;
            float state = grid[idx]; // implicit bool->float conversion (faster)

            int vidx = y * cols * 4 + x * 4; // 4 vertices per grid cell
            gridVertices[vidx].state = state;
            gridVertices[vidx + 1].state = state;
            gridVertices[vidx + 2].state = state;
            gridVertices[vidx + 3].state = state;
        }
    }
}

void setup(unsigned long randSeed, unsigned long gridVBO) {
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
    gpuAssert(
        cudaMalloc(&globalRandState,
                   gridSize * sizeof(curandState))); // gpu only, not managed
    gpuAssert(cudaMallocManaged(&grid, gridBytes));
    // prefetch grid to GPU
    gpuAssert(cudaMemPrefetchAsync(grid, gridBytes, gpuDeviceId));

    // initialize RNG
    setupRNG<<<gpuBlocks, gpuThreadsPerBlock>>>(config::rows, config::cols,
                                                globalRandState, randSeed);

    // initialize grid
    initGrid<<<gpuBlocks, gpuThreadsPerBlock>>>(
        grid, config::rows, config::cols, globalRandState, config::fill_prob);
    gpuAssert(cudaGetLastError());
    gpuAssert(cudaDeviceSynchronize());

    // register OpenGL VBO to use with CUDA
    gpuAssert(cudaGraphicsGLRegisterBuffer(&gridVBOResource, gridVBO,
                                           cudaGraphicsMapFlagsWriteDiscard));
}

void computeGrid() {
    gpuAssert(cudaMemPrefetchAsync(grid, gridBytes, gpuDeviceId));
    evolveGrid<<<gpuBlocks, gpuThreadsPerBlock>>>(grid, config::rows,
                                                  config::cols);
    gpuAssert(cudaGetLastError());
    gpuAssert(cudaMemPrefetchAsync(grid, gridBytes, cudaCpuDeviceId));
    gpuAssert(cudaDeviceSynchronize());
}

void updateGridBuffers() {
    // map OpenGL buffer object for writing from CUDA
    vec3s *gridVertices;
    gpuAssert(cudaGraphicsMapResources(1, &gridVBOResource, 0));
    size_t numBytes;
    gpuAssert(cudaGraphicsResourceGetMappedPointer((void **)&gridVertices,
                                                   &numBytes, gridVBOResource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", numBytes);

    // launch kernel
    updateGridBuffers<<<gpuBlocks, gpuThreadsPerBlock>>>(
        grid, config::rows, config::cols, gridVertices);
    gpuAssert(cudaGetLastError());

    // unmap buffer object
    gpuAssert(cudaGraphicsUnmapResources(1, &gridVBOResource, 0));
}

void cleanUp() {
    // free the grid
    gpuAssert(cudaFree(grid));
    gpuAssert(cudaFree(globalRandState));
}

} // namespace gpu