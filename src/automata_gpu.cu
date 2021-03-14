#include <stdio.h>
#include "automata_gpu.h"

namespace gpu {

__global__ void computeCell(bool* prevGrid, bool* nextGrid){
    int image_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int image_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int width = constant_parameters.grid_width;
    int height = constant_parameters.grid_height;
    // index in the grid of this thread
    int grid_index = image_y*width + image_x;

    int livingNeighbours = 0;

    // iterate over neighbourhood row by row
    for (int a = i - 1; a <= i + 1; a++) {
        // skip out-of-grid neighbours
        if (a < 0 || a >= config::rows) continue;
        // iterate over neighbourhood col by col
        for (int b = j - 1; b <= j + 1; b++) {
            // skip out-of-grid neighbours and self position
            if (b < 0 || b >= config::cols || (a == i && b == j)) continue;
            // if cell is alive, increase neighbours
            if (prevGrid[a][b]) livingNeighbours++;
        }
    }

    // 1. Any live cell with two or three live neighbours survives.
    if (prevGrid[i][j]) nextGrid[i][j] = livingNeighbours == 2 || livingNeighbours == 3;
    // 2. Any dead cell with three live neighbours becomes a live cell.
    else if (livingNeighbours == 3)
        return nextGrid[i][j] = true;
    // 3. All other live cells die in the next generation. Similarly,
    // all other dead cells stay dead.
    else
        return nextGrid[i][j] = false;
}

void computeGrid() {
    int numBlocks = 1;
    dim3 threadsPerBlock(config::cols, config::rows);

    size_t bytes = config::cols * config::rows * sizeof(bool)
    bool* deviceArray;
    
    cudaMalloc(bytes);
    cudaMemcpy(deviceArray, hostArray, bytes, cudaMemcpyHostToDevice);

    computeCell<<<numBlocks,threadsPerBlock>>>(grid);

    cudaMemcpy(hostArray, deviceArray, bytes, cudaMemcpyDeviceToHost);

    cudaFree(deviceArray);

    cudaDeviceSynchronize();
}

}