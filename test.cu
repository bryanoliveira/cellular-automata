#include <stdio.h>
#include <cuda_runtime.h>

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

class Grid {
  private:
    bool *grid;

  public:
    Grid() {
        size_t gridBytes = 2 * sizeof(bool);
        CUDA_ASSERT(cudaMallocManaged(&grid, gridBytes));
        CUDA_ASSERT(cudaMemset(grid, 0, gridBytes));
    }
    __global__ bool get() { return grid[0]; }
};

__global__ void kernel(Grid *grid) { printf("%d\n", (int)grid->get()); }

int main() {
    Grid *grid = new Grid();
    printf("%d\n", (int)grid->get());
    kernel<<<1, 1>>>(grid);
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    return 0;
}