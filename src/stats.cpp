#include <iostream>

namespace stats {

unsigned long iterations = 0;
unsigned long loadTime = 0;
unsigned long totalEvolveTime = 0;
unsigned long totalBufferTime = 0;

void print_timings() {
    std::cout << iterations;
    std::cout << "," << loadTime;
    std::cout << "," << totalEvolveTime;
    std::cout << "," << totalBufferTime;

    std::cout << "," << totalEvolveTime / iterations;
    std::cout << "," << totalBufferTime / iterations;

    std::cout << std::endl;
}

}; // namespace stats