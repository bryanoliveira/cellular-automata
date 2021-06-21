#include <iostream>

namespace stats {

ulong iterations = 0;
ulong loadTime = 0;
ulong totalEvolveTime = 0;
ulong totalBufferTime = 0;

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