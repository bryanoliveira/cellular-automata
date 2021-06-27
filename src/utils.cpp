#include <iostream>

#include "config.hpp"
#include "grid.hpp"
#include "utils.hpp"
#include "types.hpp"

namespace utils {

void print_colored(int value) {
    if (value)
        // black foreground with white background
        std::cout << "\e[0;30;47m";
    else
        // gray foreground with black background
        std::cout << "\e[0;37;40m";

    std::cout << " " << value << " "
              << "\e[0m";
}

void print_output(bool spreadBits) {
    for (uint i = 0; i < config::rows; ++i) {
        for (uint j = 0; j < config::cols; ++j) {
            if (spreadBits) {
                // print each bit as a separate value
                for (int bit = (sizeof(unsigned char) * 8) - 1; bit >= 0; --bit)
                    print_colored(!!(grid[i * config::cols + j] & (1 << bit)));
                std::cout << "  ";
            } else
                print_colored(static_cast<int>(grid[i * config::cols + j]));
        }
        std::cout << std::endl;
    }
    std::cout << "\e[0m"; // reset output formatting
}

}; // namespace utils