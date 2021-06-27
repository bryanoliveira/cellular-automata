#ifndef PATTERN_HPP_
#define PATTERN_HPP_

#include "config.hpp"
#include "grid.hpp"
#include "types.hpp"

void load_pattern(const std::string filename);
void fill_grid(const uint row, const uint col, const uint length);

#endif