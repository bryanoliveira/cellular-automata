#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include "config.hpp"
#include "grid.hpp"

namespace display {

void setup(int * argc, char ** argv, void (*loopFunc)());
void start();
void stop();
void draw();
void reshape(int w, int h);

} // namespace display

#endif