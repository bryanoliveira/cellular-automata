#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include <GL/glu.h>
#include <GL/glut.h>

#include <ctime>
#include <sstream>

#include "config.hpp"
#include "grid.hpp"

void reshape(int w, int h);
bool display();

#endif