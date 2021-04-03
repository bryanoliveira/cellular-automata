#ifndef CONFIG_HPP_
#define CONFIG_HPP_
#include <GL/glu.h>

#include <string>
#include <iostream>

namespace config {

// global configs
extern std::string programName;
extern GLint width;
extern GLint height;
extern unsigned int rows;
extern unsigned int cols;
extern bool render;
extern unsigned int renderDelayMs;
extern float fillProb;
extern float virtualFillProb;
extern unsigned long maxIterations;
extern bool cpuOnly;

void load_file();
void load_cmd(int argc, char **argv);

} // namespace config

#endif