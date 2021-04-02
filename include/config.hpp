#ifndef CONFIG_HPP_
#define CONFIG_HPP_
#include <GL/glu.h>

#include <string>
#include <iostream>

namespace config {

// global configs
extern std::string program_name;
extern unsigned long max_iterations;
extern GLint width;
extern GLint height;
extern unsigned int render_delay_ms;
extern unsigned int rows;
extern unsigned int cols;
extern float fill_prob;
extern float virtual_fill_prob;

void loadFile();
void loadCmd(int argc, char **argv);

} // namespace config

#endif