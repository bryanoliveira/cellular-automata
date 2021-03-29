#ifndef CONFIG_HPP_
#define CONFIG_HPP_
#include <GL/glu.h>

#include <string>

namespace config {

extern std::string program_name;
extern unsigned long max_iterations;
extern GLint width;
extern GLint height;
extern GLfloat top;
extern GLfloat bottom;
extern GLfloat left;
extern GLfloat right;
extern unsigned int render_delay_ms;
extern unsigned int rows;
extern unsigned int cols;
extern float fill_prob;
extern float virtual_fill_prob;

} // namespace config

#endif