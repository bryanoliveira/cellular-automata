#include "config.hpp"

namespace config {

std::string program_name = "Automata";
unsigned long max_iterations = 0;
GLint width = 600;
GLint height = 600;
GLfloat top = 0.0;
GLfloat bottom = 1.0;
GLfloat left = 0.0;
GLfloat right = 1.0;
unsigned int start_delay = 0;
unsigned int render_delay_ms = 200;
// 12000 x 12000 uses up to 2GB RAM and 8.5GB VRAM
unsigned int rows = 100;
unsigned int cols = 100;
float fill_prob = 0.08;
float virtual_fill_prob = 0; //.0001;

} // namespace config
