#include <boost/program_options.hpp>

#include "config.hpp"

namespace po = boost::program_options;

namespace config {

std::string program_name = "Automata";
unsigned long max_iterations = 0;
GLint width = 600;
GLint height = 600;
unsigned int render_delay_ms = 200;
// 12000 x 12000 uses up to 2GB RAM and 8.5GB VRAM
unsigned int rows = 100;
unsigned int cols = 100;
float fill_prob = 0.08;
float virtual_fill_prob = 0; //.0001;

void loadFile() {}
void loadCmd(int argc, char **argv) {
    po::options_description description("Usage");

    description.add_options()("help,h", "Display this help message") //
        ("width,w", po::value<GLint>(), "Window width")              //
        ("height,h", po::value<GLint>(), "Window height")            //
        ("rows,r", po::value<unsigned int>(), "Grid rows")           //
        ("cols,c", po::value<unsigned int>(), "Grid cols")           //
        ("render-delay,rd", po::value<unsigned int>(),
         "Render delay between frames (in milliseconds)") //
        ("fill-probability,fp", po::value<float>(),
         "Cell probability to start alive") //
        ("max,m", po::value<unsigned long>(), "Max iterations");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << description << std::endl;
        exit(0);
    }
    if (vm.count("width"))
        width = vm["width"].as<GLint>();
    if (vm.count("height"))
        height = vm["height"].as<GLint>();
    if (vm.count("rows"))
        rows = vm["rows"].as<unsigned int>();
    if (vm.count("cols"))
        cols = vm["cols"].as<unsigned int>();
    if (vm.count("render-delay"))
        render_delay_ms = vm["render-delay"].as<unsigned int>();
    if (vm.count("fill-probability"))
        fill_prob = vm["fill-probability"].as<float>();
    if (vm.count("max"))
        max_iterations = vm["max"].as<unsigned long>();
}

} // namespace config
