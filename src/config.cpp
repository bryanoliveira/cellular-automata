#include <boost/program_options.hpp>
#include <iostream>
#include <spdlog/spdlog.h>

#include "config.hpp"

namespace po = boost::program_options;

namespace config {

const std::string programName = "Automata";
uint width = 600;
uint height = 600;
// 12000 x 12000 uses up to 2GB RAM and 8.5GB VRAM
uint rows = 1024;
uint cols = 1024;
bool render = false;
uint renderDelayMs = 0;
float fillProb = 0.2;
float virtualFillProb = 0;
bool benchmarkMode = false;
ulong maxIterations = 0;
bool cpuOnly = false;
std::string patternFileName("random");
bool startPaused = true;
bool noDownsample = false;
bool printOutput = false;
uint gpuBlocks = 0;
uint gpuThreads = 0;
uint skipFrames = 0;

void load_file() {}
void load_cmd(const int argc, char **const argv) {
    po::options_description description("Usage");

    description.add_options()("help,h", "Display this help message") //
        ("width", po::value<uint>(), "Window width")                 //
        ("height", po::value<uint>(), "Window height")               //
        ("rows,y", po::value<uint>(), "Grid rows")                   //
        ("cols,x", po::value<uint>(), "Grid cols")                   //
        ("render,r",
         "Enable render (default is to run in headless mode)") //
        ("render-delay,d", po::value<uint>(),
         "Render delay between frames (in milliseconds)") //
        ("fill-probability,p", po::value<float>(),
         "Cell probability to start alive") //
        ("virtual-fill-probability,v", po::value<float>(),
         "Cell probability to become alive")            //
        ("max,m", po::value<ulong>(), "Max iterations") //
        ("cpu", "Enable CPU-only mode")                 //
        ("no-downsample", "Disable automatic grid to vertice downsampling"
                          " when grid size is greater than window size.") //
        ("file,f", po::value<std::string>(), "Pattern file (.rle)")       //
        ("start", "Unpause at start (default is paused when rendering, "
                  "unpaused when not rendering)")   //
        ("output,o", "Output last state to stdout") //
        ("benchmark,b", "Benchmark mode - overrides 'max' to 1000 if it is 0, "
                        "'start' to true, 'render-delay' to 0")         //
        ("gpu-blocks", po::value<uint>(), "GPU grid size in blocks")    //
        ("gpu-threads", po::value<uint>(), "GPU block size in threads") //
        ("skip-frames", po::value<uint>(),
         "Number of extra generations before render");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(description).run(),
              vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << description << std::endl;
        exit(EXIT_SUCCESS);
    }
    if (vm.count("width"))
        width = vm["width"].as<uint>();
    if (vm.count("height"))
        height = vm["height"].as<uint>();
    if (vm.count("rows"))
        rows = vm["rows"].as<uint>();
    if (vm.count("cols"))
        cols = vm["cols"].as<uint>();
    if (vm.count("render")) {
#ifdef HEADLESS_ONLY
        spdlog::critical(
            "ERROR! Rendering (-r) is disabled in HEADLESS_ONLY mode!");
        exit(EXIT_FAILURE);
#else
        render = true;
#endif
    }
    if (vm.count("render-delay"))
        renderDelayMs = vm["render-delay"].as<uint>();
    if (vm.count("fill-probability"))
        fillProb = vm["fill-probability"].as<float>();
    if (vm.count("virtual-fill-probability"))
        virtualFillProb = vm["virtual-fill-probability"].as<float>();
    if (vm.count("max"))
        maxIterations = vm["max"].as<ulong>();
    if (vm.count("cpu"))
        cpuOnly = true;
#ifdef CPU_ONLY
    spdlog::warn("cpuOnly flag overriden due to CPU_ONLY compilation.");
    cpuOnly = true;
#endif
    if (vm.count("file"))
        patternFileName = vm["file"].as<std::string>();
    if (vm.count("start") || !render)
        // by default, start the computation loop automatically if we're not
        // rendering
        startPaused = false;
    if (vm.count("no-downsample") || (width == cols && height == rows))
        // by default, don't use downsample when scale is 1:1
        noDownsample = true;
    if (vm.count("benchmark")) {
        benchmarkMode = true;
        startPaused = false;
        renderDelayMs = 0;
        if (maxIterations == 0)
            maxIterations = 1000;
    }
    if (vm.count("output"))
        printOutput = true;
    if (vm.count("gpu-blocks"))
        gpuBlocks = vm["gpu-blocks"].as<uint>();
    if (vm.count("gpu-threads"))
        gpuThreads = vm["gpu-threads"].as<uint>();
    if (vm.count("skip-frames"))
        skipFrames = vm["skip-frames"].as<uint>();
}

} // namespace config
