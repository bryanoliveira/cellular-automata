#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <string>
#include "types.hpp"

namespace config {

// global configs
extern const std::string programName;
extern uint width;
extern uint height;
extern uint rows;
extern uint cols;
extern bool render;
extern uint renderDelayMs;
extern float fillProb;
extern float virtualFillProb;
extern bool benchmarkMode;
extern ulong maxIterations;
extern bool cpuOnly;
extern std::string patternFileName;
extern bool startPaused;
extern bool noDownsample;
extern bool printOutput;
extern uint gpuBlocks;
extern uint gpuThreads;
extern uint skipFrames;

void load_file();
void load_cmd(const int argc, char **const argv);

} // namespace config

#endif