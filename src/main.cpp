/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 *
 * Using ISO C++ 17 (C++ 11 may be compatible)
 *
 * Conventions (a variation of STL/Boost Style Guides):
 *  - use spaces instead of tabs
 *  - indent with 4 spaces
 *  - variables are camelCased
 *    - params are prefixed with p (e.g. pFillProb)
 *    - member variables are prefixed with m (e.g. mFillProb)
 *    - globals are prefixed with g (e.g. gDisplay)
 *       - the 'config' namespace doesn't follow this as the 'config::' prefix
 *         is always made explicit
 *  - methods are snake_cased
 *  - CUDA kernels are prefixed with k (e.g. k_compute_grid())
 *  - Macros are UPPER_CASED (e.g. CUDA_ASSERT())
 */
#include <chrono>
#include <iostream>
#include <thread>
#include <signal.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#include "automata_interface.hpp"
#include "automata_base_cpu.hpp"
#include "config.hpp"
#include "pattern.hpp"
#include "stats.hpp"
#ifndef CPU_ONLY
    #include "automata_base_gpu.cuh"
#endif // CPU_ONLY
#ifndef HEADLESS_ONLY
    #include "controls.hpp"
    #include "display.hpp"
#endif // HEADLESS_ONLY

#ifndef HEADLESS_ONLY
Display *gDisplay;
#endif // HEADLESS_ONLY
AutomataInterface *gAutomata;

bool gLooping = true;
unsigned long gLastIterationCount = 0;
unsigned long gIterationsPerSecond = 0;
unsigned long gNsBetweenSeconds = 0;
std::chrono::steady_clock::time_point gLastPrintClock =
    std::chrono::steady_clock::now();
std::ostringstream gLiveLogBuffer;

void loop();
bool should_log();
void live_log();
void sigint_handler(int s);

int main(int argc, char **argv) {
    spdlog::cfg::load_env_levels();

    const unsigned long randSeed = time(nullptr);
    struct sigaction sigIntHandler;

    // configure interrupt signal handler
    sigIntHandler.sa_handler = sigint_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);

    // load command line arguments
    config::load_cmd(argc, argv);

#ifndef HEADLESS_ONLY
    controls::paused = config::startPaused;

    // configure display
    if (config::render)
        gDisplay = new Display(&argc, argv, loop, config::cpuOnly);
#endif // HEADLESS_ONLY

    // configure automata object
    if (config::cpuOnly)
        // the CPU implementation uses the buffer update function provided by
        // the display class and we configure it here to reduce complexity by
        // maintaining the AutomataInterface predictable
        gAutomata = static_cast<AutomataInterface *>(
            new cpu::AutomataBase(randSeed, &gLiveLogBuffer, []() {
#ifndef HEADLESS_ONLY
                gDisplay->update_grid_buffers_cpu();
#endif // HEADLESS_ONLY
            }));
#ifndef CPU_ONLY
    #ifndef HEADLESS_ONLY
    else if (config::render)
        // the GPU implementation updates the VBO using the CUDA<>GL interop
        gAutomata = static_cast<AutomataInterface *>(new gpu::AutomataBase(
            randSeed, &gLiveLogBuffer, &(gDisplay->grid_vbo())));
    #endif // HEADLESS_ONLY
    else
        gAutomata = static_cast<AutomataInterface *>(
            new gpu::AutomataBase(randSeed, &gLiveLogBuffer));
#endif // CPU_ONLY

    if (config::patternFileName != "random")
        load_pattern(config::patternFileName);

#ifndef HEADLESS_ONLY
    if (config::render)
        gDisplay->start();
    else
#endif // HEADLESS_ONLY
        while (gLooping)
            loop();

    if (config::benchmarkMode)
        stats::print_timings();
    else
        std::cout << std::endl;

    spdlog::info("Exiting after {} iterations.", stats::iterations);

    // clean up
    delete gAutomata;

#ifndef HEADLESS_ONLY
    if (config::render)
        delete gDisplay;
#endif // HEADLESS_ONLY

    return 0;
}

void loop() {
    // limit framerate
    if (config::renderDelayMs > 0)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config::renderDelayMs));

    // loop timer
    const std::chrono::steady_clock::time_point timeStart =
        std::chrono::steady_clock::now();

    // prepare logging
    const bool logEnabled = !config::benchmarkMode && should_log();
    if (logEnabled)
        // carriage return
        gLiveLogBuffer << "\r\e[KIt: " << stats::iterations;

#ifndef HEADLESS_ONLY
    // update buffers & render
    if (config::render) {
        // update display buffers
        gAutomata->update_grid_buffers();

        // display current grid
        gDisplay->draw(logEnabled, gIterationsPerSecond);
    }

    // there are controls only when rendering is enabled
    if (controls::paused && !controls::singleStep) {
        if (config::render)
            std::cout << "\r\e[KPaused. Press space to resume." << std::flush;
    } else {
        controls::singleStep = false;

#endif // HEADLESS_ONLY

        // compute next grid
        gAutomata->compute_grid(logEnabled); // count alive cells if will log
        ++stats::iterations;

#ifndef HEADLESS_ONLY
    }
#endif // HEADLESS_ONLY

    // calculate loop time and iterations per second
    gNsBetweenSeconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - timeStart)
                             .count();
    if (logEnabled)
        live_log();

    // check if number of iterations reached max
    if (!gLooping || (config::maxIterations > 0 &&
                      stats::iterations >= config::maxIterations)) {
#ifndef HEADLESS_ONLY
        if (config::render)
            gDisplay->stop();
        else
#endif // HEADLESS_ONLY
            gLooping = false;
    }
}

bool should_log() {
    // calculate loop time and iterations per second
    gIterationsPerSecond = stats::iterations - gLastIterationCount;
    // return if it's not time to update the log
    if (gIterationsPerSecond <= 0 ||
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - gLastPrintClock)
                .count() < 1)
        return false;
    return true;
}

void live_log() {
    // add main loop info to the buffer
    gLiveLogBuffer << " | It/s: " << gIterationsPerSecond
                   << " | Main Loop: "
                   // average time per iteration
                   << gNsBetweenSeconds / gIterationsPerSecond << " ns";
    // print the buffer
    std::cout << gLiveLogBuffer.str() << std::flush;
    // reset the buffer
    gLiveLogBuffer.str("");
    gLiveLogBuffer.clear();
    // update global counters
    gNsBetweenSeconds = 0;
    gLastIterationCount = stats::iterations;
    gLastPrintClock = std::chrono::steady_clock::now();
}

void sigint_handler(int s) {
    gLooping = false;
    std::cout << std::endl;
}