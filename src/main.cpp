/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
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

#include "automata.hpp"
#include "config.hpp"
#include "display.hpp"
#include "grid.hpp"
#include "kernels.hpp"

Display *gDisplay;
unsigned long gIterations = 0;

void loop();

int main(int argc, char **argv) {
    unsigned long randSeed = time(NULL);

    config::load_cmd(argc, argv);

    gDisplay = new Display(&argc, argv, loop, config::cpuOnly);

    if (config::cpuOnly)
        cpu::setup(randSeed, config::fillProb);
    else
        gpu::setup(randSeed, gDisplay->grid_vbo());

    // insert_glider(config::rows / 2 - 12, config::cols / 2 - 12);
    // insert_blinker(config::rows / 2, config::cols / 2);

    gDisplay->start();

    std::cout << "Exiting after " << gIterations << " iterations." << std::endl;

    // clean up
    if (config::cpuOnly)
        cpu::clean_up();
    else
        gpu::clean_up();

    delete gDisplay;

    return 0;
}

void loop() {
    // limit framerate
    if (config::renderDelayMs > 0)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config::renderDelayMs));

    // update display buffers
    if (config::cpuOnly)
        gDisplay->update_grid_buffers_cpu();
    else
        gpu::update_grid_buffers();

    // display current grid
    gDisplay->draw();
    // compute next grid
    if (config::cpuOnly)
        cpu::compute_grid();
    else
        gpu::compute_grid();

    gIterations++;
    if (config::maxIterations > 0 && gIterations >= config::maxIterations)
        gDisplay->stop();
}