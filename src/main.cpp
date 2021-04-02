/**
 * Author: Bryan Lincoln
 * Email: bryanufg@gmail.com
 *
 * This must be compiled with nvcc so required libraries are properly loaded
 */
#include <iostream>
#include <chrono>
#include <thread>

#include "automata.hpp"
#include "config.hpp"
#include "display.hpp"
#include "grid.hpp"
#include "kernels.hpp"

#define USE_GPU true

Display *display;
unsigned long iterations = 0;

void loop();

int main(int argc, char **argv) {
    unsigned long randSeed = time(NULL);

    display = new Display(&argc, argv, loop, USE_GPU);

    if (USE_GPU)
        gpu::setup(randSeed, display->gridVBO);
    else
        cpu::setup(randSeed, config::fill_prob);

    // insertGlider(config::rows / 2 - 12, config::cols / 2 - 12);
    // insertBlinker(config::rows / 2, config::cols / 2);

    std::this_thread::sleep_for(std::chrono::milliseconds(config::start_delay));
    display->start();

    std::cout << "Exiting after " << iterations << " iterations." << std::endl;

    // clean up
    if (USE_GPU)
        gpu::cleanUp();
    else
        cpu::cleanUp();

    delete display;

    return 0;
}

void loop() {
    // limit framerate
    if (config::render_delay_ms > 0)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config::render_delay_ms));

    // update display buffers
    if (USE_GPU)
        gpu::updateGridBuffers();
    else
        display->updateGridBuffersCPU();

    // display current grid
    display->draw();
    // compute next grid
    if (USE_GPU)
        gpu::computeGrid();
    else
        cpu::computeGrid();

    iterations++;
    if (config::max_iterations > 0 && iterations >= config::max_iterations)
        display->stop();
}