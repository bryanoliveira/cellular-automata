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

Display *display;
unsigned long iterations = 0;

void loop();

int main(int argc, char **argv) {
    unsigned long randSeed = time(NULL);
    srand(randSeed);

    // grid = initGrid(true);
    // insertGlider(config::rows / 2 - 12, config::cols / 2 - 12);
    // insertBlinker(config::rows / 2, config::cols / 2);

    display = new Display(&argc, argv, loop);

    gpu::setup(randSeed, display->gridVBO);

    display->start();

    std::cout << "Exiting after " << iterations << " iterations." << std::endl;
    gpu::cleanUp();

    return 0;
}

void loop() {
    // limit framerate
    if (config::render_delay_ms > 0)
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config::render_delay_ms));

    // update display buffers
    // display->updateGridBuffersCPU();
    gpu::updateGridBuffers();

    // display current grid
    display->draw();
    // compute next grid
    gpu::computeGrid();

    iterations++;
    if (config::max_iterations > 0 && iterations >= config::max_iterations)
        display->stop();
}