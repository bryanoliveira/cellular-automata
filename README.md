# Cellular Automata

<img src="docs/100x100.gif" align="right">

A [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) program built with C++, CUDA and OpenGL. It's built to run on a GPU but it also supports CPU-only execution (mainly for relative speedup comparisons). On the right there's an example execution of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) on a 100x100 grid randomly initialized.

The main objective of this project is to allow scaling up to a fairly large number of cells while maintaining the code legibility and allowing for further customizations. It supports command line arguments to set up quick configs (run `./automata -h` for details) like headless mode (which is significantly faster) and initial patterns (which can be loaded from the `patterns` folder). It doesn't yet support the definition of evolution rules at runtime, but I'm working on that.

This program can currently evolve a dense & high entropy 182.25 million cell Conway's Game of Life grid (13500x13500) with rendering enabled with up to 320 iterations per second on a Ryzen 7 3700X / RTX 3080 using up to 200MB RAM and 8.5GB VRAM (which is the actual scaling limiter). With rendering disabled it can evolve a 190.44 million cell grid (13800x13800) with up to 9 million iterations per second (currently the bottleneck is the rendering overhead).

<br />

<img src="docs/12000x12000.png">

> A 12000x12000 grid running Conway's Game of life.

## Requirements

-   Debian-like linux distro (I only tested this on Ubuntu 20)
-   make
-   g++ (C++ 17)
-   OpenGL (GLEW and GLUT)
-   Boost C++ Library (program_options module)
-   CUDA (nvcc) and CUDA runtime libraries

It is possible to run this program in a CPU-only mode, so if you don't have a CUDA-capable video card you may skip the last step. For that to work you will need to run the program with `./automata --cpu` and disable `*.cu` file compilation on `Makefile`.

## Usage

-   Install the requirements
-   Clone this repository
-   Building and executing:
    -   Run `make` to build and run
    -   Run `make build` to only build
    -   Run `make run` to only run
    -   Run `make clean` to remove generated build files
    -   Run `make profile` to run [NVIDIA's nsys](https://developer.nvidia.com/nsight-systems) profiling.

### Runtime Commands

-   Press `space` to start/pause the simulation
-   Press `enter/return` to run a single step of the simulation
-   Scroll to zoom in/out
-   Left click to pan, right click to rotate (not very useful yet, but it will be for the 3D version), middle click to reset the camera

## Bonus

![1000x1000 grid (click to open)](docs/1000x1000.gif)

> A 1000x1000 grid running Conway's Game of life.

---

This program was developed during the 2021/1 Parallel Computing (CCO0455) Computer Science graduate course at Universidade Federal de Goiás (UFG, Brazil).
