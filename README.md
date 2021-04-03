# Cellular Automata

<img src="docs/100x100.gif" align="right">

A [Cellular Automata](https://en.wikipedia.org/wiki/Cellular_automaton) program built with C++, CUDA and OpenGL. It's built to run on a GPU but it also supports CPU-only execution (mainly for relative speedup comparisons). On the right there's an example execution of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) on a 100x100 grid randomly initialized.

The main objective of this project is to allow scaling up to a fairly large number of cells while maintaining the code legibility and allowing for further customizations.

It doesn't yet support headless mode or definition of the evolution rules at runtime, but I'm working to add those in future versions.

This program can currently evolve a 144 million cell Conway's Game of Life grid (12000x12000) with up to 15 FPS on a Ryzen 7 3700X / RTX 3080 using up to 2GB RAM and 9GB VRAM (which is the actual scaling limiter).

<br />

<img src="docs/12000x12000.png">

> A 12000x12000 grid running Conway's Game of life.

## Requirements

-   Debian-like linux distro (I only tested this on Ubuntu 20)
-   make
-   g++ (C++ 17)
-   OpenGL (GLEW and GLUT)
-   Boost C++ Libraty (program_options)
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

## Bonus

![1000x1000 grid (click to open)](docs/1000x1000.gif)

> A 1000x1000 grid running Conway's Game of life.

---

This program was developed during the 2021/1 Parallel Computing (CCO0455) Computer Science graduate course at Universidade Federal de Goiás (UFG, Brazil).
