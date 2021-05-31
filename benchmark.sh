#!/bin/bash

# disable info logging
export SPDLOG_LEVEL=error

function run {
    # echo header
    echo "size,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $2

    # run program for each lattice size
    for size in {32,64,128,256,512,1024,2048,4096,8192}; do
        echo "Size: $size"
        echo -n "$size," >> $2
        ./automata -b -x $size -y $size -p 0.5 $1 >> $2
    done
}


echo "Benchmarking CPU"
run "--cpu" "benchmark_results_cpu.csv"
echo "Benchmarking GPU"
run "" "benchmark_results_gpu.csv"