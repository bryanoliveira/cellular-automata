#!/bin/bash

# exit on error
set -e

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


RUN="$(ls -1v res | tail -1 | cut -d "." -f1)"
RUN=$(($RUN + 1))

echo "Benchmark #$RUN"

for threads in {1,2,4,8,12,16}; do
    export OMP_NUM_THREADS=$threads
    echo "Benchmarking CPU with $OMP_NUM_THREADS threads"
    run "--cpu" "res/$RUN.benchmark_results_cpu_th_$OMP_NUM_THREADS.csv"
done

echo "Benchmarking GPU"
run "" "res/$RUN.benchmark_results_gpu.csv"