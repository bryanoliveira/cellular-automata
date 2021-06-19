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

TRIALS=5

for i in $(seq 1 $TRIALS); do
    for threads in {16,}; do # 1,2,4,8,12,
        export OMP_NUM_THREADS=$threads
        echo "Benchmarking CPU with $OMP_NUM_THREADS threads"
        run "--cpu" "res/$RUN.benchmark_results_cpu_th_$OMP_NUM_THREADS.csv"
    done
done

echo "Benchmarking GPU"
for i in $(seq 1 $TRIALS); do
    echo "Trial #$i"
    run "" "res/$RUN.$i.benchmark_results_gpu.csv"
done