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

printf "Benchmark #$RUN\n"

TRIALS=5

for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    for threads in {16,}; do # 1,2,4,8,12,
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads\n"
        run "--cpu" "res/$RUN.benchmark_results_cpu_th_$OMP_NUM_THREADS.csv"
    done
done

printf "\nBenchmarking GPU"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run "" "res/$RUN.$i.benchmark_results_gpu.csv"
done