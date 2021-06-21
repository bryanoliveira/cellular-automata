#!/bin/bash

# exit on error
set -e

# disable info logging
export SPDLOG_LEVEL=error

function run_lattice {
    # echo header
    echo "size,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $2

    # run program for each lattice size
    for size in {32,64,128,256,512,1024,2048,4096}; do # ,8192
        echo "Size: $size"
        echo "./automata -b -x $size -y $size -p 0.5 $1"
        echo -n "$size," >> $2
        ./automata -b -x $size -y $size -p 0.5 $1 >> $2
    done
}

function run_kernel_config {
    # echo header
    echo "blocks,threads,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $2

    # run program for each lattice size
    for ((blocks=68;blocks<=1000;blocks+=68)); do # in {272,544,748,1088,2176,4352,8704}
        for ((threads=32;threads<=1024;threads+=32)); do # in {64,128,256,512,768,1024}
            echo "Blocks: $blocks | Threads: $threads"
            echo "./automata -b -x 2048 -y 2048 -p 0.5 $1 --gpu-blocks $blocks --gpu-threads $threads"
            echo -n "$blocks,$threads," >> $2
            ./automata -b -x 2048 -y 2048 -p 0.5 $1 --gpu-blocks $blocks --gpu-threads $threads >> $2
        done
    done
}


RUN="$(ls -1v res | tail -1 | cut -d "." -f1)"
RUN=$(($RUN + 1))

printf "Benchmark #$RUN\n"

TRIALS=1

for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    for threads in {1,2,4,8,12,16}; do
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads\n"
        run_lattice "--cpu" "res/$RUN.$i.hl.cpu.$OMP_NUM_THREADS.csv"
    done
done

for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    for threads in {16,}; do # 1,2,4,8,12,
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads and render\n"
        run_lattice "--cpu -r" "res/$RUN.$i.render.cpu.$OMP_NUM_THREADS.csv"
    done
done

# run_lattice "--gpu-blocks 1 --gpu-threads 1" "res/$RUN.1.hl.gpu.1.csv"

printf "\nBenchmarking GPU"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run_lattice "--gpu-blocks 272 --gpu-threads 768" "res/$RUN.$i.hl.gpu.8704.csv"
done

printf "\nBenchmarking GPU with render"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run_lattice "-r" "res/$RUN.$i.render.gpu.8704.csv"
done

printf "\nBenchmarking GPU for kernel config"
run_kernel_config "" "res/$RUN.k.hl.gpu.8704.csv"