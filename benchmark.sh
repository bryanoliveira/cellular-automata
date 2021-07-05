#!/bin/bash

# exit on error
set -e

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
    echo "size,threads,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $2

    # run program for each lattice size
    for size in {32,64,128,256,512,1024,2048,4096}; do # ,8192
        for ((threads=32;threads<=1024;threads+=32)); do # in {64,128,256,512,768,1024}
            echo "Size: $size | Threads: $threads"
            echo "./automata -b -x $size -y $size -p 0.5 $1 --gpu-threads $threads"
            echo -n "$size,$threads," >> $2
            ./automata -b -x $size -y $size -p 0.5 $1 --gpu-threads $threads >> $2
        done
    done
}

# disable info logging
export SPDLOG_LEVEL=error
# how many times to repeat the same experiment
TRIALS=5
# run ID
RUN="$(ls -1v res | tail -1 | cut -d "." -f1 | rev | cut -d "-" -f1 | cut -d "," -f1 | rev)"
RUN=$(($RUN + 1))


printf "Benchmark #$RUN\n"

### KERNEL CONFIG EXPRIMENT

prefix="Exp Kernel Config:"

printf "\n$prefix Benchmarking GPU"
for i in $(seq 1 $TRIALS); do
    printf "\n$prefix Trial #$i\n"
    run_kernel_config "" "res/$RUN.$i.k.hl.gpu.8704.csv"
done

### RENDERING EXPERIMENTS

prefix="Exp Rendering:"

# CPU threads
for i in $(seq 1 $TRIALS); do
    printf "\n$prefix Trial #$i\n"
    for threads in {1,2,4,6,8,10,12,14,16}; do
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\n$prefix Benchmarking CPU with $OMP_NUM_THREADS threads and render\n"
        run_lattice "--cpu -r" "res/$RUN.render.cpu.$OMP_NUM_THREADS.$i.csv"
    done
done

# GPU
printf "\n$prefix Benchmarking GPU\n"
for i in $(seq 1 $TRIALS); do
    printf "\n$prefix Trial #$i\n"
    run_lattice "-r" "res/$RUN.render.gpu.8704.$i.csv"
done

### NEIGHBOURHOOD EXPERIMENTS

printf "\nBenchmarking Neighbourhood Radius\n"
for nh_radius in $(seq 1 5); do
    prefix="Exp Radius $nh_radius:"
    printf $"\n$prefix Recompiling...\n"
    make clean
    make automata NH_RADIUS=$nh_radius -j

    for i in $(seq 1 $TRIALS); do
        printf "\n$prefix Trial #$i\n"

        filename="res/$RUN.nh.$nh_radius.$i.csv"
        echo "hw,threads,size,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $filename

        for threads in {1,2,4,6,8,10,12,14,16}; do
            export OMP_NUM_THREADS=$threads
            export OMP_SCHEDULE=dynamic,1
            printf "\n$prefix Benchmarking CPU with $OMP_NUM_THREADS threads\n"
            echo -n "cpu,$threads,4096," >> $filename
            ./automata -b -x 4096 -y 4096 -p 0.5 --cpu >> $filename
        done

        printf "\n$prefix Benchmarking GPU\n"
        echo -n "gpu,8704,4096," >> $filename
        ./automata -b -x 4096 -y 4096 -p 0.5 >> $filename

    done
done

# reset compilation to defaults
make clean
make automata -j


## LATTICE SIZE EXPERIMENTS

prefix="Exp Lattice:"

# single GPU thread experiment
printf "\n\n$prefix Single GPU Thread\n"
run_lattice "--gpu-blocks 1 --gpu-threads 1" "res/$RUN.1.hl.gpu.1.csv"

# CPU threads
for i in $(seq 1 $TRIALS); do
    printf "\n$prefix Trial #$i\n"
    for threads in {1,2,4,6,8,10,12,14,16}; do
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\n$prefix Benchmarking CPU with $OMP_NUM_THREADS threads\n"
        run_lattice "--cpu" "res/$RUN.lat.hl.cpu.$OMP_NUM_THREADS.$i.csv"
    done
done

# GPU
printf "\n$prefix Benchmarking GPU"
for i in $(seq 1 $TRIALS); do
    printf "\n$prefix Trial #$i\n"
    run_lattice "" "res/$RUN.lat.hl.gpu.8704.$i.csv"
done