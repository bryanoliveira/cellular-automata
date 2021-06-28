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
RUN="$(ls -1v res | tail -1 | cut -d "." -f1)"
RUN=$(($RUN + 1))


printf "Benchmark #$RUN\n"

### KERNEL CONFIG EXPRIMENT

printf "\nBenchmarking GPU for kernel config"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run_kernel_config "" "res/$RUN.$i.k.hl.gpu.8704.csv"
done

### LATTICE SIZE EXPERIMENTS

# single GPU thread experiment
run_lattice "--gpu-blocks 1 --gpu-threads 1" "res/$RUN.1.hl.gpu.1.csv"

# CPU threads
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    for threads in {1,2,4,8,12,16}; do
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads\n"
        run_lattice "--cpu" "res/$RUN.$i.hl.cpu.$OMP_NUM_THREADS.csv"
    done
done

# GPU
printf "\nBenchmarking GPU"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run_lattice "" "res/$RUN.$i.hl.gpu.8704.csv"
done

### RENDERING EXPERIMENTS

# CPU threads
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    for threads in {16,}; do # 1,2,4,8,12,
        export OMP_NUM_THREADS=$threads
        export OMP_SCHEDULE=dynamic,1
        printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads and render\n"
        run_lattice "--cpu -r" "res/$RUN.$i.render.cpu.$OMP_NUM_THREADS.csv"
    done
done

# GPU
printf "\nBenchmarking GPU with render\n"
for i in $(seq 1 $TRIALS); do
    printf "\nTrial #$i\n"
    run_lattice "-r" "res/$RUN.$i.render.gpu.8704.csv"
done

### NEIGHBOURHOOD EXPERIMENTS

printf "\nBenchmarking Neighbourhood Radius\n"
for nh_radius in $(seq 1 5); do
    printf "\nRadius: $nh_radius\nRecompiling...\n"
    make clean
    make automata NH_RADIUS=$nh_radius

    for i in $(seq 1 $TRIALS); do
        printf "\nTrial #$i\n"

        filename="res/$RUN.$i.nh.$nh_radius.csv"
        echo "hw,threads,size,iterations,loadTime,totalEvolveTime,totalBufferTime,avgEvolveTime,avgBufferTime" > $filename

        for threads in {1,2,4,8,12,16}; do
            export OMP_NUM_THREADS=$threads
            export OMP_SCHEDULE=dynamic,1
            printf "\nBenchmarking CPU with $OMP_NUM_THREADS threads\n"
            echo -n "cpu,$threads,4096," >> $filename
            ./automata -b -x 4096 -y 4096 -p 0.5 --cpu >> $filename
        done

        echo -n "gpu,8704,4096," >> $filename
        ./automata -b -x 4096 -y 4096 -p 0.5 >> $filename

    done
done