#!/bin/bash

# Define the batch sizes
batch_sizes=(64 128 256 512 1024 2048)

# Loop through each batch size and run the command
for batch_size in "${batch_sizes[@]}"
do
    echo "Running with batch size $batch_size"
    ./ch train jsc-tiny jsc --max-epochs 10 --batch-size $batch_size
done
