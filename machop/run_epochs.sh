#!/bin/bash

# Define the epoch sizes
epoch_sizes=(5 10 15 20 25)

# Loop through each batch size and run the command
for epoch_size in "${epoch_sizes[@]}"
do
    echo "Running with batch size $epoch_size"
    ./ch train jsc-tiny jsc --max-epochs $epoch_size --batch-size 256 --project-dir /home/mp1820/mase/mase_output/jsc-tiny_classification_jsc_epochs
done
