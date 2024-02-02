#!/bin/bash

# Define the learning rates
lrs=(1e-06 1e-05 1e-04 1e-03 1e-02)
DIR=/home/mp1820/mase/mase_output/jsc-tiny_classification_jsc_lr

# Loop through each learning rate and run the command
for lr in "${lrs[@]}"
do
    echo "Running with learning rate $lr"
    ./ch train jsc-tiny jsc --max-epochs 10 --batch-size 256 --learning-rate $lr --project-dir "$DIR"
done
