#!/bin/bash

# conda activate opensim-tf2

for i in {1..200}
do
    echo "Run number $i:"
    python main.py opensim -n "test_nr_14" --agents=30
done
