#!/bin/bash

# conda activate opensim-tf2

for i in {1..200}
do
    echo "Run number $i:"
    # timeout 45m python main.py opensim -n "test_nr_20" --agents=30
    python main.py opensim -n "test_nr_20" --agents=30
done
