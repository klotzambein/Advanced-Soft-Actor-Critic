#!/bin/bash

# conda activate opensim-tf2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robin/.conda/envs/rug-osim-torch/lib/:/opt/anaconda/lib/
for i in {1..200}
do
    echo "Run number $i:"
    # timeout 45m python main.py opensim -n "test_nr_20" --agents=30
    python main.py opensim -n "test_nr_20" --agents=30
done
