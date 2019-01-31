#!/bin/bash -l

module load python3


echo $1

python3 run_cell.py $1 $1
