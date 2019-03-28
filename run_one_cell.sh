#!/bin/bash -l
#$ -l h_rt=140:00:00
#$ -P ecog-eeg
#$ -N cell_$1


module load python3


echo $1

python3 run_cell.py $1 $1
