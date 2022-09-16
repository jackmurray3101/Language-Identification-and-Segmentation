#!/bin/bash
 
#PBS -P wa66
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l jobfs=40GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=gdata/wa66
#PBS -l ngpus=1
 
# Load modules, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 train.py > $PBS_JOBID.log
