#!/bin/bash
 
#PBS -P wa66
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=16GB
#PBS -l jobfs=10GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=gdata/wa66
#PBS -l ngpus=1
 
# Load modules, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 train.py > $PBS_JOBID.log