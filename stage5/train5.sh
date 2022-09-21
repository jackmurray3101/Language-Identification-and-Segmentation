#!/bin/bash
 
#PBS -P wa66
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l jobfs=10GB
#PBS -l walltime=04:00:00
#PBS -l wd
#PBS -l storage=gdata/wa66
#PBS -l ngpus=1

# Load modules, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 train.py config5.json > $PBS_JOBID.log
