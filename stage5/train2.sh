#!/bin/bash
 
#PBS -P wa66
#PBS -q gpuvolta
#PBS -l ncpus=24
#PBS -l mem=16GB
#PBS -l jobfs=10GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=gdata/wa66
#PBS -l ngpus=2

# Load modules, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 train.py config2.json > $PBS_JOBID.log
