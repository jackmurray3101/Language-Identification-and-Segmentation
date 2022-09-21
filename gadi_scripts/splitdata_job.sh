#!/bin/bash
 
#PBS -P wa66
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=8GB
#PBS -l jobfs=8GB
#PBS -l walltime=01:30:00
#PBS -l wd
#PBS -l storage=gdata/wa66
 
# Load module, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 splitdata_v2.py > $PBS_JOBID.log
