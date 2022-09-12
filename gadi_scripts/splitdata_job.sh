#!/bin/bash
 
#PBS -P wa66
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=8GB
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -l storage=gdata/wa66/jm2369
 
# Load module, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 splitdata.py > $PBS_JOBID.log