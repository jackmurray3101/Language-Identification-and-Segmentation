#!/bin/bash
 
#PBS -P wa66
#PBS -q normal
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l jobfs=10GB
#PBS -l walltime=00:10:00
#PBS -l wd
#PBS -l storage=gdata/wa66

# Load modules, always specify version number.
module load python3/3.9.2
 
# Run Python applications
python3 plot_accuracy.py completed_jobs/58364391.gadi-pbs.log 10 > $PBS_JOBID.log
