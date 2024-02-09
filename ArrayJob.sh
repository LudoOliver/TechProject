#!/bin/bash

#SBATCH --job-name=array_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:10
#SBATCH --mem=500M
#SBATCH --account=cosc016682
#SBATCH --array=1-5

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

echo Started
module add lang/python/anaconda/3.7.7-2020-R-3.6.1

python BPPref.py SLURM_ARRAY_TASK_ID 
echo Ended


