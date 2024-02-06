#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:10
#SBATCH --mem=100M
#SBATCH --account=cosc016682

cd "${SLURM_SUBMIT_DIR}"
echo Started
module add lang/python/anaconda/3.7.7-2020-R-3.6.1

python BPPref.py 
echo Ended


