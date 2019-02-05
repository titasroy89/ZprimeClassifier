#!/bin/bash
#SBATCH --time    0-01:00:00
#SBATCH --nodes   1
#SBATCH --constraint GPU
#SBATCH --partition all
#SBATCH --array 1-144
#SBATCH --job-name DNN-scan
#SBATCH --workdir   /beegfs/desy/user/reimersa/ZprimeClassifier/workdir/output
#SBATCH --output    steer-%N-%j.out
#SBATCH --error     steer-%N-%j.err
# export LD_PRELOAD=""
# source /etc/profile.d/modules.sh

source ~/.setenv
cd /beegfs/desy/user/reimersa/ZprimeClassifier
./steer_array.py $SLURM_ARRAY_TASK_ID
