#!/bin/bash
#SBATCH --time    0-12:00:00
#SBATCH --nodes   1
#SBATCH --constraint GPU
#SBATCH --partition all
#SBATCH --array 1-3
#SBATCH --job-name DNN-scan
#SBATCH --workdir   /beegfs/desy/user/reimersa/ZprimeClassifier/workdir/output
#SBATCH --output    steer-%N-%j.out
#SBATCH --error     steer-%N-%j.err
# export LD_PRELOAD=""
# source /etc/profile.d/modules.sh

source ~/.setenv
cd /beegfs/desy/user/reimersa/ZprimeClassifier
./steer_array.py $SLURM_ARRAY_TASK_ID
