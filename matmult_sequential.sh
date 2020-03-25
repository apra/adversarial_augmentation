#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run collect on a decidated server in the hpcintro
# queue
#
#
#BSUB -J matmult_gpu8
#BSUB -o matmult_gpu8%J.out
#BSUB -e matmult_gpu8%J.err
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -W 20

module load gcc/8.3.0
module load cuda/10.2
module load python3/3.7.5

pip3.7 install torch torchvision --user

SCRIPT=transfer_model.py
NS="512 1024 2048 4096 8192"
NS="300"
ITER="100"
TOLERANCE="0"
START_T="10"
    # start the command with the above settings
./python3 $SCRIPT
