#!/bin/bash
#
#SBATCH --job-name=matrixMult
#SBATCH --output=mult.out
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1
 
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD:LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=0

./mult matriz1.txt matriz2.txt

