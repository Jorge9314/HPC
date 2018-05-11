#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --job-name=matrizCuda
#SBATCH --output=res-matrizCuda.out
#SBATCH --account=cgiraldo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=200

for a in {1..10}
do
  ./matrizSh m1 m2
done