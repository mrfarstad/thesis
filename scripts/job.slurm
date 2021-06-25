#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:P100:1
#SBATCH --time=5:00:00
#SBATCH --mem=125G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="martinrf/thesis"
#SBATCH --output="thesis_output.out"

source /etc/profile.d/modules.sh
module load CUDA/10.1.243-GCC-8.3.0
module load Python/2.7.16-GCCcore-8.3.0

cd thesis
python3 ./scripts/evaluate_stencil_depths.py False False
