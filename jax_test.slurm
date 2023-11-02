#!/bin/bash
#SBATCH --job-name=jax_test
#SBATCH --output=slurm_out/slurm-%A.%a.out
#SBATCH --error=slurm_out/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --array=0-0
#SBATCH --mail-type=all
#SBATCH --mail-user=rkp16384@gmail.com

echo "SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID."
echo "Executing on the machine:" $(hostname)

module purge

module load anaconda3/2023.3
module load cudnn/cuda-11.x/8.2.0

conda activate s5-gpu

python svd.py