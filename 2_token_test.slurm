#!/bin/bash
#SBATCH --job-name=2_token_test
#SBATCH --output=slurm_out/2_token_test_%A.%a.out
#SBATCH --error=slurm_out/2_token_test_%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --partition=mig
#SBATCH --mail-type=all

export HF_DATASETS_CACHE="/scratch/gpfs/rp21/cache_hf"

echo "SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID."
echo "Executing on the machine:" $(hostname)

module purge

module load anaconda3/2023.3
module load cudnn/cuda-11.x/8.2.0

conda activate s5-gpu-test

python 2_token_test.py
