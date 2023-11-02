#!/bin/bash
#SBATCH --job-name=fly_exp_test_clf
#SBATCH --output=slurm_out/fly_exp_test_clf-%A.%a.out
#SBATCH --error=slurm_out/fly_exp_test_clf-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --partition=mig
#SBATCH --mail-type=all
#SBATCH --mail-user=rkp16384@gmail.com

echo "SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID."
echo "Executing on the machine:" $(hostname)

module purge

module load anaconda3/2023.3
module load cudnn/cuda-11.x/8.2.0

conda activate s5-gpu

python run_train.py --C_init=trunc_standard_normal --batchnorm=True --bidirectional=True \
                    --blocks=16 --bsz=32 --d_model=8 --dataset=fly-courtship-clf \
                    --dt_global=True --epochs=40 --jax_seed=5464368 --lr_factor=2 --n_layers=2 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.001 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.05
