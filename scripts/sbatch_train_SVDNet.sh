#!/bin/bash 
#SBATCH --job-name=test_
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=30G
#SBATCH --time=12:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=train_SVDNet

module load miniconda
source activate CLpavan
python iBatchLearn.py --schedule 80 120 160 --exp_name 'SVDNet_BL_0.5_reg' -e 3e-6