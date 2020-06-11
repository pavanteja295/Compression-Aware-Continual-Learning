#!/bin/bash 
#SBATCH --job-name=lin_search_bn_on
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=ResNet20SVDNet_bn_tracking_on_lin_search

module load miniconda
source activate CLpavan
python iBatchLearn.py --schedule 80 120 160 --exp_name 'ResNet20SVDNet_bn_tracking_on_lin_search' -e 3e-5  --save_running_stats