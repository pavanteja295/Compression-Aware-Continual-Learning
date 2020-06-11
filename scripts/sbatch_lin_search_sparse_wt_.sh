#!/bin/bash 
#SBATCH --job-name=lin_search
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=ResNet20SVDNet_bn_track_off_sp_wt_search

module load miniconda
source activate CLpavan
python iBatchLearn.py --schedule 80 120 160 --exp_name 'ResNet20SVDNet_bn_track_off_sp_wt_search' -e 3e-5 