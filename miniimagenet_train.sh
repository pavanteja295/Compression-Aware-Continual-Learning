#!/bin/bash 
#SBATCH --job-name=test_
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:v100:1
#SBATCH --output=CNet_ImageNet_large_network_growing_3_runs_report_paper_fixed_network_finalest_one_lasttime

module load miniconda
source activate CLpavan
python iBatchLearn.py  -e 0.00003 --model_name Net_SVD --model_type customnet_SVD --exp_name miniImageNet_train --first_split_size 5 --other_split_size 5  --train_aug  --schedule 150 180 200 --batch_size 64 --dataset miniImageNet --force_out_dim 0  --sparse_wt 0.1  --train_aug    --benchmark --rand_split_order --repeat 3 --grow_network
