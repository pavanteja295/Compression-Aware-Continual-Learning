#!/bin/bash 
#SBATCH --job-name=CNet_large_cifar_100_10_10_no_dropout_fixed_wd_only_0.4
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=CNet_large_cifar_100_10_10_no_dropout_fixed_wd_only

module load miniconda
source activate CLpavan
python iBatchLearn.py  --exp_name 'CNet_large_cifar_100_10_10_no_dropout_fixed_wd_only' -e 3e-5 --model_name Net  --model_type  customnet_SVD --dataset CIFAR100 --train_aug --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 110 140 180 --sparse_wt 0.4