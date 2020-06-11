#!/bin/bash 
#SBATCH --job-name=WR_CIF100_BN_ON_same_params
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=CNet_cifar_100_10_10_same_params_new_lin_search_drop_out_off_newest

module load miniconda
source activate CLpavan
python iBatchLearn.py  --exp_name 'CNet_cifar_100_10_10_same_params_new_lin_search_drop_out_off_newest' -e 3e-5 --model_name Net  --model_type  customnet_SVD --dataset CIFAR100 --train_aug --force_out_dim 0 --first_split_size 10 --other_split_size 10 --schedule 80 120 160