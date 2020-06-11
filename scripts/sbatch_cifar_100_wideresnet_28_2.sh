#!/bin/bash 
#SBATCH --job-name=WR_CIF100_BN_ON
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=Wide_resnet_28_2_cifar_100_bn_on

module load miniconda
source activate CLpavan
python iBatchLearn.py  --exp_name 'Wide_resnet_28_2_cifar_100_bn_on' -e 3e-5 --model_name WideResNet_28_2_cifar --model_type resnet --dataset CIFAR100 --train_aug --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --save_running_stats