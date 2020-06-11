#!/bin/bash 
#SBATCH --job-name=test_
#SBATCH --ntasks=1 --nodes=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=20G
#SBATCH --time=20:00:00	
#SBATCH --gres=gpu:k80:1
#SBATCH --output=CNet_MultiDataset_large_network_growing_3_runs_report_paper_finalist_last_0.4_again

module load miniconda
source activate CLpavan
for k in  10;
do 
for e in 0.00003;
do
echo $k , $e
echo CNet_large_cifar_100_tasks_${k}_spt_wt_0.4_energy_${e}
python iBatchLearn.py  -e $e --model_name Net_SVD --model_type customnet_SVD --exp_name MultiDataset_final --first_split_size 10 --other_split_size 10  --train_aug  --schedule 120 180 200 --batch_size 64 --dataset multidataset --force_out_dim 0  --sparse_wt  0.1  --train_aug   --grow_network --benchmark --rand_split_order --repeat 1

done 
done
