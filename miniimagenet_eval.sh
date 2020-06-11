python iBatchLearn.py  -e 0.00003 --model_name Net_SVD --model_type customnet_SVD --loadmodel miniImageNet_final_sparse_wt0.1 --first_split_size 5 --other_split_size 5  --train_aug  --schedule 150 180 200 --batch_size 64 --dataset miniImageNet --force_out_dim 0  --sparse_wt 0.1  --train_aug    --benchmark --rand_split_order --repeat 1 --grow_network

