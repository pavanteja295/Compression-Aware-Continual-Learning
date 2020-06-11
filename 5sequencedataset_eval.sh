python iBatchLearn.py  -e 0.00003 --model_name Net_SVD --model_type customnet_SVD --exp_name MultiDataset_final --first_split_size 10 --other_split_size 10  --train_aug  --schedule 120 180 200 --batch_size 64 --dataset multidataset --force_out_dim 0  --sparse_wt  0.1  --train_aug   --grow_network --benchmark --rand_split_order --repeat 3

