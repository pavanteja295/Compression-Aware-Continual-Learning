# Compression-aware Continual Learning using SVD
This repository is the official implementation of Compression-aware Continual Learning using SVD.
![Pipeline of our approach](/images/neurips22.png)
## Requirements

To install requirements you need to create an anaconda environment using the following code snippet:

```setup
conda create --name <env> --file anaconda_requirements.txt
pip install -r requirements.txt
```

## Datasets
All datasets except notMNIST and miniImageNet are downloaded from the torchvision.datasets
1. notMNIST is by default downloaded from [Adversarial Continual Learning](https://github.com/facebookresearch/Adversarial-Continual-Learning/tree/master/data)
2. Please download miniImageNet from https://www.dropbox.com/s/zuyqhk290gpf1hm/miniimagenet.zip?dl=0 and unzip the train.pkl and test.pkl into to a new folder data/mini-imagenet

We provide pretrained models for CIFAR-100, miniImageNet, 5-sequence dataset. To evaluate trained model use:



## Pretrained Models

We provide pretrained models for CIFAR-100, miniImageNet, 5-sequence dataset. To evaluate trained model use:

### CIFAR-100
```
./cifar100_eval.sh
```

| Model name         | Accuracy  | Model Size(MB) |
| ------------------ |---------------- | -------------- |
| CACL_Final   |     86.58%         |      8.53       |
### miniImageNet
```
./miniimagenet_eval.sh
```
| Model name         | Accuracy  | Model Size(MB) |
| ------------------ |---------------- | -------------- |
| CACL_Final   |     70.10%         |     13.03       |
### 5-sequence dataset

```
./5sequencedataset_eval.sh
```
## Training

To train the model(s) from scratch, run the following scripts. The scripts contain hyper-parameter details used to obtain the results in this paper:
### CIFAR-100
```
./cifar100_train.sh
```
### miniImageNet

```
./miniimagenet_train.sh
```
### 5-sequence dataset

```
./5sequencedataset_train.sh
```
### Other datasets
```
python iBatchLearn.py  -e <pruning_intensity> --model_name <Net_SVD/vgg16_bn_cifar100_SVD> --model_type <customnet_SVD/vgg_16_bn> --exp_name <exp_name> --first_split_size <classes per task> --other_split_size <classes per task>  --train_aug  --schedule <lrdropepochs> --batch_size <64/128> --dataset <CIFAR100/miniImageNet/multidataset> --force_out_dim 0  --sparse_wt <sparsity weight>  --benchmark <fixatesrandomseed> --rand_split_order --repeat 3

```
