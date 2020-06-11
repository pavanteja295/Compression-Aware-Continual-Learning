# dataloading code
import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import torch
from PIL import Image
import os
import pickle
from .dataset_utlis import *

class MiniImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train, transform=None):
        super(MiniImageNet, self).__init__()
        if train:
            self.name='train'
        else:
            self.name='test'
        self.root = os.path.join(root, 'mini-imagenet')
        with open(os.path.join(self.root,f'{self.name}.pkl'), 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')

        # import pdb; pdb.set_trace()
        self.data = data_dict['images']
        self.targets = data_dict['labels']
        self.transform = transform
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.targets[i]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def multidataset(root, train_aug=False):
    # import pdb; pdb.set_trace
    mean_datasets = {
        'CIFAR10': [x/255 for x in [125.3,123.0,113.9]],
        'notMNIST': (0.4254,),
        'MNIST':  (0.1,) ,
        'SVHN':[0.4377,0.4438,0.4728] ,
        'FashionMNIST': (0.2190,),

    }

    std_datasets = {
        'CIFAR10': [x/255 for x in [63.0,62.1,66.7]],
        'notMNIST': (0.4501,),
        'MNIST':  (0.2752,),
        'SVHN': [0.198,0.201,0.197],
        'FashionMNIST': (0.3318,)
    }

    class_datasets = {
        'CIFAR10' : CIFAR10_,
        'notMNIST' : notMNIST_,
        'MNIST' : torchvision.datasets.MNIST,
        'SVHN' : SVHN_,
        'FashionMNIST' : torchvision.datasets.FashionMNIST
    }
    
    datset_keys_ = list(mean_datasets.keys())
    transforms_ = {}

    for key_ in datset_keys_:
        tmp_transforms_ = []    
        if key_ in ['notMNIST', 'MNIST', 'FashionMNIST']:
            tmp_transforms_.append(transforms.Pad(padding=2, fill=0))
        tmp_transforms_.append(transforms.ToTensor())
        tmp_transforms_.append(transforms.Normalize(mean_datasets[key_],std_datasets[key_]))

        if key_ in [ 'MNIST', 'FashionMNIST']:
            tmp_transforms_.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) ))
        transforms_[key_] = transforms.Compose(tmp_transforms_)

    datasets_train = {}
    datasets_test = {}
    for key_ in datset_keys_:
        datasets_train[key_] = CacheClassLabel(class_datasets[key_](root=root, train=True, download=True, target_transform = None, transform=transforms_[key_]))
        datasets_test[key_] = CacheClassLabel(class_datasets[key_](root=root, train=False, download=True, target_transform = None, transform=transforms_[key_]))
    # import pdb; pdb.set_trace()
    print('Loaded all datasets')
    # for key_ in datset_keys_:
    #     print(f'{key_} : Train Images are of shape {datasets_train[key_].data.shape} and Label Set is {set(datasets_train[key_].targets)}')
    #     print(f'{key_} : Test Images are of shape {datasets_test[key_].data.shape} and Label Set is {set(datasets_test[key_].targets)}')

    # train_dataset_splits = {}
    # test_dataset_splits = {}    

    # for idx_, key_ in enumerate(datset_keys_):
    #     train_dataset_splits[str(idx_)]
    return datasets_train, datasets_test




def miniImageNet(root, train_aug=False):

    # no transforms yer
    val_transform = transforms.Compose([
            # transforms.CenterCrop(224),
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(84, padding=4),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # import pdb; pdb.set_trace()
    val_dataset = MiniImageNet(root=root, train=False, transform=val_transform )
    train_dataset = MiniImageNet(root=root, train=True, transform=train_transform )

    train_dataset = CacheClassLabel(train_dataset)
    val_dataset = CacheClassLabel(val_dataset)

    # import pdb; pdb.set_trace()
    return train_dataset, val_dataset
        

def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    
    # import pdb; pdb.set_trace()
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )

    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

