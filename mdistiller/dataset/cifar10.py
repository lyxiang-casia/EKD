import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
import torch

def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

def worker_init_fn(worker_id):
    seed = os.environ.get('PYTHONHASHSEED')
    np.random.seed(int(seed) + worker_id)  
    random.seed(int(seed) + worker_id)

def get_cifar10_dataloaders(batch_size, val_batch_size, num_workers):

    data_folder = get_data_folder()
    # train_transform = get_cifar100_train_transform()
    # test_transform = get_cifar100_test_transform()
    # train_set = CIFAR100Instance(
    #     root=data_folder, download=True, train=True, transform=train_transform
    # )
    # num_data = len(train_set)
    # test_set = datasets.CIFAR100(
    #     root=data_folder, download=True, train=False, transform=test_transform
    # )
    image_transforms=[transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.RandomRotation(degrees=15)]
    train_set =  CIFAR10Instance(
        data_folder, download=True, train=True,
        transform=transforms.Compose(image_transforms + [transforms.ToTensor()])
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR10(
        data_folder, download=True, train=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    seed = os.environ.get('PYTHONHASHSEED')
    if seed:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )

    return train_loader, test_loader, num_data