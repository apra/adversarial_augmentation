import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

# ## ----   ARGS   ---- ##
dataset_name = "cifar_10"
batch_size = 4
num_workers = 4
data_dir = "{}/{}".format("data", dataset_name)
model_name = "resnet18"
# ## ---- END ARGS ---- ##

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

datasets = {
    "cifar_10": {
        "train": datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=data_transforms["train"]
        ),
        "val": datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=data_transforms["val"]
        )
    }
}

dataloaders = {
    "cifar_10": {
        "train": torch.utils.data.DataLoader(datasets["cifar_10"]["train"], batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers),
        "val": torch.utils.data.DataLoader(datasets["cifar_10"]["val"], batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)

    }
}

dataset_sizes = {cur_name: {x: len(datasets[cur_name][x]) for x in ['train', 'val']} for cur_name in
                 datasets.keys()}

class_names = {cur_name: datasets[cur_name]['train'].classes for cur_name in datasets.keys()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

selected_dataset = datasets[dataset_name]
selected_dataloader = dataloaders[dataset_name]
selected_class_names = class_names[dataset_name]
selected_dataset_sizes = dataset_sizes[dataset_name]