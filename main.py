#!/usr/bin/env python

import torch
from torchvision import datasets, models, transforms

from config import Config as config

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(
                    config.data_dir,
                    data_transforms["train"],
                    )
train_data_loader=torch.utils.data.DataLoader(
                train_dataset,
                32,
                True,
                num_workers=2)

for item in train_data_loader:
    print(item[0].shape)
    break
