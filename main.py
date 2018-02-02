#!/usr/bin/env python

import torch
from torchvision import datasets, models, transforms

from config import Config as config
from data_loaders import train_data_loader

for images, labels in train_data_loader:
    print(images.shape)
    print(labels)
    break
