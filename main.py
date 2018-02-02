#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms


from config import Config as config
from data_loaders import train_dataset, train_data_loader

"""
Initliaze params
"""
class_names = train_dataset.classes
use_gpu = torch.cuda.is_available()

"""
Initialize Model
"""
net = models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(class_names))
# Make net use parallel gpu
if use_gpu:
    net = nn.DataParallel(net).cuda()

"""
Initialize Optimizers and Schedulers
"""
optimizer_ft = optim.Adam(net.parameters(), lr=0.001)

for _idx, (images, labels) in enumerate(train_data_loader):
    print(images.shape)
    print(labels)

    images = Variable(images)
    labels = Variable(labels)

    output = net(images)
    print(output)
    print(output.shape)
