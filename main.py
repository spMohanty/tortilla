#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss


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
Initialize Optimizers, Loss, Loss Schedulers
"""
optimizer_ft = optim.Adam(net.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = CrossEntropyLoss()

"""
Train
"""
for epoch in range(config.epochs):
    running_loss = 0.0
    running_corrects = 0

    for _idx, (images, labels) in enumerate(train_data_loader):

        """
            Prepare Data
        """
        if use_gpu:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images)
            labels = Variable(labels)

        # Predict
        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)

        # Compute Loss
        loss = criterion(outputs, labels)

        #Prepare and print stats
        batch_corrects = torch.sum(preds == labels.data)
        batch_accuracy = batch_corrects*1.0/config.batch_size

        running_loss += loss.data[0] * images.size(0)
        running_corrects += batch_corrects
        if _idx % 10 == 0:
            print("Epoch : %d Iteration %d Accuracy : %.3f Loss %.3f" % (epoch, _idx, batch_accuracy, loss.data[0]))

        # Backpropagate
        net.zero_grad()
        loss.backward()


        #Update weights
        optimizer_ft.step()
