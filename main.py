	#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import CrossEntropyLoss

from config import Config as config
from data_loaders import TortillaDataset
from trainer import TortillaTrainer

from monitor import TortillaMonitor

"""
Initliaze params
"""
use_gpu = torch.cuda.is_available()

def main():
	"""
	Initialize Dataset
	"""
	dataset = TortillaDataset(	"datasets/food-101",
								batch_size=config.batch_size,
								num_cpu_workers=10
								)

	"""
	Initialize Model
	"""
	net = models.resnet50(pretrained=True)
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, len(dataset.classes))

	# Make net use parallel gpu
	if use_gpu:
		net = nn.DataParallel(net).cuda()

	"""
	Initialize Optimizers, Loss, Loss Schedulers
	"""
	optimizer_ft = optim.Adam(net.parameters(), lr=0.001)
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
	criterion = CrossEntropyLoss()

	monitor = TortillaMonitor(	experiment_name="test-food-101",
								topk=config.topk,
								classes=dataset.classes,
								use_gpu = use_gpu,
								plot=True,
								config=config
								)
	"""
	Train
	"""
	trainer = TortillaTrainer(
				dataset = dataset,
				model = net,
				loss = criterion,
				optimizer = optimizer_ft,
				monitor = monitor,
				config=config
				)

	for epoch in range(config.epochs):
		for train in [False, True]:
			end_of_epoch = False
			while not end_of_epoch:
				_loss, images, labels, \
				outputs, end_of_epoch = trainer._step(use_gpu=use_gpu, train=train)
				if end_of_epoch:
					break
				# print(epoch+trainer.dataset.get_current_pointer(train=True), _loss, images.shape)


if __name__ == "__main__":
	main()
